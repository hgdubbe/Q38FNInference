package httpapi

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"regexp"
	"strconv"
	"time"

	"github.com/hgdubbe/q38fninference/internal/appconfig"
	"github.com/hgdubbe/q38fninference/internal/gpu"
	"github.com/hgdubbe/q38fninference/internal/tuning"
)

// After a model loads, llama-server has logged exactly what it allocated on
// each GPU. If that is more than the GPU had free, Windows doesn't fail the
// allocation: it moves the excess to shared (system) memory, and generation
// drops to a fraction of its speed with no error anywhere. So the launcher
// compares the log against the free memory it saw at start, remembers the
// overshoot per model and GPU, and plans with it from then on.

// cudaBufferRe matches llama.cpp's per-device buffer lines, e.g.
// "load_tensors:        CUDA0 model buffer size =  9855.23 MiB" or
// "sched_reserve:      CUDA1 compute buffer size =   812.02 MiB".
// CUDA_Host (pinned RAM) and CPU lines don't match.
var cudaBufferRe = regexp.MustCompile(`\bCUDA(\d+) +(?:model|KV|RS|compute) buffer size *= *([\d.]+) MiB`)

// cudaBufferBytes sums llama.cpp's buffers per CUDA device ordinal (the
// order of CUDA_VISIBLE_DEVICES, i.e. the plan's device order).
func cudaBufferBytes(lines []string) map[int]uint64 {
	out := map[int]uint64{}
	for _, l := range lines {
		m := cudaBufferRe.FindStringSubmatch(l)
		if m == nil {
			continue
		}
		dev, _ := strconv.Atoi(m[1])
		mib, err := strconv.ParseFloat(m[2], 64)
		if err != nil {
			continue
		}
		out[dev] += uint64(mib * (1 << 20))
	}
	return out
}

// cudaContextBytes is llama.cpp's device memory outside the buffers it logs:
// the CUDA context, cuBLAS workspace, graphs. Matches the planner's figure.
const cudaContextBytes = 512 << 20

// launchInfo is what a VRAM check needs to know about the running load.
type launchInfo struct {
	modelPath  string
	devices    []tuning.GPU // in CUDA ordinal order, FreeBytes as seen at start
	auto       bool         // args were the plan's own, so a re-plan can replace them
	recheck    bool         // this start already is the corrected re-plan
	limited    bool         // the plan keeps experts in RAM: VRAM limits it
	hotProfile string       // hot experts' profile, "off" without
	target     *url.URL     // llama-server's address (under Server.mu)
}

// vramOverflow reports, per GPU index, how far a load went past the memory
// that GPU had free.
func vramOverflow(li launchInfo, used map[int]uint64) map[int]uint64 {
	over := map[int]uint64{}
	for ord, g := range li.devices {
		need := used[ord] + cudaContextBytes
		if g.FreeBytes > 0 && need > g.FreeBytes {
			over[g.Index] = need - g.FreeBytes
		}
	}
	return over
}

// learnVRAMOverflow adds the overshoot (plus headroom) to the model's saved
// corrections and describes what happened.
func (s *Server) learnVRAMOverflow(li launchInfo, over map[int]uint64) string {
	const headroom = 256 << 20
	s.mu.Lock()
	if s.cfg.VRAMCorrections == nil {
		s.cfg.VRAMCorrections = map[string]int64{}
	}
	msg := ""
	for _, g := range li.devices {
		o, ok := over[g.Index]
		if !ok {
			continue
		}
		k := appconfig.VRAMCorrectionKey(li.modelPath, g.Index)
		// an earlier reduction was too much: drop it, then add the overshoot
		s.cfg.VRAMCorrections[k] = max(s.cfg.VRAMCorrections[k], 0) + int64(o+headroom)
		if msg != "" {
			msg += ", "
		}
		msg += fmt.Sprintf("%s by %.1f GiB", g.Name, float64(o)/(1<<30))
	}
	cfg := s.cfg
	s.mu.Unlock()
	if err := cfg.Save(); err != nil {
		msg += fmt.Sprintf(" (could not save the correction: %v)", err)
	}
	return msg
}

// extraReserve is the learned VRAM correction for a model, by GPU index.
func (s *Server) extraReserve(modelPath string, gpus []tuning.GPU) map[int]int64 {
	cfg := s.Config()
	out := map[int]int64{}
	for _, g := range gpus {
		if v := cfg.VRAMCorrections[appconfig.VRAMCorrectionKey(modelPath, g.Index)]; v != 0 {
			out[g.Index] = v
		}
	}
	return out
}

// The planner's per-GPU reserves (CUDA context, compute buffers, safety) are
// estimates made before anything is loaded; on real cards they can leave
// gigabytes unused. After a load whose plan keeps experts in RAM, the memory
// still free on each GPU is measured: above slackThreshold, the plan was too
// careful, and the next plan may use all but slackKeep of it.
const (
	slackThreshold = 768 << 20
	slackKeep      = 512 << 20 // for allocations made on the first requests, and the desktop
)

// vramSlack is, per GPU index, how much more a plan could use, from the
// memory free now.
func vramSlack(li launchInfo, now []tuning.GPU) map[int]uint64 {
	slack := map[int]uint64{}
	for _, g := range li.devices {
		for _, n := range now {
			if n.Index == g.Index && n.FreeBytes > slackThreshold {
				slack[g.Index] = n.FreeBytes - slackKeep
			}
		}
	}
	return slack
}

// learnVRAMSlack lowers the model's saved corrections by the unused memory
// and describes it.
func (s *Server) learnVRAMSlack(li launchInfo, slack map[int]uint64) string {
	s.mu.Lock()
	if s.cfg.VRAMCorrections == nil {
		s.cfg.VRAMCorrections = map[string]int64{}
	}
	msg := ""
	for _, g := range li.devices {
		sl, ok := slack[g.Index]
		if !ok {
			continue
		}
		s.cfg.VRAMCorrections[appconfig.VRAMCorrectionKey(li.modelPath, g.Index)] -= int64(sl)
		if msg != "" {
			msg += ", "
		}
		msg += fmt.Sprintf("%.1f GiB on %s", float64(sl)/(1<<30), g.Name)
	}
	cfg := s.cfg
	s.mu.Unlock()
	if err := cfg.Save(); err != nil {
		msg += fmt.Sprintf(" (could not save: %v)", err)
	}
	return msg
}

func (s *Server) setNotice(msg string) {
	s.mu.Lock()
	s.notice = msg
	s.noticeID++
	s.mu.Unlock()
}

// loadLogs is the llama-server output of the current load: the lines after
// the last "starting llama-server" note.
func loadLogs(lines []string) []string {
	for i := len(lines) - 1; i >= 0; i-- {
		if lines[i] == "[launcher] starting llama-server" {
			return lines[i+1:]
		}
	}
	return lines
}

// checkVRAM runs once a load is ready. It returns true when it replaced the
// load with a corrected one (the caller then stops watching the old PID).
func (s *Server) checkVRAM(pid int, li launchInfo) bool {
	over := vramOverflow(li, cudaBufferBytes(loadLogs(s.llama.Logs())))
	if len(over) == 0 {
		return s.checkVRAMSlack(pid, li)
	}
	what := s.learnVRAMOverflow(li, over)
	if !li.auto || li.recheck {
		msg := "GPU memory overflowed (" + what + "): the excess runs from shared system memory, which is much slower. " +
			"The plan has been corrected; Stop and Start to use it."
		if !li.auto {
			msg = "GPU memory overflowed (" + what + "), so part of the model runs from shared system memory, which is much slower. " +
				"Your edited launch arguments were kept; press Recalculate and restart to use the corrected plan."
		}
		s.llama.Note(msg)
		s.setNotice(msg)
		return false
	}

	msg := "GPU memory overflowed (" + what + "); restarting with a corrected plan so nothing runs from shared memory."
	s.llama.Note(msg)
	s.setNotice(msg)
	return s.restartWithNewPlan(pid, li)
}

// checkVRAMSlack runs after a load that didn't overflow: when memory a
// RAM-limited plan could use is left free, it is learned and (for the plan's
// own arguments, once) the model restarts with the fuller plan.
func (s *Server) checkVRAMSlack(pid int, li launchInfo) bool {
	if !li.limited || s.autoProfiling() {
		return false
	}
	now, err := gpu.Detect()
	if err != nil {
		return false
	}
	slack := vramSlack(li, now)
	if len(slack) == 0 {
		return false
	}
	what := s.learnVRAMSlack(li, slack)
	if !li.auto || li.recheck {
		msg := "GPU memory left unused after loading (" + what + "); the next start puts more of the model on the GPU."
		s.llama.Note(msg)
		s.setNotice(msg)
		return false
	}
	msg := "GPU memory left unused after loading (" + what + "); restarting once with more of the model on the GPU."
	s.llama.Note(msg)
	s.setNotice(msg)
	return s.restartWithNewPlan(pid, li)
}

// restartWithNewPlan replaces a load with a fresh plan after a correction.
func (s *Server) restartWithNewPlan(pid int, li launchInfo) bool {
	plan, err := s.tune(li.modelPath)
	if err != nil {
		log.Printf("httpapi: re-plan: %v", err)
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if st := s.llama.Status(); st.PID != pid {
		return true // the user already stopped or restarted it
	}
	if err := s.llama.StopWithTimeout(ctx); err != nil {
		log.Printf("httpapi: stopping for re-plan: %v", err)
		return false
	}
	if _, err := s.launch(li.modelPath, plan.Args, plan.Plan.Devices, true, true); err != nil {
		s.llama.Note("could not restart: " + err.Error())
		s.setNotice("Restart with the corrected plan failed: " + err.Error())
	}
	return true
}
