package httpapi

// Expert usage recording (patches/0009, LLAMA_EXPERT_STATS). With the
// "Record expert usage" model setting, each launch writes one run file of
// per-layer, per-expert routing counts; the Run page sums a model's runs to
// show whether a few experts per layer take most of the tokens, i.e. whether
// keeping just those in VRAM would pay off.

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/hgdubbe/q38fninference/internal/appconfig"
	"github.com/hgdubbe/q38fninference/internal/gguf"
)

type expertRun struct {
	NExpert      int    `json:"n_expert"`
	GenTokens    uint64 `json:"gen_tokens"`
	PromptTokens uint64 `json:"prompt_tokens"`
	Layers       []struct {
		Layer  int      `json:"layer"`
		Gen    []uint64 `json:"gen"`
		Prompt []uint64 `json:"prompt"`
	} `json:"layers"`
}

// expertRunMeta is written next to a run file at launch: which blocks kept
// their experts in RAM (blocks 0..RAMLayers-1).
type expertRunMeta struct {
	RAMLayers int `json:"ram_layers"`
}

func expertStatsDir(modelPath string) (string, error) {
	dir, err := appconfig.Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "expert-stats", modelName(modelPath)), nil
}

// expertStatsEnv prepares a new run file for a launch and returns the
// environment variable pointing llama-server at it.
func expertStatsEnv(modelPath string, args []string) (string, error) {
	dir, err := expertStatsDir(modelPath)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	base := filepath.Join(dir, "run-"+strconv.FormatInt(time.Now().UnixNano(), 10))
	meta, _ := json.Marshal(expertRunMeta{RAMLayers: ramExpertLayers(modelPath, args)})
	if err := os.WriteFile(base+".meta", meta, 0o644); err != nil {
		return "", err
	}
	return "LLAMA_EXPERT_STATS=" + base + ".json", nil
}

// ramExpertLayers counts the blocks whose experts a command line leaves in
// RAM: the blocks -ngl doesn't offload, or the first --n-cpu-moe.
func ramExpertLayers(modelPath string, args []string) int {
	meta, err := gguf.ReadModel(modelPath)
	if err != nil {
		return 0
	}
	n, ok := meta.NLayer()
	if !ok {
		return 0
	}
	nLayer := int(n)
	val := func(names ...string) (int, bool) {
		for i := 0; i+1 < len(args); i++ {
			for _, f := range names {
				if args[i] == f {
					v, err := strconv.Atoi(args[i+1])
					return v, err == nil
				}
			}
		}
		return 0, false
	}
	ram := 0
	if ngl, ok := val("--n-gpu-layers", "-ngl", "--gpu-layers"); ok && ngl >= 0 && ngl <= nLayer {
		ram = nLayer + 1 - ngl // n_layer+1 slots; -ngl offloads the last ones
	}
	if cm, ok := val("--n-cpu-moe", "-ncmoe"); ok {
		ram = max(ram, cm)
	}
	for _, a := range args {
		if a == "--cpu-moe" || a == "-cmoe" {
			ram = nLayer
		}
	}
	return min(ram, nLayer)
}

// expertCoverage is the share of routed tokens that the busiest Share of
// each layer's experts took (averaged over layers); an even spread gives
// Share itself.
type expertCoverage struct {
	Share float64 `json:"share"`
	All   float64 `json:"all"`
	RAM   float64 `json:"ram"` // layers with experts in RAM; 0 if none
}

type expertSummary struct {
	Runs         int              `json:"runs"`
	GenTokens    uint64           `json:"gen_tokens"`
	PromptTokens uint64           `json:"prompt_tokens"`
	Source       string           `json:"source"` // "generation", or "prompt" before any generation
	NExpert      int              `json:"n_expert"`
	Layers       int              `json:"layers"`
	RAMLayers    int              `json:"ram_layers"`
	Coverage     []expertCoverage `json:"coverage"`
}

// summarizeExpertStats sums every run file in dir.
func summarizeExpertStats(dir string) (expertSummary, error) {
	var sum expertSummary
	files, _ := filepath.Glob(filepath.Join(dir, "run-*.json"))
	sort.Strings(files)
	gen, prompt := map[int][]uint64{}, map[int][]uint64{}
	add := func(m map[int][]uint64, layer int, v []uint64) {
		acc := m[layer]
		if acc == nil {
			acc = make([]uint64, len(v))
			m[layer] = acc
		}
		for i := range min(len(acc), len(v)) {
			acc[i] += v[i]
		}
	}
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		var r expertRun
		if json.Unmarshal(b, &r) != nil || r.NExpert <= 0 || (sum.NExpert != 0 && r.NExpert != sum.NExpert) {
			continue
		}
		sum.NExpert = r.NExpert
		sum.Runs++
		sum.GenTokens += r.GenTokens
		sum.PromptTokens += r.PromptTokens
		for _, l := range r.Layers {
			add(gen, l.Layer, l.Gen)
			add(prompt, l.Layer, l.Prompt)
		}
		if mb, err := os.ReadFile(strings.TrimSuffix(f, ".json") + ".meta"); err == nil {
			var m expertRunMeta
			if json.Unmarshal(mb, &m) == nil {
				sum.RAMLayers = m.RAMLayers // the latest run's placement
			}
		}
	}
	counts, src := gen, "generation"
	if sum.GenTokens == 0 {
		counts, src = prompt, "prompt"
	}
	if sum.Runs == 0 {
		return sum, nil
	}
	sum.Source = src
	sum.Layers = len(counts)
	for _, share := range []float64{0.1, 0.25, 0.5} {
		c := expertCoverage{Share: share}
		c.All, _ = meanCoverage(counts, share, func(int) bool { return true })
		c.RAM, _ = meanCoverage(counts, share, func(l int) bool { return l < sum.RAMLayers })
		sum.Coverage = append(sum.Coverage, c)
	}
	return sum, nil
}

func meanCoverage(counts map[int][]uint64, share float64, use func(layer int) bool) (float64, int) {
	var total float64
	n := 0
	for layer, hits := range counts {
		if !use(layer) || len(hits) == 0 {
			continue
		}
		s := append([]uint64(nil), hits...)
		sort.Slice(s, func(i, j int) bool { return s[i] > s[j] })
		k := int(math.Ceil(share * float64(len(s))))
		var top, all uint64
		for i, v := range s {
			all += v
			if i < k {
				top += v
			}
		}
		if all == 0 {
			continue
		}
		total += float64(top) / float64(all)
		n++
	}
	if n == 0 {
		return 0, 0
	}
	return total / float64(n), n
}

func (s *Server) handleExpertStats(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("model")
	if path == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("model is required"))
		return
	}
	dir, err := expertStatsDir(path)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	if r.Method == http.MethodDelete {
		// files only: a running model keeps writing its run into the folder
		files, _ := filepath.Glob(filepath.Join(dir, "run-*"))
		for _, f := range files {
			if err := os.Remove(f); err != nil {
				writeErr(w, http.StatusInternalServerError, err)
				return
			}
		}
		writeJSON(w, expertSummary{})
		return
	}
	sum, err := summarizeExpertStats(dir)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, sum)
}
