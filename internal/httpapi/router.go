package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hgdubbe/q38fninference/internal/appconfig"
	"github.com/hgdubbe/q38fninference/internal/models"
	"github.com/hgdubbe/q38fninference/internal/proc"
	"github.com/hgdubbe/q38fninference/internal/router"
	"github.com/hgdubbe/q38fninference/internal/tuning"
)

// On-demand mode: llama-server runs as a router (llama.cpp's multi-model
// mode) and loads whichever model a request names, one at a time
// (--models-max 1). The launcher writes a preset file giving every model
// its own offload plan, and keeps it current: local files that appear or
// disappear, and models the router itself downloads into its cache via
// POST /models, are picked up by a periodic rescan and a preset reload.

const routerRescanEvery = 30 * time.Second

// routerRun is the state of one on-demand llama-server process.
type routerRun struct {
	pid        int
	target     *url.URL
	presetPath string
	// GPU memory as measured when the router started, before any model was
	// loaded: with one model at a time, that is what each model gets
	gpus []tuning.GPU
	// option names this llama-server accepts (nil if --help couldn't be read)
	known map[string]bool

	mu      sync.Mutex
	plans   map[string]*tuneResponse // by model file + size + mtime + settings
	lastINI string
	notes   []string
	rescan  chan struct{}
}

func (s *Server) setRouter(rt *routerRun) {
	s.mu.Lock()
	s.router = rt
	s.mu.Unlock()
}

func (s *Server) currentRouter() *routerRun {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.router
}

// routerModel is one entry of the router's GET /models list.
type routerModel struct {
	ID     string `json:"id"`
	Path   string `json:"path"`
	Status struct {
		Value string `json:"value"`
	} `json:"status"`
}

func (rt *routerRun) models() ([]routerModel, error) {
	c := &http.Client{Timeout: 3 * time.Second}
	resp, err := c.Get(rt.target.String() + "/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out struct {
		Data []routerModel `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Data, nil
}

// loadedModel names the model the router currently has loaded (or is
// loading), or "none loaded".
func (rt *routerRun) loadedModel() string {
	ms, err := rt.models()
	if err != nil {
		return "on-demand"
	}
	for _, m := range ms {
		switch m.Status.Value {
		case "loaded", "loading", "sleeping":
			return m.ID + " (" + m.Status.Value + ")"
		}
	}
	return "none loaded"
}

func (s *Server) modelDirs() []string {
	dirs := models.DefaultSearchDirs()
	if d, err := appconfig.ModelsDir(); err == nil && !slices.Contains(dirs, d) {
		dirs = append(dirs, d)
	}
	return append(dirs, s.Config().ExtraModelDirs...)
}

var ggufSuffix = regexp.MustCompile(`(?i)(-\d{5}-of-\d{5})?\.gguf$`)

// modelName is the id API clients use: the file name without the shard
// suffix and ".gguf", which usually already carries the quantization. Only
// ".gguf" goes: names like "Qwen3.8-…" have dots of their own.
func modelName(path string) string {
	return ggufSuffix.ReplaceAllString(filepath.Base(path), "")
}

// planFor returns the cached launch plan for a model file, computing it
// the first time (or when the file or the model settings changed).
func (s *Server) planFor(rt *routerRun, path string) (*tuneResponse, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	settings, _ := json.Marshal(s.Config().ModelFor(modelName(path)))
	key := fmt.Sprintf("%s|%d|%d|%s", path, fi.Size(), fi.ModTime().UnixNano(), settings)

	rt.mu.Lock()
	p, ok := rt.plans[key]
	rt.mu.Unlock()
	if ok {
		return p, nil
	}
	p, err = s.tuneWith(path, rt.gpus)
	if err != nil {
		return nil, err
	}
	rt.mu.Lock()
	rt.plans[key] = p
	rt.mu.Unlock()
	return p, nil
}

// buildEntries lists every model the router should offer, with its plan:
// local GGUF files (named by file), and models the router found in its own
// cache (kept under the router's id, so API clients' names still work).
func (s *Server) buildEntries(rt *routerRun, known []routerModel) ([]router.Entry, []string) {
	var notes []string
	groups, err := models.Scan(s.modelDirs())
	if err != nil {
		notes = append(notes, "model scan failed: "+err.Error())
	}

	var paths, names []string
	for _, g := range groups {
		if ok, have, total := g.Complete(); !ok {
			// e.g. still downloading; say so rather than leave it out silently
			if len(g.Files) > 0 {
				notes = append(notes, fmt.Sprintf("%s: only %d of %d parts found", modelName(g.Files[0].Path), have, total))
			}
			continue
		}
		paths = append(paths, g.Files[0].Path)
		names = append(names, router.SafeName(modelName(g.Files[0].Path)))
	}
	names = router.UniqueNames(names)

	var entries []router.Entry
	ours := map[string]bool{}
	for i, p := range paths {
		plan, err := s.planFor(rt, p)
		if err != nil {
			notes = append(notes, fmt.Sprintf("%s: %v", names[i], err))
			continue
		}
		entries = append(entries, router.Entry{Name: names[i], Path: p, Args: plan.Args})
		ours[names[i]] = true
		ours[filepath.Clean(p)] = true
	}

	for _, m := range known {
		if m.Path == "" || ours[m.ID] || ours[filepath.Clean(m.Path)] {
			continue
		}
		plan, err := s.planFor(rt, m.Path)
		if err != nil {
			notes = append(notes, fmt.Sprintf("%s: %v", m.ID, err))
			continue
		}
		entries = append(entries, router.Entry{Name: m.ID, Args: plan.Args})
	}
	return entries, notes
}

// refreshPresets rewrites the preset file; it reports whether it changed.
func (s *Server) refreshPresets(rt *routerRun, known []routerModel) (bool, error) {
	entries, notes := s.buildEntries(rt, known)
	ini, errs := router.INI(entries, rt.known)
	for _, e := range errs {
		notes = append(notes, e.Error())
	}
	rt.mu.Lock()
	rt.notes = notes
	changed := ini != rt.lastINI
	rt.lastINI = ini
	rt.mu.Unlock()
	if !changed {
		return false, nil
	}
	return true, os.WriteFile(rt.presetPath, []byte(ini), 0o644)
}

func (s *Server) startRouter(w http.ResponseWriter) {
	cfg := s.Config()
	bin := appconfig.LocateLlamaServer(cfg)
	if bin == "" {
		writeErr(w, http.StatusPreconditionFailed, fmt.Errorf("llama-server not found: put it next to this app or set its path in Settings"))
		return
	}
	dir, err := appconfig.Dir()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	gpus, err := s.selectedGPUs()
	if err != nil {
		log.Printf("httpapi: GPU detection failed, planning CPU-only: %v", err)
	}
	port, err := freePort()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}

	rt := &routerRun{
		target:     &url.URL{Scheme: "http", Host: net.JoinHostPort("127.0.0.1", strconv.Itoa(port))},
		presetPath: filepath.Join(dir, "router-presets.ini"),
		gpus:       gpus,
		known:      serverFlags(bin),
		plans:      map[string]*tuneResponse{},
		rescan:     make(chan struct{}, 1),
	}
	if _, err := s.refreshPresets(rt, nil); err != nil {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("writing router presets: %w", err))
		return
	}

	args := []string{
		"--models-preset", rt.presetPath,
		"--models-max", "1",
		"--host", "127.0.0.1", "--port", strconv.Itoa(port),
	}
	if cfg.IdleUnloadMinutes > 0 {
		args = append(args, "--sleep-idle-seconds", strconv.Itoa(cfg.IdleUnloadMinutes*60))
	}
	if cfg.Model.APIKey != "" {
		args = append(args, "--api-key", cfg.Model.APIKey)
	}

	// every plan was made for the same GPU set, ordered the same way
	env := append([]string{"CUDA_DEVICE_ORDER=PCI_BUS_ID"}, cfg.Model.Env()...)
	if devs := planDevices(gpus); len(devs) > 0 {
		env = append(env, "CUDA_VISIBLE_DEVICES="+devs)
	}

	if err := s.llama.Start(bin, args, env...); err != nil {
		writeErr(w, http.StatusConflict, err)
		return
	}
	st := s.llama.Status()
	rt.pid = st.PID
	s.setRouter(rt)
	s.Proxy.SetTarget(rt.target)
	go s.watch(st.PID, rt.target, nil)
	go s.keepPresetsCurrent(rt)
	writeJSON(w, st)
}

// serverFlags asks the llama-server binary which options it accepts.
func serverFlags(bin string) map[string]bool {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "--help")
	proc.Hide(cmd)
	out, _ := cmd.CombinedOutput() // --help may exit non-zero; the text is what matters
	known := router.KnownFlags(string(out))
	if len(known) < 50 {
		log.Printf("httpapi: couldn't read llama-server --help; preset options are not checked")
		return nil
	}
	return known
}

// planDevices is the CUDA_VISIBLE_DEVICES order tuning.Compute uses:
// most free memory first.
func planDevices(gpus []tuning.GPU) string {
	sorted := slices.Clone(gpus)
	slices.SortStableFunc(sorted, func(a, b tuning.GPU) int {
		switch {
		case a.FreeBytes > b.FreeBytes:
			return -1
		case a.FreeBytes < b.FreeBytes:
			return 1
		}
		return 0
	})
	ids := make([]string, len(sorted))
	for i, g := range sorted {
		ids[i] = strconv.Itoa(g.Index)
	}
	return strings.Join(ids, ",")
}

// keepPresetsCurrent rescans models periodically (and on request) while
// this router runs, reloading its model list when the presets change.
func (s *Server) keepPresetsCurrent(rt *routerRun) {
	t := time.NewTicker(routerRescanEvery)
	defer t.Stop()
	for {
		select {
		case <-t.C:
		case <-rt.rescan:
		}
		st := s.llama.Status()
		if !st.Running || st.PID != rt.pid {
			return
		}
		known, err := rt.models()
		if err != nil {
			continue // router still starting
		}
		changed, err := s.refreshPresets(rt, known)
		if err != nil {
			log.Printf("httpapi: router presets: %v", err)
			continue
		}
		if changed {
			c := &http.Client{Timeout: 10 * time.Second}
			if resp, err := c.Get(rt.target.String() + "/models?reload=1"); err == nil {
				resp.Body.Close()
			}
			log.Printf("httpapi: router model list updated")
		}
	}
}

func (s *Server) handleRouterNotes(w http.ResponseWriter, r *http.Request) {
	notes := []string{}
	if rt := s.currentRouter(); rt != nil {
		rt.mu.Lock()
		notes = append(notes, rt.notes...)
		rt.mu.Unlock()
	}
	writeJSON(w, map[string]any{"notes": notes})
}

// handleRouterRescan triggers an immediate rescan and returns the notes
// from the last one (models that couldn't be offered, and why).
func (s *Server) handleRouterRescan(w http.ResponseWriter, r *http.Request) {
	rt := s.currentRouter()
	if rt == nil {
		writeErr(w, http.StatusConflict, fmt.Errorf("not running in on-demand mode"))
		return
	}
	select {
	case rt.rescan <- struct{}{}:
	default:
	}
	rt.mu.Lock()
	notes := slices.Clone(rt.notes)
	rt.mu.Unlock()
	if notes == nil {
		notes = []string{}
	}
	writeJSON(w, map[string]any{"notes": notes})
}
