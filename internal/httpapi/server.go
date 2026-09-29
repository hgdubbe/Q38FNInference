// Package httpapi wires the launcher's local control panel: a small JSON API
// plus the embedded static web UI in ./web. The model itself is served by a
// llama-server child process behind internal/proxy, which provides the
// OpenAI-compatible API and llama-server's own web UI on the public port.
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
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hgdubbe/q38fninference/internal/appconfig"
	"github.com/hgdubbe/q38fninference/internal/downloadmgr"
	"github.com/hgdubbe/q38fninference/internal/gguf"
	"github.com/hgdubbe/q38fninference/internal/gpu"
	"github.com/hgdubbe/q38fninference/internal/hf"
	"github.com/hgdubbe/q38fninference/internal/models"
	"github.com/hgdubbe/q38fninference/internal/proxy"
	"github.com/hgdubbe/q38fninference/internal/reasoning"
	"github.com/hgdubbe/q38fninference/internal/server"
	"github.com/hgdubbe/q38fninference/internal/sysmem"
	"github.com/hgdubbe/q38fninference/internal/tuning"
)

// AppID identifies this app on /api/info, for single-instance detection.
const AppID = "q38fninference"

// Hooks are the process-level actions the API can trigger.
type Hooks struct {
	OpenURL func(string)
	Quit    func()
	// HasTray: a tray icon shows the launcher is running and can quit it,
	// so closing the panel only exits when the user asked for that.
	HasTray bool
}

// Server holds every piece of state the API handlers touch.
type Server struct {
	mu  sync.Mutex
	cfg appconfig.Config

	hfClient  *hf.Client
	downloads *downloadmgr.Manager
	llama     *server.Manager
	Proxy     *proxy.Proxy
	hooks     Hooks
	apiErr    string

	router *routerRun // set while llama-server runs in on-demand (router) mode

	notice   string // see statusResponse.Notice
	noticeID int

	current *launchInfo // the single-model load that is running, for plans made meanwhile

	// open control-panel tabs, counted by their log-stream connections
	panels     int
	panelSeen  bool
	panelsGone time.Time

	mux *http.ServeMux
}

// New builds a Server with its config loaded and dependencies wired up.
func New(hooks Hooks) (*Server, error) {
	cfg, err := appconfig.Load()
	if err != nil {
		log.Printf("httpapi: config unreadable, using defaults: %v", err)
	}

	hfClient := hf.NewClient(cfg.HFToken)
	s := &Server{
		cfg:       cfg,
		hfClient:  hfClient,
		downloads: downloadmgr.NewManager(hfClient),
		llama:     server.NewManager(4000),
		Proxy:     proxy.New(),
		hooks:     hooks,
	}
	s.Proxy.SetSystemPrompt(proxy.Mode(cfg.SystemPromptMode), cfg.SystemPrompt)
	s.mux = s.routes()
	return s, nil
}

func (s *Server) Handler() http.Handler { return s.mux }

// Config returns a copy of the current settings.
func (s *Server) Config() appconfig.Config {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cfg
}

// SetAPIError records why the public API port couldn't be opened.
func (s *Server) SetAPIError(err error) {
	s.mu.Lock()
	s.apiErr = err.Error()
	s.mu.Unlock()
}

// Shutdown stops the llama-server child, if any.
func (s *Server) Shutdown(ctx context.Context) {
	if err := s.llama.StopWithTimeout(ctx); err != nil {
		log.Printf("httpapi: stopping llama-server: %v", err)
	}
}

func (s *Server) routes() *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /api/info", s.handleInfo)
	mux.HandleFunc("POST /api/quit", s.handleQuit)
	mux.HandleFunc("POST /api/open", s.handleOpen)

	mux.HandleFunc("GET /api/config", s.handleGetConfig)
	mux.HandleFunc("POST /api/config", s.handlePostConfig)

	mux.HandleFunc("GET /api/gpus", s.handleGPUs)
	mux.HandleFunc("GET /api/models/local", s.handleLocalModels)

	mux.HandleFunc("GET /api/hf/search", s.handleHFSearch)
	mux.HandleFunc("GET /api/hf/files", s.handleHFFiles)
	mux.HandleFunc("POST /api/hf/download", s.handleHFDownload)
	mux.HandleFunc("GET /api/hf/downloads", s.handleHFDownloadsList)

	mux.HandleFunc("POST /api/tune", s.handleTune)

	mux.HandleFunc("POST /api/server/start", s.handleServerStart)
	mux.HandleFunc("POST /api/server/stop", s.handleServerStop)
	mux.HandleFunc("GET /api/server/status", s.handleServerStatus)
	mux.HandleFunc("GET /api/server/logs/stream", s.handleServerLogsStream)

	mux.HandleFunc("POST /api/benchmark", s.handleBenchmark)
	mux.HandleFunc("POST /api/router/rescan", s.handleRouterRescan)
	mux.HandleFunc("GET /api/router/notes", s.handleRouterNotes)

	// same-origin access to the model API for the control panel's chat tab,
	// and to the router's model list / load / unload endpoints
	mux.Handle("GET /v1/", s.Proxy)
	mux.Handle("POST /v1/", s.Proxy)
	mux.Handle("GET /models", s.Proxy)
	mux.Handle("GET /models/", s.Proxy)
	mux.Handle("POST /models/", s.Proxy)

	mux.Handle("GET /", webFS())
	return mux
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("httpapi: encoding response: %v", err)
	}
}

func writeErr(w http.ResponseWriter, status int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
}

// --- app ------------------------------------------------------------------

func (s *Server) apiURL() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	host := s.cfg.APIHost
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	return "http://" + net.JoinHostPort(host, strconv.Itoa(s.cfg.Port))
}

func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	apiErr := s.apiErr
	s.mu.Unlock()
	mem := sysmem.Read()
	writeJSON(w, map[string]any{"app": AppID, "api_url": s.apiURL(), "api_error": apiErr,
		"ram_total": mem.Total, "ram_available": mem.Available})
}

func (s *Server) handleQuit(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]bool{"ok": true})
	if s.hooks.Quit != nil {
		go s.hooks.Quit()
	}
}

func (s *Server) handleOpen(w http.ResponseWriter, r *http.Request) {
	if s.hooks.OpenURL != nil {
		s.hooks.OpenURL(s.apiURL() + "/")
	}
	writeJSON(w, map[string]bool{"ok": true})
}

// --- config ---------------------------------------------------------------

func (s *Server) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.Config())
}

func (s *Server) handlePostConfig(w http.ResponseWriter, r *http.Request) {
	cfg := appconfig.Default()
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if err := cfg.Save(); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}

	s.mu.Lock()
	tokenChanged := cfg.HFToken != s.cfg.HFToken
	s.cfg = cfg
	if tokenChanged {
		// in-flight downloads keep their old manager/client and finish normally
		s.hfClient = hf.NewClient(cfg.HFToken)
		s.downloads = downloadmgr.NewManager(s.hfClient)
	}
	s.mu.Unlock()
	s.Proxy.SetSystemPrompt(proxy.Mode(cfg.SystemPromptMode), cfg.SystemPrompt)

	writeJSON(w, cfg)
}

// --- gpus / local models ----------------------------------------------------

func (s *Server) handleGPUs(w http.ResponseWriter, r *http.Request) {
	gpus, err := gpu.Detect()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	if gpus == nil {
		gpus = []tuning.GPU{}
	}
	writeJSON(w, gpus)
}

func (s *Server) selectedGPUs() ([]tuning.GPU, error) {
	all, err := gpu.Detect()
	if err != nil {
		return nil, err
	}
	sel := s.Config().GPUs
	if len(sel) == 0 {
		return all, nil
	}
	var out []tuning.GPU
	for _, g := range all {
		if slices.Contains(sel, g.Index) {
			out = append(out, g)
		}
	}
	return out, nil
}

func (s *Server) handleLocalModels(w http.ResponseWriter, r *http.Request) {
	groups, err := models.Scan(s.modelDirs())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	if groups == nil {
		groups = []models.Group{}
	}
	writeJSON(w, groups)
}

// --- hugging face -----------------------------------------------------------

func (s *Server) client() *hf.Client {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.hfClient
}

func (s *Server) downloadMgr() *downloadmgr.Manager {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.downloads
}

func (s *Server) handleHFSearch(w http.ResponseWriter, r *http.Request) {
	results, err := s.client().SearchModels(r.Context(), r.URL.Query().Get("q"), 25)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, results)
}

func (s *Server) handleHFFiles(w http.ResponseWriter, r *http.Request) {
	repo := r.URL.Query().Get("repo")
	if repo == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("missing repo query param"))
		return
	}
	info, err := s.client().GetModelInfo(r.Context(), repo)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	files := info.GGUFFiles()
	if files == nil {
		files = []hf.Sibling{}
	}
	writeJSON(w, files)
}

func (s *Server) handleHFDownload(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Repo     string `json:"repo"`
		Filename string `json:"filename"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if !validRepo(req.Repo) || !validRepoFile(req.Filename) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid repo or filename"))
		return
	}

	modelsDir, err := appconfig.ModelsDir()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	// models--Org--Name mirrors the HF cache layout, so discovery can map
	// the files back to their repo
	dest := filepath.Join(modelsDir, "models--"+strings.ReplaceAll(req.Repo, "/", "--"), filepath.FromSlash(req.Filename))

	// downloads outlive the HTTP request that started them
	d := s.downloadMgr().Start(context.Background(), req.Repo, req.Filename, dest)
	writeJSON(w, d)
}

func validRepo(repo string) bool {
	parts := strings.Split(repo, "/")
	if len(parts) != 2 {
		return false
	}
	for _, p := range parts {
		if p == "" || p == "." || p == ".." || strings.ContainsAny(p, `\:`) {
			return false
		}
	}
	return true
}

// validRepoFile rejects names that would escape the model directory.
func validRepoFile(name string) bool {
	if name == "" || strings.ContainsAny(name, `\:`) || strings.HasPrefix(name, "/") {
		return false
	}
	for _, p := range strings.Split(name, "/") {
		if p == "" || p == "." || p == ".." {
			return false
		}
	}
	return true
}

func (s *Server) handleHFDownloadsList(w http.ResponseWriter, r *http.Request) {
	list := s.downloadMgr().List()
	if list == nil {
		list = []downloadmgr.Download{}
	}
	writeJSON(w, list)
}

// --- tuning -----------------------------------------------------------------

type tuneResponse struct {
	Plan      *tuning.Plan `json:"plan"`
	Args      []string     `json:"args"`
	Reasoning string       `json:"reasoning"` // detected reasoning-control style
}

func (s *Server) tune(modelPath string) (*tuneResponse, error) {
	gpus, err := s.selectedGPUs()
	if err != nil {
		log.Printf("httpapi: GPU detection failed, planning CPU-only: %v", err)
	}
	return s.tuneWith(modelPath, s.withOwnUsage(gpus))
}

// withOwnUsage adds back the VRAM our own running llama-server holds: a plan
// made while a model is loaded is for the next start, when that memory is
// free again. Without this, opening the panel with a model running planned
// everything onto the CPU.
func (s *Server) withOwnUsage(gpus []tuning.GPU) []tuning.GPU {
	s.mu.Lock()
	cur := s.current
	s.mu.Unlock()
	if cur == nil || !s.llama.Status().Running || s.currentRouter() != nil {
		return gpus
	}
	used := cudaBufferBytes(loadLogs(s.llama.Logs()))
	out := slices.Clone(gpus)
	for ord, d := range cur.devices {
		for i := range out {
			if out[i].Index != d.Index {
				continue
			}
			if u, ok := used[ord]; ok {
				out[i].FreeBytes += u + cudaContextBytes
			} else {
				out[i].FreeBytes = d.FreeBytes // still loading: use what was free before it started
			}
			if out[i].TotalBytes > 0 && out[i].FreeBytes > out[i].TotalBytes {
				out[i].FreeBytes = out[i].TotalBytes
			}
		}
	}
	return out
}

func (s *Server) tuneWith(modelPath string, gpus []tuning.GPU) (*tuneResponse, error) {
	meta, err := gguf.ReadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", modelPath, err)
	}
	ms := s.Config().Model
	plan, err := tuning.Compute(meta, gpus, tuning.Options{
		ExtraReserve: s.extraReserve(modelPath, gpus),
		RequestedCtx: ms.CtxSize,
		Parallel:     ms.Parallel,
		CacheType:    ms.CacheType,
		UBatch:       ms.UBatchSize,
		NoOpOffload:  strings.Contains(" "+ms.ExtraArgs+" ", " --no-op-offload "),
	})
	if err != nil {
		return nil, err
	}
	if note := sizeCheck(modelPath, meta); note != "" {
		plan.Notes = append(plan.Notes, note)
	}
	mem := sysmem.Read()
	if note := ramCheck(plan, mem); note != "" {
		plan.Notes = append(plan.Notes, note)
	}

	tmpl, _ := meta.KV["tokenizer.chat_template"].(string)
	style := reasoning.Detect(tmpl)
	level := reasoning.Level(ms.Reasoning)
	if level == "" {
		level = reasoning.High
	}
	args := append(plan.Args(modelPath), cacheRAMArgs(plan, mem)...)
	if ms.Alias == "" {
		// without an alias llama-server reports the model as its file path, which for
		// a split model is the path of part 1; use the id on-demand mode uses instead
		args = append(args, "--alias", modelName(modelPath))
	}
	args = append(args, ms.Args()...)
	args = append(args, style.Args(level)...)
	return &tuneResponse{Plan: plan, Args: args, Reasoning: style.Name}, nil
}

// ramCheck warns when the weights a plan keeps in RAM don't fit in physical
// memory: they are memory-mapped, so the OS would keep re-reading them from
// disk for every token. Embedding tables are excluded (a few rows per token).
func ramCheck(plan *tuning.Plan, mem sysmem.Info) string {
	if !mem.OK || plan.CPUFitBytes <= plan.InputBytes {
		return ""
	}
	need := plan.CPUFitBytes - plan.InputBytes
	gib := func(b uint64) float64 { return float64(b) / (1 << 30) }
	switch {
	case need > mem.Total*9/10:
		return fmt.Sprintf("warning: %.1f GiB of weights stay in RAM but the machine has %.1f GiB: they will be re-read from disk constantly and generation will be very slow. Use a smaller quantization, or add GPUs.", gib(need), gib(mem.Total))
	case need > mem.Available:
		return fmt.Sprintf("%.1f GiB of weights stay in RAM and only %.1f GiB is free right now: close other programs, or the first answers will be slow while Windows pages", gib(need), gib(mem.Available))
	}
	return fmt.Sprintf("%.1f GiB of weights stay in RAM (%.1f GiB free)", gib(need), gib(mem.Available))
}

// cacheRAMArgs caps llama-server's prompt cache (--cache-ram, 8 GiB by
// default) to the RAM the RAM-resident weights leave over. The cache lives in
// host memory, so past that it evicts memory-mapped expert pages, which are
// then re-read from disk for every token. Nothing when the default fits.
func cacheRAMArgs(plan *tuning.Plan, mem sysmem.Info) []string {
	const llamaDefaultMiB = 8192
	if !mem.OK || plan.CPUFitBytes <= plan.InputBytes {
		return nil
	}
	// weights, plus Windows, the launcher and llama-server's CPU buffers
	need := plan.CPUFitBytes - plan.InputBytes + 4<<30
	var spare uint64
	if mem.Total > need {
		spare = (mem.Total - need) >> 20
	}
	if spare >= llamaDefaultMiB {
		return nil
	}
	return []string{"--cache-ram", strconv.FormatUint(spare, 10)}
}

// sizeCheck flags a plan built from fewer tensor bytes than the model files
// hold, so an undercount shows up as a warning instead of an OOM at load.
func sizeCheck(modelPath string, meta *gguf.Metadata) string {
	shards, err := gguf.FindShards(modelPath)
	if err != nil {
		return ""
	}
	var files uint64
	for _, p := range shards {
		if fi, err := os.Stat(p); err == nil {
			files += uint64(fi.Size())
		}
	}
	var tensors uint64
	for _, t := range meta.Tensors {
		if n, ok := t.SizeBytes(); ok {
			tensors += n
		}
	}
	if files == 0 || tensors >= files*9/10 {
		return ""
	}
	return fmt.Sprintf("warning: the model files hold %.1f GiB but only %.1f GiB of tensors were recognized; this plan likely underestimates memory",
		float64(files)/(1<<30), float64(tensors)/(1<<30))
}

func (s *Server) handleTune(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelPath string `json:"model_path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.ModelPath == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("model_path is required"))
		return
	}
	resp, err := s.tune(req.ModelPath)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, resp)
}

// --- server lifecycle ---------------------------------------------------------

type startRequest struct {
	// OnDemand starts llama-server in router mode instead: every local model
	// is offered and loaded when a request names it (see router.go)
	OnDemand  bool     `json:"on_demand"`
	ModelPath string   `json:"model_path"`
	Args      []string `json:"args"`    // optional: edited plan args; recomputed if empty
	Devices   []int    `json:"devices"` // GPU indices the args were planned for
}

func (s *Server) handleServerStart(w http.ResponseWriter, r *http.Request) {
	var req startRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.OnDemand {
		s.startRouter(w)
		return
	}
	auto := false
	if req.ModelPath != "" {
		plan, err := s.tune(req.ModelPath)
		if err != nil && len(req.Args) == 0 {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		if len(req.Args) == 0 {
			req.Args, req.Devices = plan.Args, plan.Plan.Devices
		}
		// unedited plan args can be re-planned if the load overshoots VRAM
		auto = plan != nil && slices.Equal(stripFlags(plan.Args, "--host", "--port"), stripFlags(req.Args, "--host", "--port"))
	} else if len(req.Args) == 0 {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("model_path or args required"))
		return
	}

	st, err := s.launch(req.ModelPath, req.Args, req.Devices, auto, false)
	if err != nil {
		writeErr(w, err.(httpError).status, err)
		return
	}

	if req.ModelPath != "" {
		s.mu.Lock()
		s.cfg.LastModelPath = req.ModelPath
		c := s.cfg
		s.mu.Unlock()
		if err := c.Save(); err != nil {
			log.Printf("httpapi: saving last model: %v", err)
		}
	}
	writeJSON(w, st)
}

type httpError struct {
	status int
	err    error
}

func (e httpError) Error() string { return e.err.Error() }

// launch starts llama-server for one model and watches it until ready,
// then checks its GPU memory use (see vramcheck.go).
func (s *Server) launch(modelPath string, args []string, devices []int, auto, recheck bool) (server.Status, error) {
	cfg := s.Config()
	bin := appconfig.LocateLlamaServer(cfg)
	if bin == "" {
		return server.Status{}, httpError{http.StatusPreconditionFailed, fmt.Errorf("llama-server not found: put it next to this app or set its path in Settings")}
	}

	port, err := freePort()
	if err != nil {
		return server.Status{}, httpError{http.StatusInternalServerError, err}
	}
	args = append(stripFlags(args, "--host", "--port"), "--host", "127.0.0.1", "--port", strconv.Itoa(port))

	env := append([]string{"CUDA_DEVICE_ORDER=PCI_BUS_ID"}, cfg.Model.Env()...)
	if len(devices) > 0 {
		ids := make([]string, len(devices))
		for i, d := range devices {
			ids[i] = strconv.Itoa(d)
		}
		env = append(env, "CUDA_VISIBLE_DEVICES="+strings.Join(ids, ","))
	}

	// free memory per GPU just before the load, in CUDA ordinal order
	li := launchInfo{modelPath: modelPath, auto: auto, recheck: recheck}
	if all, err := gpu.Detect(); err == nil {
		for _, d := range devices {
			for _, g := range all {
				if g.Index == d {
					li.devices = append(li.devices, g)
				}
			}
		}
	}

	s.mu.Lock()
	s.current = &li
	s.mu.Unlock()
	s.llama.Note("starting llama-server")
	if err := s.llama.Start(bin, args, env...); err != nil {
		return server.Status{}, httpError{http.StatusConflict, err}
	}
	st := s.llama.Status()
	target := &url.URL{Scheme: "http", Host: net.JoinHostPort("127.0.0.1", strconv.Itoa(port))}
	s.setRouter(nil)
	s.Proxy.SetTarget(target)
	go s.watch(st.PID, target, &li)
	return st, nil
}

// watch marks the server ready once /health answers, and detaches the proxy
// when that process exits.
func (s *Server) watch(pid int, target *url.URL, li *launchInfo) {
	client := &http.Client{Timeout: 2 * time.Second}
	ready := false
	for {
		st := s.llama.Status()
		if !st.Running || st.PID != pid {
			if !st.Running {
				s.Proxy.SetTarget(nil)
			}
			return
		}
		if !ready {
			if resp, err := client.Get(target.String() + "/health"); err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					ready = true
					if li != nil && len(li.devices) > 0 && s.checkVRAM(pid, *li) {
						return // replaced by a corrected load
					}
					s.llama.MarkReady(pid)
					if s.Config().OpenChatOnReady && s.hooks.OpenURL != nil {
						s.hooks.OpenURL(s.apiURL() + "/")
					}
				}
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
}

func freePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

// stripFlags removes the given flags and their values (the launcher owns
// where llama-server listens; clients reach it through the proxy).
func stripFlags(args []string, flags ...string) []string {
	var out []string
	for i := 0; i < len(args); i++ {
		if slices.Contains(flags, args[i]) {
			i++
			continue
		}
		out = append(out, args[i])
	}
	return out
}

func (s *Server) handleServerStop(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	if err := s.llama.StopWithTimeout(ctx); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	s.Proxy.SetTarget(nil)
	s.setRouter(nil)
	writeJSON(w, s.status())
}

type statusResponse struct {
	server.Status
	OnDemand bool `json:"OnDemand"`
	// Notice is the latest thing the launcher did on its own (e.g. a VRAM
	// re-plan); NoticeID changes with each one so the panel shows it once.
	Notice   string `json:"Notice,omitempty"`
	NoticeID int    `json:"NoticeID,omitempty"`
}

func (s *Server) status() statusResponse {
	st := s.llama.Status()
	s.mu.Lock()
	notice, id := s.notice, s.noticeID
	s.mu.Unlock()
	return statusResponse{Status: st, OnDemand: st.Running && s.currentRouter() != nil, Notice: notice, NoticeID: id}
}

func (s *Server) handleServerStatus(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.status())
}

func (s *Server) handleServerLogsStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")

	s.panelOpened()
	defer s.panelClosed()

	// subscribe before snapshotting so no line falls between the two
	ch, unsub := s.llama.Subscribe()
	defer unsub()
	for _, line := range s.llama.Logs() {
		fmt.Fprintf(w, "data: %s\n\n", jsonString(line))
	}
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			return
		case line, ok := <-ch:
			if !ok {
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", jsonString(line))
			flusher.Flush()
		}
	}
}

// ModelStatus summarizes the llama-server child for the tray: "stopped",
// "loading" or "ready", and the model file's name.
func (s *Server) ModelStatus() (state, model string) {
	st := s.llama.Status()
	if !st.Running {
		return "stopped", ""
	}
	if rt := s.currentRouter(); rt != nil {
		if !st.Ready {
			return "loading", "on-demand"
		}
		return "ready", rt.loadedModel()
	}
	for i, a := range st.Args {
		if a == "--model" && i+1 < len(st.Args) {
			model = strings.TrimSuffix(filepath.Base(st.Args[i+1]), ".gguf")
		}
	}
	if st.Ready {
		return "ready", model
	}
	return "loading", model
}

// ChatURL is where llama-server's web UI is served (through the proxy).
func (s *Server) ChatURL() string { return s.apiURL() + "/" }

func (s *Server) panelOpened() {
	s.mu.Lock()
	s.panels++
	s.panelSeen = true
	s.mu.Unlock()
}

func (s *Server) panelClosed() {
	s.mu.Lock()
	s.panels--
	if s.panels == 0 {
		s.panelsGone = time.Now()
	}
	s.mu.Unlock()
}

// panelAbandoned reports whether every control-panel tab has been closed
// for at least grace. A page reload reconnects well within it.
func (s *Server) panelAbandoned(now time.Time, grace time.Duration) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	exit := s.cfg.ExitWithPanel || !s.hooks.HasTray
	return exit && s.panelSeen && s.panels == 0 && now.Sub(s.panelsGone) >= grace
}

// QuitWhenPanelClosed calls the Quit hook once no control panel has been
// open for grace, if that behaviour applies (see Hooks.HasTray and
// Config.ExitWithPanel).
func (s *Server) QuitWhenPanelClosed(ctx context.Context, grace time.Duration) {
	t := time.NewTicker(time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-t.C:
			if s.panelAbandoned(now, grace) {
				log.Printf("control panel closed for %s; exiting", grace)
				if s.hooks.Quit != nil {
					s.hooks.Quit()
				}
				return
			}
		}
	}
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
