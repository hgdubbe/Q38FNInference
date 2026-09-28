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
	"github.com/hgdubbe/q38fninference/internal/server"
	"github.com/hgdubbe/q38fninference/internal/tuning"
)

// AppID identifies this app on /api/info, for single-instance detection.
const AppID = "q38fninference"

// Hooks are the process-level actions the API can trigger.
type Hooks struct {
	OpenURL func(string)
	Quit    func()
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

	// same-origin access to the model API for the control panel's chat tab
	mux.Handle("GET /v1/", s.Proxy)
	mux.Handle("POST /v1/", s.Proxy)

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
	writeJSON(w, map[string]string{"app": AppID, "api_url": s.apiURL(), "api_error": apiErr})
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
	dirs := models.DefaultSearchDirs()
	if d, err := appconfig.ModelsDir(); err == nil && !slices.Contains(dirs, d) {
		dirs = append(dirs, d)
	}
	dirs = append(dirs, s.Config().ExtraModelDirs...)

	groups, err := models.Scan(dirs)
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
	Plan *tuning.Plan `json:"plan"`
	Args []string     `json:"args"`
}

func (s *Server) tune(modelPath string) (*tuneResponse, error) {
	meta, err := gguf.ReadWithTensors(modelPath)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", modelPath, err)
	}
	gpus, err := s.selectedGPUs()
	if err != nil {
		log.Printf("httpapi: GPU detection failed, planning CPU-only: %v", err)
	}
	ms := s.Config().Model
	plan, err := tuning.Compute(meta, gpus, tuning.Options{
		RequestedCtx: ms.CtxSize,
		Parallel:     ms.Parallel,
		CacheType:    ms.CacheType,
	})
	if err != nil {
		return nil, err
	}
	return &tuneResponse{Plan: plan, Args: append(plan.Args(modelPath), ms.Args()...)}, nil
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
	if len(req.Args) == 0 {
		if req.ModelPath == "" {
			writeErr(w, http.StatusBadRequest, fmt.Errorf("model_path or args required"))
			return
		}
		plan, err := s.tune(req.ModelPath)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		req.Args, req.Devices = plan.Args, plan.Plan.Devices
	}

	cfg := s.Config()
	bin := appconfig.LocateLlamaServer(cfg)
	if bin == "" {
		writeErr(w, http.StatusPreconditionFailed, fmt.Errorf("llama-server not found: put it next to this app or set its path in Settings"))
		return
	}

	port, err := freePort()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	args := append(stripFlags(req.Args, "--host", "--port"), "--host", "127.0.0.1", "--port", strconv.Itoa(port))

	env := []string{"CUDA_DEVICE_ORDER=PCI_BUS_ID"}
	if len(req.Devices) > 0 {
		ids := make([]string, len(req.Devices))
		for i, d := range req.Devices {
			ids[i] = strconv.Itoa(d)
		}
		env = append(env, "CUDA_VISIBLE_DEVICES="+strings.Join(ids, ","))
	}

	if err := s.llama.Start(bin, args, env...); err != nil {
		writeErr(w, http.StatusConflict, err)
		return
	}
	st := s.llama.Status()
	target := &url.URL{Scheme: "http", Host: net.JoinHostPort("127.0.0.1", strconv.Itoa(port))}
	s.Proxy.SetTarget(target)
	go s.watch(st.PID, target)

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

// watch marks the server ready once /health answers, and detaches the proxy
// when that process exits.
func (s *Server) watch(pid int, target *url.URL) {
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
	writeJSON(w, s.llama.Status())
}

func (s *Server) handleServerStatus(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.llama.Status())
}

func (s *Server) handleServerLogsStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")

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

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
