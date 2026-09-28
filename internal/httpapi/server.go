// Package httpapi wires the launcher's local control panel: a small JSON API
// plus the embedded static web UI in ../../web. This is Q38FNInference's own
// GUI surface; it is separate from (and talks to) llama-server, which
// already provides its own OpenAI-compatible API and testing web UI once
// started — see docs/ARCHITECTURE.md.
package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"sync"
	"time"

	"github.com/hgdubbe/q38fninference/internal/appconfig"
	"github.com/hgdubbe/q38fninference/internal/downloadmgr"
	"github.com/hgdubbe/q38fninference/internal/gguf"
	"github.com/hgdubbe/q38fninference/internal/gpu"
	"github.com/hgdubbe/q38fninference/internal/hf"
	"github.com/hgdubbe/q38fninference/internal/models"
	"github.com/hgdubbe/q38fninference/internal/server"
	"github.com/hgdubbe/q38fninference/internal/tuning"
)

// Server holds every piece of state the API handlers touch.
type Server struct {
	mu  sync.Mutex
	cfg appconfig.Config

	hfClient  *hf.Client
	downloads *downloadmgr.Manager
	llama     *server.Manager

	mux *http.ServeMux
}

// New builds a Server with its config loaded and dependencies wired up.
func New() (*Server, error) {
	cfg, err := appconfig.Load()
	if err != nil {
		return nil, fmt.Errorf("loading config: %w", err)
	}

	hfClient := hf.NewClient(cfg.HFToken)
	s := &Server{
		cfg:       cfg,
		hfClient:  hfClient,
		downloads: downloadmgr.NewManager(hfClient),
		llama:     server.NewManager(4000),
	}
	s.mux = s.routes()
	return s, nil
}

func (s *Server) Handler() http.Handler { return s.mux }

func (s *Server) routes() *http.ServeMux {
	mux := http.NewServeMux()

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
	mux.HandleFunc("GET /api/server/logs", s.handleServerLogs)
	mux.HandleFunc("GET /api/server/logs/stream", s.handleServerLogsStream)

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

// --- config ---------------------------------------------------------------

func (s *Server) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	cfg := s.cfg
	s.mu.Unlock()
	writeJSON(w, cfg)
}

func (s *Server) handlePostConfig(w http.ResponseWriter, r *http.Request) {
	var cfg appconfig.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if err := cfg.Save(); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}

	s.mu.Lock()
	s.cfg = cfg
	s.hfClient = hf.NewClient(cfg.HFToken)
	s.downloads = downloadmgr.NewManager(s.hfClient)
	s.mu.Unlock()

	writeJSON(w, cfg)
}

// --- gpus / local models ----------------------------------------------------

func (s *Server) handleGPUs(w http.ResponseWriter, r *http.Request) {
	gpus, err := gpu.Detect()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, gpus)
}

func (s *Server) handleLocalModels(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	dirs := append(models.DefaultSearchDirs(), s.cfg.ExtraModelDirs...)
	s.mu.Unlock()

	groups, err := models.Scan(dirs)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, groups)
}

// --- hugging face -----------------------------------------------------------

func (s *Server) handleHFSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query().Get("q")
	results, err := s.client().SearchModels(r.Context(), q, 25)
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
	writeJSON(w, info.GGUFFiles())
}

type downloadRequest struct {
	Repo     string `json:"repo"`
	Filename string `json:"filename"`
}

func (s *Server) handleHFDownload(w http.ResponseWriter, r *http.Request) {
	var req downloadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.Repo == "" || req.Filename == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("repo and filename are required"))
		return
	}

	modelsDir, err := appconfig.ModelsDir()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	dest := filepath.Join(modelsDir, filepath.Base(req.Repo), req.Filename)

	// downloads outlive the HTTP request that started them
	d := s.downloadMgr().Start(context.Background(), req.Repo, req.Filename, dest)
	writeJSON(w, d)
}

func (s *Server) handleHFDownloadsList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.downloadMgr().List())
}

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

// --- tuning -----------------------------------------------------------------

type tuneRequest struct {
	ModelPath    string `json:"model_path"`
	RequestedCtx uint64 `json:"requested_ctx,omitempty"`
}

type tuneResponse struct {
	Plan *tuning.Plan `json:"plan"`
	Args []string     `json:"args"`
}

func (s *Server) handleTune(w http.ResponseWriter, r *http.Request) {
	var req tuneRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.ModelPath == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("model_path is required"))
		return
	}

	meta, err := gguf.ReadWithTensors(req.ModelPath)
	if err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("reading %s: %w", req.ModelPath, err))
		return
	}

	gpus, err := gpu.Detect()
	if err != nil {
		gpus = nil // fall back to CPU-only plan rather than failing the request
	}

	plan, err := tuning.Compute(meta, gpus, tuning.Options{RequestedCtx: req.RequestedCtx})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}

	writeJSON(w, tuneResponse{Plan: plan, Args: plan.Args(req.ModelPath)})
}

// --- server lifecycle ---------------------------------------------------------

type startRequest struct {
	Args []string `json:"args"`
}

func (s *Server) handleServerStart(w http.ResponseWriter, r *http.Request) {
	var req startRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}

	s.mu.Lock()
	bin := appconfig.LocateLlamaServer(s.cfg)
	port := s.cfg.Port
	s.mu.Unlock()

	if bin == "" {
		writeErr(w, http.StatusPreconditionFailed, fmt.Errorf("llama-server binary not found; set llama_server_path in settings or place it next to this app"))
		return
	}

	args := req.Args
	if !hasFlag(args, "--port") && port != 0 {
		args = append(args, "--port", fmt.Sprint(port))
	}
	if !hasFlag(args, "--host") {
		args = append(args, "--host", "127.0.0.1")
	}

	if err := s.llama.Start(bin, args); err != nil {
		writeErr(w, http.StatusConflict, err)
		return
	}
	writeJSON(w, s.llama.Status())
}

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func (s *Server) handleServerStop(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	if err := s.llama.StopWithTimeout(ctx); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, s.llama.Status())
}

func (s *Server) handleServerStatus(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.llama.Status())
}

func (s *Server) handleServerLogs(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.llama.Logs())
}

func (s *Server) handleServerLogsStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	for _, line := range s.llama.Logs() {
		fmt.Fprintf(w, "data: %s\n\n", jsonEscape(line))
	}
	flusher.Flush()

	ch, unsub := s.llama.Subscribe()
	defer unsub()

	for {
		select {
		case <-r.Context().Done():
			return
		case line, ok := <-ch:
			if !ok {
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", jsonEscape(line))
			flusher.Flush()
		}
	}
}

func jsonEscape(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
