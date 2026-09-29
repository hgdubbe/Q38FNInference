// Package appconfig persists launcher settings (llama-server binary
// location, extra model search directories, HF token, last-used model)
// between runs.
package appconfig

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// Config holds everything the launcher remembers across restarts.
type Config struct {
	// LlamaServerPath overrides auto-detection of the llama-server binary.
	LlamaServerPath string `json:"llama_server_path,omitempty"`
	// ExtraModelDirs are additional directories scanned for local *.gguf
	// files, beyond the default Hugging Face cache and app models dir.
	ExtraModelDirs []string `json:"extra_model_dirs,omitempty"`
	// HFToken is an optional Hugging Face access token, for gated repos.
	HFToken string `json:"hf_token,omitempty"`
	// Port is where the OpenAI-compatible API and llama-server's web UI are
	// served (through the system-prompt proxy). Takes effect on restart.
	Port int `json:"port,omitempty"`
	// APIHost is the API bind address; "0.0.0.0" exposes it on the LAN.
	APIHost string `json:"api_host,omitempty"`
	// LastModelPath remembers the last model launched, for convenience.
	LastModelPath string `json:"last_model_path,omitempty"`

	// GPUs lists the nvidia-smi indices to use; empty means all of them.
	GPUs []int `json:"gpus,omitempty"`

	// VRAMCorrections is extra VRAM, in bytes, the planner keeps free per
	// model and GPU ("<model path>#<gpu index>"), learned when a load used
	// more than the plan expected (see httpapi's VRAM check).
	VRAMCorrections map[string]uint64 `json:"vram_corrections,omitempty"`

	// SystemPrompt is injected by the API proxy according to SystemPromptMode
	// ("off", "default": only when a request has none, "override": always).
	SystemPrompt     string `json:"system_prompt,omitempty"`
	SystemPromptMode string `json:"system_prompt_mode,omitempty"`

	// OpenChatOnReady opens the chat web UI once a started model is loaded.
	OpenChatOnReady bool `json:"open_chat_on_ready"`

	// ExitWithPanel stops the model and exits once every control-panel tab
	// has been closed. Where there is no tray icon (outside Windows) the
	// launcher always does this, since nothing else would show it running.
	ExitWithPanel bool `json:"exit_with_panel"`

	// OnDemand runs llama-server as a router offering every local model,
	// loading whichever one a request names (one at a time).
	OnDemand bool `json:"on_demand"`
	// IdleUnloadMinutes frees the loaded model after this long without
	// requests; the next request reloads it. 0 keeps it loaded.
	IdleUnloadMinutes int `json:"idle_unload_minutes,omitempty"`

	Model ModelSettings `json:"model"`
}

// ModelSettings are llama-server load/sampling options. Zero values and nil
// pointers mean "leave llama-server's own default".
type ModelSettings struct {
	CtxSize    uint64 `json:"ctx_size,omitempty"`
	Parallel   int    `json:"parallel,omitempty"`
	CacheType  string `json:"cache_type,omitempty"`
	Threads    int    `json:"threads,omitempty"`
	BatchSize  int    `json:"batch_size,omitempty"`
	UBatchSize int    `json:"ubatch_size,omitempty"`
	// OffloadMinBatch is the prompt batch size from which llama.cpp copies
	// RAM-resident weights to the GPU instead of computing on the CPU
	// (GGML_OP_OFFLOAD_MIN_BATCH). 0 = DefaultOffloadMinBatch.
	OffloadMinBatch int `json:"offload_min_batch,omitempty"`
	// QSABlocks selects qwen4exp's sparse-attention context by whole
	// blocks instead of per cell (patches/0006, LLAMA_QWEN4EXP_QSA_BLOCKS):
	// less compute memory at long contexts, close to but not identical to
	// the reference selection. Experimental, off by default.
	QSABlocks bool `json:"qsa_blocks,omitempty"`

	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	TopK            *int     `json:"top_k,omitempty"`
	MinP            *float64 `json:"min_p,omitempty"`
	RepeatPenalty   *float64 `json:"repeat_penalty,omitempty"`
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`
	MaxTokens       *int     `json:"max_tokens,omitempty"`
	Seed            *int64   `json:"seed,omitempty"`

	// Reasoning is off/low/medium/high; internal/reasoning maps it onto
	// whatever the model's chat template understands. Empty means high.
	Reasoning string `json:"reasoning,omitempty"`
	APIKey    string `json:"api_key,omitempty"`
	Alias     string `json:"alias,omitempty"`
	ExtraArgs string `json:"extra_args,omitempty"` // whitespace-separated, appended last
}

// DefaultOffloadMinBatch replaces llama.cpp's 32. The copy moves a layer's
// whole expert tensor (every expert) over PCIe, while the CPU only reads
// the experts a batch routes to, so for sparse MoE the copy pays off only
// for longer batches; ~128 tokens is where a PCIe 4.0 x16 copy of a large
// MoE layer and a desktop CPU's prompt throughput roughly break even.
const DefaultOffloadMinBatch = 128

// Env renders the environment variables these settings need.
func (s ModelSettings) Env() []string {
	n := s.OffloadMinBatch
	if n <= 0 {
		n = DefaultOffloadMinBatch
	}
	env := []string{"GGML_OP_OFFLOAD_MIN_BATCH=" + strconv.Itoa(n)}
	if s.QSABlocks {
		env = append(env, "LLAMA_QWEN4EXP_QSA_BLOCKS=1")
	}
	return env
}

// Args renders the sampling/runtime llama-server flags for these settings.
// Reasoning is model-dependent and rendered by internal/reasoning instead.
func (s ModelSettings) Args() []string {
	var a []string
	addI := func(flag string, v int) {
		if v > 0 {
			a = append(a, flag, strconv.Itoa(v))
		}
	}
	addF := func(flag string, v *float64) {
		if v != nil {
			a = append(a, flag, strconv.FormatFloat(*v, 'f', -1, 64))
		}
	}
	addI("--threads", s.Threads)
	addI("--batch-size", s.BatchSize)
	addI("--ubatch-size", s.UBatchSize)
	addF("--temp", s.Temperature)
	addF("--top-p", s.TopP)
	if s.TopK != nil {
		a = append(a, "--top-k", strconv.Itoa(*s.TopK))
	}
	addF("--min-p", s.MinP)
	addF("--repeat-penalty", s.RepeatPenalty)
	addF("--presence-penalty", s.PresencePenalty)
	if s.MaxTokens != nil {
		a = append(a, "--n-predict", strconv.Itoa(*s.MaxTokens))
	}
	if s.Seed != nil {
		a = append(a, "--seed", strconv.FormatInt(*s.Seed, 10))
	}
	if s.APIKey != "" {
		a = append(a, "--api-key", s.APIKey)
	}
	if s.Alias != "" {
		a = append(a, "--alias", s.Alias)
	}
	return append(a, strings.Fields(s.ExtraArgs)...)
}

// Default returns a Config with sane defaults filled in.
func Default() Config {
	return Config{Port: 8080, APIHost: "127.0.0.1", SystemPromptMode: "default", OpenChatOnReady: true}
}

// Dir returns the directory config/state lives in, creating it if needed.
func Dir() (string, error) {
	base, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(base, "Q38FNInference")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return dir, nil
}

func path() (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "config.json"), nil
}

// Load reads the persisted config, returning Default() if none exists yet.
func Load() (Config, error) {
	p, err := path()
	if err != nil {
		return Default(), err
	}
	b, err := os.ReadFile(p)
	if os.IsNotExist(err) {
		return Default(), nil
	}
	if err != nil {
		return Default(), err
	}
	cfg := Default()
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Default(), err
	}
	return cfg, nil
}

// Save writes the config to disk as pretty-printed JSON.
func (c Config) Save() error {
	p, err := path()
	if err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, b, 0o600) // holds the HF token / API key
}

// ModelsDir is where downloaded models are stored by default.
func ModelsDir() (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	d := filepath.Join(dir, "models")
	if err := os.MkdirAll(d, 0o755); err != nil {
		return "", err
	}
	return d, nil
}

func serverBinaryName() string {
	if runtime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

// LocateLlamaServer resolves the llama-server binary to run, in priority
// order: an explicit config override, a copy next to this executable (or in
// its llama/ subdirectory — where the bundled Windows CUDA build lands), then
// PATH. Returns "" if none of those exist.
func LocateLlamaServer(cfg Config) string {
	name := serverBinaryName()

	if cfg.LlamaServerPath != "" {
		if fileExists(cfg.LlamaServerPath) {
			return cfg.LlamaServerPath
		}
	}

	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates := []string{
			filepath.Join(exeDir, name),
			filepath.Join(exeDir, "llama", name),
		}
		for _, c := range candidates {
			if fileExists(c) {
				return c
			}
		}
	}

	if p, err := exec.LookPath(name); err == nil {
		return p
	}

	return ""
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

// VRAMCorrectionKey is the VRAMCorrections key for a model on a GPU.
func VRAMCorrectionKey(modelPath string, gpuIndex int) string {
	return modelPath + "#" + strconv.Itoa(gpuIndex)
}
