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
	// Port is the port llama-server listens on.
	Port int `json:"port,omitempty"`
	// LastModelPath remembers the last model launched, for convenience.
	LastModelPath string `json:"last_model_path,omitempty"`
}

// Default returns a Config with sane defaults filled in.
func Default() Config {
	return Config{Port: 8080}
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
	return os.WriteFile(p, b, 0o644)
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
