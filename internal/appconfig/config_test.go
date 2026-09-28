package appconfig

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func withTempConfigDir(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir) // os.UserConfigDir() honors this on linux
}

func TestSaveLoadRoundTrip(t *testing.T) {
	withTempConfigDir(t)

	cfg := Default()
	cfg.LlamaServerPath = "/opt/llama/llama-server"
	cfg.HFToken = "hf_test_token"
	cfg.ExtraModelDirs = []string{"/mnt/models"}

	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	got, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, cfg) {
		t.Errorf("got %+v, want %+v", got, cfg)
	}
}

func TestLoadMissingReturnsDefault(t *testing.T) {
	withTempConfigDir(t)
	got, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, Default()) {
		t.Errorf("got %+v, want default %+v", got, Default())
	}
}

func TestLocateLlamaServerPrefersConfigOverride(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	cfg := Default()
	cfg.LlamaServerPath = bin
	got := LocateLlamaServer(cfg)
	if got != bin {
		t.Errorf("got %q, want %q", got, bin)
	}
}

func TestLocateLlamaServerReturnsEmptyWhenNotFound(t *testing.T) {
	t.Setenv("PATH", t.TempDir()) // empty PATH
	cfg := Default()
	cfg.LlamaServerPath = "/no/such/binary"
	if got := LocateLlamaServer(cfg); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}
