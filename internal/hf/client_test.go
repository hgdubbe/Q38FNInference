package hf

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestGetModelInfo(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/Qwen/Qwen3.8-Flash-Next" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		json.NewEncoder(w).Encode(ModelInfo{
			ID: "Qwen/Qwen3.8-Flash-Next",
			Siblings: []Sibling{
				{RFilename: "README.md"},
				{RFilename: "model-00001-of-00004.gguf", Size: 1000},
				{RFilename: "model-00002-of-00004.gguf", Size: 1000},
			},
		})
	}))
	defer srv.Close()

	c := NewClient("")
	c.BaseURL = srv.URL
	info, err := c.GetModelInfo(context.Background(), "Qwen/Qwen3.8-Flash-Next")
	if err != nil {
		t.Fatal(err)
	}
	if info.ID != "Qwen/Qwen3.8-Flash-Next" {
		t.Errorf("ID = %q", info.ID)
	}
	gguf := info.GGUFFiles()
	if len(gguf) != 2 {
		t.Fatalf("GGUFFiles() = %v", gguf)
	}
}

func TestGetModelInfoError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte("not found"))
	}))
	defer srv.Close()

	c := NewClient("")
	c.BaseURL = srv.URL
	if _, err := c.GetModelInfo(context.Background(), "nope/nope"); err == nil {
		t.Fatal("expected error for 404")
	}
}

func TestDownloadFresh(t *testing.T) {
	content := strings.Repeat("A", 1000)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		w.Write([]byte(content))
	}))
	defer srv.Close()

	c := NewClient("")
	c.BaseURL = srv.URL
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.gguf")

	var lastDone, lastTotal int64
	err := c.Download(context.Background(), "repo", "model.gguf", dest, func(done, total int64) {
		lastDone, lastTotal = done, total
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != content {
		t.Errorf("downloaded content mismatch: got %d bytes, want %d", len(got), len(content))
	}
	if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
		t.Error(".part file should be gone after a successful download")
	}
	if lastDone != int64(len(content)) || lastTotal != int64(len(content)) {
		t.Errorf("final progress callback = (%d, %d), want (%d, %d)", lastDone, lastTotal, len(content), len(content))
	}
}

func TestDownloadResumes(t *testing.T) {
	full := strings.Repeat("B", 500) + strings.Repeat("C", 500)
	var gotRangeHeader string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRangeHeader = r.Header.Get("Range")
		if gotRangeHeader == "" {
			w.Header().Set("Content-Length", strconv.Itoa(len(full)))
			w.Write([]byte(full))
			return
		}
		// bytes=500-
		w.Header().Set("Content-Length", strconv.Itoa(500))
		w.WriteHeader(http.StatusPartialContent)
		w.Write([]byte(full[500:]))
	}))
	defer srv.Close()

	c := NewClient("")
	c.BaseURL = srv.URL
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.gguf")

	if err := os.WriteFile(dest+".part", []byte(full[:500]), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := c.Download(context.Background(), "repo", "model.gguf", dest, nil); err != nil {
		t.Fatal(err)
	}
	if gotRangeHeader != "bytes=500-" {
		t.Errorf("Range header = %q, want bytes=500-", gotRangeHeader)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != full {
		t.Errorf("resumed download mismatch: got %d bytes, want %d", len(got), len(full))
	}
}

func TestDownloadServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	c := NewClient("")
	c.BaseURL = srv.URL
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.gguf")

	if err := c.Download(context.Background(), "repo", "f.gguf", dest, nil); err == nil {
		t.Fatal("expected error")
	}
	if _, err := os.Stat(dest); !os.IsNotExist(err) {
		t.Error("destPath should not exist after a failed download")
	}
}

func TestFileURLKeepsSubfolders(t *testing.T) {
	c := NewClient("")
	got := c.FileURL("Org/Repo", "Q4_K_M/model 1-00001-of-00002.gguf")
	want := "https://huggingface.co/Org/Repo/resolve/main/Q4_K_M/model%201-00001-of-00002.gguf"
	if got != want {
		t.Errorf("FileURL = %q, want %q", got, want)
	}
}
