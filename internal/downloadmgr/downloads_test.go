package downloadmgr

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/hgdubbe/q38fninference/internal/hf"
)

func TestStartTracksProgressToDone(t *testing.T) {
	content := strings.Repeat("X", 10_000)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		w.Write([]byte(content))
	}))
	defer srv.Close()

	client := hf.NewClient("")
	client.BaseURL = srv.URL
	m := NewManager(client)

	dest := filepath.Join(t.TempDir(), "model.gguf")
	d := m.Start(context.Background(), "repo", "model.gguf", dest)
	if d.State != StatePending && d.State != StateActive {
		t.Errorf("initial state = %v", d.State)
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		cur := m.Get(d.ID)
		if cur.State == StateDone {
			if cur.Done != int64(len(content)) {
				t.Errorf("Done = %d, want %d", cur.Done, len(content))
			}
			return
		}
		if cur.State == StateError {
			t.Fatalf("download errored: %s", cur.Err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("timed out waiting for download to complete")
}

func TestGetUnknownID(t *testing.T) {
	m := NewManager(hf.NewClient(""))
	if d := m.Get("nope"); d != nil {
		t.Errorf("Get(unknown) = %v, want nil", d)
	}
}

func TestListReturnsAllDownloads(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	defer srv.Close()

	client := hf.NewClient("")
	client.BaseURL = srv.URL
	m := NewManager(client)

	dir := t.TempDir()
	m.Start(context.Background(), "repo", "a.gguf", filepath.Join(dir, "a.gguf"))
	m.Start(context.Background(), "repo", "b.gguf", filepath.Join(dir, "b.gguf"))

	// wait for both to finish too: they write into dir, which the test's
	// cleanup removes
	finished := func() bool {
		l := m.List()
		for _, d := range l {
			if d.State != StateDone && d.State != StateError {
				return false
			}
		}
		return len(l) == 2
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && !finished() {
		time.Sleep(10 * time.Millisecond)
	}
	if got := len(m.List()); got != 2 {
		t.Fatalf("List() len = %d, want 2", got)
	}
}

func TestStartDedupesActiveDownload(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
		w.Write([]byte("x"))
	}))
	defer srv.Close()
	defer close(block)

	client := hf.NewClient("")
	client.BaseURL = srv.URL
	m := NewManager(client)
	dest := filepath.Join(t.TempDir(), "m.gguf")
	a := m.Start(context.Background(), "r", "m.gguf", dest)
	b := m.Start(context.Background(), "r", "m.gguf", dest)
	if a.ID != b.ID {
		t.Errorf("second Start got %s, want existing %s", b.ID, a.ID)
	}
}
