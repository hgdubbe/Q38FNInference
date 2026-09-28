package httpapi

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/hgdubbe/q38fninference/internal/gguf"
)

func newTestServer(t *testing.T) *Server {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	s, err := New(Hooks{})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestRoutes(t *testing.T) {
	h := newTestServer(t).Handler()
	cases := []struct {
		method, path string
		code         int
		contains     string
	}{
		{"GET", "/", 200, "<title>Q38FNInference</title>"},
		{"GET", "/app.js", 200, ""},
		{"GET", "/api/info", 200, `"app":"q38fninference"`},
		{"GET", "/api/config", 200, `"port":8080`},
		{"GET", "/api/server/status", 200, `"Running":false`},
		{"POST", "/v1/chat/completions", 503, "no model is running"},
	}
	for _, c := range cases {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(c.method, c.path, strings.NewReader("{}")))
		if rec.Code != c.code || !strings.Contains(rec.Body.String(), c.contains) {
			t.Errorf("%s %s = %d %q, want %d containing %q", c.method, c.path, rec.Code, rec.Body.String(), c.code, c.contains)
		}
	}
}

func TestDownloadRejectsPathTraversal(t *testing.T) {
	h := newTestServer(t).Handler()
	for _, body := range []string{
		`{"repo":"a/b","filename":"../../evil.gguf"}`,
		`{"repo":"../x","filename":"m.gguf"}`,
		`{"repo":"a/b","filename":"C:\\evil.gguf"}`,
		`{"repo":"a/b","filename":"/abs.gguf"}`,
	} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/api/hf/download", strings.NewReader(body)))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s -> %d, want 400", body, rec.Code)
		}
	}
}

func TestStripFlags(t *testing.T) {
	got := stripFlags([]string{"--model", "m", "--port", "1", "--host", "0.0.0.0", "--temp", "0.5"}, "--host", "--port")
	if want := []string{"--model", "m", "--temp", "0.5"}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestSizeCheckFlagsUndercount(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(p, make([]byte, 10000), 0o644); err != nil {
		t.Fatal(err)
	}
	// 256 f32 elements = 1024 bytes of tensors vs a 10000-byte file
	meta := &gguf.Metadata{Tensors: []gguf.TensorInfo{{Name: "x", Dims: []uint64{256}, Type: 0}}}
	if note := sizeCheck(p, meta); !strings.Contains(note, "underestimates") {
		t.Errorf("sizeCheck = %q, want an undercount warning", note)
	}
	meta.Tensors[0].Dims = []uint64{2400} // 9600 bytes: within 10%
	if note := sizeCheck(p, meta); note != "" {
		t.Errorf("sizeCheck = %q, want no warning", note)
	}
}

func TestPanelAbandoned(t *testing.T) {
	s := newTestServer(t)
	now := time.Now()
	if s.panelAbandoned(now, time.Second) {
		t.Error("must not exit before any panel was ever opened (browser may have failed to launch)")
	}
	s.panelOpened()
	s.panelOpened() // two tabs
	s.panelClosed()
	if s.panelAbandoned(now.Add(time.Hour), time.Second) {
		t.Error("one tab is still open")
	}
	s.panelClosed()
	if s.panelAbandoned(time.Now(), 15*time.Second) {
		t.Error("must wait out the grace period (page reloads reconnect within it)")
	}
	if !s.panelAbandoned(time.Now().Add(16*time.Second), 15*time.Second) {
		t.Error("should exit after the grace period with no panel open")
	}
	s.cfg.KeepRunning = true
	if s.panelAbandoned(time.Now().Add(time.Hour), 15*time.Second) {
		t.Error("keep_running must suppress the exit")
	}
}
