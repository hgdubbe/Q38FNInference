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
	"github.com/hgdubbe/q38fninference/internal/sysmem"
	"github.com/hgdubbe/q38fninference/internal/tuning"
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
	s := newTestServer(t) // no tray: closing the panel exits
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

	s.hooks.HasTray = true
	if s.panelAbandoned(time.Now().Add(time.Hour), 15*time.Second) {
		t.Error("with a tray icon the launcher keeps running unless exit_with_panel is set")
	}
	s.cfg.ExitWithPanel = true
	if !s.panelAbandoned(time.Now().Add(time.Hour), 15*time.Second) {
		t.Error("exit_with_panel must make it exit even with a tray icon")
	}
}

func TestRAMCheck(t *testing.T) {
	const G = uint64(1) << 30
	plan := &tuning.Plan{CPUFitBytes: 50 * G, InputBytes: 10 * G} // 40 GiB must be resident
	cases := []struct {
		mem  sysmem.Info
		want string
	}{
		{sysmem.Info{}, ""},
		{sysmem.Info{Total: 32 * G, Available: 20 * G, OK: true}, "very slow"},
		{sysmem.Info{Total: 64 * G, Available: 30 * G, OK: true}, "close other programs"},
		{sysmem.Info{Total: 64 * G, Available: 50 * G, OK: true}, "40.0 GiB of weights stay in RAM"},
	}
	for _, c := range cases {
		got := ramCheck(plan, c.mem)
		if (c.want == "") != (got == "") || !strings.Contains(got, c.want) {
			t.Errorf("ramCheck(%+v) = %q, want it to contain %q", c.mem, got, c.want)
		}
	}
}

func TestCacheRAMArgs(t *testing.T) {
	const G = uint64(1 << 30)
	plan := &tuning.Plan{CPUFitBytes: 45 * G, InputBytes: 5 * G} // 40 GiB of weights in RAM
	cases := []struct {
		mem  sysmem.Info
		want []string
	}{
		{sysmem.Info{}, nil},
		{sysmem.Info{Total: 128 * G, OK: true}, nil}, // the 8 GiB default fits
		{sysmem.Info{Total: 48 * G, OK: true}, []string{"--cache-ram", "4096"}},
		{sysmem.Info{Total: 32 * G, OK: true}, []string{"--cache-ram", "0"}},
	}
	for _, c := range cases {
		if got := cacheRAMArgs(plan, c.mem); !slices.Equal(got, c.want) {
			t.Errorf("cacheRAMArgs(total %d GiB) = %v, want %v", c.mem.Total/G, got, c.want)
		}
	}
}

func TestModelNameIsAPIFriendly(t *testing.T) {
	for in, want := range map[string]string{
		"/m/RVN-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00008.gguf": "RVN-Qwen3.8-Flash-Next-IQ4_XS",
		"/m/tiny.gguf": "tiny",
	} {
		if got := modelName(in); got != want {
			t.Errorf("modelName(%q) = %q, want %q", in, got, want)
		}
	}
}
