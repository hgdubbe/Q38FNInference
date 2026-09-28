package httpapi

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
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
