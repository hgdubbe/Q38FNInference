package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// backend echoes the request body it received.
func backend(t *testing.T) (*httptest.Server, *url.URL) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(b)
	}))
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	return srv, u
}

func post(t *testing.T, h http.Handler, path, body string) map[string]any {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response %q: %v", rec.Body.String(), err)
	}
	return out
}

func messages(t *testing.T, out map[string]any) []map[string]any {
	t.Helper()
	var res []map[string]any
	for _, m := range out["messages"].([]any) {
		res = append(res, m.(map[string]any))
	}
	return res
}

func TestNoTargetReturns503(t *testing.T) {
	rec := httptest.NewRecorder()
	New().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("code = %d, want 503", rec.Code)
	}
}

func TestOverrideReplacesSystemMessages(t *testing.T) {
	_, u := backend(t)
	p := New()
	p.SetTarget(u)
	p.SetSystemPrompt(ModeOverride, "be terse")

	out := post(t, p, "/v1/chat/completions", `{"seed":12345678901234567,"messages":[{"role":"system","content":"old"},{"role":"user","content":"hi"}]}`)
	msgs := messages(t, out)
	if len(msgs) != 2 || msgs[0]["role"] != "system" || msgs[0]["content"] != "be terse" || msgs[1]["content"] != "hi" {
		t.Errorf("messages = %v", msgs)
	}
}

func TestSeedPrecisionPreserved(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		w.Write(b)
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	p := New()
	p.SetTarget(u)
	p.SetSystemPrompt(ModeOverride, "x")

	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"seed":12345678901234567,"messages":[]}`)))
	if !strings.Contains(rec.Body.String(), "12345678901234567") {
		t.Errorf("seed lost precision: %s", rec.Body.String())
	}
}

func TestDefaultKeepsExistingSystem(t *testing.T) {
	_, u := backend(t)
	p := New()
	p.SetTarget(u)
	p.SetSystemPrompt(ModeDefault, "fallback")

	msgs := messages(t, post(t, p, "/v1/chat/completions", `{"messages":[{"role":"system","content":"mine"},{"role":"user","content":"hi"}]}`))
	if msgs[0]["content"] != "mine" || len(msgs) != 2 {
		t.Errorf("default mode must keep the client's system prompt: %v", msgs)
	}

	msgs = messages(t, post(t, p, "/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`))
	if msgs[0]["role"] != "system" || msgs[0]["content"] != "fallback" {
		t.Errorf("default mode must add a prompt when none is given: %v", msgs)
	}
}

func TestOffPassesThrough(t *testing.T) {
	_, u := backend(t)
	p := New()
	p.SetTarget(u)
	p.SetSystemPrompt(ModeOverride, "") // empty prompt disables

	msgs := messages(t, post(t, p, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`))
	if len(msgs) != 1 {
		t.Errorf("messages = %v, want untouched", msgs)
	}
}

func TestAnthropicMessages(t *testing.T) {
	_, u := backend(t)
	p := New()
	p.SetTarget(u)
	p.SetSystemPrompt(ModeOverride, "sys")
	out := post(t, p, "/v1/messages", `{"system":"old","messages":[]}`)
	if out["system"] != "sys" {
		t.Errorf("system = %v", out["system"])
	}
}

func TestStreamingPassesThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, tok := range []string{"a", "b"} {
			w.Write([]byte("data: " + tok + "\n\n"))
			w.(http.Flusher).Flush()
		}
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	p := New()
	p.SetTarget(u)

	front := httptest.NewServer(p)
	defer front.Close()
	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"stream":true,"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	if string(b) != "data: a\n\ndata: b\n\n" {
		t.Errorf("stream = %q", b)
	}
}

func TestCombineMergesIntoClientSystemPrompt(t *testing.T) {
	_, u := backend(t)
	p := New()
	p.SetTarget(u)
	p.SetSystemPrompt(ModeCombine, "be careful")

	msgs := messages(t, post(t, p, "/v1/chat/completions", `{"messages":[{"role":"system","content":"agent tools: ..."},{"role":"user","content":"hi"}]}`))
	if len(msgs) != 2 || msgs[0]["content"] != "be careful\n\nagent tools: ..." {
		t.Errorf("messages = %v, want one merged system message", msgs)
	}

	msgs = messages(t, post(t, p, "/v1/chat/completions", `{"messages":[{"role":"system","content":[{"type":"text","text":"tools"}]},{"role":"user","content":"hi"}]}`))
	parts := msgs[0]["content"].([]any)
	if len(parts) != 2 || parts[0].(map[string]any)["text"] != "be careful\n\n" {
		t.Errorf("content parts = %v, want the prompt prepended as a text part", parts)
	}

	msgs = messages(t, post(t, p, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`))
	if len(msgs) != 2 || msgs[0]["content"] != "be careful" {
		t.Errorf("messages = %v, want the prompt inserted when the client sent none", msgs)
	}

	out := post(t, p, "/v1/messages", `{"system":"client","messages":[]}`)
	if out["system"] != "be careful\n\nclient" {
		t.Errorf("anthropic system = %v", out["system"])
	}
}
