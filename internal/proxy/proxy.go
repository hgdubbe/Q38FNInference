// Package proxy fronts llama-server with a reverse proxy that can inject or
// override the system prompt. llama-server has no system-prompt flag, so
// this is the one place a prompt set in the launcher reaches every client:
// external OpenAI/Anthropic-style API callers and llama-server's own web UI
// (which is served through here too). Everything else, including streaming
// responses, passes through untouched.
package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
)

// Mode controls how the configured system prompt is applied.
type Mode string

const (
	ModeOff      Mode = "off"      // never touch requests
	ModeDefault  Mode = "default"  // add the prompt only if the request has none
	ModeOverride Mode = "override" // replace whatever system prompt the request has
	// ModeCombine puts the prompt in front of the request's own system prompt,
	// e.g. to steer a coding agent without dropping its tool instructions
	ModeCombine Mode = "combine"
)

const maxBody = 64 << 20

// Proxy forwards to the current llama-server, if one is running.
type Proxy struct {
	mu     sync.RWMutex
	target *url.URL
	rp     *httputil.ReverseProxy
	mode   Mode
	prompt string
}

func New() *Proxy { return &Proxy{mode: ModeOff} }

// SetTarget points the proxy at a running llama-server, or nil when stopped.
func (p *Proxy) SetTarget(u *url.URL) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.target = u
	if u == nil {
		p.rp = nil
		return
	}
	p.rp = &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(u)
			r.Out.Host = u.Host
		},
		FlushInterval: -1, // stream SSE tokens as they arrive
	}
}

// SetSystemPrompt updates the prompt policy for subsequent requests.
func (p *Proxy) SetSystemPrompt(mode Mode, prompt string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if prompt == "" {
		mode = ModeOff
	}
	p.mode, p.prompt = mode, prompt
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p.mu.RLock()
	rp, mode, prompt := p.rp, p.mode, p.prompt
	p.mu.RUnlock()

	if rp == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":{"message":"no model is running; start one from the Q38FNInference control panel","type":"unavailable"}}`))
		return
	}

	if r.Method == http.MethodPost && mode != ModeOff {
		var rewrite func(map[string]any, Mode, string) bool
		switch r.URL.Path {
		case "/v1/chat/completions", "/chat/completions":
			rewrite = rewriteOpenAI
		case "/v1/messages":
			rewrite = rewriteAnthropic
		}
		if rewrite != nil {
			if err := rewriteBody(r, func(req map[string]any) bool { return rewrite(req, mode, prompt) }); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
	}
	rp.ServeHTTP(w, r)
}

func rewriteBody(r *http.Request, fn func(map[string]any) bool) error {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBody))
	r.Body.Close()
	if err != nil {
		return err
	}
	out := body
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber() // keep seeds / token ids exact
	var req map[string]any
	if dec.Decode(&req) == nil && fn(req) {
		if b, err := json.Marshal(req); err == nil {
			out = b
		}
	}
	// malformed JSON is forwarded as-is so llama-server reports the error
	r.Body = io.NopCloser(bytes.NewReader(out))
	r.ContentLength = int64(len(out))
	r.Header.Del("Content-Length")
	return nil
}

func isSystemRole(m any) bool {
	msg, ok := m.(map[string]any)
	if !ok {
		return false
	}
	role, _ := msg["role"].(string)
	return role == "system" || role == "developer"
}

func rewriteOpenAI(req map[string]any, mode Mode, prompt string) bool {
	msgs, ok := req["messages"].([]any)
	if !ok {
		return false
	}
	if mode == ModeCombine && len(msgs) > 0 && isSystemRole(msgs[0]) {
		// merge into the first system message: many chat templates accept
		// only one, and only at the start
		first := msgs[0].(map[string]any)
		merged := make(map[string]any, len(first))
		for k, v := range first {
			merged[k] = v
		}
		merged["content"] = prependText(prompt, first["content"])
		req["messages"] = append([]any{merged}, msgs[1:]...)
		return true
	}
	hasSystem := false
	kept := make([]any, 0, len(msgs)+1)
	for _, m := range msgs {
		if isSystemRole(m) {
			hasSystem = true
			if mode == ModeOverride {
				continue
			}
		}
		kept = append(kept, m)
	}
	if mode == ModeDefault && hasSystem {
		return false
	}
	req["messages"] = append([]any{map[string]any{"role": "system", "content": prompt}}, kept...)
	return true
}

func rewriteAnthropic(req map[string]any, mode Mode, prompt string) bool {
	existing, has := req["system"]
	switch {
	case has && mode == ModeDefault:
		return false
	case has && mode == ModeCombine:
		req["system"] = prependText(prompt, existing)
	default:
		req["system"] = prompt
	}
	return true
}

// prependText puts prompt before content, which is either a plain string or
// a list of typed content parts.
func prependText(prompt string, content any) any {
	switch c := content.(type) {
	case string:
		if c == "" {
			return prompt
		}
		return prompt + "\n\n" + c
	case []any:
		return append([]any{map[string]any{"type": "text", "text": prompt + "\n\n"}}, c...)
	default:
		return prompt
	}
}
