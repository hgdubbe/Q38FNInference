package httpapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// A fixed workload so runs are comparable across settings: a ~1500-token
// prompt (long enough to cross the op-offload threshold and use a large
// micro-batch) and exactly 128 generated tokens.
const (
	benchPromptSentences = 120
	benchGenTokens       = 128
)

func benchPrompt(sentences int) string {
	var b strings.Builder
	b.WriteString("Summarize the following inventory log in one sentence.\n\n")
	for i := 1; i <= sentences; i++ {
		fmt.Fprintf(&b, "Entry %d: crate %d of batch %d moved from shelf %c%d to loading bay %d at %02d:%02d.\n",
			i, i*7%97, i%13, 'A'+rune(i%6), i%40, i%5, i%24, i*11%60)
	}
	return b.String()
}

type benchResult struct {
	Model              string  `json:"model"`
	PromptTokens       int     `json:"prompt_tokens"`
	PromptPerSecond    float64 `json:"prompt_per_second"`
	GeneratedTokens    int     `json:"generated_tokens"`
	GeneratedPerSecond float64 `json:"generated_per_second"`
	WallSeconds        float64 `json:"wall_seconds"`
}

// handleBenchmark runs the fixed workload directly against llama-server
// (bypassing the system-prompt proxy) and reports llama-server's timings.
func (s *Server) handleBenchmark(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	target := s.Proxy.Target()
	if target == nil || !s.llama.Status().Ready {
		writeErr(w, http.StatusConflict, fmt.Errorf("start a model first"))
		return
	}
	start := time.Now()
	res, status, err := runBench(target.String(), req.Model, benchPromptSentences)
	if ctxErr, ok := err.(*contextError); ok && ctxErr.nPrompt > 0 {
		// shrink the prompt to fit a small context and try once more
		fit := benchPromptSentences * (ctxErr.nCtx - benchGenTokens - 32) / ctxErr.nPrompt
		if fit < 1 {
			writeErr(w, http.StatusBadGateway, fmt.Errorf("context of %d tokens is too small to benchmark", ctxErr.nCtx))
			return
		}
		res, status, err = runBench(target.String(), req.Model, fit)
	}
	if err != nil {
		writeErr(w, status, err)
		return
	}
	res.WallSeconds = time.Since(start).Seconds()
	writeJSON(w, res)
}

type contextError struct{ nPrompt, nCtx int }

func (e *contextError) Error() string {
	return fmt.Sprintf("prompt of %d tokens exceeds the context of %d", e.nPrompt, e.nCtx)
}

func runBench(target, model string, sentences int) (benchResult, int, error) {
	body := map[string]any{
		"messages":     []map[string]string{{"role": "user", "content": benchPrompt(sentences)}},
		"max_tokens":   benchGenTokens,
		"ignore_eos":   true, // always generate exactly benchGenTokens
		"temperature":  0,
		"cache_prompt": false, // measure a full prompt every time
		"stream":       false,
	}
	if model != "" {
		body["model"] = model
	}
	buf, _ := json.Marshal(body)

	c := &http.Client{Timeout: 15 * time.Minute} // on-demand mode may load the model first
	resp, err := c.Post(target+"/v1/chat/completions", "application/json", bytes.NewReader(buf))
	if err != nil {
		return benchResult{}, http.StatusBadGateway, err
	}
	defer resp.Body.Close()

	var out struct {
		Model   string `json:"model"`
		Timings struct {
			PromptN     int     `json:"prompt_n"`
			PromptPS    float64 `json:"prompt_per_second"`
			PredictedN  int     `json:"predicted_n"`
			PredictedPS float64 `json:"predicted_per_second"`
		} `json:"timings"`
		Error json.RawMessage `json:"error"`
		// llama-server's context-size error is a top-level object
		Type    string `json:"type"`
		NPrompt int    `json:"n_prompt_tokens"`
		NCtx    int    `json:"n_ctx"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return benchResult{}, http.StatusBadGateway, fmt.Errorf("reading llama-server response: %w", err)
	}
	if out.Type == "exceed_context_size_error" {
		return benchResult{}, http.StatusBadGateway, &contextError{nPrompt: out.NPrompt, nCtx: out.NCtx}
	}
	if len(out.Error) > 0 {
		var e struct {
			Type    string `json:"type"`
			NPrompt int    `json:"n_prompt_tokens"`
			NCtx    int    `json:"n_ctx"`
		}
		if json.Unmarshal(out.Error, &e) == nil && e.Type == "exceed_context_size_error" {
			return benchResult{}, http.StatusBadGateway, &contextError{nPrompt: e.NPrompt, nCtx: e.NCtx}
		}
	}
	if resp.StatusCode != http.StatusOK || len(out.Error) > 0 {
		return benchResult{}, http.StatusBadGateway, fmt.Errorf("llama-server: %s %s", resp.Status, out.Error)
	}
	return benchResult{
		Model:              out.Model,
		PromptTokens:       out.Timings.PromptN,
		PromptPerSecond:    out.Timings.PromptPS,
		GeneratedTokens:    out.Timings.PredictedN,
		GeneratedPerSecond: out.Timings.PredictedPS,
	}, http.StatusOK, nil
}
