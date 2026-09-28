package reasoning

import (
	"slices"
	"testing"
)

// Snippets are the distinguishing lines of the templates llama.cpp ships.
var (
	qwen3    = `{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}`
	gptOSS   = `{%- if not reasoning_effort is defined %}{%- set reasoning_effort = "medium" %}{%- endif %}Reasoning: {{ reasoning_effort }}<|channel|>analysis`
	hy3      = `{%- elif reasoning_effort not in ['high', 'low', 'no_think'] %}`
	dsV4     = `{%- set thinking = enable_thinking -%}{%- if thinking and reasoning_effort == 'max' -%}`
	dsV4F    = dsV4 + `{%- if reasoning_effort == 'high' -%}`
	solar    = `{%- if reasoning_effort in ["low", "minimal"] -%}<|think|>`
	cohere   = `{%- set thinking = false if reasoning_effort is defined and reasoning_effort | lower == "none" else true -%}`
	kimi     = `{%- if thinking_effort is undefined -%}{%- set thinking_effort = 'max' -%}`
	miniMax  = `{%- if thinking_mode == "enabled" -%}`
	seed     = `{%- if thinking_budget is defined -%}<seed:think>`
	granite  = `{%- if thinking %}Think step by step.{%- endif %}`
	r1       = `<｜Assistant｜><think>\n`
	plain    = `{{ '<|im_start|>' + message['role'] }}`
	dsProse  = `{%- set tools_header = 'If thinking_mode is enabled (triggered by <think>)' -%}{%- set thinking = enable_thinking -%}`
	seedDocs = `{# thinking_budget. #}{%- if enable_thinking -%}`
)

func TestDetect(t *testing.T) {
	cases := []struct {
		tmpl string
		kind kind
	}{
		{qwen3, kindToggle},
		{gptOSS, kindEffort},
		{hy3, kindHy3},
		{dsV4, kindDeepSeekV4},
		{solar, kindSolar},
		{cohere, kindEffortNone},
		{kimi, kindKimi},
		{miniMax, kindMiniMax},
		{seed, kindSeed},
		{granite, kindThinkingBool},
		{r1, kindTags},
		{plain, kindNone},
		{"", kindTags},
		// prose mentioning a variable name must not trigger that family
		{dsProse, kindToggle},
		{seedDocs, kindToggle},
	}
	for _, c := range cases {
		if got := Detect(c.tmpl).kind; got != c.kind {
			t.Errorf("Detect(%.50q) = %v (%s), want %v", c.tmpl, got, Detect(c.tmpl).Name, c.kind)
		}
	}
}

func TestArgs(t *testing.T) {
	cases := []struct {
		tmpl  string
		level Level
		want  []string
	}{
		// on/off-only families (Qwen3/3.5, GLM, ...): budgets fill the gap
		{qwen3, Off, []string{"--reasoning", "off"}},
		{qwen3, Low, []string{"--reasoning", "on", "--reasoning-budget", "1024"}},
		{qwen3, Medium, []string{"--reasoning", "on", "--reasoning-budget", "4096"}},
		{qwen3, High, []string{"--reasoning", "on"}},
		// gpt-oss can't disable analysis
		{gptOSS, Off, []string{"--reasoning-effort", "low", "--reasoning-budget", "0"}},
		{gptOSS, Medium, []string{"--reasoning-effort", "medium"}},
		{hy3, Off, []string{"--reasoning-effort", "no_think"}},
		{hy3, Medium, []string{"--reasoning-effort", "high", "--reasoning-budget", "4096"}},
		{dsV4, High, []string{"--reasoning", "on", "--reasoning-effort", "max"}},
		{dsV4F, High, []string{"--reasoning", "on", "--reasoning-effort", "high"}},
		{solar, Off, []string{"--reasoning-effort", "minimal"}},
		{cohere, Off, []string{"--reasoning-effort", "none"}},
		{cohere, High, nil},
		{kimi, Medium, []string{"--chat-template-kwargs", `{"thinking":true,"thinking_effort":"high"}`}},
		{kimi, Off, []string{"--chat-template-kwargs", `{"thinking":false}`}},
		{miniMax, Off, []string{"--chat-template-kwargs", `{"thinking_mode":"disabled"}`}},
		{seed, Low, []string{"--reasoning-budget", "1024", "--chat-template-kwargs", `{"thinking_budget":1024}`}},
		{seed, High, []string{"--chat-template-kwargs", `{"thinking_budget":-1}`}},
		{granite, Off, []string{"--chat-template-kwargs", `{"thinking":false}`}},
		{r1, Off, []string{"--reasoning-budget", "0"}},
		{r1, High, nil},
		{plain, Off, nil},
		// legacy/unknown values behave as high
		{qwen3, Level("on"), []string{"--reasoning", "on"}},
	}
	for _, c := range cases {
		got := Detect(c.tmpl).Args(c.level)
		if !slices.Equal(got, c.want) {
			t.Errorf("%s / %s: Args = %q, want %q", Detect(c.tmpl).Name, c.level, got, c.want)
		}
	}
}
