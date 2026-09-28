// Package reasoning maps one user-facing setting (off/low/medium/high) onto
// whatever reasoning controls a model actually understands. Model families
// differ: most templates only toggle thinking (enable_thinking), some take
// an effort level with their own vocabulary (gpt-oss, Hy3, DeepSeek V4,
// Kimi K3, ...), one takes a token budget (Seed-OSS). The style is detected
// from the chat template embedded in the GGUF (tokenizer.chat_template), so
// it follows the model file rather than a list of names. Levels a template
// can't express are enforced with llama-server's --reasoning-budget, which
// works for any model that thinks in tagged blocks.
//
// Signatures were taken from the templates llama.cpp ships in
// models/templates at the pinned commit.
package reasoning

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
)

// Level is the user's choice.
type Level string

const (
	Off    Level = "off"
	Low    Level = "low"
	Medium Level = "medium"
	High   Level = "high"
)

// thinking-token caps for levels a template can't express natively
const (
	lowBudget    = 1024
	mediumBudget = 4096
)

// Style is a detected reasoning-control family.
type Style struct {
	Name string // shown in the UI
	kind kind
	// effort level used for "high" by DeepSeek V4 ("max", or "high" on
	// V4-Flash where "high" already means maximum)
	maxEffort string
}

type kind int

const (
	kindNone         kind = iota // no reasoning in the template
	kindTags                     // thinks in tags, no template switch
	kindToggle                   // enable_thinking
	kindThinkingBool             // bare `thinking` kwarg (Granite)
	kindEffortNone               // reasoning_effort, only "none" matters (Cohere2)
	kindEffort                   // free-form effort low/medium/high, can't disable (gpt-oss)
	kindStrength                 // reasoning_strength (Muse Glimmer)
	kindHy3                      // reasoning_effort no_think/low/high
	kindDeepSeekV4               // enable_thinking + reasoning_effort max
	kindSolar                    // reasoning_effort low/minimal = no thinking
	kindKimi                     // thinking + thinking_effort low/high/max
	kindMiniMax                  // thinking_mode enabled/disabled/adaptive
	kindSeed                     // thinking_budget kwarg
)

var (
	bareThinkingRe = regexp.MustCompile(`if\s+(not\s+)?thinking\b`)
	setThinkingRe  = regexp.MustCompile(`set\s+thinking\s*=`)
	thinkTagRe     = regexp.MustCompile(`<think>|<thinking>|<\|think\|>|<seed:think>|<\|channel\|>analysis|\[THINK\]|\[BEGIN FINAL RESPONSE\]`)
	// match template code, not prose that happens to mention the word
	seedRe    = regexp.MustCompile(`thinking_budget\s+is\s+defined`)
	kimiRe    = regexp.MustCompile(`thinking_effort\s+(is|in|not|==)`)
	miniMaxRe = regexp.MustCompile(`thinking_mode\s*==`)
)

// Detect classifies a chat template. Order matters: specific families
// before the generic toggle, which many of them also use.
func Detect(tmpl string) Style {
	has := func(s string) bool { return strings.Contains(tmpl, s) }
	switch {
	case tmpl == "":
		return Style{Name: "unknown (no chat template in the model)", kind: kindTags}
	case seedRe.MatchString(tmpl):
		return Style{Name: "thinking budget (Seed-OSS style)", kind: kindSeed}
	case kimiRe.MatchString(tmpl):
		return Style{Name: "thinking + effort (Kimi style)", kind: kindKimi}
	case miniMaxRe.MatchString(tmpl):
		return Style{Name: "thinking mode (MiniMax style)", kind: kindMiniMax}
	case has("reasoning_effort") && has("no_think"):
		return Style{Name: "effort no_think/low/high (Hy3 style)", kind: kindHy3}
	case has("reasoning_effort == 'max'"):
		s := Style{Name: "thinking + max effort (DeepSeek V4 style)", kind: kindDeepSeekV4, maxEffort: "max"}
		if has("reasoning_effort == 'high'") {
			s.maxEffort = "high"
		}
		return s
	case has(`reasoning_effort in ["low", "minimal"]`):
		return Style{Name: "effort, low = no thinking (Solar Open style)", kind: kindSolar}
	case has("reasoning_strength"):
		return Style{Name: "reasoning strength", kind: kindStrength}
	case has("reasoning_effort") && strings.Contains(strings.ToLower(tmpl), `"none"`):
		return Style{Name: "thinking on/off (reasoning_effort none)", kind: kindEffortNone}
	case has("reasoning_effort"):
		return Style{Name: "effort low/medium/high (gpt-oss style)", kind: kindEffort}
	case has("enable_thinking"):
		return Style{Name: "thinking on/off (enable_thinking)", kind: kindToggle}
	case bareThinkingRe.MatchString(tmpl) && !setThinkingRe.MatchString(tmpl):
		return Style{Name: "thinking on/off (thinking kwarg)", kind: kindThinkingBool}
	case thinkTagRe.MatchString(tmpl):
		return Style{Name: "thinks in tags, no switch", kind: kindTags}
	default:
		return Style{Name: "no reasoning", kind: kindNone}
	}
}

// Supported reports whether the setting has any effect for this model.
func (s Style) Supported() bool { return s.kind != kindNone }

// Args returns the llama-server flags that apply level for this style.
// An unknown level falls back to High.
func (s Style) Args(level Level) []string {
	switch level {
	case Off, Low, Medium, High:
	default:
		level = High
	}

	var (
		reasoning string // --reasoning on|off
		effort    string // --reasoning-effort
		budget    = -1   // --reasoning-budget
		kwargs    = map[string]any{}
	)
	stepBudget := func() {
		switch level {
		case Low:
			budget = lowBudget
		case Medium:
			budget = mediumBudget
		}
	}

	switch s.kind {
	case kindNone:
		return nil

	case kindTags:
		if level == Off {
			budget = 0
		}
		stepBudget()

	case kindToggle:
		if level == Off {
			reasoning = "off"
		} else {
			reasoning = "on"
			stepBudget()
		}

	case kindThinkingBool:
		kwargs["thinking"] = level != Off
		stepBudget()

	case kindEffortNone:
		if level == Off {
			effort = "none"
		} else {
			stepBudget()
		}

	case kindEffort:
		// gpt-oss can't switch analysis off: lowest effort, cut immediately
		if level == Off {
			effort, budget = "low", 0
		} else {
			effort = string(level)
		}

	case kindStrength:
		if level == Off {
			effort, budget = "low", 0
		} else {
			effort = string(level)
		}

	case kindHy3:
		switch level {
		case Off:
			effort = "no_think"
		case Low:
			effort = "low"
		case Medium:
			effort, budget = "high", mediumBudget
		case High:
			effort = "high"
		}

	case kindDeepSeekV4:
		if level == Off {
			reasoning = "off"
		} else {
			reasoning = "on"
			stepBudget()
			if level == High {
				effort = s.maxEffort
			}
		}

	case kindSolar:
		if level == Off {
			effort = "minimal"
		} else {
			effort = "high"
			stepBudget()
		}

	case kindKimi:
		kwargs["thinking"] = level != Off
		switch level {
		case Low:
			kwargs["thinking_effort"] = "low"
		case Medium:
			kwargs["thinking_effort"] = "high"
		case High:
			kwargs["thinking_effort"] = "max"
		}

	case kindMiniMax:
		switch level {
		case Off:
			kwargs["thinking_mode"] = "disabled"
		case Low:
			kwargs["thinking_mode"] = "adaptive"
			budget = lowBudget
		case Medium:
			kwargs["thinking_mode"] = "enabled"
			budget = mediumBudget
		case High:
			kwargs["thinking_mode"] = "enabled"
		}

	case kindSeed:
		stepBudget()
		if level == Off {
			budget = 0
		}
		kwargs["thinking_budget"] = budget // the template scales its own prompting to it
	}

	var args []string
	if reasoning != "" {
		args = append(args, "--reasoning", reasoning)
	}
	if effort != "" {
		args = append(args, "--reasoning-effort", effort)
	}
	if budget >= 0 {
		args = append(args, "--reasoning-budget", strconv.Itoa(budget))
	}
	if len(kwargs) > 0 {
		b, _ := json.Marshal(kwargs)
		args = append(args, "--chat-template-kwargs", string(b))
	}
	return args
}
