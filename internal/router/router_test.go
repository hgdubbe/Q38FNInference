package router

import (
	"slices"
	"strings"
	"testing"
)

func TestINI(t *testing.T) {
	ini, errs := INI([]Entry{
		{
			Name: "RVN-Qwen3.8-Flash-Next-IQ4_XS",
			Path: `C:\Users\me\models\RVN-00001-of-00008.gguf`,
			Args: []string{"--model", "ignored.gguf", "--ctx-size", "65536", "--n-cpu-moe", "30",
				"--tensor-split", "40,9", "--reasoning-budget", "-1", "--mlock",
				"--chat-template-kwargs", `{"thinking":false}`, "--host", "0.0.0.0", "--port", "1"},
		},
		{Name: "org/cached:Q4_K_M-2", Args: []string{"--temp", "0.6"}},
	}, nil)
	if len(errs) != 0 {
		t.Fatalf("errs = %v", errs)
	}
	for _, want := range []string{
		"version = 1",
		"[RVN-Qwen3.8-Flash-Next-IQ4_XS]",
		`model = C:\Users\me\models\RVN-00001-of-00008.gguf`,
		"ctx-size = 65536",
		"n-cpu-moe = 30",
		"tensor-split = 40,9",
		"reasoning-budget = -1",
		"mlock = true",
		`chat-template-kwargs = {"thinking":false}`,
		"[org/cached:Q4_K_M-2]\ntemp = 0.6",
	} {
		if !strings.Contains(ini, want) {
			t.Errorf("INI missing %q:\n%s", want, ini)
		}
	}
	for _, bad := range []string{"ignored.gguf", "host =", "port ="} {
		if strings.Contains(ini, bad) {
			t.Errorf("INI must not contain %q (router-owned):\n%s", bad, ini)
		}
	}
}

func TestINIRejectsUnrepresentableValues(t *testing.T) {
	ini, errs := INI([]Entry{{Name: "m", Path: "/m.gguf", Args: []string{"--alias", "x", "--reasoning-budget-message", "done; answer"}}}, nil)
	if len(errs) != 1 || strings.Contains(ini, "done") {
		t.Errorf("errs = %v, ini:\n%s", errs, ini)
	}
}

func TestSafeAndUniqueNames(t *testing.T) {
	if got := SafeName(" org/model:Q4 [x];#y "); got != "org/model-Q4 -x-y" {
		t.Errorf("SafeName = %q", got)
	}
	got := UniqueNames([]string{"a", "a", "b", "a", "a-2"})
	if want := []string{"a", "a-2", "b", "a-3", "a-2-2"}; !slices.Equal(got, want) {
		t.Errorf("UniqueNames = %v, want %v", got, want)
	}
}

func TestINIDropsUnknownFlags(t *testing.T) {
	known := KnownFlags("-t,    --threads N   number of threads\n-lm,   --load-mode MODE  (env: LLAMA_ARG_LOAD_MODE)\n--temp N")
	ini, errs := INI([]Entry{{Name: "m", Path: "/m.gguf", Args: []string{"--temp", "0.6", "--mlock", "--load-mode", "mmap+mlock"}}}, known)
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "--mlock") {
		t.Errorf("errs = %v, want one about --mlock", errs)
	}
	if strings.Contains(ini, "mlock = true") || !strings.Contains(ini, "load-mode = mmap+mlock") || !strings.Contains(ini, "temp = 0.6") {
		t.Errorf("ini:\n%s", ini)
	}
}
