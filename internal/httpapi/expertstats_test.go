package httpapi

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSummarizeExpertStats(t *testing.T) {
	dir := t.TempDir()
	// 4 experts; layer 0 (in RAM) sends 7 of 10 tokens to one expert, layer 1 is even
	run := `{"n_expert":4,"gen_tokens":10,"prompt_tokens":5,"layers":[
		{"layer":0,"gen":[7,1,1,1],"prompt":[5,0,0,0]},
		{"layer":1,"gen":[3,2,3,2],"prompt":[1,1,1,2]}]}`
	for _, n := range []string{"run-1", "run-2"} {
		if err := os.WriteFile(filepath.Join(dir, n+".json"), []byte(run), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	os.WriteFile(filepath.Join(dir, "run-2.meta"), []byte(`{"ram_layers":1}`), 0o644)

	s, err := summarizeExpertStats(dir)
	if err != nil {
		t.Fatal(err)
	}
	if s.Runs != 2 || s.GenTokens != 20 || s.Source != "generation" || s.RAMLayers != 1 || s.Layers != 2 {
		t.Fatalf("summary = %+v", s)
	}
	top25 := s.Coverage[1] // top 25% = 1 of 4 experts
	if math.Abs(top25.RAM-0.7) > 1e-9 || math.Abs(top25.All-(0.7+0.3)/2) > 1e-9 {
		t.Errorf("top 25%% coverage = %+v, want ram 0.7, all 0.5", top25)
	}
}
