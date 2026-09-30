package httpapi

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
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

func TestExpertProfilesMatchLatestSession(t *testing.T) {
	dir := t.TempDir()
	run := func(profile, name string, hits string) {
		p := filepath.Join(dir, profile)
		os.MkdirAll(p, 0o755)
		body := `{"n_expert":4,"gen_tokens":1000,"prompt_tokens":0,"layers":[{"layer":0,"gen":` + hits + `,"prompt":[0,0,0,0]}]}`
		if err := os.WriteFile(filepath.Join(p, name+".json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	run("chat", "run-1", "[900,50,25,25]")
	run("coding", "run-2", "[25,25,50,900]")
	// the active profile is chat, but the latest session there routed like coding
	run("chat", "run-3", "[30,20,60,890]")
	os.WriteFile(filepath.Join(dir, "active"), []byte("chat"), 0o644)

	r, err := expertStats(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if r.Profile != "chat" || r.Best != "coding" || len(r.Profiles) != 2 || r.Runs != 2 {
		t.Fatalf("got profile %q best %q, %+v", r.Profile, r.Best, r)
	}
}

func TestMigrateExpertStats(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "run-1.json"), []byte(`{}`), 0o644)
	migrateExpertStats(dir)
	if _, err := os.Stat(filepath.Join(dir, defaultExpertProfile, "run-1.json")); err != nil {
		t.Fatal("run file not moved into the default profile")
	}
}

func TestLiveRunSplitsAtProfileSwitch(t *testing.T) {
	dir := t.TempDir()
	live := filepath.Join(liveDir(dir), "run-9.json")
	os.MkdirAll(liveDir(dir), 0o755)
	write := func(gen string, tokens int) {
		body := `{"n_expert":2,"gen_tokens":` + strconv.Itoa(tokens) + `,"prompt_tokens":0,"layers":[{"layer":0,"gen":` + gen + `,"prompt":[0,0]}]}`
		if err := os.WriteFile(live, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeSegments(live, []expertSegment{{Profile: "default"}})
	write("[10,0]", 10)
	if err := switchLiveProfile(live, "coding"); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "active"), []byte("coding"), 0o644)
	write("[10,6]", 16) // 6 more tokens, all to expert 1, after the switch

	r, err := expertStats(dir, live)
	if err != nil {
		t.Fatal(err)
	}
	if r.Profile != "coding" || r.GenTokens != 6 {
		t.Fatalf("active profile while running: %+v", r)
	}

	finalizeLiveRuns(dir, "") // the model stopped
	for profile, want := range map[string]uint64{"default": 10, "coding": 6} {
		c := loadExpertCounts(filepath.Join(dir, profile), "")
		if c.genTokens != want || c.runs != 1 {
			t.Errorf("%s: %d tokens in %d runs, want %d in 1", profile, c.genTokens, c.runs, want)
		}
	}
	if left, _ := filepath.Glob(filepath.Join(liveDir(dir), "*")); len(left) != 0 {
		t.Errorf("live files left: %v", left)
	}
}

func TestAutoProfilingRunsAreNotTheLatestSession(t *testing.T) {
	dir := t.TempDir()
	write := func(profile, name, hits string, auto bool) {
		p := filepath.Join(dir, profile)
		os.MkdirAll(p, 0o755)
		os.WriteFile(filepath.Join(p, name+".json"), []byte(`{"n_expert":4,"gen_tokens":1000,"prompt_tokens":0,"layers":[{"layer":0,"gen":`+hits+`,"prompt":[0,0,0,0]}]}`), 0o644)
		if auto {
			os.WriteFile(filepath.Join(p, name+".meta"), []byte(`{"auto":true}`), 0o644)
		}
	}
	write("chat", "run-1", "[900,50,25,25]", false)
	write("chat", "run-2", "[30,20,60,890]", false) // the user's latest session, coding-like
	write("coding", "run-3", "[25,25,50,900]", true)
	write("chat", "run-4", "[900,40,30,30]", true) // newer, but auto-profiling
	os.WriteFile(filepath.Join(dir, "active"), []byte("chat"), 0o644)

	r, err := expertStats(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if r.Best != "coding" {
		t.Errorf("best = %q, want coding (matched on run-2, not the auto-profiling run-4)", r.Best)
	}
}
