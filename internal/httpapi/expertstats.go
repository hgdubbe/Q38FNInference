package httpapi

// Expert usage recording (patches/0009, LLAMA_EXPERT_STATS). With the
// "Record expert usage" model setting, each launch writes one run file of
// per-layer, per-expert routing counts into the model's active profile
// (expert-stats/<model id>/<profile>/); the Run page sums a profile's runs
// to show whether a few experts per layer take most of the tokens, i.e.
// whether keeping just those in VRAM would pay off, and which profile the
// latest session's routing resembles most.

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/hgdubbe/q38fninference/internal/appconfig"
	"github.com/hgdubbe/q38fninference/internal/gguf"
)

const defaultExpertProfile = "default"

var profileNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}$`)

type expertRun struct {
	NExpert      int    `json:"n_expert"`
	GenTokens    uint64 `json:"gen_tokens"`
	PromptTokens uint64 `json:"prompt_tokens"`
	Layers       []struct {
		Layer  int      `json:"layer"`
		Gen    []uint64 `json:"gen"`
		Prompt []uint64 `json:"prompt"`
	} `json:"layers"`
}

// expertRunMeta is written next to a run file at launch: which blocks kept
// their experts in RAM (blocks 0..RAMLayers-1).
type expertRunMeta struct {
	RAMLayers int `json:"ram_layers"`
}

// expertModelDir holds a model's profiles, one folder each, and the name of
// the active one in the file "active".
func expertModelDir(modelPath string) (string, error) {
	dir, err := appconfig.Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "expert-stats", modelName(modelPath)), nil
}

func activeExpertProfile(modelDir string) string {
	b, err := os.ReadFile(filepath.Join(modelDir, "active"))
	if name := strings.TrimSpace(string(b)); err == nil && profileNameRe.MatchString(name) {
		return name
	}
	return defaultExpertProfile
}

func expertProfiles(modelDir string) []string {
	entries, _ := os.ReadDir(modelDir)
	names := []string{}
	for _, e := range entries {
		if e.IsDir() && profileNameRe.MatchString(e.Name()) {
			names = append(names, e.Name())
		}
	}
	return names
}

// expertStatsEnv prepares a new run file in the active profile for a launch
// and returns the environment variable pointing llama-server at it.
func expertStatsEnv(modelPath string, args []string) (string, error) {
	modelDir, err := expertModelDir(modelPath)
	if err != nil {
		return "", err
	}
	dir := filepath.Join(modelDir, activeExpertProfile(modelDir))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	base := filepath.Join(dir, "run-"+strconv.FormatInt(time.Now().UnixNano(), 10))
	meta, _ := json.Marshal(expertRunMeta{RAMLayers: ramExpertLayers(modelPath, args)})
	if err := os.WriteFile(base+".meta", meta, 0o644); err != nil {
		return "", err
	}
	return "LLAMA_EXPERT_STATS=" + base + ".json", nil
}

// ramExpertLayers counts the blocks whose experts a command line leaves in
// RAM: the blocks -ngl doesn't offload, or the first --n-cpu-moe.
func ramExpertLayers(modelPath string, args []string) int {
	meta, err := gguf.ReadModel(modelPath)
	if err != nil {
		return 0
	}
	n, ok := meta.NLayer()
	if !ok {
		return 0
	}
	nLayer := int(n)
	val := func(names ...string) (int, bool) {
		for i := 0; i+1 < len(args); i++ {
			for _, f := range names {
				if args[i] == f {
					v, err := strconv.Atoi(args[i+1])
					return v, err == nil
				}
			}
		}
		return 0, false
	}
	ram := 0
	if ngl, ok := val("--n-gpu-layers", "-ngl", "--gpu-layers"); ok && ngl >= 0 && ngl <= nLayer {
		ram = nLayer + 1 - ngl // n_layer+1 slots; -ngl offloads the last ones
	}
	if cm, ok := val("--n-cpu-moe", "-ncmoe"); ok {
		ram = max(ram, cm)
	}
	for _, a := range args {
		if a == "--cpu-moe" || a == "-cmoe" {
			ram = nLayer
		}
	}
	return min(ram, nLayer)
}

// expertCoverage is the share of routed tokens that the busiest Share of
// each layer's experts took (averaged over layers); an even spread gives
// Share itself.
type expertCoverage struct {
	Share float64 `json:"share"`
	All   float64 `json:"all"`
	RAM   float64 `json:"ram"` // layers with experts in RAM; 0 if none
}

type expertSummary struct {
	Runs         int              `json:"runs"`
	GenTokens    uint64           `json:"gen_tokens"`
	PromptTokens uint64           `json:"prompt_tokens"`
	Source       string           `json:"source"` // "generation", or "prompt" before any generation
	NExpert      int              `json:"n_expert"`
	Layers       int              `json:"layers"`
	RAMLayers    int              `json:"ram_layers"`
	Coverage     []expertCoverage `json:"coverage"`
}

// expertCounts are routing counts per layer and expert.
type expertCounts struct {
	gen, prompt   map[int][]uint64
	genTokens     uint64
	promptTokens  uint64
	nExpert, runs int
	ramLayers     int
	newest        string // file name of the newest run
}

func newExpertCounts() *expertCounts {
	return &expertCounts{gen: map[int][]uint64{}, prompt: map[int][]uint64{}}
}

func (c *expertCounts) add(r expertRun) {
	addTo := func(m map[int][]uint64, layer int, v []uint64) {
		acc := m[layer]
		if acc == nil {
			acc = make([]uint64, len(v))
			m[layer] = acc
		}
		for i := range min(len(acc), len(v)) {
			acc[i] += v[i]
		}
	}
	c.nExpert = r.NExpert
	c.runs++
	c.genTokens += r.GenTokens
	c.promptTokens += r.PromptTokens
	for _, l := range r.Layers {
		addTo(c.gen, l.Layer, l.Gen)
		addTo(c.prompt, l.Layer, l.Prompt)
	}
}

// routing picks the generation counts, or the prompt counts before any
// generation was recorded.
func (c *expertCounts) routing() (map[int][]uint64, string) {
	if c.genTokens == 0 {
		return c.prompt, "prompt"
	}
	return c.gen, "generation"
}

// loadExpertCounts sums the run files in a profile folder, skipping skip
// (a file name) if set.
func loadExpertCounts(dir, skip string) *expertCounts {
	c := newExpertCounts()
	files, _ := filepath.Glob(filepath.Join(dir, "run-*.json"))
	sort.Strings(files) // run-<unix nanos>: oldest first
	for _, f := range files {
		if filepath.Base(f) == skip {
			continue
		}
		r, ok := readExpertRun(f)
		if !ok || (c.nExpert != 0 && r.NExpert != c.nExpert) {
			continue
		}
		c.add(r)
		c.newest = filepath.Base(f)
		if mb, err := os.ReadFile(strings.TrimSuffix(f, ".json") + ".meta"); err == nil {
			var m expertRunMeta
			if json.Unmarshal(mb, &m) == nil {
				c.ramLayers = m.RAMLayers // the latest run's placement
			}
		}
	}
	return c
}

func readExpertRun(path string) (expertRun, bool) {
	var r expertRun
	b, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(b, &r) != nil || r.NExpert <= 0 {
		return r, false
	}
	return r, true
}

// summarizeExpertStats sums every run file in a profile folder.
func summarizeExpertStats(dir string) (expertSummary, error) {
	c := loadExpertCounts(dir, "")
	sum := expertSummary{Runs: c.runs, GenTokens: c.genTokens, PromptTokens: c.promptTokens, NExpert: c.nExpert, RAMLayers: c.ramLayers}
	if c.runs == 0 {
		return sum, nil
	}
	counts, src := c.routing()
	sum.Source = src
	sum.Layers = len(counts)
	for _, share := range []float64{0.1, 0.25, 0.5} {
		cv := expertCoverage{Share: share}
		cv.All, _ = meanCoverage(counts, share, func(int) bool { return true })
		cv.RAM, _ = meanCoverage(counts, share, func(l int) bool { return l < c.ramLayers })
		sum.Coverage = append(sum.Coverage, cv)
	}
	return sum, nil
}

func meanCoverage(counts map[int][]uint64, share float64, use func(layer int) bool) (float64, int) {
	var total float64
	n := 0
	for layer, hits := range counts {
		if !use(layer) || len(hits) == 0 {
			continue
		}
		s := append([]uint64(nil), hits...)
		sort.Slice(s, func(i, j int) bool { return s[i] > s[j] })
		k := int(math.Ceil(share * float64(len(s))))
		var top, all uint64
		for i, v := range s {
			all += v
			if i < k {
				top += v
			}
		}
		if all == 0 {
			continue
		}
		total += float64(top) / float64(all)
		n++
	}
	if n == 0 {
		return 0, 0
	}
	return total / float64(n), n
}

// routingSimilarity compares two routing patterns: the cosine similarity of
// each layer's expert counts, averaged over the layers both have. 1 means
// the same experts in the same proportions.
func routingSimilarity(a, b map[int][]uint64) float64 {
	var total float64
	n := 0
	for layer, x := range a {
		y := b[layer]
		if len(y) != len(x) {
			continue
		}
		var dot, nx, ny float64
		for i := range x {
			fx, fy := float64(x[i]), float64(y[i])
			dot += fx * fy
			nx += fx * fx
			ny += fy * fy
		}
		if nx == 0 || ny == 0 {
			continue
		}
		total += dot / math.Sqrt(nx*ny)
		n++
	}
	if n == 0 {
		return 0
	}
	return total / float64(n)
}

// minMatchTokens is how much routing the latest session needs before it is
// compared with the profiles.
const minMatchTokens = 500

type expertProfileInfo struct {
	Name   string  `json:"name"`
	Runs   int     `json:"runs"`
	Tokens uint64  `json:"tokens"`
	Match  float64 `json:"match,omitempty"` // similarity to the latest session, if compared
}

type expertStatsResponse struct {
	expertSummary
	Profile  string              `json:"profile"`
	Profiles []expertProfileInfo `json:"profiles"`
	// Best is the profile the latest session's routing resembles most,
	// when there are at least two to compare.
	Best string `json:"best,omitempty"`
}

// expertStats reports the active profile's summary, every profile, and how
// well each matches the latest session in the active profile.
func expertStats(modelDir string) (expertStatsResponse, error) {
	active := activeExpertProfile(modelDir)
	sum, err := summarizeExpertStats(filepath.Join(modelDir, active))
	if err != nil {
		return expertStatsResponse{}, err
	}
	resp := expertStatsResponse{expertSummary: sum, Profile: active}

	var latest map[int][]uint64
	var latestFile string
	cur := loadExpertCounts(filepath.Join(modelDir, active), "")
	if cur.newest != "" {
		if r, ok := readExpertRun(filepath.Join(modelDir, active, cur.newest)); ok {
			one := newExpertCounts()
			one.add(r)
			if r.GenTokens+r.PromptTokens >= minMatchTokens {
				latest, _ = one.routing()
				latestFile = cur.newest
			}
		}
	}

	names := expertProfiles(modelDir)
	if !contains(names, active) {
		names = append(names, active)
	}
	sort.Strings(names)
	withData := 0
	for _, name := range names {
		c := loadExpertCounts(filepath.Join(modelDir, name), "")
		info := expertProfileInfo{Name: name, Runs: c.runs, Tokens: c.genTokens}
		if c.runs > 0 {
			withData++
		}
		if latest != nil {
			// the latest session itself doesn't count towards its own profile
			other := c
			if name == active {
				other = loadExpertCounts(filepath.Join(modelDir, name), latestFile)
			}
			if other.runs > 0 {
				counts, _ := other.routing()
				info.Match = routingSimilarity(latest, counts)
			}
		}
		resp.Profiles = append(resp.Profiles, info)
	}
	if latest != nil && withData >= 2 {
		best := -1.0
		for _, p := range resp.Profiles {
			if p.Match > best && p.Match > 0 {
				best, resp.Best = p.Match, p.Name
			}
		}
	}
	return resp, nil
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// migrateExpertStats moves run files from before profiles existed (directly
// in the model folder) into the default profile.
func migrateExpertStats(modelDir string) {
	files, _ := filepath.Glob(filepath.Join(modelDir, "run-*"))
	if len(files) == 0 {
		return
	}
	dst := filepath.Join(modelDir, defaultExpertProfile)
	if os.MkdirAll(dst, 0o755) != nil {
		return
	}
	for _, f := range files {
		os.Rename(f, filepath.Join(dst, filepath.Base(f)))
	}
}

// GET    /api/expert-stats?model=<path>            summary, profiles, best match
// POST   /api/expert-stats {model, profile}        make profile active (created if new)
// DELETE /api/expert-stats?model=<path>&profile=n  delete a profile's runs
func (s *Server) handleExpertStats(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model   string `json:"model"`
		Profile string `json:"profile"`
	}
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
	} else {
		req.Model, req.Profile = r.URL.Query().Get("model"), r.URL.Query().Get("profile")
	}
	if req.Model == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("model is required"))
		return
	}
	if req.Profile != "" && !profileNameRe.MatchString(req.Profile) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("profile names are 1-40 letters, digits, spaces, '.', '_' or '-'"))
		return
	}
	modelDir, err := expertModelDir(req.Model)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	migrateExpertStats(modelDir)

	switch r.Method {
	case http.MethodPost:
		if req.Profile == "" {
			writeErr(w, http.StatusBadRequest, fmt.Errorf("profile is required"))
			return
		}
		if err := os.MkdirAll(filepath.Join(modelDir, req.Profile), 0o755); err == nil {
			err = os.WriteFile(filepath.Join(modelDir, "active"), []byte(req.Profile), 0o644)
		}
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
	case http.MethodDelete:
		profile := req.Profile
		if profile == "" {
			profile = activeExpertProfile(modelDir)
		}
		// files only: a running model keeps writing its run into the folder
		files, _ := filepath.Glob(filepath.Join(modelDir, profile, "run-*"))
		for _, f := range files {
			if err := os.Remove(f); err != nil {
				writeErr(w, http.StatusInternalServerError, err)
				return
			}
		}
		if profile != activeExpertProfile(modelDir) {
			os.Remove(filepath.Join(modelDir, profile)) // gone from the list once empty
		}
	}
	resp, err := expertStats(modelDir)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, resp)
}
