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

// validProfileName: "live" holds the running model's counts (see liveDir).
func validProfileName(name string) bool {
	return profileNameRe.MatchString(name) && !strings.EqualFold(name, "live")
}

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
	// Auto marks auto-profiling runs: they fill profiles but are never
	// "the latest session" that profiles are matched against.
	Auto bool `json:"auto,omitempty"`
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
	if name := strings.TrimSpace(string(b)); err == nil && validProfileName(name) {
		return name
	}
	return defaultExpertProfile
}

func expertProfiles(modelDir string) []string {
	entries, _ := os.ReadDir(modelDir)
	names := []string{}
	for _, e := range entries {
		if e.IsDir() && validProfileName(e.Name()) {
			names = append(names, e.Name())
		}
	}
	return names
}

// A running model writes its counts into live/run-<time>.json, cumulative
// since its start. live/run-<time>.seg lists which profile each stretch of
// that run belongs to, with the counts at the moment it began, so switching
// profile takes effect at once. Once the model has stopped, the run is
// split into one finished run per stretch, in that stretch's profile.
type expertSegment struct {
	Profile string     `json:"profile"`        // "" once cleared
	Base    *expertRun `json:"base,omitempty"` // counts when the stretch began; nil = zero
}

func liveDir(modelDir string) string { return filepath.Join(modelDir, "live") }

func segPath(run string) string { return strings.TrimSuffix(run, ".json") + ".seg" }

func readSegments(run string) []expertSegment {
	var segs []expertSegment
	b, err := os.ReadFile(segPath(run))
	if err == nil {
		json.Unmarshal(b, &segs)
	}
	return segs
}

func writeSegments(run string, segs []expertSegment) error {
	b, _ := json.Marshal(segs)
	return os.WriteFile(segPath(run), b, 0o644)
}

// expertStatsEnv starts a live run for a launch in the active profile and
// returns the environment variable pointing llama-server at it, and its path.
func expertStatsEnv(modelPath string, args []string, auto bool) (string, string, error) {
	modelDir, err := expertModelDir(modelPath)
	if err != nil {
		return "", "", err
	}
	migrateExpertStats(modelDir)
	finalizeLiveRuns(modelDir, "")
	dir := liveDir(modelDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", "", err
	}
	run := filepath.Join(dir, "run-"+strconv.FormatInt(time.Now().UnixNano(), 10)+".json")
	meta, _ := json.Marshal(expertRunMeta{RAMLayers: ramExpertLayers(modelPath, args), Auto: auto})
	if err := os.WriteFile(strings.TrimSuffix(run, ".json")+".meta", meta, 0o644); err != nil {
		return "", "", err
	}
	if err := writeSegments(run, []expertSegment{{Profile: activeExpertProfile(modelDir)}}); err != nil {
		return "", "", err
	}
	return "LLAMA_EXPERT_STATS=" + run, run, nil
}

// subRuns returns a - b, per layer and expert (b nil: a itself).
func subRuns(a expertRun, b *expertRun) expertRun {
	if b == nil {
		return a
	}
	out := expertRun{NExpert: a.NExpert, GenTokens: a.GenTokens - min(a.GenTokens, b.GenTokens), PromptTokens: a.PromptTokens - min(a.PromptTokens, b.PromptTokens)}
	base := map[int]int{}
	for i, l := range b.Layers {
		base[l.Layer] = i
	}
	sub := func(x []uint64, y []uint64) []uint64 {
		r := make([]uint64, len(x))
		for i := range x {
			r[i] = x[i]
			if i < len(y) {
				r[i] -= min(x[i], y[i])
			}
		}
		return r
	}
	out.Layers = make([]struct {
		Layer  int      `json:"layer"`
		Gen    []uint64 `json:"gen"`
		Prompt []uint64 `json:"prompt"`
	}, len(a.Layers))
	for i, l := range a.Layers {
		out.Layers[i].Layer = l.Layer
		out.Layers[i].Gen, out.Layers[i].Prompt = l.Gen, l.Prompt
		if j, ok := base[l.Layer]; ok {
			out.Layers[i].Gen = sub(l.Gen, b.Layers[j].Gen)
			out.Layers[i].Prompt = sub(l.Prompt, b.Layers[j].Prompt)
		}
	}
	return out
}

// liveStretch is one stretch of a live run, as counts of its own.
type liveStretch struct {
	profile string
	counts  expertRun
	last    bool // the stretch still being recorded
}

// liveStretches splits a live run into its stretches.
func liveStretches(run string) []liveStretch {
	cur, ok := readExpertRun(run)
	if !ok {
		return nil
	}
	segs := readSegments(run)
	var out []liveStretch
	for i, sg := range segs {
		end := cur
		if i+1 < len(segs) && segs[i+1].Base != nil {
			end = *segs[i+1].Base
		}
		out = append(out, liveStretch{profile: sg.Profile, counts: subRuns(end, sg.Base), last: i == len(segs)-1})
	}
	return out
}

// finalizeLiveRuns turns every live run except running (still being
// written) into finished runs in the profiles its stretches belong to.
func finalizeLiveRuns(modelDir, running string) {
	runs, _ := filepath.Glob(filepath.Join(liveDir(modelDir), "run-*.json"))
	segs, _ := filepath.Glob(filepath.Join(liveDir(modelDir), "run-*.seg"))
	for _, sg := range segs { // a run that never wrote counts
		if r := strings.TrimSuffix(sg, ".seg") + ".json"; !contains(runs, r) {
			runs = append(runs, r)
		}
	}
	for _, run := range runs {
		if run == running {
			continue
		}
		base := strings.TrimSuffix(run, ".json")
		meta, _ := os.ReadFile(base + ".meta")
		for i, st := range liveStretches(run) {
			if st.profile == "" || st.counts.GenTokens+st.counts.PromptTokens == 0 {
				continue
			}
			dir := filepath.Join(modelDir, st.profile)
			if os.MkdirAll(dir, 0o755) != nil {
				continue
			}
			name := filepath.Join(dir, filepath.Base(base)+"-"+strconv.Itoa(i))
			b, _ := json.Marshal(st.counts)
			if os.WriteFile(name+".json", b, 0o644) == nil && meta != nil {
				os.WriteFile(name+".meta", meta, 0o644)
			}
		}
		for _, ext := range []string{".json", ".json.tmp", ".meta", ".seg"} {
			os.Remove(base + ext)
		}
	}
}

// liveCounts returns the running run's current counts, if it has any.
func liveCounts(running string) (*expertRun, bool) {
	if running == "" {
		return nil, false
	}
	cur, ok := readExpertRun(running)
	if !ok {
		return nil, true // started, nothing counted yet
	}
	return &cur, true
}

// switchLiveProfile starts a new stretch of the running run in profile.
func switchLiveProfile(running, profile string) error {
	cur, ok := liveCounts(running)
	if !ok {
		return nil
	}
	segs := readSegments(running)
	if len(segs) > 0 && segs[len(segs)-1].Profile == profile {
		return nil
	}
	return writeSegments(running, append(segs, expertSegment{Profile: profile, Base: cur}))
}

// clearLiveProfile drops the running run's stretches in profile, and
// restarts the current stretch from now if it was one of them.
func clearLiveProfile(running, profile string) error {
	cur, ok := liveCounts(running)
	if !ok {
		return nil
	}
	segs := readSegments(running)
	current := len(segs) > 0 && segs[len(segs)-1].Profile == profile
	for i := range segs {
		if segs[i].Profile == profile {
			segs[i].Profile = ""
		}
	}
	if current {
		segs = append(segs, expertSegment{Profile: profile, Base: cur})
	}
	return writeSegments(running, segs)
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
	newest        string // file name of the newest run that isn't auto-profiling
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
		var m expertRunMeta
		if mb, err := os.ReadFile(strings.TrimSuffix(f, ".json") + ".meta"); err == nil && json.Unmarshal(mb, &m) == nil {
			c.ramLayers = m.RAMLayers // the latest run's placement
		}
		if !m.Auto {
			c.newest = filepath.Base(f)
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
	return summarizeCounts(loadExpertCounts(dir, "")), nil
}

func summarizeCounts(c *expertCounts) expertSummary {
	sum := expertSummary{Runs: c.runs, GenTokens: c.genTokens, PromptTokens: c.promptTokens, NExpert: c.nExpert, RAMLayers: c.ramLayers}
	if c.runs == 0 || c.genTokens+c.promptTokens == 0 {
		return sum
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
	return sum
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
	Live bool   `json:"live"` // the model is running and recording
	// HotProfile is where hot experts come from ("" = auto), HotResolved
	// the profile that choice means right now.
	HotProfile  string `json:"hot_profile"`
	HotResolved string `json:"hot_resolved"`
	// the model's switches (expertModelSettings)
	Record bool `json:"record"`
	Hot    bool `json:"hot"`
}

// expertStats reports the active profile's summary, every profile, and how
// well each matches the latest session. running is the live run file of the
// model now running, if any.
func expertStats(modelDir, running string) (expertStatsResponse, error) {
	active := activeExpertProfile(modelDir)
	var stretches []liveStretch
	var liveMeta expertRunMeta
	if running != "" {
		stretches = liveStretches(running)
		if b, err := os.ReadFile(strings.TrimSuffix(running, ".json") + ".meta"); err == nil {
			json.Unmarshal(b, &liveMeta)
		}
	}

	// the latest session: the stretch being recorded, or else the active
	// profile's newest finished run; it isn't compared with itself
	var latest map[int][]uint64
	skipFile, skipLive := "", false
	if n := len(stretches); n > 0 && stretches[n-1].profile == active && !liveMeta.Auto {
		if r := stretches[n-1].counts; r.GenTokens+r.PromptTokens >= minMatchTokens {
			one := newExpertCounts()
			one.add(r)
			latest, _ = one.routing()
			skipLive = true
		}
	}
	if latest == nil {
		if c := loadExpertCounts(filepath.Join(modelDir, active), ""); c.newest != "" {
			if r, ok := readExpertRun(filepath.Join(modelDir, active, c.newest)); ok && r.GenTokens+r.PromptTokens >= minMatchTokens {
				one := newExpertCounts()
				one.add(r)
				latest, _ = one.routing()
				skipFile = c.newest
			}
		}
	}

	// a profile's counts: its finished runs plus its stretches of the live run
	counts := func(name string, forMatch bool) *expertCounts {
		skip := ""
		if forMatch && name == active {
			skip = skipFile
		}
		c := loadExpertCounts(filepath.Join(modelDir, name), skip)
		for _, st := range stretches {
			if st.profile != name || st.counts.NExpert == 0 || (c.nExpert != 0 && st.counts.NExpert != c.nExpert) {
				continue
			}
			if forMatch && st.last && skipLive {
				continue
			}
			c.add(st.counts)
			c.ramLayers = liveMeta.RAMLayers
		}
		return c
	}

	resp := expertStatsResponse{expertSummary: summarizeCounts(counts(active, false)), Profile: active}
	names := expertProfiles(modelDir)
	extra := []string{active}
	for _, st := range stretches {
		extra = append(extra, st.profile) // may have no finished runs yet
	}
	for _, n := range extra {
		if n != "" && !contains(names, n) {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	withData := 0
	for _, name := range names {
		c := counts(name, false)
		info := expertProfileInfo{Name: name, Runs: c.runs, Tokens: c.genTokens}
		if c.genTokens+c.promptTokens > 0 {
			withData++
		}
		if latest != nil {
			if m := counts(name, true); m.genTokens+m.promptTokens > 0 {
				r, _ := m.routing()
				info.Match = routingSimilarity(latest, r)
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

// expertLiveRun is the live run file of the model now running, if it records.
func (s *Server) expertLiveRun() string {
	s.mu.Lock()
	run := s.expertLive
	s.mu.Unlock()
	if run == "" || !s.llama.Status().Running {
		return ""
	}
	return run
}

// GET    /api/expert-stats?model=<path>            summary, profiles, best match
// POST   /api/expert-stats {model, profile}        make profile active (created if new)
// DELETE /api/expert-stats?model=<path>&profile=n  delete a profile's runs
func (s *Server) handleExpertStats(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model      string  `json:"model"`
		Profile    string  `json:"profile"`
		HotProfile *string `json:"hot_profile"` // POST: set where hot experts come from ("" = auto)
		Record     *bool   `json:"record"`      // POST: record this model's expert usage
		Hot        *bool   `json:"hot"`         // POST: plan this model with hot experts
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
	if req.Profile != "" && !validProfileName(req.Profile) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("profile names are 1-40 letters, digits, spaces, '.', '_' or '-', and not \"live\""))
		return
	}
	modelDir, err := expertModelDir(req.Model)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	migrateExpertStats(modelDir)
	running := s.expertLiveRun()
	if running != "" && filepath.Dir(filepath.Dir(running)) != modelDir {
		running = "" // another model is running
	}
	finalizeLiveRuns(modelDir, running)

	switch r.Method {
	case http.MethodPost:
		if req.Record != nil || req.Hot != nil {
			es := readExpertSettings(modelDir, s.Config().ModelFor(modelName(req.Model)))
			if req.Record != nil {
				es.Record = *req.Record
			}
			if req.Hot != nil {
				es.Hot = *req.Hot
			}
			if err := writeExpertSettings(modelDir, es); err != nil {
				writeErr(w, http.StatusInternalServerError, err)
				return
			}
			break
		}
		if req.HotProfile != nil {
			if *req.HotProfile != "" && !validProfileName(*req.HotProfile) {
				writeErr(w, http.StatusBadRequest, fmt.Errorf("unknown profile %q", *req.HotProfile))
				return
			}
			if err := os.WriteFile(filepath.Join(modelDir, "hot-profile"), []byte(*req.HotProfile), 0o644); err != nil {
				writeErr(w, http.StatusInternalServerError, err)
				return
			}
			break
		}
		if req.Profile == "" {
			writeErr(w, http.StatusBadRequest, fmt.Errorf("profile is required"))
			return
		}
		err := os.MkdirAll(filepath.Join(modelDir, req.Profile), 0o755)
		if err == nil {
			err = os.WriteFile(filepath.Join(modelDir, "active"), []byte(req.Profile), 0o644)
		}
		if err == nil {
			err = switchLiveProfile(running, req.Profile)
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
		files, _ := filepath.Glob(filepath.Join(modelDir, profile, "run-*"))
		for _, f := range files {
			if err := os.Remove(f); err != nil {
				writeErr(w, http.StatusInternalServerError, err)
				return
			}
		}
		if err := clearLiveProfile(running, profile); err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		if profile != activeExpertProfile(modelDir) {
			os.Remove(filepath.Join(modelDir, profile)) // gone from the list once empty
		}
	}
	resp, err := expertStats(modelDir, running)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	resp.Live = running != ""
	es := readExpertSettings(modelDir, s.Config().ModelFor(modelName(req.Model)))
	resp.Record, resp.Hot = es.Record, es.Hot
	resp.HotProfile = hotExpertProfile(modelDir)
	resp.HotResolved = resp.HotProfile
	if resp.HotResolved == "" {
		resp.HotResolved = autoHotProfile(resp)
	}
	writeJSON(w, resp)
}

// hotExpertsFlag carries the hot-expert list file from the plan to launch,
// which turns it into LLAMA_MOE_HOT; llama-server never sees it.
const hotExpertsFlag = "--q38-moe-hot"

// minHotTokens is how much recorded routing hot experts need.
const minHotTokens = 1000

// hotExpertProfile is the profile hot experts come from; "" means auto:
// the one the latest session resembled most (else the recording profile).
// Kept apart from the recording profile, so a profile used for hot experts
// only changes when it is also recorded into.
func hotExpertProfile(modelDir string) string {
	b, err := os.ReadFile(filepath.Join(modelDir, "hot-profile"))
	if name := strings.TrimSpace(string(b)); err == nil && validProfileName(name) {
		return name
	}
	return ""
}

// autoHotProfile is what "auto" means: the profile the latest session
// resembled most, else the recording profile if it has enough data, else the
// profile with the most.
func autoHotProfile(r expertStatsResponse) string {
	if r.Best != "" {
		return r.Best
	}
	pick, most := r.Profile, uint64(0)
	for _, p := range r.Profiles {
		if p.Name == r.Profile && p.Tokens >= minHotTokens {
			return p.Name
		}
		if p.Tokens > most {
			pick, most = p.Name, p.Tokens
		}
	}
	return pick
}

// expertHits returns the routing counts per block and expert (generation,
// else prompt) of the profile hot experts come from, and its name; nil if it
// has too few recorded.
func (s *Server) expertHits(modelPath string) (map[int][]uint64, string) {
	modelDir, err := expertModelDir(modelPath)
	if err != nil {
		return nil, ""
	}
	migrateExpertStats(modelDir)
	running := s.expertLiveRun()
	if running != "" && filepath.Dir(filepath.Dir(running)) != modelDir {
		running = ""
	}
	finalizeLiveRuns(modelDir, running)
	profile := hotExpertProfile(modelDir)
	if profile == "" {
		r, err := expertStats(modelDir, running)
		if err != nil {
			return nil, ""
		}
		profile = autoHotProfile(r)
	}
	c := loadExpertCounts(filepath.Join(modelDir, profile), "")
	if running != "" {
		for _, st := range liveStretches(running) {
			if st.profile == profile && (c.nExpert == 0 || st.counts.NExpert == c.nExpert) {
				c.add(st.counts)
			}
		}
	}
	if c.genTokens+c.promptTokens < minHotTokens {
		return nil, profile
	}
	counts, _ := c.routing()
	return counts, profile
}

// writeHotExperts saves a plan's hot experts in patches/0010's format, one
// "<block>: <expert> ..." line per block.
func writeHotExperts(modelPath string, hot map[int][]int) (string, error) {
	modelDir, err := expertModelDir(modelPath)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return "", err
	}
	blocks := make([]int, 0, len(hot))
	for b := range hot {
		blocks = append(blocks, b)
	}
	sort.Ints(blocks)
	var sb strings.Builder
	for _, b := range blocks {
		sb.WriteString(strconv.Itoa(b) + ":")
		for _, e := range hot[b] {
			sb.WriteString(" " + strconv.Itoa(e))
		}
		sb.WriteString("\n")
	}
	f := filepath.Join(modelDir, "hot-experts.txt")
	return f, os.WriteFile(f, []byte(sb.String()), 0o644)
}

func flagValue(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

// expertModelSettings are a model's recording and hot-expert switches, kept
// with its profiles (settings.json in its expert-stats folder).
type expertModelSettings struct {
	Record bool `json:"record"`
	Hot    bool `json:"hot"`
}

// readExpertSettings falls back to the model settings' flags, where these
// switches lived before, while the model has no settings.json.
func readExpertSettings(modelDir string, ms appconfig.ModelSettings) expertModelSettings {
	var es expertModelSettings
	b, err := os.ReadFile(filepath.Join(modelDir, "settings.json"))
	if err != nil || json.Unmarshal(b, &es) != nil {
		return expertModelSettings{Record: ms.ExpertStats, Hot: ms.HotExperts}
	}
	return es
}

func writeExpertSettings(modelDir string, es expertModelSettings) error {
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return err
	}
	b, _ := json.Marshal(es)
	return os.WriteFile(filepath.Join(modelDir, "settings.json"), b, 0o644)
}

func (s *Server) recordExperts(modelPath string, ms appconfig.ModelSettings) bool {
	dir, err := expertModelDir(modelPath)
	return err == nil && readExpertSettings(dir, ms).Record
}

func (s *Server) hotExperts(modelPath string, ms appconfig.ModelSettings) bool {
	dir, err := expertModelDir(modelPath)
	return err == nil && readExpertSettings(dir, ms).Hot
}
