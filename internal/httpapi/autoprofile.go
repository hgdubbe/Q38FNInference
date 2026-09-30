package httpapi

// Auto-profiling: records an expert-usage profile per use case by loading
// the model with recording on and hot experts off (so the placement can't
// matter), then sending each use case's prompts straight to llama-server
// while the recording profile is switched to that use case. The model that
// ran before is started again afterwards.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type useCase struct {
	ID      string   `json:"id"` // also the profile name
	Name    string   `json:"name"`
	Desc    string   `json:"desc"`
	System  string   `json:"-"`
	Prompts []string `json:"-"`
}

var useCases = []useCase{
	{
		ID: "assistant", Name: "Assistant", Desc: "everyday questions, explanations, advice",
		System: "You are a helpful, knowledgeable assistant. Answer clearly and thoroughly.",
		Prompts: []string{
			"Explain how a heat pump works and whether it makes sense for an old, poorly insulated house.",
			"I have 20 minutes a day to learn Spanish. Make me a realistic 3-month plan, with what to do each week.",
			"What are the main differences between a Roth IRA and a traditional IRA, and how would I decide between them?",
			"My sourdough bread comes out dense and flat. What could be going wrong? Walk me through the likely causes.",
			"Compare electric cars and hybrids for someone who drives 60 km a day and sometimes takes long trips.",
			"How does the immune system remember an infection, and why do some vaccines need boosters?",
		},
	},
	{
		ID: "coding", Name: "Coding", Desc: "writing, explaining and fixing code",
		System: "You are an expert software engineer. Write correct, idiomatic, well-structured code and explain your reasoning briefly.",
		Prompts: []string{
			"Write a Python function that parses a CSV file of transactions (date, description, amount), groups them by month and category keywords, and prints a summary table. Include type hints and tests with pytest.",
			"Implement an LRU cache in Go with Get and Put in O(1), safe for concurrent use. Explain the design choices.",
			"This JavaScript debounce function sometimes fires twice. Find the bug and fix it:\n\nfunction debounce(fn, ms) {\n  let t;\n  return (...args) => {\n    if (t) clearTimeout(t);\n    t = setTimeout(fn(...args), ms);\n  };\n}",
			"Write a SQL schema for a library system (books, members, loans, reservations) and queries for: overdue loans, most borrowed books this year, and members with more than 3 active loans.",
			"Refactor this Rust code to avoid cloning and handle errors with the ? operator:\n\nfn read_config(path: &str) -> Config {\n    let text = std::fs::read_to_string(path.to_string()).unwrap();\n    let cfg: Config = toml::from_str(&text.clone()).unwrap();\n    cfg.clone()\n}",
			"Write a React component with TypeScript for a searchable, sortable table that loads data from an API with loading and error states.",
		},
	},
	{
		ID: "agentic", Name: "Agentic coding", Desc: "tool calls, multi-step coding tasks",
		System: "You are an autonomous coding agent working in a repository. You can call tools by writing JSON objects like {\"tool\": \"read_file\", \"path\": \"...\"}, {\"tool\": \"run\", \"cmd\": \"...\"} or {\"tool\": \"edit\", \"path\": \"...\", \"old\": \"...\", \"new\": \"...\"}. Think step by step, state your plan, then issue tool calls.",
		Prompts: []string{
			"The test suite fails with `TypeError: Cannot read properties of undefined (reading 'map')` in src/components/UserList.test.tsx. Investigate and fix it.",
			"Add a --dry-run flag to the CLI in cmd/sync/main.go that prints what would be copied without copying anything. Update the tests.",
			"The CI build broke after upgrading the ORM from v4 to v5. Here is the log:\n\nerror: Property 'findOne' does not exist on type 'Repository<User>'\n  at src/services/auth.ts:42\n\nFind all affected call sites and migrate them.",
			"Profile why the /api/search endpoint takes 4 seconds and propose and implement an index or caching fix.",
			"Set up a GitHub Actions workflow that runs lint, unit tests and builds a Docker image on every pull request.",
		},
	},
	{
		ID: "roleplay", Name: "Roleplay", Desc: "staying in character, dialogue, scenes",
		System: "You are Captain Mara Voss, the sardonic but loyal captain of the smuggling ship Kestrel in a gritty space-opera setting. Stay in character at all times, write vivid dialogue and actions in the present tense, and react to what the user's character does. Never speak or act for the user's character.",
		Prompts: []string{
			"*I stumble onto the bridge, holding my bleeding arm.* Captain, the customs cutter is hailing us. They say they'll board in ten minutes.",
			"*I lean against the cargo crate and cross my arms.* So, are you going to tell me what's actually in these boxes, or do I have to open one?",
			"*At the bar on Tessaly Station, I slide into the seat across from you.* I heard you're looking for a pilot. I'm the best you'll find, and I don't ask questions.",
			"*The engines sputter and die. Red emergency lights flood the corridor.* Mara, we're drifting toward the gas giant. What do we do?",
			"*I hand you a sealed data chip.* Your old crewmate Dax gave me this before he died. He said only you would know what it means.",
		},
	},
	{
		ID: "storytelling", Name: "Storytelling", Desc: "long-form fiction and narration",
		System: "You are a skilled novelist. Write immersive prose with strong pacing, sensory detail and natural dialogue.",
		Prompts: []string{
			"Write the opening chapter of a mystery novel set in a snowbound mountain hotel in 1920s Switzerland, where a guest disappears overnight.",
			"Continue this story: The lighthouse keeper had not seen another person in three hundred days when the boat washed ashore, empty except for a child's shoe.",
			"Write a fairy tale about a clockmaker who builds a heart for a mechanical bird, told in the style of a bedtime story.",
			"Write a tense scene from a heist thriller where the crew realizes one of them has been working for the police.",
			"Write a short science-fiction story about the first message received from another civilization, and why nobody can agree on what it says.",
		},
	},
	{
		ID: "writing", Name: "Writing & editing", Desc: "emails, rewriting, proofreading, marketing text",
		System: "You are a professional editor and copywriter. Write clearly, match the requested tone and keep the author's meaning.",
		Prompts: []string{
			"Rewrite this email to sound professional but friendly, and make it shorter:\n\nhey, so the thing we talked about last week, the budget stuff, i still didn't get the numbers from finance and honestly its getting late so can you maybe push them or something, we need it by friday otherwise the whole plan is off",
			"Write a product description for a handmade ceramic coffee mug for an online shop, in three versions: playful, minimalist, and luxury.",
			"Proofread and improve this paragraph for a cover letter: I am applying for the position as I have many experience in project management and I think I would be fitting good in your team because of my skills.",
			"Write a LinkedIn post announcing that our small bakery won a regional award, thanking customers and staff.",
			"Turn these bullet notes into a clear one-page meeting summary with decisions and action items: - launch moved to May - marketing needs assets by Apr 10 - Tom to check supplier contract - budget ok but no new hires",
		},
	},
	{
		ID: "research", Name: "Research & analysis", Desc: "comparisons, reports, reasoning over facts",
		System: "You are a careful research analyst. Structure your answers, weigh evidence and state uncertainty.",
		Prompts: []string{
			"Analyze the pros and cons of a four-day work week for a 200-person software company, including effects on productivity, hiring and customer support.",
			"Compare solid-state batteries and lithium-iron-phosphate batteries for electric vehicles: energy density, cost, safety, and likely timeline.",
			"A city wants to reduce traffic in its center. Evaluate congestion charges, car-free zones and better public transport, with examples from real cities.",
			"Summarize the main arguments for and against nuclear power in the context of decarbonization, and what the evidence says about cost and safety.",
			"Our online shop's conversion rate dropped from 3.1% to 2.4% after a redesign. List the likely causes and how to test each one.",
		},
	},
	{
		ID: "math", Name: "Math & reasoning", Desc: "step-by-step problem solving",
		System: "You are a patient math tutor. Solve problems step by step and check your results.",
		Prompts: []string{
			"A tank is filled by two pipes in 6 and 9 hours respectively and emptied by a third in 12 hours. If all three are open, how long does it take to fill the tank? Show your work.",
			"Prove that the square root of 2 is irrational, then explain the idea of the proof in plain words.",
			"What is the probability of getting at least two sixes in five rolls of a fair die? Explain each step.",
			"Find all real solutions of x^4 - 5x^2 + 4 = 0 and verify them.",
			"A loan of 20,000 at 6% annual interest is repaid monthly over 5 years. Compute the monthly payment and the total interest, explaining the formula.",
			"Five people shake hands with each other exactly once. How many handshakes are there? Generalize to n people and prove the formula.",
		},
	},
	{
		ID: "translation", Name: "Translation", Desc: "between languages, keeping tone",
		System: "You are a professional translator. Translate accurately and naturally, keeping tone and formatting.",
		Prompts: []string{
			"Translate into German, keeping the friendly tone: Thanks so much for your order! Your package is on its way and should arrive within three business days. If anything is wrong, just reply to this email and we'll sort it out.",
			"Translate into French and Spanish: The meeting has been moved to Thursday at 3 pm. Please review the attached report beforehand and bring your questions.",
			"Translate this German text into English: Die Bauarbeiten an der Brücke verzögern sich voraussichtlich bis zum Herbst, da bei der Prüfung der Fundamente unerwartete Schäden festgestellt wurden.",
			"Translate into Japanese and explain any choices where the meaning could shift: I'm sorry for the late reply. I was traveling last week and only just saw your message.",
			"Translate this Spanish poem into English, preserving the rhythm as much as possible: Caminante, son tus huellas el camino y nada más; caminante, no hay camino, se hace camino al andar.",
		},
	},
	{
		ID: "summarization", Name: "Summaries", Desc: "condensing long text",
		System: "You summarize text accurately and concisely, keeping the key facts.",
		Prompts: []string{
			"Summarize in five bullet points, then in one sentence:\n\nThe city council met on Tuesday to discuss the proposed expansion of the northern tram line. Supporters argued that the extension would connect three growing neighborhoods to the center, reduce car traffic by an estimated 8,000 trips per day and support new housing projects. Opponents raised concerns about the cost, which has risen from 310 to 420 million since the first estimate, and about two years of construction on the main shopping street. The transport department presented a revised plan that phases the work to keep one lane open, and proposed financing a quarter of the cost through a regional infrastructure fund. After four hours of debate, the council voted 27 to 18 to approve the planning phase, with a final decision on construction expected next spring once updated cost figures are available.",
			"Write an executive summary of this incident report:\n\nAt 02:14 the primary database became unresponsive after a disk on the storage node filled up with transaction logs. Automatic failover to the replica did not trigger because the health check only tested network reachability. The on-call engineer was paged at 02:21, identified the full disk at 02:40 and freed space by archiving old logs. Service was restored at 02:58. About 12% of checkout attempts failed during the outage. Follow-ups: alert on disk usage above 80%, make the health check run a real query, and rotate transaction logs daily.",
			"Summarize the plot of Shakespeare's Hamlet for a 12-year-old, then give three themes of the play with one example each.",
			"Condense these meeting notes into a short status update for the team chat: Frontend is 80% done, blocked on the new login API. Backend team says the API ships Wednesday. QA found 14 bugs, 3 critical, all assigned. Design review for the settings page moved to next week. Release still planned for the 28th if the API lands on time.",
		},
	},
}

func findUseCase(id string) (useCase, bool) {
	for _, u := range useCases {
		if u.ID == id {
			return u, true
		}
	}
	return useCase{}, false
}

type autoProfileStep struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Tokens int    `json:"tokens"`
	Target int    `json:"target"`
	State  string `json:"state"` // waiting, running, done, failed, canceled
}

type autoProfileStatus struct {
	Running bool              `json:"running"`
	Model   string            `json:"model,omitempty"`
	Phase   string            `json:"phase,omitempty"` // what it is doing now, for display
	Steps   []autoProfileStep `json:"steps,omitempty"`
	Error   string            `json:"error,omitempty"`
	Note    string            `json:"note,omitempty"`
}

type autoProfileJob struct {
	mu     sync.Mutex
	st     autoProfileStatus
	cancel context.CancelFunc
}

func (j *autoProfileJob) update(f func(*autoProfileStatus)) {
	j.mu.Lock()
	f(&j.st)
	j.mu.Unlock()
}

func (j *autoProfileJob) status() autoProfileStatus {
	j.mu.Lock()
	defer j.mu.Unlock()
	st := j.st
	st.Steps = append([]autoProfileStep(nil), j.st.Steps...)
	return st
}

// autoProfiling reports whether an auto-profiling run is active.
func (s *Server) autoProfiling() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.autoProf != nil && s.autoProf.status().Running
}

// GET: status of the current or last run, plus the use cases.
// POST {model, use_cases, tokens, replace}: start. DELETE: cancel.
func (s *Server) handleAutoProfile(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	job := s.autoProf
	s.mu.Unlock()
	switch r.Method {
	case http.MethodGet:
		resp := struct {
			autoProfileStatus
			UseCases []useCase `json:"use_cases"`
		}{UseCases: useCases}
		if job != nil {
			resp.autoProfileStatus = job.status()
		}
		writeJSON(w, resp)
	case http.MethodDelete:
		if job != nil && job.cancel != nil {
			job.cancel()
		}
		writeJSON(w, map[string]bool{"ok": true})
	case http.MethodPost:
		var req struct {
			Model    string   `json:"model"`
			UseCases []string `json:"use_cases"`
			Tokens   int      `json:"tokens"`
			Replace  bool     `json:"replace"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		if _, err := os.Stat(req.Model); err != nil {
			writeErr(w, http.StatusBadRequest, fmt.Errorf("model not found: %w", err))
			return
		}
		var steps []autoProfileStep
		for _, id := range req.UseCases {
			u, ok := findUseCase(id)
			if !ok {
				writeErr(w, http.StatusBadRequest, fmt.Errorf("unknown use case %q", id))
				return
			}
			steps = append(steps, autoProfileStep{ID: u.ID, Name: u.Name, Target: max(req.Tokens, minHotTokens), State: "waiting"})
		}
		if len(steps) == 0 {
			writeErr(w, http.StatusBadRequest, fmt.Errorf("choose at least one use case"))
			return
		}
		ctx, cancel := context.WithCancel(context.Background())
		nj := &autoProfileJob{cancel: cancel, st: autoProfileStatus{Running: true, Model: req.Model, Phase: "starting", Steps: steps}}
		s.mu.Lock()
		if s.autoProf != nil && s.autoProf.status().Running {
			s.mu.Unlock()
			cancel()
			writeErr(w, http.StatusConflict, fmt.Errorf("auto-profiling is already running"))
			return
		}
		s.autoProf = nj
		s.mu.Unlock()
		go s.runAutoProfile(ctx, nj, req.Model, req.Replace)
		writeJSON(w, nj.status())
	}
}

func (s *Server) runAutoProfile(ctx context.Context, job *autoProfileJob, modelPath string, replace bool) {
	fail := func(err error) {
		job.update(func(st *autoProfileStatus) {
			if ctx.Err() != nil {
				st.Error = "canceled"
			} else {
				st.Error = err.Error()
			}
			for i := range st.Steps {
				if st.Steps[i].State == "running" || st.Steps[i].State == "waiting" {
					st.Steps[i].State = map[bool]string{true: "canceled", false: "failed"}[ctx.Err() != nil]
				}
			}
		})
	}
	phase := func(p string) { job.update(func(st *autoProfileStatus) { st.Phase = p }) }

	// what ran before, to start it again afterwards
	prev := s.llama.Status()
	prevModel, prevRouter := "", prev.Running && s.currentRouter() != nil
	s.mu.Lock()
	if prev.Running && !prevRouter && s.current != nil {
		prevModel = s.current.modelPath
	}
	s.mu.Unlock()

	modelDir, err := expertModelDir(modelPath)
	if err != nil {
		fail(err)
		s.finishAutoProfile(job, "", prevModel, prevRouter, "")
		return
	}
	prevActive := activeExpertProfile(modelDir)

	defer func() {
		s.finishAutoProfile(job, modelDir, prevModel, prevRouter, prevActive)
	}()

	phase("stopping the running model")
	stopCtx, cancelStop := context.WithTimeout(context.Background(), 30*time.Second)
	err = s.llama.StopWithTimeout(stopCtx)
	cancelStop()
	if err != nil {
		fail(fmt.Errorf("stopping the running model: %w", err))
		return
	}
	s.Proxy.SetTarget(nil)
	s.setRouter(nil)
	migrateExpertStats(modelDir)
	finalizeLiveRuns(modelDir, "")

	steps := job.status().Steps
	for _, st := range steps {
		dir := filepath.Join(modelDir, st.ID)
		if replace {
			files, _ := filepath.Glob(filepath.Join(dir, "run-*"))
			for _, f := range files {
				os.Remove(f)
			}
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fail(err)
			return
		}
	}
	if err := os.WriteFile(filepath.Join(modelDir, "active"), []byte(steps[0].ID), 0o644); err != nil {
		fail(err)
		return
	}

	// recording on and hot experts off: autoProfiling() makes launch and
	// tuneWith do both
	phase("loading the model")
	gpus, _ := s.selectedGPUs()
	plan, err := s.tuneWith(modelPath, gpus, false)
	if err != nil {
		fail(err)
		return
	}
	if _, err := s.launch(modelPath, plan.Args, plan.Plan.Devices, false, false); err != nil {
		fail(err)
		return
	}
	target, live, err := s.waitReady(ctx)
	if err != nil {
		fail(err)
		return
	}

	client := &http.Client{Timeout: 15 * time.Minute}
	for i, st := range steps {
		u, _ := findUseCase(st.ID)
		if i > 0 {
			// let llama-server write out the last counts before the switch
			if !sleepCtx(ctx, 1500*time.Millisecond) {
				fail(ctx.Err())
				return
			}
			if err := os.WriteFile(filepath.Join(modelDir, "active"), []byte(st.ID), 0o644); err == nil {
				err = switchLiveProfile(live, st.ID)
			}
			if err != nil {
				fail(err)
				return
			}
		}
		job.update(func(s *autoProfileStatus) { s.Phase = "profiling " + u.Name; s.Steps[i].State = "running" })
		tokens := 0
		for p := 0; tokens < st.Target; p++ {
			n, err := chatOnce(ctx, client, target, u.System, u.Prompts[p%len(u.Prompts)], min(700, st.Target-tokens+100))
			if err != nil {
				fail(fmt.Errorf("%s: %w", u.Name, err))
				return
			}
			if n == 0 && p >= len(u.Prompts) {
				fail(fmt.Errorf("%s: the model returns no tokens", u.Name))
				return
			}
			tokens += n
			job.update(func(s *autoProfileStatus) { s.Steps[i].Tokens = tokens })
		}
		job.update(func(s *autoProfileStatus) { s.Steps[i].State = "done" })
	}
	sleepCtx(ctx, 1500*time.Millisecond)
	phase("done")
}

// finishAutoProfile stops the profiling load, files its counts into the
// profiles, and restores what ran before.
func (s *Server) finishAutoProfile(job *autoProfileJob, modelDir, prevModel string, prevRouter bool, prevActive string) {
	stopCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	s.llama.StopWithTimeout(stopCtx)
	cancel()
	s.Proxy.SetTarget(nil)
	s.mu.Lock()
	s.expertLive = ""
	s.mu.Unlock()
	if modelDir != "" {
		finalizeLiveRuns(modelDir, "")
		if prevActive != "" {
			os.WriteFile(filepath.Join(modelDir, "active"), []byte(prevActive), 0o644)
		}
	}
	job.update(func(st *autoProfileStatus) { st.Running = false })

	switch {
	case prevModel != "":
		job.update(func(st *autoProfileStatus) { st.Phase = "starting the previous model again" })
		t, err := s.tune(prevModel)
		if err == nil {
			_, err = s.launch(prevModel, t.Args, t.Plan.Devices, true, false)
		}
		if err != nil {
			job.update(func(st *autoProfileStatus) { st.Note = "could not start the previous model again: " + err.Error() })
		}
	case prevRouter:
		job.update(func(st *autoProfileStatus) {
			st.Note = "on-demand mode was stopped for profiling; start it again on the Run page"
		})
	}
	job.update(func(st *autoProfileStatus) {
		if st.Error == "" {
			st.Phase = "done"
		} else {
			st.Phase = "stopped"
		}
	})
}

// waitReady waits for the profiling load and returns its address and live
// run file.
func (s *Server) waitReady(ctx context.Context) (string, string, error) {
	for {
		st := s.llama.Status()
		if !st.Running {
			return "", "", fmt.Errorf("llama-server stopped while loading; see the server log on the Run page")
		}
		if st.Ready {
			s.mu.Lock()
			defer s.mu.Unlock()
			if s.current == nil || s.current.target == nil {
				return "", "", fmt.Errorf("llama-server address unknown")
			}
			if s.expertLive == "" {
				return "", "", fmt.Errorf("expert usage recording could not start; see the app log")
			}
			return s.current.target.String(), s.expertLive, nil
		}
		if !sleepCtx(ctx, 500*time.Millisecond) {
			return "", "", ctx.Err()
		}
	}
}

// chatOnce sends one chat request straight to llama-server and returns the
// number of generated tokens.
func chatOnce(ctx context.Context, client *http.Client, target, system, prompt string, maxTokens int) (int, error) {
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": prompt},
		},
		"max_tokens": maxTokens,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	var out struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, fmt.Errorf("llama-server answered %s: %w", resp.Status, err)
	}
	if out.Error != nil {
		return 0, fmt.Errorf("llama-server: %s", out.Error.Message)
	}
	return out.Usage.CompletionTokens, nil
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	select {
	case <-ctx.Done():
		return false
	case <-time.After(d):
		return true
	}
}
