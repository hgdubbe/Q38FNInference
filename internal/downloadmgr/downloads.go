// Package downloadmgr tracks in-progress and finished Hugging Face model
// downloads so the web UI can show live progress bars and survive a page
// reload without losing that state.
package downloadmgr

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/hgdubbe/q38fninference/internal/hf"
)

// State is a download's lifecycle stage.
type State string

const (
	StatePending State = "pending"
	StateActive  State = "active"
	StateDone    State = "done"
	StateError   State = "error"
)

// Download is a snapshot of one file transfer.
type Download struct {
	ID        string    `json:"id"`
	Repo      string    `json:"repo"`
	Filename  string    `json:"filename"`
	DestPath  string    `json:"dest_path"`
	State     State     `json:"state"`
	Done      int64     `json:"done_bytes"`
	Total     int64     `json:"total_bytes"`
	Err       string    `json:"error,omitempty"`
	StartedAt time.Time `json:"started_at"`
}

// Manager tracks all downloads started this session.
type Manager struct {
	client *hf.Client

	mu   sync.Mutex
	byID map[string]*Download
	next int
}

// NewManager wraps an hf.Client with progress tracking.
func NewManager(client *hf.Client) *Manager {
	return &Manager{client: client, byID: make(map[string]*Download)}
}

// Start begins downloading repo/filename to destPath in the background and
// returns its tracking snapshot immediately. Starting a file that is already
// downloading returns the existing transfer rather than racing on its .part.
func (m *Manager) Start(ctx context.Context, repo, filename, destPath string) *Download {
	m.mu.Lock()
	for _, d := range m.byID {
		if d.DestPath == destPath && (d.State == StatePending || d.State == StateActive) {
			cp := *d
			m.mu.Unlock()
			return &cp
		}
	}
	m.next++
	id := fmt.Sprintf("dl-%d", m.next)
	d := &Download{ID: id, Repo: repo, Filename: filename, DestPath: destPath, State: StatePending, StartedAt: time.Now()}
	m.byID[id] = d
	cp := *d
	m.mu.Unlock()

	go m.run(ctx, d)
	return &cp
}

func (m *Manager) run(ctx context.Context, d *Download) {
	m.setState(d.ID, StateActive)

	err := m.client.Download(ctx, d.Repo, d.Filename, d.DestPath, func(done, total int64) {
		m.mu.Lock()
		if cur, ok := m.byID[d.ID]; ok {
			cur.Done = done
			cur.Total = total
		}
		m.mu.Unlock()
	})

	m.mu.Lock()
	defer m.mu.Unlock()
	cur, ok := m.byID[d.ID]
	if !ok {
		return
	}
	if err != nil {
		cur.State = StateError
		cur.Err = err.Error()
		return
	}
	cur.State = StateDone
}

func (m *Manager) setState(id string, s State) {
	m.mu.Lock()
	if d, ok := m.byID[id]; ok {
		d.State = s
	}
	m.mu.Unlock()
}

// Get returns a snapshot of one download, or nil if the ID is unknown.
func (m *Manager) Get(id string) *Download {
	m.mu.Lock()
	defer m.mu.Unlock()
	d, ok := m.byID[id]
	if !ok {
		return nil
	}
	cp := *d
	return &cp
}

// List returns a snapshot of every tracked download, newest first.
func (m *Manager) List() []Download {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Download, 0, len(m.byID))
	for _, d := range m.byID {
		out = append(out, *d)
	}
	return out
}
