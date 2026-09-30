// Package server manages the llama-server child process: starting it with a
// computed argument list, capturing its combined stdout/stderr for the web
// UI's log tail, and stopping it again.
package server

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/hgdubbe/q38fninference/internal/proc"
)

// Status is a snapshot of the managed process's state.
type Status struct {
	Running   bool
	Ready     bool // model loaded and serving (set by the caller's health check)
	PID       int
	StartedAt time.Time
	Args      []string
	ExitErr   string // non-empty if the process exited on its own since the last Start
}

// Manager owns at most one llama-server child process at a time.
type Manager struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	status Status
	// stopped: Stop killed the current process, so its exit is not an error
	stopped bool

	logMu  sync.Mutex
	logBuf []string // ring buffer of recent log lines
	logCap int
	subs   map[chan string]struct{}
}

// NewManager returns a Manager that keeps the last logCapLines of output.
func NewManager(logCapLines int) *Manager {
	if logCapLines <= 0 {
		logCapLines = 2000
	}
	return &Manager{
		logCap: logCapLines,
		subs:   make(map[chan string]struct{}),
	}
}

// Start launches binPath with args and extra environment variables. Returns
// an error if a process is already running or the binary fails to start.
func (m *Manager) Start(binPath string, args []string, env ...string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cmd != nil && m.status.Running {
		return fmt.Errorf("server already running (pid %d)", m.status.PID)
	}

	cmd := exec.Command(binPath, args...)
	cmd.Env = append(os.Environ(), env...)
	proc.Hide(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting %s: %w", binPath, err)
	}

	m.cmd = cmd
	m.stopped = false
	m.status = Status{Running: true, PID: cmd.Process.Pid, StartedAt: time.Now(), Args: append([]string{binPath}, args...)}

	var pumps sync.WaitGroup
	pumps.Add(2)
	go func() { defer pumps.Done(); m.pump(stdout) }()
	go func() { defer pumps.Done(); m.pump(stderr) }()
	go m.wait(cmd, &pumps)

	return nil
}

func (m *Manager) pump(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		m.appendLog(sc.Text())
	}
}

func (m *Manager) wait(cmd *exec.Cmd, pumps *sync.WaitGroup) {
	// Wait closes the pipes, so drain them first or the last lines (often
	// the crash reason) are lost
	pumps.Wait()
	err := cmd.Wait()
	m.mu.Lock()
	if m.cmd != cmd {
		m.mu.Unlock()
		return
	}
	m.status.Running = false
	m.status.Ready = false
	if m.stopped {
		m.appendLogLocked("[launcher] server stopped")
	} else if err != nil {
		m.status.ExitErr = err.Error()
		m.appendLogLocked(fmt.Sprintf("[launcher] server exited: %v", err))
	} else {
		m.appendLogLocked("[launcher] server exited cleanly")
	}
	m.mu.Unlock()
}

// Stop terminates the running process, if any. It's a no-op if nothing is
// running. Kill is immediate rather than a graceful shutdown request: Go's
// os.Process.Signal only supports os.Kill on Windows, so there is no portable
// SIGTERM-equivalent to send llama-server here.
func (m *Manager) Stop() error {
	m.mu.Lock()
	cmd := m.cmd
	running := m.status.Running
	if cmd != nil && running {
		m.stopped = true
	}
	m.mu.Unlock()

	if cmd == nil || !running {
		return nil
	}
	if err := cmd.Process.Kill(); err != nil {
		return fmt.Errorf("stopping server: %w", err)
	}
	return nil
}

// StopWithTimeout calls Stop and waits (bounded by ctx) for the exit to be
// observed by wait(), so callers can be sure the port is free again.
func (m *Manager) StopWithTimeout(ctx context.Context) error {
	if err := m.Stop(); err != nil {
		return err
	}
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		if !m.Status().Running {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// MarkReady records that the process with this PID finished loading.
func (m *Manager) MarkReady(pid int) {
	m.mu.Lock()
	if m.status.Running && m.status.PID == pid {
		m.status.Ready = true
	}
	m.mu.Unlock()
}

// Status returns a snapshot of the current process state.
func (m *Manager) Status() Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.status
}

func (m *Manager) appendLog(line string) {
	m.mu.Lock()
	m.appendLogLocked(line)
	m.mu.Unlock()
}

// appendLogLocked requires m.mu to be held (it's called both from the pump
// goroutines directly and from wait(), which already holds it).
func (m *Manager) appendLogLocked(line string) {
	m.logMu.Lock()
	m.logBuf = append(m.logBuf, line)
	if len(m.logBuf) > m.logCap {
		m.logBuf = m.logBuf[len(m.logBuf)-m.logCap:]
	}
	m.logMu.Unlock()

	m.logMu.Lock()
	for ch := range m.subs {
		select {
		case ch <- line:
		default: // slow subscriber: drop rather than block log capture
		}
	}
	m.logMu.Unlock()
}

// Logs returns a snapshot of recently captured output lines.
// Note adds a launcher message to the log shown with llama-server's output.
func (m *Manager) Note(msg string) {
	m.appendLog("[launcher] " + msg)
}

func (m *Manager) Logs() []string {
	m.logMu.Lock()
	defer m.logMu.Unlock()
	out := make([]string, len(m.logBuf))
	copy(out, m.logBuf)
	return out
}

// Subscribe returns a channel that receives new log lines as they arrive,
// and an unsubscribe function the caller must call when done (e.g. when an
// SSE client disconnects).
func (m *Manager) Subscribe() (<-chan string, func()) {
	ch := make(chan string, 256)
	m.logMu.Lock()
	m.subs[ch] = struct{}{}
	m.logMu.Unlock()

	return ch, func() {
		m.logMu.Lock()
		delete(m.subs, ch)
		m.logMu.Unlock()
		close(ch)
	}
}
