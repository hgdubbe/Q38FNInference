package server

import (
	"context"
	"strings"
	"testing"
	"time"
)

func waitForLog(t *testing.T, m *Manager, contains string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		for _, l := range m.Logs() {
			if strings.Contains(l, contains) {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for log line containing %q; got %v", contains, m.Logs())
}

func TestStartCapturesOutputAndExit(t *testing.T) {
	m := NewManager(100)
	err := m.Start("/bin/sh", []string{"-c", "echo hello-from-child; echo err-line 1>&2"})
	if err != nil {
		t.Fatal(err)
	}

	st := m.Status()
	if !st.Running || st.PID == 0 {
		t.Fatalf("Status() = %+v, want Running with a PID", st)
	}

	waitForLog(t, m, "hello-from-child", time.Second)
	waitForLog(t, m, "err-line", time.Second)

	deadline := time.Now().Add(time.Second)
	for m.Status().Running && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if m.Status().Running {
		t.Fatal("process should have exited on its own")
	}
}

func TestStartRejectsDoubleStart(t *testing.T) {
	m := NewManager(100)
	if err := m.Start("/bin/sh", []string{"-c", "sleep 2"}); err != nil {
		t.Fatal(err)
	}
	defer m.Stop()

	if err := m.Start("/bin/sh", []string{"-c", "sleep 2"}); err == nil {
		t.Fatal("expected error starting a second process while one is running")
	}
}

func TestStop(t *testing.T) {
	m := NewManager(100)
	if err := m.Start("/bin/sh", []string{"-c", "sleep 30"}); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := m.StopWithTimeout(ctx); err != nil {
		t.Fatal(err)
	}
	if m.Status().Running {
		t.Fatal("Status().Running should be false after StopWithTimeout")
	}
}

func TestSubscribeReceivesNewLines(t *testing.T) {
	m := NewManager(100)
	ch, unsub := m.Subscribe()
	defer unsub()

	if err := m.Start("/bin/sh", []string{"-c", "echo live-line"}); err != nil {
		t.Fatal(err)
	}

	select {
	case line := <-ch:
		if !strings.Contains(line, "live-line") {
			t.Errorf("got %q", line)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for subscribed log line")
	}
}

func TestStartMissingBinary(t *testing.T) {
	m := NewManager(10)
	if err := m.Start("/no/such/binary/anywhere", nil); err == nil {
		t.Fatal("expected error for missing binary")
	}
}
