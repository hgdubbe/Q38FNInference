//go:build !windows

package proc

import "os/exec"

// Hide is a no-op outside Windows.
func Hide(cmd *exec.Cmd) {}
