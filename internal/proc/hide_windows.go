//go:build windows

package proc

import (
	"os/exec"
	"syscall"
)

const createNoWindow = 0x08000000

// Hide stops a console window from popping up for cmd: the launcher is built
// as a GUI-subsystem exe, so every console child would otherwise get its own.
func Hide(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
}
