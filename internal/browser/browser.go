// Package browser opens URLs in the user's default browser.
package browser

import (
	"os/exec"
	"runtime"

	"github.com/hgdubbe/q38fninference/internal/proc"
)

// Open launches url in the default browser without flashing a console.
func Open(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	proc.Hide(cmd)
	return cmd.Start()
}
