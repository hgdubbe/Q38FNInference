// Package tray shows a notification-area icon on Windows, so the launcher
// (which has no window of its own) can be reopened and quit from there.
package tray

// App is what the tray needs from the launcher.
type App interface {
	OpenPanel()
	OpenChat()
	// ModelStatus is "stopped", "loading" or "ready", plus the model's name.
	ModelStatus() (state, model string)
	Quit()
}
