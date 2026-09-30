//go:build windows

package tray

import (
	"context"
	"time"

	"fyne.io/systray"
)

// Available reports whether this platform shows a tray icon.
const Available = true

// Run shows the tray icon and blocks until ctx is done or Quit is chosen.
// It must be called from the main goroutine: systray locks it to the main
// OS thread, which owns the icon's message loop.
func Run(ctx context.Context, app App) {
	onReady := func() {
		systray.SetIcon(Icon())
		systray.SetTooltip("Q38FNInference")
		// left click opens the panel; right click shows the menu
		systray.SetOnTapped(app.OpenPanel)

		status := systray.AddMenuItem("Model: stopped", "")
		status.Disable()
		systray.AddSeparator()
		openPanel := systray.AddMenuItem("Open control panel", "")
		openChat := systray.AddMenuItem("Open chat (llama.cpp web UI)", "")
		openChat.Disable()
		systray.AddSeparator()
		quit := systray.AddMenuItem("Quit", "Stop the model and exit")

		go func() {
			tick := time.NewTicker(2 * time.Second)
			defer tick.Stop()
			last := ""
			for {
				state, model := app.ModelStatus()
				if cur := state + model; cur != last {
					last = cur
					label := "Model: " + state
					if model != "" {
						label += " (" + model + ")"
					}
					status.SetTitle(label)
					systray.SetTooltip("Q38FNInference — " + label)
					if state == "ready" {
						openChat.Enable()
					} else {
						openChat.Disable()
					}
				}
				select {
				case <-ctx.Done():
					systray.Quit()
					return
				case <-openPanel.ClickedCh:
					app.OpenPanel()
				case <-openChat.ClickedCh:
					app.OpenChat()
				case <-quit.ClickedCh:
					app.Quit()
				case <-tick.C:
				}
			}
		}()
	}
	systray.Run(onReady, nil)
}
