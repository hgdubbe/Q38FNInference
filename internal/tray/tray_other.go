//go:build !windows

package tray

import "context"

// Available reports whether this platform shows a tray icon.
const Available = false

// Run blocks until ctx is done; there is no tray outside Windows.
func Run(ctx context.Context, app App) { <-ctx.Done() }
