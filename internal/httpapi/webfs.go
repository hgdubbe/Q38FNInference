package httpapi

import (
	"embed"
	"io/fs"
	"net/http"
)

//go:embed all:web
var webFiles embed.FS

// webFS serves the embedded control-panel UI (index.html, app.js, style.css)
// rooted at the "web" subdirectory rather than the module root.
func webFS() http.Handler {
	sub, err := fs.Sub(webFiles, "web")
	if err != nil {
		panic(err) // programmer error: web/ is embedded at build time, always present
	}
	return http.FileServer(http.FS(sub))
}
