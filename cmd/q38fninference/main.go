// Command q38fninference is the launcher's entrypoint: it starts the local
// control-panel HTTP server (internal/httpapi) and opens it in the default
// browser. The actual model inference happens in a separately managed
// llama-server child process — see internal/server and docs/ARCHITECTURE.md
// for why this app doesn't reimplement an OpenAI API or web chat UI itself.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/hgdubbe/q38fninference/internal/httpapi"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8787", "address the control panel listens on")
	noBrowser := flag.Bool("no-browser", false, "don't auto-open the control panel in a browser")
	flag.Parse()

	srv, err := httpapi.New()
	if err != nil {
		log.Fatalf("q38fninference: %v", err)
	}

	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("q38fninference: listening on %s: %v", *addr, err)
	}

	httpServer := &http.Server{Handler: srv.Handler()}

	go func() {
		log.Printf("q38fninference: control panel on http://%s", ln.Addr())
		if err := httpServer.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Fatalf("q38fninference: %v", err)
		}
	}()

	if !*noBrowser {
		openBrowser(fmt.Sprintf("http://%s", ln.Addr()))
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownCtx)
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", "", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	if err := cmd.Start(); err != nil {
		log.Printf("q38fninference: couldn't auto-open a browser (%v); open %s manually", err, url)
	}
}
