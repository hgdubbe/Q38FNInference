// Command q38fninference is the launcher: it serves the local control panel,
// fronts llama-server with the OpenAI-compatible API port, and opens the
// control panel in the default browser. Release builds are linked as a
// Windows GUI app (no console window); logs go to launcher.log in the
// config directory, and the app exits via the panel's Quit button.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/hgdubbe/q38fninference/internal/appconfig"
	"github.com/hgdubbe/q38fninference/internal/browser"
	"github.com/hgdubbe/q38fninference/internal/httpapi"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8787", "control panel address")
	noBrowser := flag.Bool("no-browser", false, "don't open the control panel in a browser")
	flag.Parse()

	setupLog()

	open := func(url string) {
		if err := browser.Open(url); err != nil {
			log.Printf("couldn't open a browser (%v); open %s manually", err, url)
		}
	}
	panelURL := "http://" + *addr

	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		// double-clicking the exe again should just bring the panel back
		if isRunningInstance(panelURL) {
			if !*noBrowser {
				open(panelURL)
			}
			return
		}
		log.Fatalf("listening on %s: %v", *addr, err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv, err := httpapi.New(httpapi.Hooks{OpenURL: open, Quit: stop})
	if err != nil {
		log.Fatal(err)
	}

	panel := &http.Server{Handler: srv.Handler()}
	go serve(panel, ln)
	log.Printf("control panel on %s", panelURL)

	cfg := srv.Config()
	apiAddr := net.JoinHostPort(cfg.APIHost, strconv.Itoa(cfg.Port))
	api := &http.Server{Handler: srv.Proxy}
	if apiLn, err := net.Listen("tcp", apiAddr); err != nil {
		log.Printf("API port unavailable: %v", err)
		srv.SetAPIError(fmt.Errorf("could not listen on %s: %w (change the port in Settings and restart)", apiAddr, err))
	} else {
		go serve(api, apiLn)
		log.Printf("OpenAI-compatible API on http://%s/v1", apiAddr)
	}

	if !*noBrowser {
		open(panelURL)
	}

	go srv.QuitWhenPanelClosed(ctx, 15*time.Second)

	<-ctx.Done()
	log.Printf("shutting down")
	stopCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	srv.Shutdown(stopCtx) // stop llama-server first so its VRAM is released
	// open log streams and in-flight completions never go idle, so don't
	// wait on them for long
	for _, h := range []*http.Server{api, panel} {
		c, cancel := context.WithTimeout(context.Background(), time.Second)
		if h.Shutdown(c) != nil {
			h.Close()
		}
		cancel()
	}
}

func serve(s *http.Server, ln net.Listener) {
	if err := s.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Printf("server: %v", err)
	}
}

func isRunningInstance(panelURL string) bool {
	c := &http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get(panelURL + "/api/info")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var info struct{ App string }
	return json.NewDecoder(resp.Body).Decode(&info) == nil && info.App == httpapi.AppID
}

// setupLog tees logs to launcher.log: a GUI-subsystem exe has no console.
func setupLog() {
	dir, err := appconfig.Dir()
	if err != nil {
		return
	}
	f, err := os.OpenFile(filepath.Join(dir, "launcher.log"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return
	}
	log.SetOutput(io.MultiWriter(f, os.Stderr)) // file first: stderr is invalid in a GUI exe
}
