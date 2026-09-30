// Package hf is a minimal client for the parts of the Hugging Face Hub API
// this launcher needs: listing a repo's files and resumably downloading one
// with progress. It deliberately doesn't pull in the full huggingface_hub
// surface — just enough to browse a model repo and pull its GGUF files.
//
// API reference (stable, unauthenticated, documented):
//
//	GET https://huggingface.co/api/models/{repo}        -> repo info + file list
//	GET https://huggingface.co/api/models?search=...     -> search
//	GET https://huggingface.co/{repo}/resolve/main/{file} -> file content (redirects to CDN, supports Range)
package hf

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const defaultBaseURL = "https://huggingface.co"

// Client talks to the Hugging Face Hub.
type Client struct {
	BaseURL    string
	Token      string // optional, for gated/private repos
	HTTPClient *http.Client
}

// NewClient returns a Client configured with sane defaults.
func NewClient(token string) *Client {
	return &Client{
		BaseURL: defaultBaseURL,
		Token:   token,
		HTTPClient: &http.Client{
			Timeout: 0, // downloads can be large/slow; callers pass context deadlines instead
		},
	}
}

// Sibling is one file in a model repo.
type Sibling struct {
	RFilename string `json:"rfilename"`
	Size      int64  `json:"size,omitempty"`
}

// ModelInfo is the subset of the Hub's model-info response this client uses.
type ModelInfo struct {
	ID       string    `json:"id"`
	Siblings []Sibling `json:"siblings"`
}

// ModelSummary is one hit from a repo search.
type ModelSummary struct {
	ID     string   `json:"id"`
	Author string   `json:"author"`
	Likes  int      `json:"likes"`
	Tags   []string `json:"tags"`
}

func (c *Client) baseURL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return defaultBaseURL
}

func (c *Client) newRequest(ctx context.Context, method, rawURL string, extraHeaders map[string]string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, rawURL, nil)
	if err != nil {
		return nil, err
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	req.Header.Set("User-Agent", "q38fninference-launcher/1.0")
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
	}
	return req, nil
}

// GetModelInfo fetches a repo's metadata, including its full file list.
func (c *Client) GetModelInfo(ctx context.Context, repo string) (*ModelInfo, error) {
	u := fmt.Sprintf("%s/api/models/%s", c.baseURL(), repo)
	req, err := c.newRequest(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("hf: GET %s: %s: %s", u, resp.Status, string(body))
	}

	var info ModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("hf: decoding model info for %s: %w", repo, err)
	}
	return &info, nil
}

// GGUFFiles filters a repo's file list down to .gguf files.
func (m *ModelInfo) GGUFFiles() []Sibling {
	var out []Sibling
	for _, s := range m.Siblings {
		if strings.HasSuffix(strings.ToLower(s.RFilename), ".gguf") {
			out = append(out, s)
		}
	}
	return out
}

// SearchModels searches the Hub for model repos.
func (c *Client) SearchModels(ctx context.Context, query string, limit int) ([]ModelSummary, error) {
	q := url.Values{}
	q.Set("search", query)
	q.Set("limit", strconv.Itoa(limit))
	q.Set("full", "false")
	u := fmt.Sprintf("%s/api/models?%s", c.baseURL(), q.Encode())

	req, err := c.newRequest(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("hf: GET %s: %s: %s", u, resp.Status, string(body))
	}

	var out []ModelSummary
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("hf: decoding search results: %w", err)
	}
	return out, nil
}

// FileURL is the direct download URL for one file in a repo's main revision.
func (c *Client) FileURL(repo, filename string) string {
	// GGUF repos often keep split quants in subfolders: escape per segment
	segs := strings.Split(filename, "/")
	for i, seg := range segs {
		segs[i] = url.PathEscape(seg)
	}
	return fmt.Sprintf("%s/%s/resolve/main/%s", c.baseURL(), repo, strings.Join(segs, "/"))
}

// Progress is called periodically during Download with the bytes downloaded
// so far and the total size (0 if unknown).
type Progress func(downloaded, total int64)

// Download resumably fetches repo/filename into destPath. If destPath (or a
// same-named ".part" file next to it) already has bytes, it resumes with a
// Range request instead of starting over. The download lands in a ".part"
// file and is only renamed to destPath once fully and successfully written,
// so a killed/failed download is never mistaken for a complete model file by
// the local-model scanner.
func (c *Client) Download(ctx context.Context, repo, filename, destPath string, progress Progress) error {
	partPath := destPath + ".part"

	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return err
	}

	var startOffset int64
	if fi, err := os.Stat(partPath); err == nil {
		startOffset = fi.Size()
	}

	headers := map[string]string{}
	if startOffset > 0 {
		headers["Range"] = fmt.Sprintf("bytes=%d-", startOffset)
	}

	req, err := c.newRequest(ctx, http.MethodGet, c.FileURL(repo, filename), headers)
	if err != nil {
		return err
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var out *os.File
	switch resp.StatusCode {
	case http.StatusOK:
		// server ignored/doesn't support our Range request: start over.
		startOffset = 0
		out, err = os.Create(partPath)
	case http.StatusPartialContent:
		out, err = os.OpenFile(partPath, os.O_WRONLY|os.O_APPEND, 0o644)
	default:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("hf: download %s/%s: %s: %s", repo, filename, resp.Status, string(body))
	}
	if err != nil {
		return err
	}
	defer out.Close()

	total := startOffset + resp.ContentLength
	if resp.ContentLength <= 0 {
		total = 0
	}

	pw := &progressWriter{w: out, done: startOffset, total: total, cb: progress, last: time.Now()}
	if _, err := io.Copy(pw, resp.Body); err != nil {
		return fmt.Errorf("hf: downloading %s/%s: %w", repo, filename, err)
	}
	if err := out.Close(); err != nil {
		return err
	}

	if err := os.Rename(partPath, destPath); err != nil {
		return fmt.Errorf("hf: finalizing download of %s: %w", destPath, err)
	}
	if progress != nil {
		progress(pw.done, pw.total)
	}
	return nil
}

type progressWriter struct {
	w           io.Writer
	done, total int64
	cb          Progress
	last        time.Time
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	if p.cb != nil && time.Since(p.last) > 200*time.Millisecond {
		p.cb(p.done, p.total)
		p.last = time.Now()
	}
	return n, err
}
