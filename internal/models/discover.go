// Package models discovers GGUF model files already present on disk, so the
// launcher doesn't force a re-download of something the user (or another
// tool: llama.cpp, LM Studio, text-generation-webui, huggingface-cli, ...)
// already fetched.
package models

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
)

// Found is one discovered GGUF file, possibly one shard of a split model.
type Found struct {
	Path       string // full path to this shard
	Repo       string // best-effort "org/name" inferred from a HF cache layout, else ""
	IsShard    bool
	ShardIndex int // 1-based; 0 if not a split file
	ShardTotal int
	SizeBytes  int64
}

// Group is one logical model: all shards of a split GGUF, or a single file.
type Group struct {
	Name      string // display name: repo if known, else the base filename
	Files     []Found
	TotalSize int64
}

var shardRe = regexp.MustCompile(`^(.+)-(\d{5})-of-(\d{5})\.gguf$`)

// DefaultSearchDirs returns the directories this launcher checks by default:
// the standard Hugging Face cache, and this app's own models directory.
func DefaultSearchDirs() []string {
	var dirs []string

	if hfHome := os.Getenv("HF_HOME"); hfHome != "" {
		dirs = append(dirs, filepath.Join(hfHome, "hub"))
	} else if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs, filepath.Join(home, ".cache", "huggingface", "hub"))
	}

	if appData := appModelsDir(); appData != "" {
		dirs = append(dirs, appData)
	}

	return dirs
}

func appModelsDir() string {
	switch runtime.GOOS {
	case "windows":
		if base := os.Getenv("LOCALAPPDATA"); base != "" {
			return filepath.Join(base, "Q38FNInference", "models")
		}
	default:
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, ".local", "share", "q38fninference", "models")
		}
	}
	return ""
}

// Scan walks the given directories (non-recursive descent into arbitrarily
// deep trees is fine; symlink loops are not followed) and returns every
// *.gguf file found, grouped into logical models by split-shard naming.
func Scan(dirs []string) ([]Group, error) {
	var found []Found

	for _, dir := range dirs {
		info, err := os.Stat(dir)
		if err != nil || !info.IsDir() {
			continue // missing search dir is not an error: e.g. no HF cache yet
		}
		err = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil // skip unreadable entries rather than aborting the whole scan
			}
			if d.IsDir() {
				return nil
			}
			if !strings.EqualFold(filepath.Ext(d.Name()), ".gguf") {
				return nil
			}
			fi, err := d.Info()
			if err != nil {
				return nil
			}
			found = append(found, newFound(path, fi.Size(), dir))
			return nil
		})
		if err != nil {
			return nil, err
		}
	}

	return group(found), nil
}

func newFound(path string, size int64, searchRoot string) Found {
	base := filepath.Base(path)
	f := Found{Path: path, SizeBytes: size, Repo: inferRepo(path, searchRoot)}
	if m := shardRe.FindStringSubmatch(base); m != nil {
		f.IsShard = true
		f.ShardIndex = atoi(m[2])
		f.ShardTotal = atoi(m[3])
	}
	return f
}

// inferRepo recognizes the Hugging Face hub cache layout:
//
//	<hub>/models--Org--Name/snapshots/<rev>/file.gguf
//
// and turns it back into "Org/Name". Returns "" for anything else (e.g. a
// flat models folder), which is fine — the caller falls back to filename.
func inferRepo(path, searchRoot string) string {
	rel, err := filepath.Rel(searchRoot, path)
	if err != nil {
		return ""
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	if len(parts) == 0 {
		return ""
	}
	top := parts[0]
	if !strings.HasPrefix(top, "models--") {
		return ""
	}
	orgName := strings.TrimPrefix(top, "models--")
	return strings.ReplaceAll(orgName, "--", "/")
}

func group(found []Found) []Group {
	byKey := map[string]*Group{}
	var order []string

	for _, f := range found {
		base := filepath.Base(f.Path)
		key := base
		name := base
		if f.IsShard {
			if m := shardRe.FindStringSubmatch(base); m != nil {
				key = filepath.Dir(f.Path) + "/" + m[1]
				name = m[1]
			}
		}
		if f.Repo != "" {
			// group by repo, so unrelated files with the same shard prefix
			// under different repos don't collide
			key = f.Repo + "|" + key
			name = f.Repo
		}

		g, ok := byKey[key]
		if !ok {
			g = &Group{Name: name}
			byKey[key] = g
			order = append(order, key)
		}
		g.Files = append(g.Files, f)
		g.TotalSize += f.SizeBytes
	}

	groups := make([]Group, 0, len(byKey))
	for _, k := range order {
		g := *byKey[k]
		sort.Slice(g.Files, func(i, j int) bool { return g.Files[i].ShardIndex < g.Files[j].ShardIndex })
		groups = append(groups, g)
	}
	sort.Slice(groups, func(i, j int) bool { return groups[i].Name < groups[j].Name })
	return groups
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return n
		}
		n = n*10 + int(c-'0')
	}
	return n
}
