package models

import (
	"os"
	"path/filepath"
	"testing"
)

func touch(t *testing.T, path string, size int) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestScanFlatDir(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "model.gguf"), 100)
	touch(t, filepath.Join(dir, "README.md"), 10)

	groups, err := Scan([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 1 {
		t.Fatalf("groups = %v", groups)
	}
	if groups[0].Name != "model.gguf" {
		t.Errorf("Name = %q", groups[0].Name)
	}
	if groups[0].TotalSize != 100 {
		t.Errorf("TotalSize = %d", groups[0].TotalSize)
	}
}

func TestScanSplitShards(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "qwen4exp-00001-of-00002.gguf"), 100)
	touch(t, filepath.Join(dir, "qwen4exp-00002-of-00002.gguf"), 200)

	groups, err := Scan([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 1 {
		t.Fatalf("groups = %v, want 1 (shards should merge)", groups)
	}
	g := groups[0]
	if len(g.Files) != 2 {
		t.Fatalf("Files = %v", g.Files)
	}
	if g.TotalSize != 300 {
		t.Errorf("TotalSize = %d, want 300", g.TotalSize)
	}
	if g.Files[0].ShardIndex != 1 || g.Files[1].ShardIndex != 2 {
		t.Errorf("shard order wrong: %+v", g.Files)
	}
}

func TestScanHFCacheLayoutInfersRepo(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "models--Qwen--Qwen3.8-Flash-Next", "snapshots", "abcdef", "model-00001-of-00001.gguf")
	touch(t, p, 42)

	groups, err := Scan([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 1 {
		t.Fatalf("groups = %v", groups)
	}
	if groups[0].Name != "Qwen/Qwen3.8-Flash-Next" {
		t.Errorf("Name = %q, want Qwen/Qwen3.8-Flash-Next", groups[0].Name)
	}
}

func TestScanMissingDirIsNotError(t *testing.T) {
	groups, err := Scan([]string{"/no/such/path/anywhere"})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 0 {
		t.Errorf("groups = %v, want empty", groups)
	}
}

func TestScanIgnoresNonGGUF(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "notes.txt"), 5)
	touch(t, filepath.Join(dir, "weights.safetensors"), 5)

	groups, err := Scan([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 0 {
		t.Errorf("groups = %v, want empty", groups)
	}
}

func TestScanOverlappingDirsCountsShardsOnce(t *testing.T) {
	root := t.TempDir()
	sub := filepath.Join(root, "IQ4_XS")
	for i := 1; i <= 3; i++ {
		touch(t, filepath.Join(sub, "m-0000"+string(rune('0'+i))+"-of-00003.gguf"), 10)
	}
	// a folder and its parent both searched, one spelled with a trailing separator
	groups, err := Scan([]string{root, sub + string(filepath.Separator)})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 1 || len(groups[0].Files) != 3 {
		t.Fatalf("groups = %+v, want one model with 3 parts", groups)
	}
	if ok, have, total := groups[0].Complete(); !ok || have != 3 || total != 3 {
		t.Errorf("Complete() = %v, %d, %d; want true, 3, 3", ok, have, total)
	}
}

func TestCompleteReportsMissingParts(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "m-00001-of-00003.gguf"), 10)
	touch(t, filepath.Join(dir, "m-00003-of-00003.gguf"), 10)
	groups, _ := Scan([]string{dir})
	if ok, have, total := groups[0].Complete(); ok || have != 2 || total != 3 {
		t.Errorf("Complete() = %v, %d, %d; want false, 2, 3", ok, have, total)
	}
	single := Group{Files: []Found{{Path: "x.gguf"}}}
	if ok, _, _ := single.Complete(); !ok {
		t.Error("a single-file model is always complete")
	}
}
