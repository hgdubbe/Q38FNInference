package gpu

import (
	"testing"

	"github.com/hgdubbe/q38fninference/internal/tuning"
)

func TestParseNvidiaSMI(t *testing.T) {
	out := "NVIDIA GeForce RTX 4090, 20480, 24564\n"
	gpus, err := parseNvidiaSMI(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("len(gpus) = %d, want 1", len(gpus))
	}
	if gpus[0].Name != "NVIDIA GeForce RTX 4090" {
		t.Errorf("Name = %q", gpus[0].Name)
	}
	want := uint64(20480) * tuning.MiB
	if gpus[0].FreeBytes != want {
		t.Errorf("FreeBytes = %d, want %d", gpus[0].FreeBytes, want)
	}
}

func TestParseNvidiaSMIMultiGPU(t *testing.T) {
	out := "GPU 0, 10240, 24564\nGPU 1, 24000, 24564\n"
	gpus, err := parseNvidiaSMI(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 2 {
		t.Fatalf("len(gpus) = %d, want 2", len(gpus))
	}
}

func TestParseNvidiaSMIEmpty(t *testing.T) {
	gpus, err := parseNvidiaSMI("")
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 0 {
		t.Errorf("len(gpus) = %d, want 0", len(gpus))
	}
}

func TestParseNvidiaSMIMalformed(t *testing.T) {
	if _, err := parseNvidiaSMI("garbage line without commas\n"); err == nil {
		t.Fatal("expected error for malformed line")
	}
}
