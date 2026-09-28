package gpu

import (
	"testing"

	"github.com/hgdubbe/q38fninference/internal/tuning"
)

func TestParseNvidiaSMI(t *testing.T) {
	gpus, err := parseNvidiaSMI("0, 20480, 24564, NVIDIA GeForce RTX 4090\n1, 10000, 12288, NVIDIA RTX A2000, Rev 2\n")
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 2 {
		t.Fatalf("len(gpus) = %d, want 2", len(gpus))
	}
	g := gpus[0]
	if g.Index != 0 || g.Name != "NVIDIA GeForce RTX 4090" || g.FreeBytes != 20480*tuning.MiB || g.TotalBytes != 24564*tuning.MiB {
		t.Errorf("gpu0 = %+v", g)
	}
	if gpus[1].Index != 1 || gpus[1].Name != "NVIDIA RTX A2000, Rev 2" {
		t.Errorf("gpu1 = %+v (name with comma must survive)", gpus[1])
	}
}

func TestParseNvidiaSMIEmpty(t *testing.T) {
	gpus, err := parseNvidiaSMI("")
	if err != nil || len(gpus) != 0 {
		t.Errorf("got %v, %v", gpus, err)
	}
}

func TestParseNvidiaSMIMalformed(t *testing.T) {
	if _, err := parseNvidiaSMI("garbage line without commas\n"); err == nil {
		t.Fatal("expected error for malformed line")
	}
}
