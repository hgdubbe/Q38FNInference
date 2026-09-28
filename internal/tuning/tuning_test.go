package tuning

import (
	"fmt"
	"testing"

	"github.com/hgdubbe/q38fninference/internal/gguf"
)

// synthModel builds a fake qwen4exp-shaped model: nLayer layers, each with
// one expert tensor of expertBytes and one shared/attn tensor of sharedBytes
// (both F32 so the byte math is exact and easy to reason about), plus a
// small global tensor.
func synthModel(nLayer int, expertBytes, sharedBytes uint64) *gguf.Metadata {
	elemsExpert := expertBytes / 4
	elemsShared := sharedBytes / 4

	tensors := []gguf.TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{256}, Type: 0}, // F32 = type id 0
	}
	for i := 0; i < nLayer; i++ {
		tensors = append(tensors,
			gguf.TensorInfo{Name: fmt.Sprintf("blk.%d.ffn_gate_exps.weight", i), Dims: []uint64{elemsExpert}, Type: 0},
			gguf.TensorInfo{Name: fmt.Sprintf("blk.%d.attn_q.weight", i), Dims: []uint64{elemsShared}, Type: 0},
		)
	}

	return &gguf.Metadata{
		KV: map[string]any{
			"general.architecture":    "qwen4exp",
			"qwen4exp.block_count":    uint32(nLayer),
			"qwen4exp.context_length": uint32(262144),
		},
		Tensors: tensors,
	}
}

func TestComputeFitsEntirelyOnGPU(t *testing.T) {
	m := synthModel(4, 100*MiB, 10*MiB)
	gpus := []GPU{{Name: "test-gpu", FreeBytes: 10 * GiB}}

	p, err := Compute(m, gpus, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if p.NCPUMoE != 0 {
		t.Errorf("NCPUMoE = %d, want 0 (everything should fit)", p.NCPUMoE)
	}
	if p.NGpuLayers != 4 {
		t.Errorf("NGpuLayers = %d, want 4", p.NGpuLayers)
	}
	if !p.FullyOnGPU {
		t.Error("FullyOnGPU = false, want true")
	}
}

func TestComputeOffloadsExpertsToFit(t *testing.T) {
	// 8 layers, 1 GiB of experts each (8 GiB total experts) + 10 MiB shared each.
	// A 3 GiB GPU (minus ~1 GiB+ reserve) can only hold a couple of layers' experts.
	m := synthModel(8, 1*GiB, 10*MiB)
	gpus := []GPU{{Name: "small-gpu", FreeBytes: 3 * GiB}}

	p, err := Compute(m, gpus, Options{RequestedCtx: 4096, ReserveBytes: 512 * MiB})
	if err != nil {
		t.Fatal(err)
	}
	if p.NGpuLayers != 8 {
		t.Errorf("NGpuLayers = %d, want 8 (all layers present on GPU, just with experts offloaded)", p.NGpuLayers)
	}
	if p.NCPUMoE <= 0 || p.NCPUMoE >= 8 {
		t.Errorf("NCPUMoE = %d, want somewhere in (0, 8) for a GPU that fits some but not all experts", p.NCPUMoE)
	}
	// verify the plan actually respects the budget
	budget := int64(3*GiB) - int64(512*MiB)
	if int64(p.GPUFitBytes) > budget {
		t.Errorf("GPUFitBytes = %d exceeds budget %d", p.GPUFitBytes, budget)
	}
}

func TestComputeFallsBackToPartialLayerOffload(t *testing.T) {
	// Even zero experts on GPU won't fit: dense/shared weight alone (80 MiB/layer * 8) exceeds a tiny GPU.
	m := synthModel(8, 1*GiB, 80*MiB)
	gpus := []GPU{{Name: "tiny-gpu", FreeBytes: 200 * MiB}}

	p, err := Compute(m, gpus, Options{RequestedCtx: 4096, ReserveBytes: 16 * MiB})
	if err != nil {
		t.Fatal(err)
	}
	if p.NGpuLayers >= 8 {
		t.Errorf("NGpuLayers = %d, want < 8 (GPU too small to hold every layer's dense weights)", p.NGpuLayers)
	}
	if p.NCPUMoE != 0 {
		t.Errorf("NCPUMoE = %d, want 0 in the partial-offload fallback branch", p.NCPUMoE)
	}
}

func TestComputeNoGPU(t *testing.T) {
	m := synthModel(2, 1*MiB, 1*MiB)
	p, err := Compute(m, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if p.NGpuLayers != 0 {
		t.Errorf("NGpuLayers = %d, want 0 with no GPU detected", p.NGpuLayers)
	}
}

func TestArgsRendering(t *testing.T) {
	p := &Plan{NGpuLayers: 48, NCPUMoE: 10, CtxSize: 32768, FlashAttn: true, CacheTypeKV: "q8_0"}
	args := p.Args("/models/model.gguf")
	joined := ""
	for _, a := range args {
		joined += a + " "
	}
	for _, want := range []string{"--model", "/models/model.gguf", "--n-cpu-moe", "10", "--flash-attn", "on", "--cache-type-k", "q8_0"} {
		if !contains(args, want) {
			t.Errorf("args %v missing %q (joined: %s)", args, want, joined)
		}
	}
}

func contains(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}
