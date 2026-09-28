package tuning

import (
	"fmt"
	"slices"
	"testing"

	"github.com/hgdubbe/q38fninference/internal/gguf"
)

// synthModel builds a qwen4exp-shaped model with F32 tensors (exact byte
// math): per block one expert tensor and one shared tensor, plus an input
// embedding (CPU-resident) and an output head. No cache metadata, so cache
// bytes are zero unless a test adds keys.
func synthModel(nLayer int, expertBytes, sharedBytes, outputBytes uint64) *gguf.Metadata {
	tensors := []gguf.TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{1 << 20}, Type: 0},
		{Name: "output.weight", Dims: []uint64{outputBytes / 4}, Type: 0},
	}
	for i := 0; i < nLayer; i++ {
		tensors = append(tensors,
			gguf.TensorInfo{Name: fmt.Sprintf("blk.%d.ffn_gate_up_exps.weight", i), Dims: []uint64{expertBytes / 4}, Type: 0},
			gguf.TensorInfo{Name: fmt.Sprintf("blk.%d.attn_q.weight", i), Dims: []uint64{sharedBytes / 4}, Type: 0},
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

// fixed per-GPU overhead the planner subtracts (context + compute + reserve)
func overhead(main bool) uint64 {
	c := uint64(computeOtherBytes)
	if main {
		c = computeMainBytes
	}
	return cudaContextBytes + c + 512*MiB
}

func TestFitsEntirelyOnOneGPU(t *testing.T) {
	m := synthModel(4, 100*MiB, 10*MiB, 50*MiB)
	p, err := Compute(m, []GPU{{Index: 0, Name: "g", FreeBytes: 10 * GiB}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if p.NGpuLayers != 5 || p.NCPUMoE != 0 || !p.FullyOnGPU {
		t.Errorf("plan = %+v, want all 5 slots on GPU, no CPU experts", p)
	}
	if p.TensorSplit != nil {
		t.Errorf("TensorSplit = %v, want nil for a single GPU", p.TensorSplit)
	}
	if want := uint64(4*110*MiB + 50*MiB); p.GPUFitBytes != want {
		t.Errorf("GPUFitBytes = %d, want %d (token_embd must stay on CPU)", p.GPUFitBytes, want)
	}
}

func TestPinsMinimalExpertsToFit(t *testing.T) {
	// 8 blocks x (1 GiB experts + 10 MiB shared) + 10 MiB output.
	m := synthModel(8, 1*GiB, 10*MiB, 10*MiB)
	free := overhead(true) + 3*GiB + 100*MiB // room for dense part + 3 blocks' experts
	p, err := Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if p.NGpuLayers != 9 || p.NCPUMoE != 5 {
		t.Errorf("ngl=%d ncmoe=%d, want 9 and 5", p.NGpuLayers, p.NCPUMoE)
	}
}

func TestPartialOffloadPinsExpertsAndUsesTrailingSlots(t *testing.T) {
	// dense part alone (8 x 80 MiB + output) can't fit a ~250 MiB budget
	m := synthModel(8, 1*GiB, 80*MiB, 20*MiB)
	free := overhead(true) + 250*MiB
	p, err := Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	// output (20) + 3 blocks (240) = 260 > 250; output + 2 blocks = 180 fits
	if p.NGpuLayers != 3 {
		t.Errorf("NGpuLayers = %d, want 3 (output + last 2 blocks)", p.NGpuLayers)
	}
	if p.NCPUMoE != 8 {
		t.Errorf("NCPUMoE = %d, want 8: GPU-resident blocks must not pull their experts onto the GPU", p.NCPUMoE)
	}
	if p.GPUFitBytes != 180*MiB {
		t.Errorf("GPUFitBytes = %d MiB, want 180", p.GPUFitBytes/MiB)
	}
}

func TestMultiGPUSplitsBySlotBytes(t *testing.T) {
	// 6 blocks x (500 MiB experts + 100 MiB shared) + 100 MiB output = 3.7 GiB
	m := synthModel(6, 500*MiB, 100*MiB, 100*MiB)
	gpus := []GPU{
		{Index: 0, Name: "a", FreeBytes: overhead(true) + 1300*MiB},  // 2 blocks
		{Index: 2, Name: "b", FreeBytes: overhead(false) + 2600*MiB}, // 4 blocks + output
	}
	p, err := Compute(m, gpus, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !p.FullyOnGPU {
		t.Fatalf("expected full fit across two GPUs, got %+v", p)
	}
	if !slices.Equal(p.TensorSplit, []int{2, 5}) {
		t.Errorf("TensorSplit = %v, want [2 5]", p.TensorSplit)
	}
	if !slices.Equal(p.Devices, []int{0, 2}) {
		t.Errorf("Devices = %v, want [0 2]", p.Devices)
	}
	args := p.Args("m.gguf")
	if !slices.Contains(args, "--tensor-split") || !slices.Contains(args, "2,5") {
		t.Errorf("args %v missing --tensor-split 2,5", args)
	}
}

func TestMultiGPUPooledMemoryAvoidsCPUExperts(t *testing.T) {
	m := synthModel(4, 1*GiB, 10*MiB, 10*MiB)
	one := []GPU{{Index: 0, FreeBytes: overhead(true) + 2200*MiB}}
	two := append(one, GPU{Index: 1, FreeBytes: overhead(false) + 2200*MiB})

	p1, _ := Compute(m, one, Options{})
	p2, _ := Compute(m, two, Options{})
	if p1.NCPUMoE == 0 {
		t.Fatal("single GPU should need CPU experts")
	}
	if p2.NCPUMoE != 0 {
		t.Errorf("two GPUs: NCPUMoE = %d, want 0", p2.NCPUMoE)
	}
}

func TestCacheBytesFromMetadata(t *testing.T) {
	m := synthModel(4, 1*MiB, 1*MiB, 1*MiB)
	m.KV["qwen4exp.attention.recurrent_layers"] = []any{true, true, true, false}
	m.KV["qwen4exp.attention.head_count_kv"] = uint32(2)
	m.KV["qwen4exp.attention.key_length"] = uint32(128)
	m.KV["qwen4exp.attention.value_length"] = uint32(128)
	m.KV["qwen4exp.ssm.conv_kernel"] = uint32(4)
	m.KV["qwen4exp.ssm.inner_size"] = uint32(1024)
	m.KV["qwen4exp.ssm.state_size"] = uint32(128)
	m.KV["qwen4exp.ssm.group_count"] = uint32(16)

	p, err := Compute(m, []GPU{{FreeBytes: 64 * GiB}}, Options{RequestedCtx: 1024, CacheType: "f16"})
	if err != nil {
		t.Fatal(err)
	}
	kv := uint64(1024 * 2 * (128 + 128) * 2)               // one attention block
	recr := uint64(3 * (3*(1024+2*16*128) + 128*1024) * 4) // three GDN blocks, 1 slot
	weights := uint64(4*2*MiB + 1*MiB)
	if want := weights + kv + recr; p.GPUFitBytes != want {
		t.Errorf("GPUFitBytes = %d, want %d", p.GPUFitBytes, want)
	}
}

func TestNoGPU(t *testing.T) {
	p, err := Compute(synthModel(2, MiB, MiB, MiB), nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if p.NGpuLayers != 0 || p.GPUFitBytes != 0 {
		t.Errorf("plan = %+v, want CPU-only", p)
	}
}

func TestArgsSingleGPU(t *testing.T) {
	p := &Plan{NGpuLayers: 49, NCPUMoE: 10, CtxSize: 32768, Parallel: 1, FlashAttn: true, CacheTypeKV: "q8_0"}
	args := p.Args("/m.gguf")
	for _, want := range []string{"--model", "/m.gguf", "--n-cpu-moe", "10", "--fit", "off", "--cache-type-k", "q8_0"} {
		if !slices.Contains(args, want) {
			t.Errorf("args %v missing %q", args, want)
		}
	}
	if slices.Contains(args, "--tensor-split") {
		t.Errorf("single GPU args must not set --tensor-split: %v", args)
	}
}
