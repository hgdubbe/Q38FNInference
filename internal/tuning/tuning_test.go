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

	// without op offload: only the weights compete for VRAM
	p, err := Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{NoOpOffload: true, UBatch: 512})
	if err != nil {
		t.Fatal(err)
	}
	if p.NGpuLayers != 9 || p.NCPUMoE != 5 || p.StagingBytes != 0 {
		t.Errorf("no-op-offload: ngl=%d ncmoe=%d staging=%d, want 9, 5, 0", p.NGpuLayers, p.NCPUMoE, p.StagingBytes)
	}

	// default: experts in RAM get a staging area for one block (1 GiB) and
	// the micro-batch goes to 2048 (+512 MiB compute), leaving room for one
	// block's experts on the GPU
	p, err = Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if p.NGpuLayers != 9 || p.NCPUMoE != 7 {
		t.Errorf("ngl=%d ncmoe=%d, want 9 and 7", p.NGpuLayers, p.NCPUMoE)
	}
	if p.StagingBytes != 1*GiB || p.UBatch != 2048 || p.NoOpOffload {
		t.Errorf("staging=%d ubatch=%d noOpOffload=%v, want 1 GiB, 2048, false", p.StagingBytes, p.UBatch, p.NoOpOffload)
	}
	args := p.Args("m.gguf")
	for _, want := range []string{"--ubatch-size", "2048"} {
		if !slices.Contains(args, want) {
			t.Errorf("args %v missing %q", args, want)
		}
	}
}

func TestPartialOffloadPinsExpertsAndUsesTrailingSlots(t *testing.T) {
	// dense part alone (8 x 80 MiB + output) can't fit a ~250 MiB budget,
	// and a 1 GiB staging area certainly can't: the planner drops op offload
	m := synthModel(8, 1*GiB, 80*MiB, 20*MiB)
	free := overhead(true) + 512*MiB + 250*MiB // +512 MiB: compute at ubatch 2048
	p, err := Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !p.NoOpOffload || !slices.Contains(p.Args("m"), "--no-op-offload") {
		t.Errorf("NoOpOffload = %v, want true (staging can't fit)", p.NoOpOffload)
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

func TestFullFitNeedsNoStagingOrBiggerBatch(t *testing.T) {
	p, err := Compute(synthModel(4, 100*MiB, 10*MiB, 10*MiB), []GPU{{FreeBytes: 10 * GiB}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if p.StagingBytes != 0 || p.UBatch != 0 || slices.Contains(p.Args("m"), "--ubatch-size") {
		t.Errorf("staging=%d ubatch=%d: nothing in RAM, so neither should be set", p.StagingBytes, p.UBatch)
	}
}

func TestMultiGPUSplitsBySlotBytes(t *testing.T) {
	// 6 blocks x (500 MiB experts + 100 MiB shared) + 100 MiB output
	m := synthModel(6, 500*MiB, 100*MiB, 100*MiB)
	gpus := []GPU{
		{Index: 0, Name: "small", FreeBytes: overhead(false) + 1300*MiB},
		{Index: 2, Name: "big", FreeBytes: overhead(true) + 2600*MiB},
	}
	p, err := Compute(m, gpus, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !p.FullyOnGPU {
		t.Fatalf("expected full fit across two GPUs, got %+v", p)
	}
	// the bigger GPU leads (main device), taking 4 blocks; the small one
	// takes the last 2 blocks plus the output layer
	if !slices.Equal(p.Devices, []int{2, 0}) {
		t.Errorf("Devices = %v, want [2 0]", p.Devices)
	}
	if !slices.Equal(p.TensorSplit, []int{4, 3}) {
		t.Errorf("TensorSplit = %v, want [4 3]", p.TensorSplit)
	}
	args := p.Args("m.gguf")
	if !slices.Contains(args, "--tensor-split") || !slices.Contains(args, "4,3") {
		t.Errorf("args %v missing --tensor-split 4,3", args)
	}
}

func TestLargeSplitModelPinsExpertsAcrossTwoGPUs(t *testing.T) {
	// roughly the reported setup: ~64 GiB of weights, 48 blocks, an 8 GB
	// and a 16 GB card listed in that order
	m := synthModel(48, 1300*MiB, 60*MiB, 600*MiB)
	gpus := []GPU{
		{Index: 0, Name: "RTX 3050", FreeBytes: 7900 * MiB},
		{Index: 1, Name: "RTX 4070 Ti SUPER", FreeBytes: 14700 * MiB},
	}
	p, err := Compute(m, gpus, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if p.Devices[0] != 1 {
		t.Errorf("Devices = %v, want the 16 GB card first", p.Devices)
	}
	if p.NGpuLayers != 49 || p.NCPUMoE < 30 || p.NCPUMoE >= 48 {
		t.Errorf("ngl=%d ncmoe=%d; want all slots on GPU with most experts in RAM", p.NGpuLayers, p.NCPUMoE)
	}
	for d, b := range p.DeviceBytes {
		if b > gpus[1-d].FreeBytes { // DeviceBytes follow Devices order
			t.Errorf("device %d planned %d MiB, more than it has", p.Devices[d], b/MiB)
		}
	}
	if p.GPUFitBytes+p.CPUFitBytes != p.TotalBytes {
		t.Errorf("bytes don't add up: %d + %d != %d", p.GPUFitBytes, p.CPUFitBytes, p.TotalBytes)
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

func TestIndexerCacheIsOneKeyPerCell(t *testing.T) {
	m := synthModel(1, 1*MiB, 1*MiB, 1*MiB)
	m.KV["qwen4exp.attention.head_count_kv"] = uint32(2)
	m.KV["qwen4exp.attention.key_length"] = uint32(128)
	m.KV["qwen4exp.attention.indexer.head_count"] = uint32(64)
	m.KV["qwen4exp.attention.indexer.key_length"] = uint32(128)

	p, err := Compute(m, []GPU{{FreeBytes: 64 * GiB}}, Options{RequestedCtx: 1024, CacheType: "f16"})
	if err != nil {
		t.Fatal(err)
	}
	cache := uint64(1024 * (2*(128+128) + 128) * 2) // KV + one indexer key per cell, not 64 heads' worth
	if want := uint64(2*MiB+1*MiB) + cache; p.GPUFitBytes != want {
		t.Errorf("GPUFitBytes = %d, want %d", p.GPUFitBytes, want)
	}
}

func TestQSAComputeReserve(t *testing.T) {
	// 8 blocks x (1 GiB experts + 10 MiB shared); the indexer budget is 2048+4-1 cells
	m := synthModel(8, 1*GiB, 10*MiB, 10*MiB)
	m.KV["qwen4exp.attention.indexer.top_k"] = uint32(2048)
	m.KV["qwen4exp.attention.compress_ratios"] = []any{uint32(4), uint32(4)}
	free := overhead(true) + 3*GiB + 100*MiB
	gpu := []GPU{{FreeBytes: free}}

	// a context within the budget never runs the sparse path: same plan as without QSA
	short, err := Compute(m, gpu, Options{RequestedCtx: 2048})
	if err != nil {
		t.Fatal(err)
	}
	if short.NCPUMoE != 7 || short.UBatch != 2048 {
		t.Errorf("ctx 2048: ncmoe=%d ubatch=%d, want 7 and 2048", short.NCPUMoE, short.UBatch)
	}

	// at 64k cells the selection needs 13 B x 65536 x ubatch: 1.6 GiB at 2048
	// would cost two more blocks' experts, 0.4 GiB at 512 costs one
	long, err := Compute(m, gpu, Options{RequestedCtx: 65536, NoOpOffload: true})
	if err != nil {
		t.Fatal(err)
	}
	if long.UBatch == 2048 {
		t.Errorf("ctx 64k: ubatch 2048 chosen although it pushes more than one extra block to RAM (ncmoe=%d)", long.NCPUMoE)
	}
	if got := qsaComputeBytes(qsaCells(m, 65536), 512); got != 65536*512*13 {
		t.Errorf("qsaComputeBytes = %d", got)
	}
	if qsaCells(m, 2051) != 0 || qsaCells(m, 2052) != 2052 {
		t.Errorf("sparse threshold must be top_k + ratio - 1 < ctx")
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

func TestExtraReserveAndMarginMoveExpertsOffGPU(t *testing.T) {
	m := synthModel(8, 1*GiB, 10*MiB, 10*MiB)
	free := overhead(true) + 4*GiB + 100*MiB
	base, err := Compute(m, []GPU{{Index: 3, FreeBytes: free}}, Options{NoOpOffload: true, UBatch: 512})
	if err != nil {
		t.Fatal(err)
	}
	// a learned 1 GiB correction for this GPU costs exactly one block's experts
	corr, _ := Compute(m, []GPU{{Index: 3, FreeBytes: free}}, Options{NoOpOffload: true, UBatch: 512, ExtraReserve: map[int]int64{3: 1 * GiB}})
	if corr.NCPUMoE != base.NCPUMoE+1 {
		t.Errorf("with correction ncmoe = %d, want %d", corr.NCPUMoE, base.NCPUMoE+1)
	}
	// corrections for another GPU don't apply
	other, _ := Compute(m, []GPU{{Index: 3, FreeBytes: free}}, Options{NoOpOffload: true, UBatch: 512, ExtraReserve: map[int]int64{0: 1 * GiB}})
	if other.NCPUMoE != base.NCPUMoE {
		t.Errorf("another GPU's correction changed the plan: ncmoe %d vs %d", other.NCPUMoE, base.NCPUMoE)
	}
	// the 4% margin: a 25 GiB card keeps 1 GiB more free
	marg, _ := Compute(m, []GPU{{Index: 3, FreeBytes: free, TotalBytes: 25 * GiB}}, Options{NoOpOffload: true, UBatch: 512})
	if marg.NCPUMoE != base.NCPUMoE+1 {
		t.Errorf("with margin ncmoe = %d, want %d", marg.NCPUMoE, base.NCPUMoE+1)
	}
}

func TestHotExpertsReplaceWholeBlocks(t *testing.T) {
	// 8 blocks x 8 experts of 128 MiB; the GPU has room for 24 experts
	m := synthModel(8, 1*GiB, 10*MiB, 10*MiB)
	m.KV["qwen4exp.expert_count"] = uint32(8)
	free := overhead(true) + 3*GiB + 100*MiB
	hits := map[int][]uint64{}
	for b := 0; b < 8; b++ {
		hits[b] = []uint64{70, 5, 5, 5, 5, 5, 3, 2}
	}

	p, err := Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{NoOpOffload: true, UBatch: 512, ExpertHits: hits})
	if err != nil {
		t.Fatal(err)
	}
	if p.NCPUMoE != 8 || p.NGpuLayers != 9 {
		t.Fatalf("ncmoe=%d ngl=%d, want all experts in RAM (8) and 9 slots on GPU", p.NCPUMoE, p.NGpuLayers)
	}
	n := 0
	for b, es := range p.HotExperts {
		n += len(es)
		if es[0] != 0 {
			t.Errorf("block %d: busiest hot expert %d, want 0", b, es[0])
		}
	}
	// expert 0 of all 8 blocks (560 hits) + 16 of the 5-hit ones (80) of 800
	if n != 24 || p.HotBytes != 3*GiB || p.HotShare != 0.8 || p.WholeLayerShare != 0.375 {
		t.Errorf("hot: %d experts, %d bytes, share %.3f (whole blocks %.3f); want 24, 3 GiB, 0.8, 0.375", n, p.HotBytes, p.HotShare, p.WholeLayerShare)
	}
}

func TestHotExpertsFillSpareMemoryWhenAllExpertsAreInRAM(t *testing.T) {
	// room for the dense part and 4 experts, not for a whole block's 8
	m := synthModel(8, 1*GiB, 10*MiB, 10*MiB)
	m.KV["qwen4exp.expert_count"] = uint32(8)
	free := overhead(true) + 512*MiB + 100*MiB
	hits := map[int][]uint64{0: {1, 1, 1, 1, 1, 1, 1, 90}}
	p, err := Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{NoOpOffload: true, UBatch: 512, ExpertHits: hits})
	if err != nil {
		t.Fatal(err)
	}
	if p.NCPUMoE != 8 || p.WholeLayerShare != 0 || len(p.HotExperts[0]) != 4 || p.HotExperts[0][0] != 7 {
		t.Errorf("ncmoe=%d hot=%v whole=%.2f, want 8, block 0's 4 busiest starting with 7, 0", p.NCPUMoE, p.HotExperts, p.WholeLayerShare)
	}
}

func TestNegativeExtraReserveGivesMoreRoom(t *testing.T) {
	m := synthModel(8, 1*GiB, 10*MiB, 10*MiB)
	free := overhead(true) + 3*GiB + 100*MiB
	base, _ := Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{NoOpOffload: true, UBatch: 512})
	more, _ := Compute(m, []GPU{{Index: 0, FreeBytes: free}}, Options{NoOpOffload: true, UBatch: 512, ExtraReserve: map[int]int64{0: -1 * GiB}})
	if more.NCPUMoE != base.NCPUMoE-1 {
		t.Errorf("ncmoe %d with 1 GiB measured unused, want %d", more.NCPUMoE, base.NCPUMoE-1)
	}
}
