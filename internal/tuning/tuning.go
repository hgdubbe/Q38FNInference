// Package tuning turns a parsed GGUF model (see internal/gguf) plus detected
// GPU memory into a concrete llama-server offload configuration, tailored to
// the qwen4exp (Qwen3.8-Flash-Next) hybrid MoE + gated-delta-net/QSA
// architecture, on one or several GPUs.
//
// It reproduces llama.cpp's own placement rules (src/llama-model.cpp,
// load_tensors) so the plan is exact rather than proportional guesswork:
//   - the input layer (token_embd) always stays on CPU;
//   - there are n_layer+1 offloadable slots: every block plus the output
//     layer; -ngl k puts the LAST k slots on GPUs;
//   - --tensor-split c0,c1,... assigns those GPU slots to devices in order,
//     by cumulative proportion, so passing slot counts places exactly c_d
//     consecutive slots on device d;
//   - --n-cpu-moe N pins the MoE expert tensors of blocks 0..N-1 to CPU RAM,
//     wherever the rest of the block lives.
//
// Each slot's GPU cost is its weights (minus experts if pinned) plus its
// share of the context cache: KV + indexer keys for full (QSA) attention
// blocks, conv + delta-rule state for gated-delta-net blocks.
package tuning

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"

	"github.com/hgdubbe/q38fninference/internal/gguf"
)

const (
	MiB = 1024 * 1024
	GiB = 1024 * MiB
)

// GPU is one detected accelerator. Index is the nvidia-smi / PCI-bus-order
// index, used for CUDA_VISIBLE_DEVICES.
type GPU struct {
	Index      int
	Name       string
	FreeBytes  uint64
	TotalBytes uint64
}

// Plan is the computed launch configuration and the reasoning behind it.
type Plan struct {
	NGpuLayers  int      // -ngl (slots, incl. the output layer)
	NCPUMoE     int      // --n-cpu-moe
	TensorSplit []int    // --tensor-split, slot count per device (multi-GPU only)
	Devices     []int    // GPU indices used, in llama.cpp device order
	DeviceNames []string // for display
	DeviceBytes []uint64 // estimated bytes placed on each device
	CtxSize     uint64
	Parallel    int
	// UBatch is the micro-batch size to pass (0 = llama.cpp's default, 512).
	// Raised when weights stay in RAM, see Compute.
	UBatch int
	// StagingBytes is VRAM kept free on the main GPU for llama.cpp's
	// per-batch copies of RAM-resident weights (op offload).
	StagingBytes uint64
	// NoOpOffload: the GPU is too small for that staging area, so prompt
	// processing keeps RAM-resident weights on the CPU (--no-op-offload).
	NoOpOffload bool
	FlashAttn   bool
	CacheTypeKV string
	Notes       []string
	GPUFitBytes uint64
	CPUFitBytes uint64
	TotalBytes  uint64
	// InputBytes are embedding tables (token_embd, the PLE n-gram table):
	// memory-mapped and read a few rows per token, so they needn't fit in RAM.
	InputBytes uint64
	FullyOnGPU bool
}

// Options lets the caller override what would otherwise be auto-picked.
type Options struct {
	RequestedCtx uint64 // 0 = auto (native context, capped)
	Parallel     int    // server slots; recurrent state scales with it. 0 = 1
	CacheType    string // KV cache type; "" = q8_0
	// ReserveBytes is extra safety headroom kept free on every GPU on top
	// of the modelled CUDA context and compute buffers. 0 = 512 MiB.
	ReserveBytes uint64
	// UBatch forces the micro-batch size; 0 lets the planner pick.
	UBatch int
	// NoOpOffload: llama-server runs with --no-op-offload, so RAM-resident
	// weights are never copied to the GPU and need no staging VRAM.
	NoOpOffload bool

	// qsaCells is the context size when the QSA sparse path can run (the
	// cache can outgrow the indexer budget), else 0; see qsaComputeBytes.
	qsaCells uint64
}

// llama.cpp's LLM_FFN_EXPS_REGEX (common/common.h), anchored to a block index.
var expertTensorRe = regexp.MustCompile(`^blk\.(\d+)\.ffn_(up|down|gate|gate_up)_(ch|)exps`)
var layerTensorRe = regexp.MustCompile(`^blk\.(\d+)\.`)
var inputTensorRe = regexp.MustCompile(`^(token_embd|pos_embd|token_types|per_layer_token_embd)\.`)

const (
	cudaContextBytes  = 512 * MiB // CUDA context + cuBLAS workspace, per device
	computeMainBytes  = 1 * GiB   // compute buffer on the first GPU at ubatch 512 (holds logits)
	computeOtherBytes = 256 * MiB
	defaultCtxCap     = 65536

	defaultUBatch = 512
	// With weights in RAM, llama.cpp copies each RAM-resident layer's whole
	// weight tensor to the main GPU once per micro-batch during prompt
	// processing (ggml-backend.cpp, "1.off" op offload). A larger micro-batch
	// spreads that fixed PCIe cost over more tokens.
	offloadUBatch = 2048
)

// qsaBytesPerCellToken is what the QSA top-k selection needs in the compute
// buffer per KV cell and micro-batch token once the cache outgrows the
// indexer budget: the block scores expanded to every cell (f32) plus the
// attention mask built from the selection (f16), measured with
// patches/0004 at 32k cells (825 MiB at ubatch 2048, 207 MiB at 512). Every
// GPU holding a full-attention layer needs it.
const qsaBytesPerCellToken = 13

func qsaComputeBytes(cells uint64, ub int) uint64 {
	return cells * uint64(ub) * qsaBytesPerCellToken
}

// qsaCells: the context size if the sparse path can run, i.e. top-k keeps
// fewer cells than the context holds (llama.cpp keeps top_k + ratio - 1).
func qsaCells(m *gguf.Metadata, ctx uint64) uint64 {
	topK, ok := m.IndexerTopK()
	r := m.MinCompressRatio()
	if !ok || topK == 0 || r == 0 {
		return 0
	}
	if topK+r-1 >= ctx {
		return 0
	}
	return ctx
}

// computeMain is the main GPU's compute buffer estimate for a micro-batch
// size: activations grow with it (roughly +512 MiB going 512 -> 2048).
func computeMain(ub int) uint64 {
	if ub <= defaultUBatch {
		return computeMainBytes
	}
	return computeMainBytes + uint64(ub-defaultUBatch)*512*MiB/uint64(offloadUBatch-defaultUBatch)
}

type slot struct {
	shared, expert, cache uint64
}

func (s slot) bytes(expertsOnGPU bool) uint64 {
	b := s.shared + s.cache
	if expertsOnGPU {
		b += s.expert
	}
	return b
}

// Compute builds a plan for a model read with gguf.ReadModel. gpus are the
// devices the user allowed; Plan.Devices is the order llama-server must see
// them in (via CUDA_VISIBLE_DEVICES), largest free memory first.
func Compute(m *gguf.Metadata, gpus []GPU, opt Options) (*Plan, error) {
	nLayer64, ok := m.NLayer()
	if !ok || nLayer64 == 0 {
		return nil, fmt.Errorf("model is missing %s.block_count", m.Arch())
	}
	n := int(nLayer64)
	if opt.Parallel <= 0 {
		opt.Parallel = 1
	}
	if opt.CacheType == "" {
		opt.CacheType = "q8_0"
	}
	if opt.ReserveBytes == 0 {
		opt.ReserveBytes = 512 * MiB
	}

	ctx := pickCtx(m, opt)
	slots := make([]slot, n+1) // slots[n] is the output layer
	var inputBytes uint64
	var sized, unsized int

	for _, t := range m.Tensors {
		sz, ok := t.SizeBytes()
		if !ok {
			unsized++
			continue
		}
		sized++
		if em := expertTensorRe.FindStringSubmatch(t.Name); em != nil {
			if il, err := strconv.Atoi(em[1]); err == nil && il < n {
				slots[il].expert += sz
				continue
			}
		}
		if lm := layerTensorRe.FindStringSubmatch(t.Name); lm != nil {
			if il, err := strconv.Atoi(lm[1]); err == nil && il < n {
				slots[il].shared += sz
				continue
			}
		}
		if inputTensorRe.MatchString(t.Name) {
			inputBytes += sz
			continue
		}
		slots[n].shared += sz
	}

	cacheNote := addCacheBytes(m, slots[:n], ctx, opt)
	opt.qsaCells = qsaCells(m, ctx)

	plan := &Plan{
		CtxSize:     ctx,
		Parallel:    opt.Parallel,
		CacheTypeKV: opt.CacheType,
		TotalBytes:  inputBytes,
		InputBytes:  inputBytes,
	}
	for _, s := range slots {
		plan.TotalBytes += s.bytes(true)
	}
	if cacheNote != "" {
		plan.Notes = append(plan.Notes, cacheNote)
	}
	if unsized > 0 {
		plan.Notes = append(plan.Notes, fmt.Sprintf("%d tensor(s) use a quant type this launcher doesn't know; they were left out of the sizing", unsized))
	}

	if sized == 0 || len(gpus) == 0 {
		if sized == 0 {
			plan.Notes = append(plan.Notes, "model tensor sizes unavailable; running CPU-only")
		} else {
			plan.Notes = append(plan.Notes, "no usable GPU selected; running CPU-only")
		}
		plan.CacheTypeKV = "f16"
		plan.CPUFitBytes = plan.TotalBytes
		return plan, nil
	}

	// The first device is llama.cpp's main GPU (logits, compute buffers) and
	// takes the first slots, so lead with the one that has the most room.
	gpus = slices.Clone(gpus)
	slices.SortStableFunc(gpus, func(a, b GPU) int {
		switch {
		case a.FreeBytes > b.FreeBytes:
			return -1
		case a.FreeBytes < b.FreeBytes:
			return 1
		}
		return 0
	})

	for _, g := range gpus {
		plan.Devices = append(plan.Devices, g.Index)
		plan.DeviceNames = append(plan.DeviceNames, g.Name)
	}

	ub := opt.UBatch
	if ub == 0 {
		ub = defaultUBatch
	}
	nCPUMoE, k, staging := solve(slots, gpus, opt, ub)
	if opt.UBatch == 0 && (nCPUMoE > 0 || k < n+1) {
		// weights stay in RAM: re-plan with a larger micro-batch, whose
		// bigger compute buffer may push one more block's experts to RAM.
		// With QSA at long contexts the buffer grows with context x
		// micro-batch, so stop at the largest size that costs at most that.
		for _, try := range []int{offloadUBatch, offloadUBatch / 2} {
			N2, k2, st2 := solve(slots, gpus, opt, try)
			if k2 == k && N2 <= nCPUMoE+1 {
				ub, nCPUMoE, k, staging = try, N2, k2, st2
				plan.UBatch = ub
				break
			}
		}
	} else if opt.UBatch != 0 {
		plan.UBatch = opt.UBatch
	}
	if !opt.NoOpOffload && (nCPUMoE > 0 || k < n+1) {
		// Copying RAM-resident weights to the GPU speeds up long prompts but
		// needs a staging area; on a GPU too small for it, keeping whole
		// slots on the GPU matters more, so plan without op offload then.
		noOff := opt
		noOff.NoOpOffload = true
		N2, k2, _ := solve(slots, gpus, noOff, ub)
		if k2 > k {
			opt = noOff
			nCPUMoE, k, staging = N2, k2, 0
			plan.NoOpOffload = true
			plan.Notes = append(plan.Notes, "GPU too small to also stage RAM-resident weights for prompt processing: prompts are processed on the CPU for those layers (--no-op-offload)")
		}
	}
	plan.StagingBytes = staging
	budgets := deviceBudgets(gpus, opt, ub, staging)

	counts, _ := place(slots, nCPUMoE, k, budgets)
	plan.NGpuLayers = k
	plan.NCPUMoE = nCPUMoE
	plan.FlashAttn = true
	plan.DeviceBytes = make([]uint64, len(gpus))

	first := n + 1 - k
	j := first
	for d, c := range counts {
		for i := 0; i < c; i++ {
			plan.DeviceBytes[d] += slots[j].bytes(j >= nCPUMoE)
			j++
		}
		plan.GPUFitBytes += plan.DeviceBytes[d]
	}
	plan.CPUFitBytes = plan.TotalBytes - plan.GPUFitBytes
	if len(gpus) > 1 {
		plan.TensorSplit = counts
	}
	plan.FullyOnGPU = k == n+1 && nCPUMoE == 0

	switch {
	case plan.FullyOnGPU:
		plan.Notes = append(plan.Notes, "all weights and cache fit on GPU")
	case k == n+1:
		plan.Notes = append(plan.Notes, fmt.Sprintf("MoE experts of the first %d/%d blocks stay in CPU RAM; attention, gated-delta-net and shared weights are all on GPU", nCPUMoE, n))
		if staging > 0 {
			plan.Notes = append(plan.Notes, fmt.Sprintf("%.1f GiB kept free on %s for copying RAM-resident experts to the GPU during long prompts; micro-batch %d so each copy serves more tokens",
				float64(staging)/GiB, gpus[0].Name, ub))
		}
	case k == 0:
		plan.Notes = append(plan.Notes, "the GPU(s) can't hold even one block's non-expert weights plus cache; running CPU-only")
		plan.FlashAttn = false
		plan.CacheTypeKV = "f16"
	default:
		plan.Notes = append(plan.Notes,
			fmt.Sprintf("GPU memory is too small for every block's non-expert weights: offloading the last %d of %d slots, all experts in CPU RAM", k, n+1),
			"expect slow generation: most compute runs on CPU")
	}
	if len(gpus) > 1 {
		for d, c := range counts {
			if c == 0 {
				plan.Notes = append(plan.Notes, fmt.Sprintf("%s got no layers; consider deselecting it", gpus[d].Name))
			}
		}
	}
	return plan, nil
}

// deviceBudgets is what each GPU can hold for weights and cache: free
// memory minus CUDA context, compute buffers, safety reserve and, on the
// main GPU, the op-offload staging area.
func deviceBudgets(gpus []GPU, opt Options, ub int, staging uint64) []int64 {
	budgets := make([]int64, len(gpus))
	for d, g := range gpus {
		compute := uint64(computeOtherBytes)
		if d == 0 {
			compute = computeMain(ub) + staging
		}
		compute += qsaComputeBytes(opt.qsaCells, ub)
		budgets[d] = int64(g.FreeBytes) - int64(cudaContextBytes+compute+opt.ReserveBytes)
	}
	return budgets
}

// stagingFor is the largest single block's worth of RAM-resident weights:
// llama.cpp sizes the main GPU's compute buffer for copying it over during
// prompt processing. Zero when nothing stays in RAM or op offload is off.
func stagingFor(slots []slot, nCPUMoE, k int, opt Options) uint64 {
	if opt.NoOpOffload {
		return 0
	}
	first := len(slots) - k // slots before this are wholly in RAM
	var most uint64
	for j := 0; j < len(slots)-1; j++ {
		var ram uint64
		if j < first {
			ram = slots[j].shared + slots[j].expert
		} else if j < nCPUMoE {
			ram = slots[j].expert
		}
		most = max(most, ram)
	}
	return most
}

// solve finds the placement for one micro-batch size: the fewest blocks'
// experts pinned to RAM (phase 1), or failing that the most trailing slots
// on GPU with all experts in RAM (phase 2). The staging reserve depends on
// what ends up in RAM, so each candidate is checked with its own.
func solve(slots []slot, gpus []GPU, opt Options, ub int) (nCPUMoE, k int, staging uint64) {
	n := len(slots) - 1
	fits := func(N, k int) (uint64, bool) {
		st := stagingFor(slots, N, k, opt)
		_, ok := place(slots, N, k, deviceBudgets(gpus, opt, ub, st))
		return st, ok
	}
	for N := 0; N <= n; N++ {
		if st, ok := fits(N, n+1); ok {
			return N, n + 1, st
		}
	}
	for k = n; k > 0; k-- {
		if st, ok := fits(n, k); ok {
			return n, k, st
		}
	}
	return n, 0, 0
}

// place packs the last k slots (with experts of blocks < nCPUMoE on CPU)
// into devices in order, filling each as far as it goes, the way contiguous
// --tensor-split ranges must be laid out. Greedy fill is optimal for
// feasibility of an ordered contiguous partition.
func place(slots []slot, nCPUMoE, k int, budgets []int64) ([]int, bool) {
	counts := make([]int, len(budgets))
	first := len(slots) - k
	d := 0
	var used int64
	for j := first; j < len(slots); j++ {
		need := int64(slots[j].bytes(j >= nCPUMoE))
		for d < len(budgets) && used+need > budgets[d] {
			d++
			used = 0
		}
		if d == len(budgets) {
			return nil, false
		}
		used += need
		counts[d]++
	}
	return counts, true
}

// addCacheBytes charges every block its context-cache memory, from GGUF
// metadata. Estimates are deliberately on the high side: KV compression
// ratios on QSA layers are ignored, and recurrent state is counted at f32.
func addCacheBytes(m *gguf.Metadata, blocks []slot, ctx uint64, opt Options) string {
	kvElem := cacheElemBytes(opt.CacheType)
	recr := m.RecurrentLayers()
	keyLen, _ := m.Uint("attention.key_length")
	valLen, _ := m.Uint("attention.value_length")
	// the indexer caches one pooled-from key per cell, not one per indexer head
	idxKeyLen, _ := m.Uint("attention.indexer.key_length")

	dConv, _ := m.Uint("ssm.conv_kernel")
	dInner, _ := m.Uint("ssm.inner_size")
	dState, _ := m.Uint("ssm.state_size")
	nGroup, _ := m.Uint("ssm.group_count")
	var recrPerSeq uint64
	if dInner > 0 && dState > 0 {
		var conv uint64
		if dConv > 0 {
			conv = (dConv - 1) * (dInner + 2*nGroup*dState)
		}
		recrPerSeq = (conv + dState*dInner) * 4
	}

	var missingKV bool
	for il := range blocks {
		if il < len(recr) && recr[il] {
			blocks[il].cache += recrPerSeq * uint64(opt.Parallel)
			continue
		}
		nKV := m.HeadCountKV(il)
		if nKV == 0 || keyLen == 0 {
			missingKV = true
			continue
		}
		v := valLen
		if v == 0 {
			v = keyLen
		}
		perToken := float64(nKV*(keyLen+v)) * kvElem
		perToken += float64(idxKeyLen) * kvElem
		blocks[il].cache += uint64(perToken * float64(ctx))
	}
	if missingKV {
		return "attention head metadata missing for some layers; their KV cache is not included in the plan"
	}
	return ""
}

func cacheElemBytes(t string) float64 {
	switch t {
	case "f32":
		return 4
	case "q8_0":
		return 34.0 / 32
	case "q4_0", "iq4_nl":
		return 18.0 / 32
	case "q4_1":
		return 20.0 / 32
	case "q5_0":
		return 22.0 / 32
	case "q5_1":
		return 24.0 / 32
	default: // f16, bf16
		return 2
	}
}

// pickCtx: the model's native context, capped so a first run doesn't
// reserve cache for 262k tokens nobody asked for.
func pickCtx(m *gguf.Metadata, opt Options) uint64 {
	if opt.RequestedCtx != 0 {
		return opt.RequestedCtx
	}
	trained, ok := m.NCtxTrain()
	if !ok || trained == 0 || trained > defaultCtxCap {
		return defaultCtxCap
	}
	return trained
}

// Args renders the offload part of the plan as llama-server CLI arguments.
func (p *Plan) Args(modelPath string) []string {
	args := []string{
		"--model", modelPath,
		"--ctx-size", strconv.FormatUint(p.CtxSize, 10),
		"--n-gpu-layers", strconv.Itoa(p.NGpuLayers),
		"--parallel", strconv.Itoa(max(p.Parallel, 1)),
		"--fit", "off",
	}
	if p.NCPUMoE > 0 && p.NGpuLayers > 0 {
		args = append(args, "--n-cpu-moe", strconv.Itoa(p.NCPUMoE))
	}
	if p.NoOpOffload {
		args = append(args, "--no-op-offload")
	}
	if p.UBatch > 0 {
		args = append(args, "--batch-size", strconv.Itoa(max(p.UBatch, 2048)), "--ubatch-size", strconv.Itoa(p.UBatch))
	}
	if len(p.TensorSplit) > 1 {
		s := ""
		for i, c := range p.TensorSplit {
			if i > 0 {
				s += ","
			}
			s += strconv.Itoa(c)
		}
		args = append(args, "--split-mode", "layer", "--tensor-split", s)
	}
	if p.FlashAttn {
		args = append(args, "--flash-attn", "on")
	} else {
		args = append(args, "--flash-attn", "off")
	}
	if p.CacheTypeKV != "" {
		args = append(args, "--cache-type-k", p.CacheTypeKV, "--cache-type-v", p.CacheTypeKV)
	}
	return args
}
