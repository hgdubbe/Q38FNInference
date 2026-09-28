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
	FlashAttn   bool
	CacheTypeKV string
	Notes       []string
	GPUFitBytes uint64
	CPUFitBytes uint64
	TotalBytes  uint64
	FullyOnGPU  bool
}

// Options lets the caller override what would otherwise be auto-picked.
type Options struct {
	RequestedCtx uint64 // 0 = auto (native context, capped)
	Parallel     int    // server slots; recurrent state scales with it. 0 = 1
	CacheType    string // KV cache type; "" = q8_0
	// ReserveBytes is extra safety headroom kept free on every GPU on top
	// of the modelled CUDA context and compute buffers. 0 = 512 MiB.
	ReserveBytes uint64
}

// llama.cpp's LLM_FFN_EXPS_REGEX (common/common.h), anchored to a block index.
var expertTensorRe = regexp.MustCompile(`^blk\.(\d+)\.ffn_(up|down|gate|gate_up)_(ch|)exps`)
var layerTensorRe = regexp.MustCompile(`^blk\.(\d+)\.`)
var inputTensorRe = regexp.MustCompile(`^(token_embd|pos_embd|token_types|per_layer_token_embd)\.`)

const (
	cudaContextBytes  = 512 * MiB // CUDA context + cuBLAS workspace, per device
	computeMainBytes  = 1 * GiB   // compute buffer on the first GPU (holds logits)
	computeOtherBytes = 256 * MiB
	defaultCtxCap     = 65536
)

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

// Compute builds a plan for a model read with gguf.ReadWithTensors. gpus
// are the devices the user allowed, in the order llama.cpp will see them.
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

	plan := &Plan{
		CtxSize:     ctx,
		Parallel:    opt.Parallel,
		CacheTypeKV: opt.CacheType,
		TotalBytes:  inputBytes,
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

	budgets := make([]int64, len(gpus))
	for d, g := range gpus {
		compute := uint64(computeOtherBytes)
		if d == 0 {
			compute = computeMainBytes
		}
		budgets[d] = int64(g.FreeBytes) - int64(cudaContextBytes+compute+opt.ReserveBytes)
		plan.Devices = append(plan.Devices, g.Index)
		plan.DeviceNames = append(plan.DeviceNames, g.Name)
	}

	// Phase 1: every slot on GPU; pin as few blocks' experts to CPU as needed.
	nCPUMoE, k := -1, n+1
	for N := 0; N <= n; N++ {
		if _, ok := place(slots, N, k, budgets); ok {
			nCPUMoE = N
			break
		}
	}
	// Phase 2: even all experts on CPU don't fit; offload as many trailing
	// slots as possible (experts stay pinned, since they can't fit anyway).
	if nCPUMoE < 0 {
		nCPUMoE = n
		for k = n; k > 0; k-- {
			if _, ok := place(slots, nCPUMoE, k, budgets); ok {
				break
			}
		}
	}

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
	idxHeads, _ := m.Uint("attention.indexer.head_count")
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
		perToken += float64(idxHeads*idxKeyLen) * kvElem
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
