// Package tuning turns a parsed GGUF model (see internal/gguf) plus detected
// GPU memory into a concrete llama-server command line, tailored to the
// qwen4exp (Qwen3.8-Flash-Next) hybrid MoE + gated-delta-net/QSA-attention
// architecture.
//
// Upstream llama.cpp already ships the mechanism this needs (`--n-cpu-moe`,
// see tools/server/README.md), but decides nothing on its own: the user picks
// -ngl/-ncmoe by hand, and the generic `--fit` auto-planner (common/fit.cpp)
// does not yet account for this architecture's recurrent/SSM cache or its
// PLE n-gram hash table (see docs/ARCHITECTURE.md). This package fills that
// gap by reading the model's real per-tensor sizes and picking the minimal
// "keep first N layers' experts on CPU" split that fits the detected VRAM,
// with headroom held back for the KV cache/compute buffers.
package tuning

import (
	"fmt"
	"regexp"
	"sort"

	"github.com/hgdubbe/q38fninference/internal/gguf"
)

// MiB and GiB are byte-count constants for readability.
const (
	MiB = 1024 * 1024
	GiB = 1024 * MiB
)

// Plan is the computed launch configuration and the reasoning behind it.
type Plan struct {
	NGpuLayers  int    // -ngl
	NCPUMoE     int    // --n-cpu-moe (0 if not applicable/needed)
	CtxSize     uint64 // --ctx-size
	FlashAttn   bool   // --flash-attn
	CacheTypeKV string // --cache-type-k / --cache-type-v
	Notes       []string
	GPUFitBytes uint64 // estimated bytes placed on GPU
	CPUFitBytes uint64 // estimated bytes left on CPU
	TotalBytes  uint64
	FullyOnGPU  bool
}

// GPU describes one detected accelerator's memory, in bytes.
type GPU struct {
	Name      string
	FreeBytes uint64
}

// Options lets the caller override what would otherwise be auto-picked.
type Options struct {
	// RequestedCtx, if non-zero, is used verbatim instead of the auto default.
	RequestedCtx uint64
	// ReserveBytes is headroom subtracted from free VRAM for the KV cache,
	// compute buffers and CUDA context overhead. Zero means "pick a sane
	// default based on context size".
	ReserveBytes uint64
}

var expertTensorRe = regexp.MustCompile(`^blk\.(\d+)\.ffn_(gate|up|down)_exps(\.weight)?$`)
var layerTensorRe = regexp.MustCompile(`^blk\.(\d+)\.`)

type layerBytes struct {
	expert uint64
	shared uint64
}

// Plan computes a launch plan for a model whose tensor list has already been
// read (gguf.ReadWithTensors). gpus should be sorted GPU-0-first; only the
// first (largest-priority) device is used for now — see docs/ARCHITECTURE.md
// TODO for multi-GPU tensor-split support.
func Compute(m *gguf.Metadata, gpus []GPU, opt Options) (*Plan, error) {
	nLayer, ok := m.NLayer()
	if !ok || nLayer == 0 {
		return nil, fmt.Errorf("model is missing %s.block_count", m.Arch())
	}

	layers := make([]layerBytes, nLayer)
	var globalBytes uint64
	var sizedTensors, unsizedTensors int

	for _, t := range m.Tensors {
		sz, ok := t.SizeBytes()
		if !ok {
			unsizedTensors++
			continue
		}
		sizedTensors++

		if em := expertTensorRe.FindStringSubmatch(t.Name); em != nil {
			il := mustAtoi(em[1])
			if il < len(layers) {
				layers[il].expert += sz
			} else {
				globalBytes += sz
			}
			continue
		}
		if lm := layerTensorRe.FindStringSubmatch(t.Name); lm != nil {
			il := mustAtoi(lm[1])
			if il < len(layers) {
				layers[il].shared += sz
			} else {
				globalBytes += sz
			}
			continue
		}
		globalBytes += sz
	}

	var totalBytes uint64 = globalBytes
	for _, l := range layers {
		totalBytes += l.expert + l.shared
	}

	plan := &Plan{TotalBytes: totalBytes}
	if sizedTensors == 0 {
		plan.Notes = append(plan.Notes, "model has no tensor-info section loaded (call gguf.ReadWithTensors); falling back to CPU-only defaults")
		plan.NGpuLayers = 0
		plan.NCPUMoE = 0
		plan.CtxSize = pickCtx(m, opt)
		plan.FlashAttn = true
		plan.CacheTypeKV = "q8_0"
		return plan, nil
	}
	if unsizedTensors > 0 {
		plan.Notes = append(plan.Notes, fmt.Sprintf("%d tensor(s) used an unrecognized quant type and were excluded from sizing; the plan below may be slightly optimistic", unsizedTensors))
	}

	var freeBytes uint64
	var gpuName string
	if len(gpus) > 0 {
		freeBytes = gpus[0].FreeBytes
		gpuName = gpus[0].Name
	}

	if freeBytes == 0 {
		plan.Notes = append(plan.Notes, "no GPU with usable free memory detected; running CPU-only")
		plan.NGpuLayers = 0
		plan.NCPUMoE = 0
		plan.CtxSize = pickCtx(m, opt)
		plan.FlashAttn = false
		plan.CacheTypeKV = "f16"
		plan.CPUFitBytes = totalBytes
		return plan, nil
	}

	ctx := pickCtx(m, opt)
	reserve := opt.ReserveBytes
	if reserve == 0 {
		reserve = reserveForCtx(ctx)
	}
	budget := int64(freeBytes) - int64(reserve)
	if budget < 0 {
		budget = 0
	}

	// prefixShared[i] = sum of shared/attn bytes for layers [0, i)
	// suffixExpert[i] = sum of expert bytes for layers [i, n) -- what stays
	// on GPU if the first i layers' experts are pinned to CPU (n-cpu-moe=i).
	prefixShared := make([]uint64, nLayer+1)
	suffixExpert := make([]uint64, nLayer+1)
	for i := uint64(0); i < nLayer; i++ {
		prefixShared[i+1] = prefixShared[i] + layers[i].shared
	}
	for i := int(nLayer) - 1; i >= 0; i-- {
		suffixExpert[i] = suffixExpert[i+1] + layers[i].expert
	}

	fixedAllLayers := int64(globalBytes) + int64(prefixShared[nLayer])

	if fixedAllLayers <= budget {
		// every layer offloads to GPU; find the minimal n-cpu-moe (fewest
		// experts pinned to CPU) that still fits.
		n := sort.Search(int(nLayer)+1, func(n int) bool {
			return fixedAllLayers+int64(suffixExpert[n]) <= budget
		})
		plan.NGpuLayers = int(nLayer)
		plan.NCPUMoE = n
		plan.FullyOnGPU = n == 0
		plan.GPUFitBytes = uint64(fixedAllLayers) + suffixExpert[n]
		plan.CPUFitBytes = totalBytes - plan.GPUFitBytes
		if n == 0 {
			plan.Notes = append(plan.Notes, fmt.Sprintf("model fits entirely on %s with headroom to spare", gpuName))
		} else {
			plan.Notes = append(plan.Notes, fmt.Sprintf("keeping MoE expert weights of the first %d/%d layers on CPU RAM to fit %s (all attention/GDN/shared weights stay on GPU)", n, nLayer, gpuName))
		}
	} else {
		// Even with every expert on CPU, the dense per-layer weights alone
		// don't fit: fall back to partial layer offload (classic -ngl < n_layer).
		k := sort.Search(int(nLayer)+1, func(k int) bool {
			return int64(globalBytes)+int64(prefixShared[k]) > budget
		})
		if k > 0 {
			k--
		}
		plan.NGpuLayers = k
		plan.NCPUMoE = 0
		plan.GPUFitBytes = globalBytes + prefixShared[k]
		plan.CPUFitBytes = totalBytes - plan.GPUFitBytes
		plan.Notes = append(plan.Notes,
			fmt.Sprintf("%s is too small to hold even the non-expert weights of all %d layers; offloading only %d/%d layers to GPU", gpuName, nLayer, k, nLayer),
			"expect noticeably slower generation: most of this model's compute is happening on CPU",
		)
	}

	plan.CtxSize = ctx
	plan.FlashAttn = true
	plan.CacheTypeKV = "q8_0"
	return plan, nil
}

// pickCtx picks a default context size when the caller didn't request one:
// the model's native training context, capped to a testing-friendly size so
// a first run doesn't reserve hundreds of MiB of KV cache nobody asked for.
func pickCtx(m *gguf.Metadata, opt Options) uint64 {
	if opt.RequestedCtx != 0 {
		return opt.RequestedCtx
	}
	const defaultCap = 65536
	trained, ok := m.NCtxTrain()
	if !ok || trained == 0 {
		return defaultCap
	}
	if trained < defaultCap {
		return trained
	}
	return defaultCap
}

// reserveForCtx estimates headroom for KV cache + compute buffers. This is
// deliberately conservative (see docs/ARCHITECTURE.md: llama.cpp's own
// --fit estimator doesn't yet model this arch's recurrent-state/PLE memory,
// so we'd rather leave VRAM on the table than OOM mid-load).
func reserveForCtx(ctx uint64) uint64 {
	base := uint64(1) * GiB
	// rough per-token compute-buffer scaling; generous on purpose.
	perToken := uint64(64 * 1024) // 64 KiB/token
	return base + ctx*perToken
}

// Args renders the plan as llama-server CLI arguments.
func (p *Plan) Args(modelPath string) []string {
	args := []string{
		"--model", modelPath,
		"--ctx-size", fmt.Sprint(p.CtxSize),
		"--n-gpu-layers", fmt.Sprint(p.NGpuLayers),
	}
	if p.NCPUMoE > 0 {
		args = append(args, "--n-cpu-moe", fmt.Sprint(p.NCPUMoE))
	}
	if p.FlashAttn {
		args = append(args, "--flash-attn", "on")
	}
	if p.CacheTypeKV != "" {
		args = append(args, "--cache-type-k", p.CacheTypeKV, "--cache-type-v", p.CacheTypeKV)
	}
	return args
}

func mustAtoi(s string) int {
	n := 0
	for _, c := range s {
		n = n*10 + int(c-'0')
	}
	return n
}
