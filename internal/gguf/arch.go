package gguf

import "fmt"

// Arch returns the general.architecture GGUF key ("qwen4exp" for
// Qwen3.8-Flash-Next, per llama.cpp's src/llama-arch.cpp).
func (m *Metadata) Arch() string {
	s, _ := m.KV["general.architecture"].(string)
	return s
}

func (m *Metadata) key(suffix string) string {
	return fmt.Sprintf("%s.%s", m.Arch(), suffix)
}

func (m *Metadata) getUint(suffix string) (uint64, bool) {
	v, ok := m.KV[m.key(suffix)]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case uint8:
		return uint64(n), true
	case uint16:
		return uint64(n), true
	case uint32:
		return uint64(n), true
	case uint64:
		return n, true
	case int32:
		return uint64(n), true
	case int64:
		return uint64(n), true
	default:
		return 0, false
	}
}

// Uint reads an arch-prefixed integer key, e.g. Uint("ssm.state_size").
func (m *Metadata) Uint(suffix string) (uint64, bool) { return m.getUint(suffix) }

// HeadCountKV returns the KV head count for layer il; the key may be a
// scalar or a per-layer array (0 for recurrent layers in hybrid models).
func (m *Metadata) HeadCountKV(il int) uint64 {
	v, ok := m.KV[m.key("attention.head_count_kv")]
	if !ok {
		v, ok = m.KV[m.key("attention.head_count")]
		if !ok {
			return 0
		}
	}
	if arr, ok := v.([]any); ok {
		if il < len(arr) {
			return asUint(arr[il])
		}
		return 0
	}
	return asUint(v)
}

// NLayer is the total transformer block count (arch.block_count).
func (m *Metadata) NLayer() (uint64, bool) { return m.getUint("block_count") }

// NEmbd is the model's hidden/embedding size.
func (m *Metadata) NEmbd() (uint64, bool) { return m.getUint("embedding_length") }

// NCtxTrain is the model's native training context length.
func (m *Metadata) NCtxTrain() (uint64, bool) { return m.getUint("context_length") }

// NExpert is the MoE expert count (0 for dense models).
func (m *Metadata) NExpert() uint64 {
	n, _ := m.getUint("expert_count")
	return n
}

// NExpertUsed is the MoE active-experts-per-token count.
func (m *Metadata) NExpertUsed() uint64 {
	n, _ := m.getUint("expert_used_count")
	return n
}

// IsMoE reports whether the model routes to a subset of experts per token.
func (m *Metadata) IsMoE() bool { return m.NExpert() > 0 }

// RecurrentLayers returns a per-layer mask: true = gated-delta-net
// (linear/recurrent) layer, false = full QSA attention layer. This is the
// qwen4exp/qwen3next hybrid-attention pattern. Falls back to reconstructing
// the pattern from full_attention_interval (every Nth layer is full
// attention) when the explicit per-layer array isn't present in the file.
func (m *Metadata) RecurrentLayers() []bool {
	if v, ok := m.KV[m.key("attention.recurrent_layers")]; ok {
		if arr, ok := v.([]any); ok {
			out := make([]bool, len(arr))
			for i, x := range arr {
				out[i] = asBool(x)
			}
			return out
		}
	}

	nLayer, ok := m.NLayer()
	if !ok {
		return nil
	}
	interval, ok := m.getUint("full_attention_interval")
	if !ok || interval == 0 {
		return nil
	}
	out := make([]bool, nLayer)
	for i := range out {
		out[i] = (uint64(i)+1)%interval != 0
	}
	return out
}

// PLELayers returns the (0-based) layer indices carrying the PLE n-gram hash
// embedding table, if any.
func (m *Metadata) PLELayers() []uint64 {
	v, ok := m.KV[m.key("ple.layers")]
	if !ok {
		return nil
	}
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]uint64, 0, len(arr))
	for _, x := range arr {
		out = append(out, asUint(x))
	}
	return out
}

// HyperConnectionCount is the qwen4exp gated-residual branch count (1 means
// disabled / plain residual).
func (m *Metadata) HyperConnectionCount() uint64 {
	n, ok := m.getUint("hyper_connection.count")
	if !ok {
		return 1
	}
	return n
}

// IndexerTopK is the QSA sparse-attention token budget per query.
func (m *Metadata) IndexerTopK() (uint64, bool) { return m.getUint("attention.indexer.top_k") }

// MinCompressRatio is the smallest QSA block size (tokens pooled per indexer
// key) over the layers that have one; 0 when the model has none.
func (m *Metadata) MinCompressRatio() uint64 {
	v, ok := m.KV[m.key("attention.compress_ratios")]
	if !ok {
		return 0
	}
	arr, ok := v.([]any)
	if !ok {
		return asUint(v)
	}
	var least uint64
	for _, x := range arr {
		if r := asUint(x); r > 0 && (least == 0 || r < least) {
			least = r
		}
	}
	return least
}

// NFFExp is the per-expert feed-forward hidden size.
func (m *Metadata) NFFExp() (uint64, bool) {
	if v, ok := m.KV[m.key("expert_feed_forward_length")]; ok {
		if arr, ok := v.([]any); ok && len(arr) > 0 {
			return asUint(arr[0]), true
		}
		return asUint(v), true
	}
	return 0, false
}

func asBool(v any) bool {
	switch b := v.(type) {
	case bool:
		return b
	default:
		return asUint(v) != 0
	}
}

func asUint(v any) uint64 {
	switch n := v.(type) {
	case uint8:
		return uint64(n)
	case uint16:
		return uint64(n)
	case uint32:
		return uint64(n)
	case uint64:
		return n
	case int8:
		return uint64(n)
	case int16:
		return uint64(n)
	case int32:
		return uint64(n)
	case int64:
		return uint64(n)
	default:
		return 0
	}
}
