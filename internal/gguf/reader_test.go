package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// ggufBuilder writes a minimal valid GGUF v3 file for tests, without
// depending on any external gguf-writing library.
type ggufBuilder struct {
	buf     bytes.Buffer
	kv      bytes.Buffer
	nKV     uint64
	tensors bytes.Buffer
	nT      uint64
}

func newGGUFBuilder() *ggufBuilder { return &ggufBuilder{} }

func (b *ggufBuilder) writeStr(w *bytes.Buffer, s string) {
	binary.Write(w, binary.LittleEndian, uint64(len(s)))
	w.WriteString(s)
}

func (b *ggufBuilder) u32KV(key string, v uint32) {
	b.writeStr(&b.kv, key)
	binary.Write(&b.kv, binary.LittleEndian, int32(typeUint32))
	binary.Write(&b.kv, binary.LittleEndian, v)
	b.nKV++
}

func (b *ggufBuilder) strKV(key, v string) {
	b.writeStr(&b.kv, key)
	binary.Write(&b.kv, binary.LittleEndian, int32(typeString))
	b.writeStr(&b.kv, v)
	b.nKV++
}

func (b *ggufBuilder) boolArrKV(key string, v []bool) {
	b.writeStr(&b.kv, key)
	binary.Write(&b.kv, binary.LittleEndian, int32(typeArray))
	binary.Write(&b.kv, binary.LittleEndian, int32(typeBool))
	binary.Write(&b.kv, binary.LittleEndian, uint64(len(v)))
	for _, x := range v {
		if x {
			b.kv.WriteByte(1)
		} else {
			b.kv.WriteByte(0)
		}
	}
	b.nKV++
}

func (b *ggufBuilder) tensor(name string, dims []uint64, ttype uint32, offset uint64) {
	b.writeStr(&b.tensors, name)
	binary.Write(&b.tensors, binary.LittleEndian, uint32(len(dims)))
	for _, d := range dims {
		binary.Write(&b.tensors, binary.LittleEndian, d)
	}
	binary.Write(&b.tensors, binary.LittleEndian, ttype)
	binary.Write(&b.tensors, binary.LittleEndian, offset)
	b.nT++
}

func (b *ggufBuilder) bytes() []byte {
	var out bytes.Buffer
	out.WriteString(magic)
	binary.Write(&out, binary.LittleEndian, uint32(3))
	binary.Write(&out, binary.LittleEndian, b.nT)
	binary.Write(&out, binary.LittleEndian, b.nKV)
	out.Write(b.kv.Bytes())
	out.Write(b.tensors.Bytes())
	return out.Bytes()
}

func writeTemp(t *testing.T, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestReadBasicKV(t *testing.T) {
	b := newGGUFBuilder()
	b.strKV("general.architecture", "qwen4exp")
	b.u32KV("qwen4exp.block_count", 48)
	b.u32KV("qwen4exp.context_length", 262144)
	b.boolArrKV("qwen4exp.attention.recurrent_layers", []bool{true, true, true, false})
	p := writeTemp(t, b.bytes())

	m, err := Read(p)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if m.Arch() != "qwen4exp" {
		t.Errorf("Arch() = %q, want qwen4exp", m.Arch())
	}
	nl, ok := m.NLayer()
	if !ok || nl != 48 {
		t.Errorf("NLayer() = %v, %v; want 48, true", nl, ok)
	}
	ctx, ok := m.NCtxTrain()
	if !ok || ctx != 262144 {
		t.Errorf("NCtxTrain() = %v, %v; want 262144, true", ctx, ok)
	}
	rl := m.RecurrentLayers()
	if len(rl) != 4 || rl[3] != false || rl[0] != true {
		t.Errorf("RecurrentLayers() = %v", rl)
	}
}

func TestReadRejectsBadMagic(t *testing.T) {
	p := writeTemp(t, []byte("NOTG\x00\x00\x00\x00"))
	if _, err := Read(p); err == nil {
		t.Fatal("expected error for bad magic")
	}
}

func TestFullAttentionIntervalFallback(t *testing.T) {
	b := newGGUFBuilder()
	b.strKV("general.architecture", "qwen4exp")
	b.u32KV("qwen4exp.block_count", 8)
	b.u32KV("qwen4exp.full_attention_interval", 4)
	p := writeTemp(t, b.bytes())

	m, err := Read(p)
	if err != nil {
		t.Fatal(err)
	}
	rl := m.RecurrentLayers()
	want := []bool{true, true, true, false, true, true, true, false}
	if len(rl) != len(want) {
		t.Fatalf("len(RecurrentLayers()) = %d, want %d", len(rl), len(want))
	}
	for i := range want {
		if rl[i] != want[i] {
			t.Errorf("layer %d: got %v want %v", i, rl[i], want[i])
		}
	}
}

func TestReadWithTensorsAndSizeBytes(t *testing.T) {
	b := newGGUFBuilder()
	b.strKV("general.architecture", "qwen4exp")
	b.u32KV("qwen4exp.block_count", 2)
	// Q4_K tensor: 256-elem block, 144 bytes/block. 512 elements -> 2 blocks -> 288 bytes.
	b.tensor("blk.0.ffn_gate_exps.weight", []uint64{512}, typeQ4_K, 0)
	// F32 tensor: 4 bytes/elem, no blocking.
	b.tensor("blk.0.attn_norm.weight", []uint64{128}, typeF32, 0)
	p := writeTemp(t, b.bytes())

	m, err := ReadWithTensors(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Tensors) != 2 {
		t.Fatalf("len(Tensors) = %d, want 2", len(m.Tensors))
	}
	sz, ok := m.Tensors[0].SizeBytes()
	if !ok || sz != 288 {
		t.Errorf("expert tensor SizeBytes() = %d, %v; want 288, true", sz, ok)
	}
	sz2, ok := m.Tensors[1].SizeBytes()
	if !ok || sz2 != 512 {
		t.Errorf("f32 tensor SizeBytes() = %d, %v; want 512, true", sz2, ok)
	}
}

func TestFindShards(t *testing.T) {
	got, err := FindShards("/models/qwen4exp-00002-of-00004.gguf")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"/models/qwen4exp-00001-of-00004.gguf",
		"/models/qwen4exp-00002-of-00004.gguf",
		"/models/qwen4exp-00003-of-00004.gguf",
		"/models/qwen4exp-00004-of-00004.gguf",
	}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("shard %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestFindShardsNonSplit(t *testing.T) {
	got, err := FindShards("/models/model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != "/models/model.gguf" {
		t.Errorf("got %v", got)
	}
}
