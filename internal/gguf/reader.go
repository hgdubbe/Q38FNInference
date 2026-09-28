// Package gguf reads just the header and key-value metadata block of a GGUF
// file. It intentionally never reads tensor data, so it stays fast even on
// huge (100GB+) MoE checkpoints.
//
// Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
)

const magic = "GGUF"

// gguf_type ids (subset needed for KV values); see docs/gguf.md.
const (
	typeUint8   = 0
	typeInt8    = 1
	typeUint16  = 2
	typeInt16   = 3
	typeUint32  = 4
	typeInt32   = 5
	typeFloat32 = 6
	typeBool    = 7
	typeString  = 8
	typeArray   = 9
	typeUint64  = 10
	typeInt64   = 11
	typeFloat64 = 12
)

// Metadata holds the parsed header + key/value section of a GGUF file.
type Metadata struct {
	Version  uint32
	NTensors uint64
	KV       map[string]any
	// Tensors is nil unless ReadTensors read the tensor-info section too.
	Tensors []TensorInfo
}

// TensorInfo describes one tensor's shape/type/offset, without its data.
type TensorInfo struct {
	Name   string
	Dims   []uint64
	Type   uint32
	Offset uint64
}

type reader struct {
	r   *bufio.Reader
	err error
}

func (r *reader) read(n int) []byte {
	if r.err != nil {
		return nil
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r.r, buf); err != nil {
		r.err = fmt.Errorf("unexpected EOF reading %d bytes: %w", n, err)
		return nil
	}
	return buf
}

func (r *reader) u32() uint32 {
	b := r.read(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (r *reader) i32() int32 { return int32(r.u32()) }

func (r *reader) u64() uint64 {
	b := r.read(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (r *reader) str() string {
	n := r.u64()
	if r.err != nil || n > (1<<32) {
		if r.err == nil {
			r.err = fmt.Errorf("implausible string length %d", n)
		}
		return ""
	}
	b := r.read(int(n))
	return string(b)
}

func (r *reader) scalar(gtype uint32) any {
	switch gtype {
	case typeUint8:
		b := r.read(1)
		return uint8b(b)
	case typeInt8:
		b := r.read(1)
		return int8(b[0])
	case typeUint16:
		b := r.read(2)
		return binary.LittleEndian.Uint16(b)
	case typeInt16:
		b := r.read(2)
		return int16(binary.LittleEndian.Uint16(b))
	case typeUint32:
		return r.u32()
	case typeInt32:
		return r.i32()
	case typeFloat32:
		return math.Float32frombits(r.u32())
	case typeBool:
		b := r.read(1)
		if b == nil {
			return false
		}
		return b[0] != 0
	case typeString:
		return r.str()
	case typeUint64:
		return r.u64()
	case typeInt64:
		return int64(r.u64())
	case typeFloat64:
		return math.Float64frombits(r.u64())
	default:
		if r.err == nil {
			r.err = fmt.Errorf("unsupported gguf scalar type id %d", gtype)
		}
		return nil
	}
}

func (r *reader) value(gtype uint32) any {
	if gtype == typeArray {
		elemType := uint32(r.i32())
		n := r.u64()
		if r.err != nil {
			return nil
		}
		out := make([]any, 0, n)
		for i := uint64(0); i < n; i++ {
			out = append(out, r.value(elemType))
		}
		return out
	}
	return r.scalar(gtype)
}

// Read parses the header and KV section of a GGUF file. It does not read
// tensor info or tensor data.
func Read(path string) (*Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readFrom(f, false)
}

// ReadWithTensors parses the header, KV section, and tensor-info table
// (names/shapes/types/offsets, not tensor data). Used for sizing offload
// plans precisely.
func ReadWithTensors(path string) (*Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readFrom(f, true)
}

func readFrom(f *os.File, withTensors bool) (*Metadata, error) {
	r := &reader{r: bufio.NewReaderSize(f, 1<<20)}

	m := r.read(4)
	if r.err != nil {
		return nil, r.err
	}
	if string(m) != magic {
		return nil, fmt.Errorf("not a GGUF file (bad magic %q)", m)
	}

	version := r.u32()
	if version < 2 {
		return nil, fmt.Errorf("unsupported GGUF version %d", version)
	}

	nTensors := r.u64()
	nKV := r.u64()
	if r.err != nil {
		return nil, r.err
	}

	kv := make(map[string]any, nKV)
	for i := uint64(0); i < nKV; i++ {
		key := r.str()
		gtype := uint32(r.i32())
		kv[key] = r.value(gtype)
		if r.err != nil {
			return nil, fmt.Errorf("reading kv %q: %w", key, r.err)
		}
	}

	meta := &Metadata{Version: version, NTensors: nTensors, KV: kv}

	if withTensors {
		tensors := make([]TensorInfo, 0, nTensors)
		for i := uint64(0); i < nTensors; i++ {
			name := r.str()
			nDims := r.u32()
			dims := make([]uint64, nDims)
			for d := range dims {
				dims[d] = r.u64()
			}
			ttype := r.u32()
			offset := r.u64()
			if r.err != nil {
				return nil, fmt.Errorf("reading tensor info %q: %w", name, r.err)
			}
			tensors = append(tensors, TensorInfo{Name: name, Dims: dims, Type: ttype, Offset: offset})
		}
		meta.Tensors = tensors
	}

	return meta, nil
}

func uint8b(b []byte) uint8 {
	if b == nil {
		return 0
	}
	return b[0]
}

var shardRe = regexp.MustCompile(`^(.+)-(\d{5})-of-(\d{5})\.gguf$`)

// FindShards resolves llama.cpp's `name-00001-of-00004.gguf` split naming to
// the full list of sibling shard paths. Returns just the input path if it
// doesn't look like a split file.
func FindShards(firstShard string) ([]string, error) {
	dir := filepath.Dir(firstShard)
	base := filepath.Base(firstShard)
	m := shardRe.FindStringSubmatch(base)
	if m == nil {
		return []string{firstShard}, nil
	}
	prefix := m[1]
	total := 0
	if _, err := fmt.Sscanf(m[3], "%d", &total); err != nil {
		return []string{firstShard}, nil
	}
	shards := make([]string, 0, total)
	for i := 1; i <= total; i++ {
		shards = append(shards, filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, i, total)))
	}
	return shards, nil
}
