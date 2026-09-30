package gguf

// GGML tensor type ids, mirroring enum ggml_type in ggml/include/ggml.h.
// Values and block/type sizes below were read directly out of
// ggml-org/llama.cpp's ggml/include/ggml.h and ggml/src/ggml-common.h
// (sizeof(block_*) compiled and printed, not hand-computed) at the commit
// this project targets — see docs/ARCHITECTURE.md.
const (
	typeF32     = 0
	typeF16     = 1
	typeQ4_0    = 2
	typeQ4_1    = 3
	typeQ5_0    = 6
	typeQ5_1    = 7
	typeQ8_0    = 8
	typeQ8_1    = 9
	typeQ2_K    = 10
	typeQ3_K    = 11
	typeQ4_K    = 12
	typeQ5_K    = 13
	typeQ6_K    = 14
	typeQ8_K    = 15
	typeIQ2_XXS = 16
	typeIQ2_XS  = 17
	typeIQ3_XXS = 18
	typeIQ1_S   = 19
	typeIQ4_NL  = 20
	typeIQ3_S   = 21
	typeIQ2_S   = 22
	typeIQ4_XS  = 23
	typeI8      = 24
	typeI16     = 25
	typeI32     = 26
	typeI64     = 27
	typeF64     = 28
	typeIQ1_M   = 29
	typeBF16    = 30
	typeTQ1_0   = 34
	typeTQ2_0   = 35
	typeMXFP4   = 39
	typeNVFP4   = 40
	typeQ1_0    = 41
	typeQ2_0    = 42
)

type typeInfo struct {
	blockSize uint64
	typeSize  uint64
}

var ggmlTypeTraits = map[uint32]typeInfo{
	typeF32:     {1, 4},
	typeF16:     {1, 2},
	typeQ4_0:    {32, 18},
	typeQ4_1:    {32, 20},
	typeQ5_0:    {32, 22},
	typeQ5_1:    {32, 24},
	typeQ8_0:    {32, 34},
	typeQ8_1:    {32, 36},
	typeQ2_K:    {256, 84},
	typeQ3_K:    {256, 110},
	typeQ4_K:    {256, 144},
	typeQ5_K:    {256, 176},
	typeQ6_K:    {256, 210},
	typeQ8_K:    {256, 292},
	typeIQ2_XXS: {256, 66},
	typeIQ2_XS:  {256, 74},
	typeIQ3_XXS: {256, 98},
	typeIQ1_S:   {256, 50},
	typeIQ4_NL:  {32, 18},
	typeIQ3_S:   {256, 110},
	typeIQ2_S:   {256, 82},
	typeIQ4_XS:  {256, 136},
	typeI8:      {1, 1},
	typeI16:     {1, 2},
	typeI32:     {1, 4},
	typeI64:     {1, 8},
	typeF64:     {1, 8},
	typeIQ1_M:   {256, 56},
	typeBF16:    {1, 2},
	typeTQ1_0:   {256, 54},
	typeTQ2_0:   {256, 66},
	typeMXFP4:   {32, 17},
	typeNVFP4:   {64, 36},
	typeQ1_0:    {128, 18},
	typeQ2_0:    {64, 18},
}

// NElements returns the number of scalar elements a tensor holds.
func (t TensorInfo) NElements() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// SizeBytes returns the on-disk byte size of a tensor given its GGUF dims
// and quantization type. Returns 0, false for an unrecognized type id.
func (t TensorInfo) SizeBytes() (uint64, bool) {
	info, ok := ggmlTypeTraits[t.Type]
	if !ok || info.blockSize == 0 {
		return 0, false
	}
	n := t.NElements()
	nBlocks := (n + info.blockSize - 1) / info.blockSize
	return nBlocks * info.typeSize, true
}
