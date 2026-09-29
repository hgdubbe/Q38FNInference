package httpapi

import (
	"testing"

	"github.com/hgdubbe/q38fninference/internal/tuning"
)

const mib = 1 << 20

func TestCudaBufferBytes(t *testing.T) {
	lines := []string{
		"load_tensors:   CPU_Mapped model buffer size = 40000.00 MiB",
		"load_tensors:    CUDA_Host model buffer size =  1234.00 MiB",
		"load_tensors:        CUDA0 model buffer size = 10000.50 MiB",
		"load_tensors:        CUDA1 model buffer size =  5000.00 MiB",
		"llama_kv_cache:      CUDA0 KV buffer size =   500.00 MiB",
		"llama_kv_cache:      CUDA0 KV buffer size =    20.00 MiB", // indexer cache
		"llama_memory_recurrent:      CUDA1 RS buffer size =    75.00 MiB",
		"sched_reserve:      CUDA0 compute buffer size =  1500.00 MiB",
		"sched_reserve:  CUDA_Host compute buffer size =    80.00 MiB",
	}
	got := cudaBufferBytes(lines)
	if want := uint64((10000.5 + 500 + 20 + 1500) * mib); got[0] != want {
		t.Errorf("CUDA0 = %d, want %d", got[0], want)
	}
	if want := uint64(5075 * mib); got[1] != want {
		t.Errorf("CUDA1 = %d, want %d", got[1], want)
	}
	if len(got) != 2 {
		t.Errorf("devices = %v, want only CUDA0 and CUDA1 (no host buffers)", got)
	}
}

func TestVRAMOverflow(t *testing.T) {
	// CUDA ordinal 0 is nvidia-smi GPU 1 (plan order), ordinal 1 is GPU 0
	li := launchInfo{devices: []tuning.GPU{
		{Index: 1, FreeBytes: 15000 * mib},
		{Index: 0, FreeBytes: 7000 * mib},
	}}
	used := map[int]uint64{0: 15000 * mib, 1: 6000 * mib}
	over := vramOverflow(li, used)
	// 15000 + 512 context - 15000 free = 512 MiB over on GPU 1; GPU 0 fits
	if over[1] != 512*mib || len(over) != 1 {
		t.Errorf("overflow = %v, want 512 MiB on GPU 1 only", over)
	}
}

func TestLoadLogsStartsAtLastLaunch(t *testing.T) {
	lines := []string{"old CUDA0 model buffer size = 1.00 MiB", "[launcher] starting llama-server", "a", "[launcher] starting llama-server", "b"}
	if got := loadLogs(lines); len(got) != 1 || got[0] != "b" {
		t.Errorf("loadLogs = %v, want [b]", got)
	}
}
