// Package gpu detects NVIDIA GPUs and their free VRAM via nvidia-smi, the
// only dependency-free way to query this from outside the CUDA runtime
// itself (avoids linking cgo/CUDA headers into the launcher binary).
package gpu

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/hgdubbe/q38fninference/internal/proc"
	"github.com/hgdubbe/q38fninference/internal/tuning"
)

// Detect runs nvidia-smi and returns one tuning.GPU per device, ordered by
// nvidia-smi index (PCI bus order; llama-server is started with
// CUDA_DEVICE_ORDER=PCI_BUS_ID so its device numbering matches). Returns an
// empty slice, not an error, when nvidia-smi isn't on PATH.
func Detect() ([]tuning.GPU, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=index,memory.free,memory.total,name",
		"--format=csv,noheader,nounits",
	)
	proc.Hide(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if _, ok := err.(*exec.Error); ok {
			return nil, nil
		}
		return nil, fmt.Errorf("nvidia-smi: %w (%s)", err, strings.TrimSpace(stderr.String()))
	}

	return parseNvidiaSMI(stdout.String())
}

func parseNvidiaSMI(out string) ([]tuning.GPU, error) {
	var gpus []tuning.GPU
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// name last and SplitN, so a name containing commas stays intact
		fields := strings.SplitN(line, ",", 4)
		if len(fields) != 4 {
			return nil, fmt.Errorf("unexpected nvidia-smi output line: %q", line)
		}
		var nums [3]uint64
		for i := range nums {
			n, err := strconv.ParseUint(strings.TrimSpace(fields[i]), 10, 64)
			if err != nil {
				return nil, fmt.Errorf("parsing %q: %w", line, err)
			}
			nums[i] = n
		}
		gpus = append(gpus, tuning.GPU{
			Index:      int(nums[0]),
			FreeBytes:  nums[1] * tuning.MiB,
			TotalBytes: nums[2] * tuning.MiB,
			Name:       strings.TrimSpace(fields[3]),
		})
	}
	return gpus, nil
}
