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

	"github.com/hgdubbe/q38fninference/internal/tuning"
)

// Detect runs `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` and
// returns one tuning.GPU per device. Returns an empty (non-nil-error) slice,
// not an error, when nvidia-smi isn't on PATH (e.g. no NVIDIA driver) so
// callers can fall back to CPU-only without special-casing.
func Detect() ([]tuning.GPU, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=name,memory.free,memory.total",
		"--format=csv,noheader,nounits",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if isNotFound(err) {
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
		fields := strings.Split(line, ",")
		if len(fields) != 3 {
			return nil, fmt.Errorf("unexpected nvidia-smi output line: %q", line)
		}
		name := strings.TrimSpace(fields[0])
		freeMiB, err := strconv.ParseUint(strings.TrimSpace(fields[1]), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parsing free memory in %q: %w", line, err)
		}
		gpus = append(gpus, tuning.GPU{
			Name:      name,
			FreeBytes: freeMiB * tuning.MiB,
		})
	}
	return gpus, nil
}

func isNotFound(err error) bool {
	_, ok := err.(*exec.Error)
	return ok // binary not found / not executable on PATH
}
