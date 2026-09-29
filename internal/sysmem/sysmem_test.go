//go:build !windows

package sysmem

import (
	"strings"
	"testing"
)

func TestParseMeminfo(t *testing.T) {
	info := parseMeminfo(strings.NewReader("MemTotal:       65536000 kB\nMemFree:  100 kB\nMemAvailable:   32768000 kB\n"))
	if !info.OK || info.Total != 65536000*1024 || info.Available != 32768000*1024 {
		t.Errorf("got %+v", info)
	}
	if parseMeminfo(strings.NewReader("garbage\n")).OK {
		t.Error("OK must be false without MemTotal/MemAvailable")
	}
}

func TestReadThisMachine(t *testing.T) {
	if info := Read(); !info.OK || info.Total == 0 {
		t.Errorf("Read() = %+v", info)
	}
}
