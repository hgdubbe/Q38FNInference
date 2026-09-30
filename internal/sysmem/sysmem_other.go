//go:build !windows

package sysmem

import (
	"bufio"
	"io"
	"os"
	"strconv"
	"strings"
)

// Read returns the machine's physical memory (from /proc/meminfo on Linux).
func Read() Info {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return Info{}
	}
	defer f.Close()
	return parseMeminfo(f)
}

func parseMeminfo(r io.Reader) Info {
	var info Info
	var haveTotal, haveAvail bool
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 {
			continue
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			info.Total, haveTotal = kb*1024, true
		case "MemAvailable:":
			info.Available, haveAvail = kb*1024, true
		}
	}
	info.OK = haveTotal && haveAvail
	return info
}
