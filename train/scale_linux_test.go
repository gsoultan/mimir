//go:build linux

package train_test

import (
	"os"
	"strconv"
	"strings"
)

// residentBytes returns the process's resident set from /proc/self/statm,
// or 0 if it cannot be read.
func residentBytes() uint64 {
	b, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(b))
	if len(fields) < 2 {
		return 0
	}
	pages, err := strconv.ParseUint(fields[1], 10, 64)
	if err != nil {
		return 0
	}
	return pages * uint64(os.Getpagesize())
}
