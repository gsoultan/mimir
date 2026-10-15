// Package metal implements the mimir backend for Apple GPUs through
// Metal, without cgo.
//
// It calls the Objective-C runtime and Metal through purego and runs every
// kernel as Metal Shading Language source that it generates and compiles
// at runtime, once per kernel structure. Buffers live in memory the CPU
// and GPU share, so uploads and downloads are copies within RAM.
//
// The backend is available on Apple silicon Macs. Elsewhere New reports
// ErrUnavailable.
package metal

import (
	"errors"
	"fmt"
	"strings"
)

// ErrUnavailable reports that no Metal device can be used.
var ErrUnavailable = errors.New("metal unavailable")

// numberedSource returns src with line numbers, for compiler errors.
func numberedSource(src string) string {
	var b strings.Builder
	for i, l := range strings.Split(src, "\n") {
		fmt.Fprintf(&b, "%4d %s\n", i+1, l)
	}
	return b.String()
}
