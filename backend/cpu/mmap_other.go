//go:build !unix

package cpu

import "errors"

// Windows, Plan 9 and WebAssembly have no syscall.Mmap, so every buffer
// lives on the Go heap and Free leaves it to the garbage collector.
const offHeap = false

func mapAnon(int) ([]byte, func() error, error) {
	return nil, nil, errors.ErrUnsupported
}
