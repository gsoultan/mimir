//go:build unix

package cpu

import "syscall"

// offHeap reports whether Alloc can map large buffers outside the Go heap.
const offHeap = true

// mapAnon returns n zeroed bytes in a private anonymous mapping and the
// function that returns them to the operating system.
func mapAnon(n int) ([]byte, func() error, error) {
	mem, err := syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return nil, nil, err
	}
	return mem, func() error { return syscall.Munmap(mem) }, nil
}
