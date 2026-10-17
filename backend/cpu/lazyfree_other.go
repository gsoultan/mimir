//go:build !linux

package cpu

// lazyFree does nothing where the syscall package cannot advise the
// kernel: cached mappings stay resident until they are unmapped.
func lazyFree([]byte) {}
