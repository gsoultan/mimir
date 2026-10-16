//go:build !linux

package train_test

// residentBytes returns 0: the resident set is read on Linux only.
func residentBytes() uint64 { return 0 }
