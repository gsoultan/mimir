//go:build unix

package safetensors

import (
	"os"
	"syscall"
)

// mapFile maps the size bytes of fd privately, readable and writable, and
// returns the mapping with the function that releases it. Writes copy the
// pages written and never reach the file.
func mapFile(fd *os.File, size int) ([]byte, func() error, error) {
	data, err := syscall.Mmap(int(fd.Fd()), 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
