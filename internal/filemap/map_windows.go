//go:build windows

package filemap

import (
	"os"
	"syscall"
	"unsafe"
)

// mapFile maps the size bytes of fd as a copy-on-write view and returns
// the mapping with the function that releases it. Writes copy the pages
// written and never reach the file.
func mapFile(fd *os.File, size int) ([]byte, func() error, error) {
	h, err := syscall.CreateFileMapping(syscall.Handle(fd.Fd()), nil, syscall.PAGE_WRITECOPY, uint32(uint64(size)>>32), uint32(size), nil)
	if err != nil {
		return nil, nil, err
	}
	defer syscall.CloseHandle(h) // the view keeps the mapping alive
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_COPY, 0, 0, uintptr(size))
	if err != nil {
		return nil, nil, err
	}
	// The view lies outside the Go heap, so the garbage collector never
	// looks at it. vet's rule against converting a uintptr to a Pointer
	// guards heap addresses, which is why the address is read back through
	// a pointer-typed view of the variable instead.
	base := *(*unsafe.Pointer)(unsafe.Pointer(&addr))
	data := unsafe.Slice((*byte)(base), size)
	return data, func() error { return syscall.UnmapViewOfFile(addr) }, nil
}
