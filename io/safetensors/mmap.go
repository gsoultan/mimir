package safetensors

import (
	"fmt"
	"io"
	"math"
	"os"
	"sync"
)

// mapping is a file mapped into memory that the tensors of a File alias.
type mapping struct {
	mu    sync.Mutex
	unmap func() error // nil once unmapped
}

// Open parses the file at path. On Unix systems, Linux and macOS among
// them, and on Windows, Open maps the file into memory instead of reading
// it, and the tensors alias the mapping: a file larger than memory opens
// at once, and its pages are read from disk as the tensors are. Mapped
// reports it. The mapping is private: a write to a tensor's bytes, as an
// in-place update of a parameter loaded from the file makes (see nn.Load),
// copies the page and never changes the file. Where the platform cannot
// map files, or the mapping fails, Open reads the file into memory.
//
// Close releases the mapping. A file that another process truncates while
// it is mapped faults when the truncated pages are read.
func Open(path string) (*File, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	defer fd.Close()
	info, err := fd.Stat()
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	if info.Size() > math.MaxInt {
		return nil, fmt.Errorf("safetensors: %s holds %d bytes, more than can be addressed: %w", path, info.Size(), ErrFormat)
	}
	var data []byte
	var unmap func() error
	if size := int(info.Size()); size > 0 {
		// A file that cannot be mapped, as on a file system without
		// mappings, is read instead.
		data, unmap, _ = mapFile(fd, size)
	}
	if unmap == nil {
		if data, err = io.ReadAll(fd); err != nil {
			return nil, fmt.Errorf("safetensors: %w", err)
		}
	}
	f, err := Parse(data)
	if err != nil {
		if unmap != nil {
			_ = unmap()
		}
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if unmap != nil {
		f.mapping = &mapping{unmap: unmap}
	}
	return f, nil
}

// Mapped reports whether the tensors alias a mapping of the file, which
// Close releases, rather than holding memory of their own.
func (f *File) Mapped() bool { return f.mapping != nil }

// Close releases the mapping of a file that Open mapped. Afterwards the
// file has no tensors, and the tensors read from it before, along with
// every tensor built over their bytes, must not be used. Closing a file
// that is not mapped, or closing again, does nothing.
func (f *File) Close() error {
	m := f.mapping
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.unmap == nil {
		return nil
	}
	err := m.unmap()
	m.unmap, f.tensors = nil, nil
	return err
}
