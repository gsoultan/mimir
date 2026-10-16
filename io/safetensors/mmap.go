package safetensors

import (
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sync"
)

// mapping is a file mapped into memory that the tensors of a File alias.
// It counts its holders, the File until Close and each reader that Hold
// added, and is unmapped when the last lets go.
type mapping struct {
	mu    sync.Mutex
	holds int          // the File's, until Close, and each of Hold's
	unmap func() error // nil once unmapped
}

// release lets go of one hold, and unmaps the file if it was the last.
func (m *mapping) release() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.holds--; m.holds > 0 || m.unmap == nil {
		return nil
	}
	err := m.unmap()
	m.unmap = nil
	return err
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
// The mapping lasts until Close, or until the File is garbage, and past
// that for as long as a reader that Hold added keeps it. A file that
// another process truncates while it is mapped faults when the truncated
// pages are read.
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
		m := &mapping{holds: 1, unmap: unmap}
		f.mapping = m
		f.cleanup = runtime.AddCleanup(f, func(m *mapping) { _ = m.release() }, m)
	}
	return f, nil
}

// Mapped reports whether the tensors alias a mapping of the file, which
// lasts until Close and the readers that Hold added let go of it, rather
// than holding memory of their own.
func (f *File) Mapped() bool { return f.mapping != nil }

// Hold keeps the mapping of a file that Open mapped until the returned
// function is called, even past Close, for a reader of the tensors' bytes
// that outlives the File, as a tensor built over them with
// tensor.FromBytesFunc does. The function lets go once, however often it
// is called. For a file that is not mapped, or is closed, Hold returns a
// function that does nothing: the bytes of a file that is not mapped stay
// valid as long as they are referenced, and a closed file has no tensors.
func (f *File) Hold() func() {
	m := f.mapping
	if m == nil {
		return func() {}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if f.closed || m.unmap == nil {
		return func() {}
	}
	m.holds++
	return sync.OnceFunc(func() { _ = m.release() })
}

// Close lets go of the File's hold on the mapping of a file that Open
// mapped, which unmaps it unless a reader that Hold added still keeps it.
// Afterwards the file has no tensors, and the bytes of the tensors read
// from it before must not be used except through such a reader. Closing
// a file that is not mapped, or closing again, does nothing.
func (f *File) Close() error {
	m := f.mapping
	if m == nil {
		return nil
	}
	m.mu.Lock()
	if f.closed {
		m.mu.Unlock()
		return nil
	}
	f.closed, f.tensors = true, nil
	m.mu.Unlock()
	f.cleanup.Stop()
	return m.release()
}
