// Package filemap maps the files that model readers read in place, such
// as safetensors and GGUF files, and counts the readers of each mapping.
package filemap

import (
	"errors"
	"io"
	"math"
	"os"
	"sync"
)

// ErrTooLarge reports a file larger than the process can address.
var ErrTooLarge = errors.New("file larger than can be addressed")

// Read returns the contents of the file at path. On Unix systems and on
// Windows it maps the file privately, readable and writable, and returns
// the Mapping that owns the memory: a write copies the page written and
// never reaches the file. Where the platform cannot map files, or the
// mapping fails, it reads the file into memory and returns a nil Mapping.
func Read(path string) ([]byte, *Mapping, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer fd.Close()
	info, err := fd.Stat()
	if err != nil {
		return nil, nil, err
	}
	if info.Size() > math.MaxInt {
		return nil, nil, ErrTooLarge
	}
	if size := int(info.Size()); size > 0 {
		// A file that cannot be mapped, as on a file system without
		// mappings, is read instead.
		if data, unmap, err := mapFile(fd, size); err == nil {
			return data, &Mapping{holds: 1, unmap: unmap}, nil
		}
	}
	data, err := io.ReadAll(fd)
	if err != nil {
		return nil, nil, err
	}
	return data, nil, nil
}

// Mapping is a mapped file. It counts its holders, the owner that Read
// returned it to, until Close, and each reader that Hold added, and is
// unmapped when the last lets go.
type Mapping struct {
	mu     sync.Mutex
	holds  int
	unmap  func() error // nil once unmapped
	closed bool         // the owner let go
}

// Hold keeps the mapping until the returned function is called, for a
// reader of its bytes that outlives the owner. The function lets go once,
// however often it is called. Once Close has run, Hold adds no hold and
// returns a function that does nothing.
func (m *Mapping) Hold() func() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed || m.unmap == nil {
		return func() {}
	}
	m.holds++
	return sync.OnceFunc(func() { _ = m.release() })
}

// Close lets go of the owner's hold, which unmaps the file unless a
// reader that Hold added still keeps it. Closing again does nothing.
func (m *Mapping) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	m.mu.Unlock()
	return m.release()
}

// release lets go of one hold, and unmaps the file if it was the last.
func (m *Mapping) release() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.holds--; m.holds > 0 || m.unmap == nil {
		return nil
	}
	err := m.unmap()
	m.unmap = nil
	return err
}
