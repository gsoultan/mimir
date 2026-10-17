package safetensors

import (
	"errors"
	"fmt"
	"runtime"

	"github.com/gsoultan/mimir/internal/filemap"
)

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
	data, m, err := filemap.Read(path)
	if errors.Is(err, filemap.ErrTooLarge) {
		return nil, fmt.Errorf("safetensors: %s: %w: %w", path, err, ErrFormat)
	}
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	f, err := Parse(data)
	if err != nil {
		if m != nil {
			_ = m.Close()
		}
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if m != nil {
		f.mapping = m
		f.cleanup = runtime.AddCleanup(f, func(m *filemap.Mapping) { _ = m.Close() }, m)
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
	if f.mapping == nil {
		return func() {}
	}
	return f.mapping.Hold()
}

// Close lets go of the File's hold on the mapping of a file that Open
// mapped, which unmaps it unless a reader that Hold added still keeps it.
// Afterwards the file has no tensors, and the bytes of the tensors read
// from it before must not be used except through such a reader. Closing
// a file that is not mapped, or closing again, does nothing.
func (f *File) Close() error {
	if f.mapping == nil {
		return nil
	}
	f.cleanup.Stop()
	f.tensors = nil
	return f.mapping.Close()
}
