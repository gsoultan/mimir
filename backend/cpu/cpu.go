// Package cpu implements the mimir backend for the host CPU without cgo.
//
// Kernels read their inputs through strided views, write contiguous outputs
// and split work across up to GOMAXPROCS goroutines. They are the reference
// implementation that every other backend is tested against.
//
// Float32 matrix products run on assembly micro-kernels where the CPU
// supports them: AVX-512 or AVX2 with FMA on amd64, chosen at startup, and
// NEON on arm64. Elsewhere, or when built with the purego tag, they run on
// a portable Go kernel.
package cpu

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// Backend runs kernels on the host CPU. It is safe for concurrent use.
type Backend struct {
	workers int
}

// New returns a CPU backend that uses up to GOMAXPROCS goroutines per kernel.
func New() *Backend {
	return &Backend{workers: runtime.GOMAXPROCS(0)}
}

var _ backend.Backend = (*Backend)(nil)

// Name returns "cpu".
func (*Backend) Name() string { return "cpu" }

type elem interface{ float32 | int32 }

// buffer is host memory for one dtype.
type buffer[T elem] struct {
	data []T
}

func (b *buffer[T]) DType() backend.DType { return dtypeOf[T]() }
func (b *buffer[T]) Len() int             { return len(b.data) }

func dtypeOf[T elem]() backend.DType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return backend.Float32
	default:
		return backend.Int32
	}
}

// Alloc returns a zeroed buffer of n elements. A request larger than the
// runtime can address is reported as backend.ErrOutOfMemory.
func (*Backend) Alloc(dt backend.DType, n int) (buf backend.Buffer, err error) {
	if n < 0 {
		return nil, fmt.Errorf("cpu: alloc %d elements: %w", n, backend.ErrShape)
	}
	defer func() {
		if r := recover(); r != nil {
			buf, err = nil, fmt.Errorf("cpu: alloc %d x %s: %v: %w", n, dt, r, backend.ErrOutOfMemory)
		}
	}()
	switch dt {
	case backend.Float32:
		return &buffer[float32]{data: make([]float32, n)}, nil
	case backend.Int32:
		return &buffer[int32]{data: make([]int32, n)}, nil
	default:
		return nil, fmt.Errorf("cpu: alloc %s: %w", dt, backend.ErrDType)
	}
}

// Free is a no-op: CPU buffers are reclaimed by the garbage collector once
// unreferenced.
func (*Backend) Free(backend.Buffer) {}

// Upload copies src into dst.
func (*Backend) Upload(ctx context.Context, dst backend.Buffer, src []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b, err := bytesOf(dst)
	if err != nil {
		return err
	}
	if len(b) != len(src) {
		return fmt.Errorf("cpu: upload %d bytes into %d-byte buffer: %w", len(src), len(b), backend.ErrShape)
	}
	copy(b, src)
	return nil
}

// Download copies src into dst.
func (*Backend) Download(ctx context.Context, dst []byte, src backend.Buffer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b, err := bytesOf(src)
	if err != nil {
		return err
	}
	if len(b) != len(dst) {
		return fmt.Errorf("cpu: download %d-byte buffer into %d bytes: %w", len(b), len(dst), backend.ErrShape)
	}
	copy(dst, b)
	return nil
}

// Sync returns immediately: CPU kernels finish before Run returns.
func (*Backend) Sync(ctx context.Context) error { return ctx.Err() }

func bytesOf(b backend.Buffer) ([]byte, error) {
	switch b := b.(type) {
	case *buffer[float32]:
		return asBytes(b.data), nil
	case *buffer[int32]:
		return asBytes(b.data), nil
	default:
		return nil, fmt.Errorf("cpu: foreign buffer %T: %w", b, backend.ErrDType)
	}
}

func asBytes[T elem](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(s[0])))
}

func data[T elem](b backend.Buffer) ([]T, error) {
	tb, ok := b.(*buffer[T])
	if !ok {
		return nil, fmt.Errorf("cpu: got %T, want %s buffer: %w", b, dtypeOf[T](), backend.ErrDType)
	}
	return tb.data, nil
}
