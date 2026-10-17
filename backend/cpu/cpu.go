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
//
// Buffers of a megabyte or more live outside the Go heap, in anonymous
// mappings that Free returns at once to a bounded cache, from which
// Alloc reuses them before it maps new memory (see mapcache.go). A
// mapped buffer that becomes garbage instead, as a dropped tensor's
// does, goes back by a cleanup after the next collection, and since such
// buffers barely grow the Go heap, Alloc runs a collection itself once
// enough bytes have been allocated off the heap since the last (see
// Alloc). Smaller buffers live on the heap. Wrap and WrapFunc make a buffer over memory the caller
// owns, such as a mapped file. Where the syscall package cannot map
// memory, on Windows, Plan 9 and WebAssembly, every buffer Alloc makes
// lives on the heap.
package cpu

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// Backend runs kernels on the host CPU. It is safe for concurrent use.
type Backend struct {
	workers int
	// team runs the chunks of parallel loops, or is nil for the
	// process-wide team (see team.go). A Stream has a team of its own.
	team *team
}

// New returns a CPU backend that uses up to GOMAXPROCS goroutines per kernel.
func New() *Backend {
	return &Backend{workers: runtime.GOMAXPROCS(0)}
}

var _ backend.Backend = (*Backend)(nil)

// Name returns "cpu".
func (*Backend) Name() string { return "cpu" }

// elem is the element types of buffers. Kernels compute in the number
// types; the half types bf16 and f16 are converted as they are read and
// rounded as they are written.
type elem interface{ float32 | int32 | bf16 | f16 }

// number is the element types kernels compute in.
type number interface{ float32 | int32 }

// buffer is host memory for one dtype.
type buffer[T elem] struct {
	data []T
	// free returns data to the operating system, or to the owner that
	// WrapFunc names. It is nil for a heap buffer, which the garbage
	// collector reclaims, and for one of Wrap, whose memory the caller
	// owns.
	free func()
}

func (b *buffer[T]) DType() backend.DType { return dtypeOf[T]() }
func (b *buffer[T]) Len() int             { return len(b.data) }

func dtypeOf[T elem]() backend.DType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return backend.Float32
	case int32:
		return backend.Int32
	case bf16:
		return backend.BFloat16
	default:
		return backend.Float16
	}
}

// offHeapMin is the size in bytes from which Alloc takes a buffer from an
// anonymous mapping instead of the Go heap, where the platform has them.
const offHeapMin = 1 << 20

// offHeapBytes counts the bytes of buffers in anonymous mappings.
var offHeapBytes atomic.Int64

// OffHeapBytes returns the number of bytes held by buffers outside the Go
// heap: those Alloc mapped that are not freed yet.
func OffHeapBytes() int64 { return offHeapBytes.Load() }

// gcPaceMin is the fewest bytes Alloc allocates off the heap between the
// collections it runs.
const gcPaceMin = 256 << 20

var (
	mappedSinceGC atomic.Int64 // bytes allocated off the heap since Alloc last ran a collection
	collecting    atomic.Bool  // a collection that Alloc runs is under way
)

// paceGC counts size more bytes allocated off the heap and runs a
// collection once the count since the last reaches the larger of
// gcPaceMin and half the bytes off the heap now, as GOGC=50 would for
// memory the collector cannot see. The collection finds the mapped
// buffers that became garbage since, whose cleanups give their mappings
// back. Without it they would wait for a collection that the heap paces,
// which their few heap bytes may not bring for a long time, and mapped
// memory would grow without bound. One goroutine runs it while the
// others go on.
//
// The cleanups that give the mappings back run after the collection, on
// goroutines of their own, and where they fall behind, the bytes off the
// heap grow, which puts the next collection off further. So the
// goroutine that ran the collection waits, up to cleanupWait, for the
// cleanup of an object the collection found, which runs among theirs.
func paceGC(size int) {
	if mappedSinceGC.Add(int64(size)) < max(gcPaceMin, offHeapBytes.Load()/2) || !collecting.CompareAndSwap(false, true) {
		return
	}
	defer collecting.Store(false)
	mappedSinceGC.Store(0)
	done := make(chan struct{})
	runtime.AddCleanup(new(sentinel), func(done chan struct{}) { close(done) }, done)
	runtime.GC()
	wait := time.NewTimer(cleanupWait)
	defer wait.Stop()
	select {
	case <-done:
	case <-wait.C:
	}
}

// cleanupWait is the longest a collection that Alloc runs waits for the
// cleanups the collection queued.
const cleanupWait = time.Second

// Alloc returns a zeroed buffer of n elements. A buffer of offHeapMin bytes
// or more comes from an anonymous mapping on platforms that have them,
// one that a freed buffer left if there is one of its size class, and a
// smaller one from the Go heap. Each time the bytes allocated off the
// heap since its last collection reach the larger of 256 MiB and half of
// OffHeapBytes, Alloc runs a garbage collection, so that the mappings of
// buffers that became garbage are reused or unmapped. A request larger
// than the runtime can address, or than the operating system can map, is
// reported as backend.ErrOutOfMemory.
func (*Backend) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	if n < 0 {
		return nil, fmt.Errorf("cpu: alloc %d elements: %w", n, backend.ErrShape)
	}
	if size := dt.Size(); size > 0 && n > math.MaxInt/size {
		return nil, fmt.Errorf("cpu: alloc %d x %s: %w", n, dt, backend.ErrOutOfMemory)
	}
	switch dt {
	case backend.Q8_0, backend.Q4_K, backend.Q6_K:
		return allocBlocks(dt, n)
	case backend.Float32:
		return alloc[float32](n)
	case backend.Int32:
		return alloc[int32](n)
	case backend.BFloat16:
		return alloc[bf16](n)
	case backend.Float16:
		return alloc[f16](n)
	default:
		return nil, fmt.Errorf("cpu: alloc %s: %w", dt, backend.ErrDType)
	}
}

// alloc returns a zeroed buffer of n elements of T, in an anonymous
// mapping when it is at least offHeapMin bytes and the platform maps
// memory, and on the heap otherwise.
func alloc[T elem](n int) (buf backend.Buffer, err error) {
	var zero T
	size := n * int(unsafe.Sizeof(zero))
	if offHeap && size >= offHeapMin {
		return mapBuffer[T](n, size)
	}
	defer func() {
		if r := recover(); r != nil {
			buf, err = nil, fmt.Errorf("cpu: alloc %d x %s: %v: %w", n, dtypeOf[T](), r, backend.ErrOutOfMemory)
		}
	}()
	return &buffer[T]{data: make([]T, n)}, nil
}

// mapBuffer returns a buffer of n elements of T, size bytes in all, in
// an anonymous mapping of its size class, cached or new, with the bytes
// of the buffer zeroed. Free gives the mapping back to the cache, and so
// does the garbage collector if the buffer becomes unreachable first, as
// a tensor's does when the tensor is dropped.
func mapBuffer[T elem](n, size int) (backend.Buffer, error) {
	mem, unmap, reused, err := mappings.take(mapClass(size))
	if err != nil {
		return nil, fmt.Errorf("cpu: alloc %d x %s: %v: %w", n, dtypeOf[T](), err, backend.ErrOutOfMemory)
	}
	if reused {
		clear(mem[:size])
	}
	offHeapBytes.Add(int64(size))
	paceGC(size)
	release := func() {
		offHeapBytes.Add(-int64(size))
		mappings.put(mem, unmap)
	}
	b := &buffer[T]{data: unsafe.Slice((*T)(unsafe.Pointer(&mem[0])), n)}
	cleanup := runtime.AddCleanup(b, func(release func()) { release() }, release)
	b.free = func() {
		cleanup.Stop()
		release()
	}
	return b, nil
}

// Free gives a buffer's mapping back for reuse if Alloc mapped it (see
// mapcache.go), runs the release of one that WrapFunc made, and does nothing
// otherwise: a heap buffer is reclaimed by the garbage collector, and the
// memory of one that Wrap made is the caller's. The buffer
// must not be used afterwards. The scheduler's pool frees a buffer only
// after no schedule has taken it for many schedules, and releasing a
// node's memory makes the node stale, so a use after Free is a bug in the
// scheduler, not something user code can cause.
func (*Backend) Free(b backend.Buffer) {
	if r, ok := b.(interface{ release() }); ok {
		r.release()
	}
}

// release frees the buffer's memory, once, and empties the buffer so that
// a use afterwards fails a bounds check instead of touching freed memory.
func (b *buffer[T]) release() {
	if b.free == nil {
		return
	}
	b.free()
	b.free, b.data = nil, nil
}

// ErrUnaligned reports bytes that Wrap cannot read in place because they
// do not start at a multiple of the element size.
var ErrUnaligned = errors.New("cpu: unaligned bytes")

// Wrap returns a buffer that reads and writes data in place, as elements
// of dt in the host's byte order. data must hold a whole number of
// elements, or Wrap fails with backend.ErrShape, and must start at an
// address that is a multiple of the element size, or it fails with
// ErrUnaligned. A buffer of a block-quantized dtype holds whole blocks in
// ggml's little-endian layout at any address. The caller keeps data valid for as long as the buffer is
// in use. data may be a mapped file: a kernel that writes the buffer, as
// an in-place update does, then writes the mapping. Free does nothing to
// a wrapped buffer.
func Wrap(dt backend.DType, data []byte) (backend.Buffer, error) {
	return WrapFunc(dt, data, nil)
}

// WrapFunc is Wrap for memory that the buffer holds until it is done
// with it, such as a mapping whose owner counts its readers: release runs
// once, when Free frees the buffer or after the garbage collector finds
// it unreachable, whichever comes first. A nil release makes it Wrap.
// When WrapFunc fails, it does not run release.
func WrapFunc(dt backend.DType, data []byte, release func()) (backend.Buffer, error) {
	if dt.IsQuantized() {
		return wrapBlocks(dt, data, release)
	}
	size := dt.Size()
	if size == 0 {
		return nil, fmt.Errorf("cpu: wrap %s: %w", dt, backend.ErrDType)
	}
	if len(data)%size != 0 {
		return nil, fmt.Errorf("cpu: wrap %d bytes as %s: %w", len(data), dt, backend.ErrShape)
	}
	if len(data) > 0 && uintptr(unsafe.Pointer(&data[0]))%uintptr(size) != 0 {
		return nil, fmt.Errorf("cpu: wrap %s at %p: %w", dt, &data[0], ErrUnaligned)
	}
	switch dt {
	case backend.Float32:
		return wrapBuffer[float32](data, release), nil
	case backend.Int32:
		return wrapBuffer[int32](data, release), nil
	case backend.BFloat16:
		return wrapBuffer[bf16](data, release), nil
	case backend.Float16:
		return wrapBuffer[f16](data, release), nil
	default:
		return nil, fmt.Errorf("cpu: wrap %s: %w", dt, backend.ErrDType)
	}
}

// wrapBuffer returns a buffer over data, which is aligned and holds a
// whole number of elements, that runs release, if any, once Free frees it
// or it is garbage. Run, Upload and Download keep their buffers reachable
// until they return, as they do for a mapped one, so release never runs
// while a kernel reads data.
func wrapBuffer[T elem](data []byte, release func()) backend.Buffer {
	b := &buffer[T]{data: wrapSlice[T](data)}
	if release != nil {
		cleanup := runtime.AddCleanup(b, func(release func()) { release() }, release)
		b.free = func() {
			cleanup.Stop()
			release()
		}
	}
	return b
}

// wrapSlice reads data, which is aligned and holds a whole number of
// elements, as a []T.
func wrapSlice[T elem](data []byte) []T {
	if len(data) == 0 {
		return nil
	}
	var zero T
	return unsafe.Slice((*T)(unsafe.Pointer(&data[0])), len(data)/int(unsafe.Sizeof(zero)))
}

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
	runtime.KeepAlive(dst) // see Run
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
	runtime.KeepAlive(src) // see Run
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
	case *buffer[bf16]:
		return asBytes(b.data), nil
	case *buffer[f16]:
		return asBytes(b.data), nil
	case *blocks:
		return b.data, nil
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
