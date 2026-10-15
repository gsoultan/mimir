//go:build darwin && arm64

package metal

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// Backend runs kernels on the default Metal device. It is safe for
// concurrent use.
//
// Run encodes each kernel into an open command buffer and returns; Metal
// runs a command buffer's dispatches in order, each after the last
// finished. The buffer is committed, and waited for, when the host next
// reads or writes device memory, or at Sync. Errors a kernel finds, such
// as an index out of range, surface there.
type Backend struct {
	dev   id
	queue id
	name  string

	mu        sync.Mutex
	pipelines map[string]id // by kernel source
	cb, enc   id            // the open command buffer and its encoder, or 0
	last      id            // the last committed command buffer, or 0
	errWord   *buffer       // where kernels report errors
}

// buffer is a shared MTLBuffer, which the CPU and GPU address alike.
type buffer struct {
	obj     id
	data    unsafe.Pointer
	dt      backend.DType
	n       int
	cleanup runtime.Cleanup
}

func (b *buffer) DType() backend.DType { return b.dt }
func (b *buffer) Len() int             { return b.n }

// bytes returns the buffer's memory.
func (b *buffer) bytes() []byte { return unsafe.Slice((*byte)(b.data), b.n*b.dt.Size()) }

var _ backend.Backend = (*Backend)(nil)

// New returns a backend for the system's default Metal device, or an
// error wrapping ErrUnavailable if there is none.
func New() (*Backend, error) {
	if err := loadObjC(); err != nil {
		return nil, fmt.Errorf("metal: %v: %w", err, ErrUnavailable)
	}
	b := &Backend{pipelines: map[string]id{}}
	autoreleased(func() {
		b.dev = createDev()
		if b.dev == 0 {
			return
		}
		b.queue = id(send(b.dev, selNewCommandQueue))
		b.name = goString(id(send(b.dev, selName)))
	})
	if b.dev == 0 || b.queue == 0 {
		return nil, fmt.Errorf("metal: no default device: %w", ErrUnavailable)
	}
	w, err := b.Alloc(backend.Int32, 1)
	if err != nil {
		return nil, err
	}
	b.errWord = w.(*buffer)
	return b, nil
}

// Name returns "metal".
func (*Backend) Name() string { return "metal" }

// DeviceName returns the GPU's name, such as "Apple M5 Pro".
func (b *Backend) DeviceName() string { return b.name }

// Alloc returns a zeroed buffer of n elements in memory the CPU and GPU
// share.
func (b *Backend) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	if n < 0 || n > maxElems {
		return nil, fmt.Errorf("metal: alloc %d elements: %w", n, backend.ErrShape)
	}
	if dt.Size() == 0 {
		return nil, fmt.Errorf("metal: alloc %s: %w", dt, backend.ErrDType)
	}
	const storageShared = 0 // MTLResourceStorageModeShared
	var obj id
	autoreleased(func() {
		obj = id(send(b.dev, selNewBuffer, uintptr(max(n*dt.Size(), 4)), storageShared))
	})
	if obj == 0 {
		return nil, fmt.Errorf("metal: alloc %d x %s: %w", n, dt, backend.ErrOutOfMemory)
	}
	buf := &buffer{obj: obj, data: cPointer(send(obj, selContents)), dt: dt, n: n}
	buf.cleanup = runtime.AddCleanup(buf, release, obj)
	return buf, nil
}

func release(obj id) { send(obj, selRelease) }

// Free releases b. Work already encoded that uses b keeps it alive until
// that work finishes.
func (b *Backend) Free(buf backend.Buffer) {
	mb, ok := buf.(*buffer)
	if !ok || mb.obj == 0 {
		return
	}
	mb.cleanup.Stop()
	release(mb.obj)
	mb.obj, mb.data = 0, nil
}

func (b *Backend) own(buf backend.Buffer) (*buffer, error) {
	mb, ok := buf.(*buffer)
	if !ok || mb.obj == 0 {
		return nil, fmt.Errorf("metal: foreign or freed buffer %T: %w", buf, backend.ErrDType)
	}
	return mb, nil
}

// Upload copies src into dst once the work encoded so far has finished.
func (b *Backend) Upload(ctx context.Context, dst backend.Buffer, src []byte) error {
	mb, err := b.own(dst)
	if err != nil {
		return err
	}
	if len(src) != mb.n*mb.dt.Size() {
		return fmt.Errorf("metal: upload %d bytes into %d-byte buffer: %w", len(src), mb.n*mb.dt.Size(), backend.ErrShape)
	}
	if err := b.Sync(ctx); err != nil {
		return err
	}
	copy(mb.bytes(), src)
	return nil
}

// Download copies src into dst once the work encoded so far has finished.
func (b *Backend) Download(ctx context.Context, dst []byte, src backend.Buffer) error {
	mb, err := b.own(src)
	if err != nil {
		return err
	}
	if len(dst) != mb.n*mb.dt.Size() {
		return fmt.Errorf("metal: download %d-byte buffer into %d bytes: %w", mb.n*mb.dt.Size(), len(dst), backend.ErrShape)
	}
	if err := b.Sync(ctx); err != nil {
		return err
	}
	copy(dst, mb.bytes())
	return nil
}

// Sync commits the work encoded so far and waits for it to finish. It
// reports the errors kernels found since the last Sync.
func (b *Backend) Sync(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.commit()
	if b.last == 0 {
		return nil
	}
	var err error
	autoreleased(func() {
		send(b.last, selWait)
		const statusError = 5 // MTLCommandBufferStatusError
		if send(b.last, selStatus) == statusError {
			err = fmt.Errorf("metal: %w: %w", nsError("command buffer failed", id(send(b.last, selError))), backend.ErrInternal)
		}
	})
	release(b.last)
	b.last = 0
	word := (*uint32)(b.errWord.data)
	switch w := *word; {
	case err != nil:
	case w&errDivideByZero != 0:
		err = fmt.Errorf("metal: %w", backend.ErrDivideByZero)
	case w&errIndex != 0:
		err = fmt.Errorf("metal: %w", backend.ErrIndex)
	}
	*word = 0
	return err
}

// commit ends and commits the open command buffer, if any. b.mu is held.
func (b *Backend) commit() {
	if b.cb == 0 {
		return
	}
	autoreleased(func() {
		send(b.enc, selEndEncoding)
		send(b.cb, selCommit)
	})
	release(b.enc)
	if b.last != 0 {
		release(b.last)
	}
	b.last, b.cb, b.enc = b.cb, 0, 0
}

// Run encodes k into the open command buffer.
func (b *Backend) Run(ctx context.Context, k *backend.Kernel, outs []backend.Buffer, ins []backend.Buffer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := backend.CheckRun(k, outs, ins); err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	bufs := make([]*buffer, 0, len(outs)+len(ins))
	for _, buf := range append(append([]backend.Buffer{}, outs...), ins...) {
		mb, err := b.own(buf)
		if err != nil {
			return err
		}
		bufs = append(bufs, mb)
	}
	l, err := plan(k, outs, ins)
	if err != nil {
		return fmt.Errorf("metal: %s: %w", k.Op, err)
	}
	if l == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	pso, err := b.pipeline(l.src)
	if err != nil {
		return fmt.Errorf("metal: %s: %w", k.Op, err)
	}
	autoreleased(func() {
		if b.cb == 0 {
			b.cb = id(send(id(send(b.queue, selCommandBuffer)), selRetain))
			b.enc = id(send(id(send(b.cb, selComputeEncoder)), selRetain))
		}
		send(b.enc, selSetPipeline, uintptr(pso))
		for i, mb := range bufs {
			send(b.enc, selSetBuffer, uintptr(mb.obj), 0, uintptr(i))
		}
		send(b.enc, selSetBytes, uintptr(unsafe.Pointer(&l.params[0])), uintptr(4*len(l.params)), uintptr(len(bufs)))
		send(b.enc, selSetBuffer, uintptr(b.errWord.obj), 0, uintptr(len(bufs)+1))
		groups := &[3]uint{uint(l.groups), 1, 1}
		threads := &[3]uint{uint(l.threads), 1, 1}
		send(b.enc, selDispatch, uintptr(unsafe.Pointer(groups)), uintptr(unsafe.Pointer(threads)))
	})
	runtime.KeepAlive(l)
	runtime.KeepAlive(bufs)
	return nil
}

// pipeline returns the compiled pipeline of src, compiling it the first
// time. b.mu is held.
func (b *Backend) pipeline(src string) (id, error) {
	if pso, ok := b.pipelines[src]; ok {
		return pso, nil
	}
	var pso id
	var err error
	autoreleased(func() {
		opts := id(send(id(getClass("MTLCompileOptions")), selNew))
		defer release(opts)
		send(opts, selSetFastMath, 0)
		source := nsString(src)
		defer release(source)
		var nsErr id
		lib := id(send(b.dev, selNewLibrary, uintptr(source), uintptr(opts), uintptr(unsafe.Pointer(&nsErr))))
		if lib == 0 {
			err = fmt.Errorf("%w: %w\n%s", nsError("compile kernel", nsErr), backend.ErrInternal, numberedSource(src))
			return
		}
		defer release(lib)
		name := nsString("k")
		defer release(name)
		fn := id(send(lib, selNewFunction, uintptr(name)))
		if fn == 0 {
			err = fmt.Errorf("no kernel function: %w", backend.ErrInternal)
			return
		}
		defer release(fn)
		nsErr = 0
		pso = id(send(b.dev, selNewPipeline, uintptr(fn), uintptr(unsafe.Pointer(&nsErr))))
		if pso == 0 {
			err = fmt.Errorf("%w: %w", nsError("create pipeline", nsErr), backend.ErrInternal)
			return
		}
		if max := int(send(pso, selMaxThreads)); max < groupSize {
			release(pso)
			pso = 0
			err = fmt.Errorf("pipeline runs %d threads per threadgroup, want %d: %w", max, groupSize, backend.ErrInternal)
		}
	})
	if err != nil {
		return 0, err
	}
	b.pipelines[src] = pso
	return pso, nil
}
