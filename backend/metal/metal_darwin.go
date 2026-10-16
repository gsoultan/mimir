//go:build darwin && arm64

package metal

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
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
	cb, enc   id            // the open command buffer and compute encoder, or 0
	last      id            // the last committed command buffer, or 0
	errWord   *buffer       // where kernels report errors

	// hasMPS is whether Metal Performance Shaders runs on the device, and
	// mpsMin the work from which float32 products run through it, or -1
	// if they never do.
	hasMPS bool
	mpsMin int
	gemms  map[gemmKey]id // MPSMatrixMultiplication kernels

	// hasTensor is whether the device runs Metal 4 tensor operations in
	// hardware, and tensorMin the work from which float32 products run on
	// them, or -1 if they never do. They take precedence over MPS.
	hasTensor bool
	tensorMin int
}

// mpsMinWork is the number of multiply-adds, m·n·k per matrix times the
// batch, from which a float32 product runs through Metal Performance
// Shaders instead of the generated kernel. Below it, ending the compute
// encoder for the MPS kernel costs more than it saves.
const mpsMinWork = 1 << 18

// gemmKey identifies an MPSMatrixMultiplication kernel.
type gemmKey struct {
	transLeft, transRight bool
	m, n, k               int
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

// size returns the buffer's size in bytes.
func (b *buffer) size() int {
	n, _ := b.dt.Bytes(b.n)
	return n
}

// bytes returns the buffer's memory.
func (b *buffer) bytes() []byte { return unsafe.Slice((*byte)(b.data), b.size()) }

var _ backend.Backend = (*Backend)(nil)

// New returns a backend for the system's default Metal device, or an
// error wrapping ErrUnavailable if there is none.
func New() (*Backend, error) {
	if err := loadObjC(); err != nil {
		return nil, fmt.Errorf("metal: %v: %w", err, ErrUnavailable)
	}
	b := &Backend{pipelines: map[string]id{}, gemms: map[gemmKey]id{}, mpsMin: -1, tensorMin: -1}
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
	if b.hasMPS = loadMPS() == nil && mpsSupports(b.dev); b.hasMPS {
		b.mpsMin = mpsMinWork
	}
	if b.hasTensor = b.tensorOpsSupported(); b.hasTensor {
		b.tensorMin = tensorMinWork
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
	if n < 0 || n > gen.MaxElems {
		return nil, fmt.Errorf("metal: alloc %d elements: %w", n, backend.ErrShape)
	}
	if dt.BlockBytes() == 0 {
		return nil, fmt.Errorf("metal: alloc %s: %w", dt, backend.ErrDType)
	}
	size, ok := dt.Bytes(n)
	if !ok {
		return nil, fmt.Errorf("metal: alloc %d elements of %s, in blocks of %d: %w", n, dt, dt.BlockSize(), backend.ErrShape)
	}
	const storageShared = 0 // MTLResourceStorageModeShared
	var obj id
	autoreleased(func() {
		obj = id(send(b.dev, selNewBuffer, uintptr(max(size, 4)), storageShared))
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
	if len(src) != mb.size() {
		return fmt.Errorf("metal: upload %d bytes into %d-byte buffer: %w", len(src), mb.size(), backend.ErrShape)
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
	if len(dst) != mb.size() {
		return fmt.Errorf("metal: download %d-byte buffer into %d bytes: %w", mb.size(), len(dst), backend.ErrShape)
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
	case w&gen.ErrDivideByZero != 0:
		err = fmt.Errorf("metal: %w", backend.ErrDivideByZero)
	case w&gen.ErrIndex != 0:
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
	b.endCompute()
	autoreleased(func() { send(b.cb, selCommit) })
	if b.last != 0 {
		release(b.last)
	}
	b.last, b.cb = b.cb, 0
}

// commandBuffer returns the open command buffer, opening one if there is
// none. b.mu is held, inside an autorelease pool.
func (b *Backend) commandBuffer() id {
	if b.cb == 0 {
		b.cb = id(send(id(send(b.queue, selCommandBuffer)), selRetain))
	}
	return b.cb
}

// computeEncoder returns the open compute encoder, opening one if there is
// none. b.mu is held, inside an autorelease pool.
func (b *Backend) computeEncoder() id {
	if b.enc == 0 {
		b.enc = id(send(id(send(b.commandBuffer(), selComputeEncoder)), selRetain))
	}
	return b.enc
}

// endCompute ends the open compute encoder, if any, so that another kind
// of work can be encoded after it. b.mu is held.
func (b *Backend) endCompute() {
	if b.enc == 0 {
		return
	}
	autoreleased(func() { send(b.enc, selEndEncoding) })
	release(b.enc)
	b.enc = 0
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
	b.mu.Lock()
	defer b.mu.Unlock()
	var c call
	if err := b.plan(&c, k, outs, ins); err != nil {
		return err
	}
	b.issue(&c, bufs)
	return nil
}

// call is how the backend encodes one kernel: a product through Metal
// Performance Shaders, if mps is set, then its launches in order, which
// are a tensor product, a product's epilogue or the kernel itself.
type call struct {
	mps      *gen.GEMM
	launches []launch
}

// launch is a generated kernel's launch and its compiled pipeline.
type launch struct {
	l   *gen.Launch
	pso id
}

// plan sets c to the call that encodes k over outs and ins, compiling the
// pipelines it needs. b.mu is held.
func (b *Backend) plan(c *call, k *backend.Kernel, outs, ins []backend.Buffer) error {
	*c = call{launches: c.launches[:0]}
	lens := make([]int, len(ins))
	for i, buf := range ins {
		lens[i] = buf.Len()
	}
	add := func(l *gen.Launch, metal4 bool) error {
		pso, err := b.pipeline(l.Src, metal4, l.Threads)
		if err != nil {
			return fmt.Errorf("metal: %s: %w", k.Op, err)
		}
		c.launches = append(c.launches, launch{l, pso})
		return nil
	}
	g, isGEMM := gen.PlanGEMM(k, lens)
	work := g.Batch * g.M * g.N * g.K
	switch {
	case isGEMM && b.tensorMin >= 0 && work >= b.tensorMin:
		l, err := planTensorGEMM(g, len(ins))
		if err != nil {
			return fmt.Errorf("metal: %s: %w", k.Op, err)
		}
		if err := add(l, true); err != nil {
			return err
		}
	case isGEMM && b.mpsMin >= 0 && work >= b.mpsMin && packed(g):
		c.mps = &g
	default:
		l, err := gen.Plan(gen.Metal, k, outs, ins)
		if err != nil {
			return fmt.Errorf("metal: %s: %w", k.Op, err)
		}
		if l == nil {
			return nil
		}
		return add(l, false)
	}
	if k.Attrs.Program == nil {
		return nil
	}
	epilogue, err := gen.PlanEpilogue(gen.Metal, k, outs, ins)
	if err != nil {
		return fmt.Errorf("metal: %s: %w", k.Op, err)
	}
	return add(epilogue, false)
}

// issue encodes the call c over bufs, the outputs then the inputs. b.mu
// is held.
func (b *Backend) issue(c *call, bufs []*buffer) {
	if c.mps != nil {
		b.encodeGEMM(*c.mps, bufs)
	}
	for _, l := range c.launches {
		b.encode(l, bufs)
	}
}

// tensorOpsSupported reports whether the device runs Metal 4 tensor
// operations in hardware, as the neural accelerators of Apple10 GPUs
// (M5) do, and the system compiles the tensor product kernel, as macOS 26
// does. Elsewhere products keep running through MPS.
func (b *Backend) tensorOpsSupported() bool {
	const (
		familyApple10 = 1010 // MTLGPUFamilyApple10
		familyMetal4  = 5002 // MTLGPUFamilyMetal4
	)
	var ok bool
	autoreleased(func() {
		ok = send(b.dev, selSupportsFamily, familyApple10)&0xff != 0 &&
			send(b.dev, selSupportsFamily, familyMetal4)&0xff != 0
	})
	if !ok {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	_, err := b.pipeline(tensorSource(false, false, 2), true, tensorSIMDGroups*simdWidth)
	return err == nil
}

// encode encodes the launch l over bufs, the outputs then the inputs.
// b.mu is held.
func (b *Backend) encode(l launch, bufs []*buffer) {
	autoreleased(func() {
		enc := b.computeEncoder()
		send(enc, selSetPipeline, uintptr(l.pso))
		for i, mb := range bufs {
			send(enc, selSetBuffer, uintptr(mb.obj), 0, uintptr(i))
		}
		send(enc, selSetBytes, uintptr(unsafe.Pointer(&l.l.Params[0])), uintptr(4*len(l.l.Params)), uintptr(len(bufs)))
		send(enc, selSetBuffer, uintptr(b.errWord.obj), 0, uintptr(len(bufs)+1))
		groups := &[3]uint{uint(l.l.Groups), 1, 1}
		threads := &[3]uint{uint(l.l.Threads), 1, 1}
		send(enc, selDispatch, uintptr(unsafe.Pointer(groups)), uintptr(unsafe.Pointer(threads)))
	})
	runtime.KeepAlive(l.l)
	runtime.KeepAlive(bufs)
}

// encodeGEMM encodes the product g with Metal Performance Shaders, from
// bufs[1] and bufs[2] into bufs[0]. b.mu is held.
func (b *Backend) encodeGEMM(g gen.GEMM, bufs []*buffer) {
	const float32Type = 0x10000000 | 32 // MPSDataTypeFloat32
	key := gemmKey{g.A.Trans, g.B.Trans, g.M, g.N, g.K}
	autoreleased(func() {
		kern := b.gemms[key]
		if kern == 0 {
			kern = initMatMul(id(send(id(getClass("MPSMatrixMultiplication")), selAlloc)), selInitMatMul,
				b.dev, g.A.Trans, g.B.Trans, uint(g.M), uint(g.N), uint(g.K), 1, 0)
			b.gemms[key] = kern
		}
		mat := func(buf *buffer, offset, rows, cols, ld, stride int) id {
			desc := id(send(id(getClass("MPSMatrixDescriptor")), selMatrixDescriptor,
				uintptr(rows), uintptr(cols), uintptr(g.Batch), uintptr(4*ld), uintptr(4*stride), float32Type))
			return id(send(id(send(id(getClass("MPSMatrix")), selAlloc)), selInitMatrix, uintptr(buf.obj), uintptr(4*offset), uintptr(desc)))
		}
		left := mat(bufs[1], g.A.Offset, g.A.Rows, g.A.Cols, g.A.LD, g.A.Stride)
		defer release(left)
		right := mat(bufs[2], g.B.Offset, g.B.Rows, g.B.Cols, g.B.LD, g.B.Stride)
		defer release(right)
		result := mat(bufs[0], 0, g.M, g.N, g.N, g.ResultSize)
		defer release(result)
		b.endCompute()
		send(kern, selSetBatchStart, 0)
		send(kern, selSetBatchSize, uintptr(g.Batch))
		send(kern, selEncodeMatMul, uintptr(b.commandBuffer()), uintptr(left), uintptr(right), uintptr(result))
	})
	runtime.KeepAlive(bufs)
}

// pipeline returns the compiled pipeline of src, compiling it the first
// time as Metal Shading Language 4 if metal4 is set, and checking it runs
// threads threads per threadgroup. b.mu is held.
func (b *Backend) pipeline(src string, metal4 bool, threads int) (id, error) {
	if pso, ok := b.pipelines[src]; ok {
		return pso, nil
	}
	var pso id
	var err error
	autoreleased(func() {
		opts := id(send(id(getClass("MTLCompileOptions")), selNew))
		defer release(opts)
		send(opts, selSetFastMath, 0)
		if metal4 {
			const languageVersion4 = 4 << 16 // MTLLanguageVersion4_0
			send(opts, selSetLanguageVersion, languageVersion4)
		}
		source := nsString(src)
		defer release(source)
		var nsErr id
		lib := id(send(b.dev, selNewLibrary, uintptr(source), uintptr(opts), uintptr(unsafe.Pointer(&nsErr))))
		if lib == 0 {
			err = fmt.Errorf("%w: %w\n%s", nsError("compile kernel", nsErr), backend.ErrInternal, gen.Numbered(src))
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
		if max := int(send(pso, selMaxThreads)); max < threads {
			release(pso)
			pso = 0
			err = fmt.Errorf("pipeline runs %d threads per threadgroup, want %d: %w", max, threads, backend.ErrInternal)
		}
	})
	if err != nil {
		return 0, err
	}
	b.pipelines[src] = pso
	return pso, nil
}
