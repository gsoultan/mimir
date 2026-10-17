//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"context"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/diskcache"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// Backend runs kernels on one CUDA device. It is safe for concurrent
// use.
//
// Run launches each kernel on the backend's stream and returns; the
// stream runs them in order, each after the last finished. Sync, Upload
// and Download wait for the stream. Errors a kernel finds, such as an
// index out of range, surface there.
type Backend struct {
	dev    int32
	ctx    uintptr // the device's primary context
	stream uintptr
	name   string
	cc     int // compute capability, such as 86 for 8.6
	sms    int // multiprocessors
	pools  bool

	// arch is the architecture NVRTC compiles for, and cubin whether it
	// emits machine code for it rather than PTX the driver compiles.
	arch  string
	cubin bool
	disk  *diskcache.Cache

	blas uintptr // cuBLAS handle
	// blasMin is the work, in multiply-adds, from which float32 products
	// run on cuBLAS, or -1 if they never do.
	blasMin int

	mu    sync.Mutex
	funcs map[string]uintptr // compiled kernels by source
	// errWord is where kernels report errors: host memory the device
	// writes through errDev.
	errWord *uint32
	errDev  uint64

	// groups counts the open NCCL communicators over the device, which
	// make it asynchronous (see nccl_linux.go).
	groups atomic.Int32

	// peers holds the devices whose access to this device's memory pool
	// CopyPeer has settled, granted or not (see peer_linux.go).
	peers map[int32]bool
}

// buffer is device memory.
type buffer struct {
	ptr     uint64
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

var _ backend.Backend = (*Backend)(nil)

// New returns a backend for the first CUDA device, or an error wrapping
// ErrUnavailable if there is none or its libraries are missing.
func New() (*Backend, error) { return NewDevice(0) }

// NewDevice returns a backend for the CUDA device of the given ordinal,
// counted from 0 as DeviceCount counts them. Each call returns a new
// backend with a stream of its own; tensor.GPUs keeps one per device.
func NewDevice(ordinal int) (*Backend, error) {
	if err := loadDriver(); err != nil {
		return nil, fmt.Errorf("cuda: %v: %w", err, ErrUnavailable)
	}
	var count int32
	if r := cuDeviceGetCount(&count); r != cudaSuccess && r != cudaNoDevice {
		return nil, fmt.Errorf("cuda: count devices: %s: %w", errName(r), ErrUnavailable)
	}
	if count == 0 {
		return nil, fmt.Errorf("cuda: no device: %w", ErrUnavailable)
	}
	if ordinal < 0 || ordinal >= int(count) {
		return nil, fmt.Errorf("cuda: device %d of %d: %w", ordinal, count, ErrUnavailable)
	}
	for _, load := range []func() error{loadNVRTC, loadBLAS} {
		if err := load(); err != nil {
			return nil, fmt.Errorf("cuda: %v: %w", err, ErrUnavailable)
		}
	}
	b := &Backend{funcs: map[string]uintptr{}, disk: diskcache.Open("cuda")}
	if r := cuDeviceGet(&b.dev, int32(ordinal)); r != cudaSuccess {
		return nil, check(r, "open device")
	}
	var name [256]byte
	if r := cuDeviceGetName(&name[0], int32(len(name)), b.dev); r != cudaSuccess {
		return nil, check(r, "device name")
	}
	b.name = goString(&name[0])
	var major, minor, pools, sms int32
	for _, a := range []struct {
		v    *int32
		attr int32
	}{{&major, attrComputeMajor}, {&minor, attrComputeMinor}, {&pools, attrMemoryPools}, {&sms, attrMultiprocessors}} {
		if r := cuDeviceGetAttribute(a.v, a.attr, b.dev); r != cudaSuccess {
			return nil, check(r, "device attribute")
		}
	}
	b.cc, b.pools, b.sms = int(major)*10+int(minor), pools != 0, int(sms)
	if r := cuDevicePrimaryCtxRetain(&b.ctx, b.dev); r != cudaSuccess {
		return nil, check(r, "retain context")
	}
	defer b.enter()()
	if r := cuStreamCreate(&b.stream, 0); r != cudaSuccess {
		return nil, check(r, "create stream")
	}
	if b.pools {
		// Keep freed memory for the next allocation instead of returning
		// it to the driver at every synchronization.
		var pool uintptr
		threshold := ^uint64(0)
		if r := cuDeviceGetDefaultMemPool(&pool, b.dev); r != cudaSuccess {
			return nil, check(r, "memory pool")
		}
		if r := cuMemPoolSetAttribute(pool, poolAttrReleaseThreshold, unsafe.Pointer(&threshold)); r != cudaSuccess {
			return nil, check(r, "memory pool threshold")
		}
	}
	var word unsafe.Pointer
	if r := cuMemHostAlloc(&word, 4, hostAllocDeviceMap); r != cudaSuccess {
		return nil, check(r, "error word")
	}
	b.errWord = (*uint32)(word)
	if r := cuMemHostGetDevicePointer(&b.errDev, word, 0); r != cudaSuccess {
		return nil, check(r, "error word")
	}
	if st := cublasCreate(&b.blas); st != blasSuccess {
		return nil, fmt.Errorf("cuda: create cuBLAS handle: status %d: %w", st, backend.ErrInternal)
	}
	if st := cublasSetStream(b.blas, b.stream); st != blasSuccess {
		return nil, fmt.Errorf("cuda: cuBLAS stream: status %d: %w", st, backend.ErrInternal)
	}
	var err error
	if b.arch, b.cubin, err = architecture(b.cc); err != nil {
		return nil, err
	}
	return b, nil
}

// architecture returns the NVRTC option for a device of compute
// capability cc and whether NVRTC then emits machine code for it: it does
// when it knows the architecture; for a newer device it emits PTX for the
// newest architecture it knows, which the driver compiles.
func architecture(cc int) (string, bool, error) {
	var n int32
	if r := nvrtcGetNumSupportedArchs(&n); r != nvrtcSuccess || n == 0 {
		return "", false, fmt.Errorf("cuda: NVRTC architectures: %s: %w", nvrtcGetErrorString(r), ErrUnavailable)
	}
	archs := make([]int32, n)
	if r := nvrtcGetSupportedArchs(&archs[0]); r != nvrtcSuccess {
		return "", false, fmt.Errorf("cuda: NVRTC architectures: %s: %w", nvrtcGetErrorString(r), ErrUnavailable)
	}
	best := -1
	for _, a := range archs {
		if int(a) <= cc && int(a) > best {
			best = int(a)
		}
	}
	switch {
	case best < 0:
		return "", false, fmt.Errorf("cuda: compute capability %d.%d is older than NVRTC supports: %w", cc/10, cc%10, ErrUnavailable)
	case best == cc:
		return fmt.Sprintf("--gpu-architecture=sm_%d", cc), true, nil
	default:
		return fmt.Sprintf("--gpu-architecture=compute_%d", best), false, nil
	}
}

// check returns nil if r is success and otherwise an error naming what
// failed.
func check(r result, what string) error {
	if r == cudaSuccess {
		return nil
	}
	kind := backend.ErrInternal
	if r == cudaOutOfMemory {
		kind = backend.ErrOutOfMemory
	}
	return fmt.Errorf("cuda: %s: %s: %w", what, errName(r), kind)
}

// enter makes the device's context current on the calling thread and
// keeps the goroutine on that thread until the returned function is
// called.
func (b *Backend) enter() func() {
	runtime.LockOSThread()
	cuCtxSetCurrent(b.ctx)
	return runtime.UnlockOSThread
}

// Name returns "cuda".
func (*Backend) Name() string { return "cuda" }

// DeviceName returns the GPU's name, such as "NVIDIA A10".
func (b *Backend) DeviceName() string { return b.name }

// Alloc returns an uninitialized buffer of n elements of device memory.
func (b *Backend) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	if n < 0 || n > gen.MaxElems {
		return nil, fmt.Errorf("cuda: alloc %d elements: %w", n, backend.ErrShape)
	}
	if dt.BlockBytes() == 0 {
		return nil, fmt.Errorf("cuda: alloc %s: %w", dt, backend.ErrDType)
	}
	size, ok := dt.Bytes(n)
	if !ok {
		return nil, fmt.Errorf("cuda: alloc %d elements of %s, in blocks of %d: %w", n, dt, dt.BlockSize(), backend.ErrShape)
	}
	bytes := uintptr(max(size, 4))
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	var ptr uint64
	var r result
	if b.pools {
		r = cuMemAllocAsync(&ptr, bytes, b.stream)
	} else {
		r = cuMemAlloc(&ptr, bytes)
	}
	if r != cudaSuccess {
		return nil, check(r, fmt.Sprintf("alloc %d x %s", n, dt))
	}
	buf := &buffer{ptr: ptr, dt: dt, n: n}
	buf.cleanup = runtime.AddCleanup(buf, b.free, ptr)
	return buf, nil
}

// free releases device memory. With a memory pool the release is
// ordered after the work queued so far, so a buffer may be freed while
// kernels that read it are still queued.
func (b *Backend) free(ptr uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	if b.pools {
		cuMemFreeAsync(ptr, b.stream)
	} else {
		cuMemFree(ptr)
	}
}

// Free releases buf. Work already queued that uses buf runs first.
func (b *Backend) Free(buf backend.Buffer) {
	cb, ok := buf.(*buffer)
	if !ok || cb.ptr == 0 {
		return
	}
	cb.cleanup.Stop()
	b.free(cb.ptr)
	cb.ptr = 0
}

func (b *Backend) own(buf backend.Buffer) (*buffer, error) {
	cb, ok := buf.(*buffer)
	if !ok || cb.ptr == 0 {
		return nil, fmt.Errorf("cuda: foreign or freed buffer %T: %w", buf, backend.ErrDType)
	}
	return cb, nil
}

// Upload copies src into dst after the work queued so far.
func (b *Backend) Upload(ctx context.Context, dst backend.Buffer, src []byte) error {
	cb, err := b.own(dst)
	if err != nil {
		return err
	}
	if len(src) != cb.size() {
		return fmt.Errorf("cuda: upload %d bytes into %d-byte buffer: %w", len(src), cb.size(), backend.ErrShape)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	if len(src) > 0 {
		if r := cuMemcpyHtoDAsync(cb.ptr, unsafe.Pointer(&src[0]), uintptr(len(src)), b.stream); r != cudaSuccess {
			return check(r, "upload")
		}
	}
	err = b.sync()
	runtime.KeepAlive(src)
	return err
}

// Download copies src into dst after the work queued so far.
func (b *Backend) Download(ctx context.Context, dst []byte, src backend.Buffer) error {
	cb, err := b.own(src)
	if err != nil {
		return err
	}
	if len(dst) != cb.size() {
		return fmt.Errorf("cuda: download %d-byte buffer into %d bytes: %w", cb.size(), len(dst), backend.ErrShape)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	if len(dst) > 0 {
		if r := cuMemcpyDtoHAsync(unsafe.Pointer(&dst[0]), cb.ptr, uintptr(len(dst)), b.stream); r != cudaSuccess {
			return check(r, "download")
		}
	}
	err = b.sync()
	runtime.KeepAlive(dst)
	return err
}

// Sync waits for the work queued so far to finish. It reports the errors
// kernels found since the last Sync.
func (b *Backend) Sync(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	return b.sync()
}

// sync waits for the stream and reports the errors kernels found. b.mu is
// held and the context is current.
func (b *Backend) sync() error {
	r := cuStreamSynchronize(b.stream)
	word := atomic.SwapUint32(b.errWord, 0)
	switch {
	case r != cudaSuccess:
		return check(r, "wait for the device")
	case word&gen.ErrDivideByZero != 0:
		return fmt.Errorf("cuda: %w", backend.ErrDivideByZero)
	case word&gen.ErrIndex != 0:
		return fmt.Errorf("cuda: %w", backend.ErrIndex)
	}
	return nil
}

// Run queues k on the stream.
func (b *Backend) Run(ctx context.Context, k *backend.Kernel, outs []backend.Buffer, ins []backend.Buffer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := backend.CheckRun(k, outs, ins); err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	ptrs := make([]uint64, 0, len(outs)+len(ins))
	for _, buf := range slices.Concat(outs, ins) {
		cb, err := b.own(buf)
		if err != nil {
			return err
		}
		ptrs = append(ptrs, cb.ptr)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	var c call
	if err := b.plan(&c, k, outs, ins); err != nil {
		return err
	}
	return b.issue(&c, ptrs, nil)
}

// call is how the backend runs one kernel: a cuBLAS product, if gemm is
// set, then the generated kernel l, if it is set, which is the product's
// epilogue or the kernel itself.
type call struct {
	gemm *gen.ColumnMajor
	l    *gen.Launch
	fn   uintptr // the compiled kernel of l
}

// plan sets c to the call that runs k over outs and ins. b.mu is held
// and the context is current.
func (b *Backend) plan(c *call, k *backend.Kernel, outs, ins []backend.Buffer) error {
	*c = call{}
	lens := make([]int, len(ins))
	for i, buf := range ins {
		lens[i] = buf.Len()
	}
	var err error
	if g, isGEMM := gen.PlanGEMM(k, lens); isGEMM && b.blasMin >= 0 && g.Batch*g.M*g.N*g.K >= b.blasMin {
		s := g.ColumnMajor()
		c.gemm = &s
		if k.Attrs.Program != nil {
			c.l, err = gen.PlanEpilogue(gen.CUDA, k, outs, ins)
		}
	} else {
		c.l, err = gen.Plan(gen.CUDA, k, outs, ins)
	}
	if err != nil {
		return fmt.Errorf("cuda: %s: %w", k.Op, err)
	}
	if c.l != nil {
		if c.fn, err = b.function(c.l.Src, c.l.Threads); err != nil {
			return fmt.Errorf("cuda: %s: %w", k.Op, err)
		}
	}
	return nil
}

// issue queues the call c over the device pointers ptrs of its outputs,
// then its inputs, building the kernel's arguments in args, which may be
// nil. b.mu is held and the context is current.
func (b *Backend) issue(c *call, ptrs []uint64, args []unsafe.Pointer) error {
	if c.gemm != nil {
		if err := b.sgemm(*c.gemm, ptrs); err != nil {
			return err
		}
	}
	if c.l == nil {
		return nil
	}
	args = kernelArgs(args, ptrs, c.l, &b.errDev)
	r := cuLaunchKernel(c.fn, uint32(c.l.Groups), 1, 1, uint32(c.l.Threads), 1, 1, 0, b.stream, &args[0], 0)
	runtime.KeepAlive(ptrs)
	runtime.KeepAlive(c)
	return check(r, "launch")
}

// kernelArgs returns, in args, the arguments of a launch of l as the
// driver takes them: the address of each buffer pointer, of the launch's
// parameters and of the error word's pointer.
func kernelArgs(args []unsafe.Pointer, ptrs []uint64, l *gen.Launch, errDev *uint64) []unsafe.Pointer {
	args = args[:0]
	for i := range ptrs {
		args = append(args, unsafe.Pointer(&ptrs[i]))
	}
	return append(args, unsafe.Pointer(&l.Params[0]), unsafe.Pointer(errDev))
}

// sgemm queues the cuBLAS call s over the device pointers ptrs of the
// output, then the inputs. b.mu is held and the context is current.
func (b *Backend) sgemm(s gen.ColumnMajor, ptrs []uint64) error {
	alpha, beta := float32(1), float32(0)
	op := func(trans bool) int32 {
		if trans {
			return blasOpT
		}
		return blasOpN
	}
	at := func(o gen.Operand) uint64 { return ptrs[1+o.In] + 4*uint64(o.Offset) }
	st := cublasSgemmStridedBatched(b.blas, op(s.A.Trans), op(s.B.Trans), int32(s.M), int32(s.N), int32(s.K),
		&alpha, at(s.A), int32(s.A.LD), int64(s.A.Stride), at(s.B), int32(s.B.LD), int64(s.B.Stride),
		&beta, ptrs[0], int32(s.LDC), int64(s.StrideC), int32(s.Batch))
	if st != blasSuccess {
		return fmt.Errorf("cuda: cuBLAS sgemm: status %d: %w", st, backend.ErrInternal)
	}
	return nil
}

// function returns the compiled kernel of src, compiling it the first
// time and checking it runs threads threads per block. b.mu is held and
// the context is current.
func (b *Backend) function(src string, threads int) (uintptr, error) {
	if fn, ok := b.funcs[src]; ok {
		return fn, nil
	}
	image, err := b.compile(src)
	if err != nil {
		return 0, err
	}
	var module, fn uintptr
	if r := cuModuleLoadData(&module, &image[0]); r != cudaSuccess {
		return 0, check(r, "load kernel")
	}
	runtime.KeepAlive(image)
	if r := cuModuleGetFunction(&fn, module, "k"); r != cudaSuccess {
		return 0, check(r, "kernel function")
	}
	var maxThreads int32
	if r := cuFuncGetAttribute(&maxThreads, funcAttrMaxThreadsPerBlock, fn); r != cudaSuccess {
		return 0, check(r, "kernel attribute")
	}
	if int(maxThreads) < threads {
		return 0, fmt.Errorf("kernel runs %d threads per block, want %d: %w", maxThreads, threads, backend.ErrInternal)
	}
	b.funcs[src] = fn
	return fn, nil
}

// compile returns the machine code or PTX of src for the device, from
// the disk cache if it has it.
func (b *Backend) compile(src string) ([]byte, error) {
	options := []string{b.arch, "--fmad=false"}
	key := b.disk.Key(compilerVersion(), options, src)
	if image, ok := b.disk.Get(key); ok {
		return image, nil
	}
	image, err := compileNVRTC(src, options, b.cubin)
	if err != nil {
		return nil, err
	}
	b.disk.Put(key, image)
	return image, nil
}

// compilerVersion returns the NVRTC version, such as "nvrtc 12.9".
func compilerVersion() string {
	var major, minor int32
	nvrtcVersion(&major, &minor)
	return fmt.Sprintf("nvrtc %d.%d", major, minor)
}

// compileNVRTC compiles src with NVRTC and returns its machine code if
// cubin is set, otherwise its PTX. loadNVRTC must have succeeded.
func compileNVRTC(src string, options []string, cubin bool) ([]byte, error) {
	var prog uintptr
	if r := nvrtcCreateProgram(&prog, src, "k.cu", 0, 0, 0); r != nvrtcSuccess {
		return nil, fmt.Errorf("create program: %s: %w", nvrtcGetErrorString(r), backend.ErrInternal)
	}
	defer nvrtcDestroyProgram(&prog)
	cstrs := make([][]byte, len(options))
	opts := make([]*byte, len(options))
	for i, o := range options {
		cstrs[i] = append([]byte(o), 0)
		opts[i] = &cstrs[i][0]
	}
	r := nvrtcCompileProgram(prog, int32(len(opts)), &opts[0])
	runtime.KeepAlive(cstrs)
	if r != nvrtcSuccess {
		var n uintptr
		log := "no log"
		if nvrtcGetProgramLogSize(prog, &n) == nvrtcSuccess && n > 1 {
			buf := make([]byte, n)
			if nvrtcGetProgramLog(prog, &buf[0]) == nvrtcSuccess {
				log = goString(&buf[0])
			}
		}
		return nil, fmt.Errorf("compile kernel: %s: %s: %w\n%s", nvrtcGetErrorString(r), log, backend.ErrInternal, gen.Numbered(src))
	}
	size, get := nvrtcGetPTXSize, nvrtcGetPTX
	if cubin {
		size, get = nvrtcGetCUBINSize, nvrtcGetCUBIN
	}
	var n uintptr
	if r := size(prog, &n); r != nvrtcSuccess || n == 0 {
		return nil, fmt.Errorf("compiled kernel size: %s: %w", nvrtcGetErrorString(r), backend.ErrInternal)
	}
	image := make([]byte, n)
	if r := get(prog, &image[0]); r != nvrtcSuccess {
		return nil, fmt.Errorf("compiled kernel: %s: %w", nvrtcGetErrorString(r), backend.ErrInternal)
	}
	return image, nil
}
