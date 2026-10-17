//go:build linux && (amd64 || arm64) && !norocm

package rocm

import (
	"bytes"
	"context"
	"fmt"
	"runtime"
	"slices"
	"strings"
	"sync"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/diskcache"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// Backend runs kernels on the first ROCm device. It is safe for
// concurrent use.
//
// Run launches each kernel on the backend's stream and returns; the
// stream runs them in order, each after the last finished. Sync, Upload
// and Download wait for the stream. Errors a kernel finds, such as an
// index out of range, surface there.
type Backend struct {
	dev    int32
	stream uintptr
	name   string
	arch   string // the device's architecture, such as "gfx90a:sramecc+:xnack-"
	sms    int    // compute units
	pools  bool

	// dialect is the HIP dialect of the device's wavefront size.
	dialect *gen.Dialect
	options []string // hiprtc's options
	disk    *diskcache.Cache

	blas uintptr // hipBLAS handle, or 0 if there is no hipBLAS
	// blasMin is the work, in multiply-adds, from which float32 products
	// run on hipBLAS, or -1 if they never do.
	blasMin int

	mu    sync.Mutex
	funcs map[string]uintptr // compiled kernels by source
	// errDev is the device memory where kernels report errors, and
	// errHost where Sync copies it.
	errDev  uint64
	errHost *uint32
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

// New returns a backend for the first ROCm device, or an error wrapping
// ErrUnavailable if there is none or its libraries are missing.
func New() (*Backend, error) {
	if err := loadRuntime(); err != nil {
		return nil, fmt.Errorf("rocm: %v: %w", err, ErrUnavailable)
	}
	var count int32
	if r := hipGetDeviceCount(&count); r != hipSuccess && r != hipNoDevice {
		return nil, fmt.Errorf("rocm: count devices: %s: %w", hipGetErrorName(r), ErrUnavailable)
	}
	if count == 0 {
		return nil, fmt.Errorf("rocm: no device: %w", ErrUnavailable)
	}
	if err := loadRTC(); err != nil {
		return nil, fmt.Errorf("rocm: hiprtc: %v: %w", err, ErrUnavailable)
	}
	b := &Backend{funcs: map[string]uintptr{}, disk: diskcache.Open("rocm"), blasMin: -1, errHost: new(uint32)}
	defer b.enter()()
	var name [256]byte
	if r := hipDeviceGetName(&name[0], int32(len(name)), b.dev); r != hipSuccess {
		return nil, check(r, "device name")
	}
	b.name = goString(&name[0])
	var wave, pools, sms int32
	for _, a := range []struct {
		v    *int32
		attr int32
	}{{&wave, attrWarpSize}, {&pools, attrMemoryPools}, {&sms, attrMultiprocessors}} {
		if r := hipDeviceGetAttribute(a.v, a.attr, b.dev); r != hipSuccess {
			return nil, check(r, "device attribute")
		}
	}
	b.pools, b.sms = pools != 0, int(sms)
	if b.dialect = gen.HIP(int(wave)); b.dialect == nil {
		return nil, fmt.Errorf("rocm: wavefronts of %d threads: %w", wave, ErrUnavailable)
	}
	var err error
	if b.arch, err = architecture(b.dev, int(wave)); err != nil {
		return nil, err
	}
	b.options = compileOptions(b.arch)
	if r := hipStreamCreate(&b.stream); r != hipSuccess {
		return nil, check(r, "create stream")
	}
	if b.pools {
		// Keep freed memory for the next allocation instead of returning
		// it to the runtime at every synchronization.
		var pool uintptr
		threshold := ^uint64(0)
		if r := hipDeviceGetDefaultMemPool(&pool, b.dev); r != hipSuccess {
			return nil, check(r, "memory pool")
		}
		if r := hipMemPoolSetAttribute(pool, poolAttrReleaseThreshold, unsafe.Pointer(&threshold)); r != hipSuccess {
			return nil, check(r, "memory pool threshold")
		}
	}
	// The error word lives in device memory: atomics on host memory need
	// PCIe atomics, which not every host has.
	if r := hipMalloc(&b.errDev, 4); r != hipSuccess {
		return nil, check(r, "error word")
	}
	if r := hipMemsetAsync(b.errDev, 0, 4, b.stream); r != hipSuccess {
		return nil, check(r, "error word")
	}
	if err := b.sync(); err != nil {
		return nil, err
	}
	if loadBLAS() == nil {
		if st := hipblasCreate(&b.blas); st != blasSuccess {
			return nil, fmt.Errorf("rocm: create hipBLAS handle: status %d: %w", st, backend.ErrInternal)
		}
		if st := hipblasSetStream(b.blas, b.stream); st != blasSuccess {
			return nil, fmt.Errorf("rocm: hipBLAS stream: status %d: %w", st, backend.ErrInternal)
		}
		b.blasMin = 0
	}
	return b, nil
}

// architecture returns the architecture of device dev, with its target
// features, such as "gfx90a:sramecc+:xnack-", which only the device's
// properties hold. Their layout depends on the ROCm version, so it
// checks the layout's wavefront size against wave, the device's.
func architecture(dev int32, wave int) (string, error) {
	l := deviceProps
	// Twice the layout's size, aligned for any field.
	props := make([]uint64, l.size/4)
	if r := hipGetDeviceProperties(unsafe.Pointer(&props[0]), dev); r != hipSuccess {
		return "", check(r, "device properties")
	}
	raw := unsafe.Slice((*byte)(unsafe.Pointer(&props[0])), 8*len(props))
	got := *(*int32)(unsafe.Pointer(&raw[l.warpSize]))
	name, _, _ := bytes.Cut(raw[l.archName:l.archName+256], []byte{0})
	if int(got) != wave || !bytes.HasPrefix(name, []byte("gfx")) {
		return "", fmt.Errorf("rocm: unknown layout of the device properties (wavefront %d, want %d; architecture %q): %w", got, wave, name, ErrUnavailable)
	}
	return string(name), nil
}

// compileOptions returns hiprtc's options for a device of architecture
// arch: its code, without contracting products and sums into fused
// multiply-adds, which the CPU backend does not either.
func compileOptions(arch string) []string {
	return []string{"--offload-arch=" + arch, "-ffp-contract=off"}
}

// check returns nil if r is success and otherwise an error naming what
// failed.
func check(r hipError, what string) error {
	if r == hipSuccess {
		return nil
	}
	kind := backend.ErrInternal
	if r == hipOutOfMemory {
		kind = backend.ErrOutOfMemory
	}
	return fmt.Errorf("rocm: %s: %s: %w", what, hipGetErrorName(r), kind)
}

// enter makes the device current on the calling thread and keeps the
// goroutine on that thread until the returned function is called.
func (b *Backend) enter() func() {
	runtime.LockOSThread()
	hipSetDevice(b.dev)
	return runtime.UnlockOSThread
}

// Name returns "rocm".
func (*Backend) Name() string { return "rocm" }

// DeviceName returns the GPU's name, such as "AMD Instinct MI300X".
func (b *Backend) DeviceName() string { return b.name }

// Alloc returns an uninitialized buffer of n elements of device memory.
func (b *Backend) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	if n < 0 || n > gen.MaxElems {
		return nil, fmt.Errorf("rocm: alloc %d elements: %w", n, backend.ErrShape)
	}
	if dt.BlockBytes() == 0 {
		return nil, fmt.Errorf("rocm: alloc %s: %w", dt, backend.ErrDType)
	}
	size, ok := dt.Bytes(n)
	if !ok {
		return nil, fmt.Errorf("rocm: alloc %d elements of %s, in blocks of %d: %w", n, dt, dt.BlockSize(), backend.ErrShape)
	}
	bytes := uintptr(max(size, 4))
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	var ptr uint64
	var r hipError
	if b.pools {
		r = hipMallocAsync(&ptr, bytes, b.stream)
	} else {
		r = hipMalloc(&ptr, bytes)
	}
	if r != hipSuccess {
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
		hipFreeAsync(ptr, b.stream)
	} else {
		hipFree(ptr)
	}
}

// Free releases buf. Work already queued that uses buf runs first.
func (b *Backend) Free(buf backend.Buffer) {
	rb, ok := buf.(*buffer)
	if !ok || rb.ptr == 0 {
		return
	}
	rb.cleanup.Stop()
	b.free(rb.ptr)
	rb.ptr = 0
}

func (b *Backend) own(buf backend.Buffer) (*buffer, error) {
	rb, ok := buf.(*buffer)
	if !ok || rb.ptr == 0 {
		return nil, fmt.Errorf("rocm: foreign or freed buffer %T: %w", buf, backend.ErrDType)
	}
	return rb, nil
}

// Upload copies src into dst after the work queued so far.
func (b *Backend) Upload(ctx context.Context, dst backend.Buffer, src []byte) error {
	rb, err := b.own(dst)
	if err != nil {
		return err
	}
	if len(src) != rb.size() {
		return fmt.Errorf("rocm: upload %d bytes into %d-byte buffer: %w", len(src), rb.size(), backend.ErrShape)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	if len(src) > 0 {
		if r := hipMemcpyHtoDAsync(rb.ptr, unsafe.Pointer(&src[0]), uintptr(len(src)), b.stream); r != hipSuccess {
			return check(r, "upload")
		}
	}
	err = b.sync()
	runtime.KeepAlive(src)
	return err
}

// Download copies src into dst after the work queued so far.
func (b *Backend) Download(ctx context.Context, dst []byte, src backend.Buffer) error {
	rb, err := b.own(src)
	if err != nil {
		return err
	}
	if len(dst) != rb.size() {
		return fmt.Errorf("rocm: download %d-byte buffer into %d bytes: %w", rb.size(), len(dst), backend.ErrShape)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	if len(dst) > 0 {
		if r := hipMemcpyDtoHAsync(unsafe.Pointer(&dst[0]), rb.ptr, uintptr(len(dst)), b.stream); r != hipSuccess {
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

// sync waits for the stream and reports the errors kernels found: it
// copies the error word out after the work queued so far, and clears it
// when a kernel set it, which is rare. b.mu is held and the device is
// current.
func (b *Backend) sync() error {
	r := hipMemcpyDtoHAsync(unsafe.Pointer(b.errHost), b.errDev, 4, b.stream)
	if r == hipSuccess {
		r = hipStreamSynchronize(b.stream)
	}
	if r != hipSuccess {
		return check(r, "wait for the device")
	}
	word := *b.errHost
	if word == 0 {
		return nil
	}
	*b.errHost = 0
	r = hipMemsetAsync(b.errDev, 0, 4, b.stream)
	if r == hipSuccess {
		r = hipStreamSynchronize(b.stream)
	}
	switch {
	case r != hipSuccess:
		return check(r, "clear the error word")
	case word&gen.ErrDivideByZero != 0:
		return fmt.Errorf("rocm: %w", backend.ErrDivideByZero)
	case word&gen.ErrIndex != 0:
		return fmt.Errorf("rocm: %w", backend.ErrIndex)
	}
	return nil
}

// Run queues k on the stream.
func (b *Backend) Run(ctx context.Context, k *backend.Kernel, outs []backend.Buffer, ins []backend.Buffer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := backend.CheckRun(k, outs, ins); err != nil {
		return fmt.Errorf("rocm: %w", err)
	}
	ptrs := make([]uint64, 0, len(outs)+len(ins))
	for _, buf := range slices.Concat(outs, ins) {
		rb, err := b.own(buf)
		if err != nil {
			return err
		}
		ptrs = append(ptrs, rb.ptr)
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

// call is how the backend runs one kernel: a hipBLAS product, if gemm is
// set, then the generated kernel l, if it is set, which is the product's
// epilogue or the kernel itself.
type call struct {
	gemm *gen.ColumnMajor
	l    *gen.Launch
	fn   uintptr // the compiled kernel of l
}

// plan sets c to the call that runs k over outs and ins. b.mu is held
// and the device is current.
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
			c.l, err = gen.PlanEpilogue(b.dialect, k, outs, ins)
		}
	} else {
		c.l, err = gen.Plan(b.dialect, k, outs, ins)
	}
	if err != nil {
		return fmt.Errorf("rocm: %s: %w", k.Op, err)
	}
	if c.l != nil {
		if c.fn, err = b.function(c.l.Src, c.l.Threads); err != nil {
			return fmt.Errorf("rocm: %s: %w", k.Op, err)
		}
	}
	return nil
}

// issue queues the call c over the device pointers ptrs of its outputs,
// then its inputs, building the kernel's arguments in args, which may be
// nil. b.mu is held and the device is current.
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
	r := hipModuleLaunchKernel(c.fn, uint32(c.l.Groups), 1, 1, uint32(c.l.Threads), 1, 1, 0, b.stream, &args[0], 0)
	runtime.KeepAlive(ptrs)
	runtime.KeepAlive(c)
	return check(r, "launch")
}

// kernelArgs returns, in args, the arguments of a launch of l as the
// runtime takes them: the address of each buffer pointer, of the
// launch's parameters and of the error word's pointer.
func kernelArgs(args []unsafe.Pointer, ptrs []uint64, l *gen.Launch, errDev *uint64) []unsafe.Pointer {
	args = args[:0]
	for i := range ptrs {
		args = append(args, unsafe.Pointer(&ptrs[i]))
	}
	return append(args, unsafe.Pointer(&l.Params[0]), unsafe.Pointer(errDev))
}

// sgemm queues the hipBLAS call s over the device pointers ptrs of the
// output, then the inputs. b.mu is held and the device is current.
func (b *Backend) sgemm(s gen.ColumnMajor, ptrs []uint64) error {
	alpha, beta := float32(1), float32(0)
	op := func(trans bool) int32 {
		if trans {
			return blasOpT
		}
		return blasOpN
	}
	at := func(o gen.Operand) uint64 { return ptrs[1+o.In] + 4*uint64(o.Offset) }
	st := hipblasSgemmStridedBatched(b.blas, op(s.A.Trans), op(s.B.Trans), int32(s.M), int32(s.N), int32(s.K),
		&alpha, at(s.A), int32(s.A.LD), int64(s.A.Stride), at(s.B), int32(s.B.LD), int64(s.B.Stride),
		&beta, ptrs[0], int32(s.LDC), int64(s.StrideC), int32(s.Batch))
	if st != blasSuccess {
		return fmt.Errorf("rocm: hipBLAS sgemm: status %d: %w", st, backend.ErrInternal)
	}
	return nil
}

// function returns the compiled kernel of src, compiling it the first
// time and checking it runs threads threads per block. b.mu is held and
// the device is current.
func (b *Backend) function(src string, threads int) (uintptr, error) {
	if fn, ok := b.funcs[src]; ok {
		return fn, nil
	}
	image, err := b.compile(src)
	if err != nil {
		return 0, err
	}
	var module, fn uintptr
	if r := hipModuleLoadData(&module, &image[0]); r != hipSuccess {
		return 0, check(r, "load kernel")
	}
	runtime.KeepAlive(image)
	if r := hipModuleGetFunction(&fn, module, "k"); r != hipSuccess {
		return 0, check(r, "kernel function")
	}
	var maxThreads int32
	if r := hipFuncGetAttribute(&maxThreads, funcAttrMaxThreadsPerBlock, fn); r != hipSuccess {
		return 0, check(r, "kernel attribute")
	}
	if int(maxThreads) < threads {
		return 0, fmt.Errorf("kernel runs %d threads per block, want %d: %w", maxThreads, threads, backend.ErrInternal)
	}
	b.funcs[src] = fn
	return fn, nil
}

// compile returns the code object of src for the device, from the disk
// cache if it has it.
func (b *Backend) compile(src string) ([]byte, error) {
	key := b.disk.Key(compilerVersion(), b.options, src)
	if image, ok := b.disk.Get(key); ok {
		return image, nil
	}
	image, err := compileHIPRTC(src, b.options)
	if err != nil {
		return nil, err
	}
	b.disk.Put(key, image)
	return image, nil
}

// compilerVersion returns the versions of hiprtc and, if it is loaded,
// the HIP runtime, such as "hiprtc 6.4, hip 60443483".
func compilerVersion() string {
	var major, minor, rt int32
	hiprtcVersion(&major, &minor)
	v := fmt.Sprintf("hiprtc %d.%d", major, minor)
	if loadRuntime() == nil && hipRuntimeGetVersion(&rt) == hipSuccess {
		v += fmt.Sprintf(", hip %d", rt)
	}
	return v
}

// rtcMu serializes compilations: the hiprtc of ROCm 5 crashes compiling
// two programs at once.
var rtcMu sync.Mutex

// compileHIPRTC compiles src with hiprtc and returns its code object.
// loadRTC must have succeeded.
func compileHIPRTC(src string, options []string) ([]byte, error) {
	rtcMu.Lock()
	defer rtcMu.Unlock()
	var prog uintptr
	if r := hiprtcCreateProgram(&prog, src, "k.hip", 0, 0, 0); r != rtcSuccess {
		return nil, fmt.Errorf("create program: %s: %w", hiprtcGetErrorString(r), backend.ErrInternal)
	}
	defer hiprtcDestroyProgram(&prog)
	cstrs := make([][]byte, len(options))
	opts := make([]*byte, len(options))
	for i, o := range options {
		cstrs[i] = append([]byte(o), 0)
		opts[i] = &cstrs[i][0]
	}
	r := hiprtcCompileProgram(prog, int32(len(opts)), &opts[0])
	runtime.KeepAlive(cstrs)
	if r != rtcSuccess {
		var n uintptr
		log := "no log"
		if hiprtcGetProgramLogSize(prog, &n) == rtcSuccess && n > 1 {
			buf := make([]byte, n)
			if hiprtcGetProgramLog(prog, &buf[0]) == rtcSuccess {
				log = strings.TrimSpace(goString(&buf[0]))
			}
		}
		return nil, fmt.Errorf("compile kernel: %s: %s: %w\n%s", hiprtcGetErrorString(r), log, backend.ErrInternal, gen.Numbered(src))
	}
	var n uintptr
	if r := hiprtcGetCodeSize(prog, &n); r != rtcSuccess || n == 0 {
		return nil, fmt.Errorf("compiled kernel size: %s: %w", hiprtcGetErrorString(r), backend.ErrInternal)
	}
	image := make([]byte, n)
	if r := hiprtcGetCode(prog, &image[0]); r != rtcSuccess {
		return nil, fmt.Errorf("compiled kernel: %s: %w", hiprtcGetErrorString(r), backend.ErrInternal)
	}
	return image, nil
}
