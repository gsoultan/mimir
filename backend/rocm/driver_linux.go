//go:build linux && (amd64 || arm64) && !norocm

package rocm

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The HIP runtime, hiprtc and hipBLAS are called through purego, without
// cgo. Each library loads once, on first use, and stays loaded. ROCm 6
// and later ship hiprtc as a library of its own; ROCm 5 builds it into
// the runtime, so the runtime serves when there is no libhiprtc.

// hipError is a hipError_t.
type hipError uint32

// HIP results and constants, as in hip_runtime_api.h, the same from
// ROCm 5 to 7.
const (
	hipSuccess     hipError = 0
	hipOutOfMemory hipError = 2
	hipNoDevice    hipError = 100

	// hipDeviceAttribute_t values.
	attrComputeMajor    = 23
	attrMultiprocessors = 63
	attrWarpSize        = 87
	attrMemoryPools     = 88

	// hipFunction_attribute values, numbered apart from the device's.
	funcAttrMaxThreadsPerBlock = 0

	poolAttrReleaseThreshold = 4 // hipMemPoolAttrReleaseThreshold
)

// libraries returns the names to try for a library: the one the
// environment variable env names, if it is set, otherwise names.
func libraries(env string, names ...string) []string {
	if path := os.Getenv(env); path != "" {
		return []string{path}
	}
	return names
}

// open loads the first of names the dynamic loader finds.
func open(names []string) (uintptr, error) {
	var err error
	for _, name := range names {
		var lib uintptr
		if lib, err = purego.Dlopen(name, purego.RTLD_GLOBAL|purego.RTLD_NOW); err == nil {
			return lib, nil
		}
	}
	if len(names) == 1 {
		return 0, err
	}
	return 0, fmt.Errorf("none of %s loads: %w", strings.Join(names, ", "), err)
}

// register binds each function pointer to the symbol of lib it names, or
// reports the first that is missing.
func register(lib uintptr, fns []symbol) error {
	for _, f := range fns {
		sym, err := purego.Dlsym(lib, f.name)
		if err != nil {
			return err
		}
		purego.RegisterFunc(f.fptr, sym)
	}
	return nil
}

// symbol is a function pointer to bind and the name of its symbol.
type symbol struct {
	fptr any
	name string
}

// Loaded functions of the HIP runtime API, named as in
// hip_runtime_api.h.
var (
	runtimeOnce sync.Once
	runtimeErr  error
	runtimeLib  uintptr

	hipInit                    func(flags uint32) hipError
	hipGetErrorName            func(e hipError) string
	hipRuntimeGetVersion       func(v *int32) hipError
	hipGetDeviceCount          func(n *int32) hipError
	hipSetDevice               func(dev int32) hipError
	hipDeviceGetName           func(name *byte, n int32, dev int32) hipError
	hipDeviceGetAttribute      func(v *int32, attr int32, dev int32) hipError
	hipStreamCreate            func(s *uintptr) hipError
	hipStreamSynchronize       func(s uintptr) hipError
	hipModuleLoadData          func(m *uintptr, image *byte) hipError
	hipModuleGetFunction       func(f *uintptr, m uintptr, name string) hipError
	hipFuncGetAttribute        func(v *int32, attr int32, f uintptr) hipError
	hipModuleLaunchKernel      func(f uintptr, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, params *unsafe.Pointer, extra uintptr) hipError
	hipMallocAsync             func(p *uint64, n uintptr, stream uintptr) hipError
	hipFreeAsync               func(p uint64, stream uintptr) hipError
	hipMalloc                  func(p *uint64, n uintptr) hipError
	hipFree                    func(p uint64) hipError
	hipMemcpyHtoDAsync         func(dst uint64, src unsafe.Pointer, n uintptr, stream uintptr) hipError
	hipMemcpyDtoHAsync         func(dst unsafe.Pointer, src uint64, n uintptr, stream uintptr) hipError
	hipMemsetAsync             func(dst uint64, value int32, n uintptr, stream uintptr) hipError
	hipDeviceGetDefaultMemPool func(pool *uintptr, dev int32) hipError
	hipMemPoolSetAttribute     func(pool uintptr, attr int32, value unsafe.Pointer) hipError

	// hipGetDeviceProperties fills a hipDeviceProp_t, whose layout
	// depends on the ROCm version, and deviceProps is that layout.
	hipGetDeviceProperties func(props unsafe.Pointer, dev int32) hipError
	deviceProps            propsLayout
)

// propsLayout is where a hipDeviceProp_t holds the fields the backend
// reads.
type propsLayout struct {
	size, warpSize, archName int
}

// The layouts of hipDeviceProp_t: ROCm 6 and later fill the one of
// hipGetDevicePropertiesR0600, ROCm 5 the one of hipGetDeviceProperties.
var (
	propsR0600 = propsLayout{size: 1472, warpSize: 308, archName: 1160}
	propsR0000 = propsLayout{size: 792, warpSize: 276, archName: 396}
)

// loadRuntime loads libamdhip64 and initializes it once.
func loadRuntime() error {
	runtimeOnce.Do(func() {
		lib, err := open(libraries("MIMIR_HIP", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so"))
		if err != nil {
			runtimeErr = err
			return
		}
		runtimeLib = lib
		if runtimeErr = register(lib, []symbol{
			{&hipInit, "hipInit"},
			{&hipGetErrorName, "hipGetErrorName"},
			{&hipRuntimeGetVersion, "hipRuntimeGetVersion"},
			{&hipGetDeviceCount, "hipGetDeviceCount"},
			{&hipSetDevice, "hipSetDevice"},
			{&hipDeviceGetName, "hipDeviceGetName"},
			{&hipDeviceGetAttribute, "hipDeviceGetAttribute"},
			{&hipStreamCreate, "hipStreamCreate"},
			{&hipStreamSynchronize, "hipStreamSynchronize"},
			{&hipModuleLoadData, "hipModuleLoadData"},
			{&hipModuleGetFunction, "hipModuleGetFunction"},
			{&hipFuncGetAttribute, "hipFuncGetAttribute"},
			{&hipModuleLaunchKernel, "hipModuleLaunchKernel"},
			{&hipMallocAsync, "hipMallocAsync"},
			{&hipFreeAsync, "hipFreeAsync"},
			{&hipMalloc, "hipMalloc"},
			{&hipFree, "hipFree"},
			{&hipMemcpyHtoDAsync, "hipMemcpyHtoDAsync"},
			{&hipMemcpyDtoHAsync, "hipMemcpyDtoHAsync"},
			{&hipMemsetAsync, "hipMemsetAsync"},
			{&hipDeviceGetDefaultMemPool, "hipDeviceGetDefaultMemPool"},
			{&hipMemPoolSetAttribute, "hipMemPoolSetAttribute"},
		}); runtimeErr != nil {
			return
		}
		deviceProps = propsR0600
		if register(lib, []symbol{{&hipGetDeviceProperties, "hipGetDevicePropertiesR0600"}}) != nil {
			deviceProps = propsR0000
			if runtimeErr = register(lib, []symbol{{&hipGetDeviceProperties, "hipGetDeviceProperties"}}); runtimeErr != nil {
				return
			}
		}
		if r := hipInit(0); r != hipSuccess && r != hipNoDevice {
			runtimeErr = fmt.Errorf("hipInit: %s", hipGetErrorName(r))
		}
	})
	return runtimeErr
}

// HIP graph constants.
const (
	captureModeThreadLocal = 1 // hipStreamCaptureModeThreadLocal
	captureStatusActive    = 1 // hipStreamCaptureStatusActive
)

// kernelNodeParams is a hipKernelNodeParams.
type kernelNodeParams struct {
	blockX, blockY, blockZ uint32
	_                      uintptr // extra
	fn                     uintptr
	gridX, gridY, gridZ    uint32
	params                 *unsafe.Pointer
	_                      uint32 // sharedMemBytes
}

// Loaded functions of HIP's graph API. Kernels launch one by one if any
// is missing.
var (
	graphOnce sync.Once
	graphErr  error

	hipStreamBeginCapture           func(s uintptr, mode int32) hipError
	hipStreamEndCapture             func(s uintptr, graph *uintptr) hipError
	hipStreamGetCaptureInfo         func(s uintptr, status *int32, id *uint64, graph *uintptr, deps **uintptr, n *uintptr) hipError
	hipGraphInstantiate             func(exec *uintptr, graph uintptr, flags uint64) hipError
	hipGraphExecKernelNodeSetParams func(exec, node uintptr, params *kernelNodeParams) hipError
	hipGraphLaunch                  func(exec, s uintptr) hipError
	hipGraphExecDestroy             func(exec uintptr) hipError
	hipGraphDestroy                 func(graph uintptr) hipError
)

// loadGraphs loads the graph API once. loadRuntime must have succeeded.
func loadGraphs() error {
	graphOnce.Do(func() {
		graphErr = register(runtimeLib, []symbol{
			{&hipStreamBeginCapture, "hipStreamBeginCapture"},
			{&hipStreamEndCapture, "hipStreamEndCapture"},
			{&hipStreamGetCaptureInfo, "hipStreamGetCaptureInfo_v2"},
			{&hipGraphInstantiate, "hipGraphInstantiateWithFlags"},
			{&hipGraphExecKernelNodeSetParams, "hipGraphExecKernelNodeSetParams"},
			{&hipGraphLaunch, "hipGraphLaunch"},
			{&hipGraphExecDestroy, "hipGraphExecDestroy"},
			{&hipGraphDestroy, "hipGraphDestroy"},
		})
	})
	return graphErr
}

// goString returns the NUL-terminated string at p.
func goString(p *byte) string {
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

// rtcResult is a hiprtcResult.
type rtcResult uint32

const rtcSuccess rtcResult = 0

// Loaded functions of hiprtc, named as in hiprtc.h.
var (
	rtcOnce sync.Once
	rtcErr  error

	hiprtcVersion           func(major, minor *int32) rtcResult
	hiprtcGetErrorString    func(r rtcResult) string
	hiprtcCreateProgram     func(prog *uintptr, src, name string, headers int32, sources, names uintptr) rtcResult
	hiprtcDestroyProgram    func(prog *uintptr) rtcResult
	hiprtcCompileProgram    func(prog uintptr, n int32, options **byte) rtcResult
	hiprtcGetProgramLogSize func(prog uintptr, n *uintptr) rtcResult
	hiprtcGetProgramLog     func(prog uintptr, log *byte) rtcResult
	hiprtcGetCodeSize       func(prog uintptr, n *uintptr) rtcResult
	hiprtcGetCode           func(prog uintptr, code *byte) rtcResult
)

// loadRTC loads hiprtc once: the library MIMIR_HIPRTC names, or
// libhiprtc, or, for ROCm 5, the HIP runtime. It needs no device, so
// kernels compile on machines without one.
func loadRTC() error {
	rtcOnce.Do(func() {
		lib, err := open(libraries("MIMIR_HIPRTC", "libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so",
			"libamdhip64.so.5", "libamdhip64.so"))
		if err != nil {
			rtcErr = err
			return
		}
		rtcErr = register(lib, []symbol{
			{&hiprtcVersion, "hiprtcVersion"},
			{&hiprtcGetErrorString, "hiprtcGetErrorString"},
			{&hiprtcCreateProgram, "hiprtcCreateProgram"},
			{&hiprtcDestroyProgram, "hiprtcDestroyProgram"},
			{&hiprtcCompileProgram, "hiprtcCompileProgram"},
			{&hiprtcGetProgramLogSize, "hiprtcGetProgramLogSize"},
			{&hiprtcGetProgramLog, "hiprtcGetProgramLog"},
			{&hiprtcGetCodeSize, "hiprtcGetCodeSize"},
			{&hiprtcGetCode, "hiprtcGetCode"},
		})
	})
	return rtcErr
}

// blasStatus is a hipblasStatus_t.
type blasStatus uint32

const (
	blasSuccess blasStatus = 0

	blasOpN int32 = 111 // HIPBLAS_OP_N
	blasOpT int32 = 112 // HIPBLAS_OP_T
)

// Loaded functions of hipBLAS, named as in hipblas.h.
var (
	blasOnce sync.Once
	blasErr  error

	hipblasCreate              func(h *uintptr) blasStatus
	hipblasSetStream           func(h, stream uintptr) blasStatus
	hipblasSgemmStridedBatched func(h uintptr, transA, transB, m, n, k int32, alpha *float32,
		a uint64, lda int32, strideA int64, b uint64, ldb int32, strideB int64,
		beta *float32, c uint64, ldc int32, strideC int64, batch int32) blasStatus
)

// loadBLAS loads libhipblas once. The backend runs without it, on the
// generated kernels, if it is missing.
func loadBLAS() error {
	blasOnce.Do(func() {
		lib, err := open(libraries("MIMIR_HIPBLAS", "libhipblas.so.3", "libhipblas.so.2", "libhipblas.so.1", "libhipblas.so.0", "libhipblas.so"))
		if err != nil {
			blasErr = err
			return
		}
		blasErr = register(lib, []symbol{
			{&hipblasCreate, "hipblasCreate"},
			{&hipblasSetStream, "hipblasSetStream"},
			{&hipblasSgemmStridedBatched, "hipblasSgemmStridedBatched"},
		})
	})
	return blasErr
}
