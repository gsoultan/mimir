//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The CUDA driver, NVRTC and cuBLAS are called through purego, without
// cgo. Each library loads once, on first use, and stays loaded.

// result is a CUresult.
type result uint32

// CUDA driver results and constants.
const (
	cudaSuccess     result = 0
	cudaOutOfMemory result = 2
	cudaNoDevice    result = 100

	// CUdevice_attribute values.
	attrMultiprocessors = 16
	attrComputeMajor    = 75
	attrComputeMinor    = 76
	attrMemoryPools     = 115

	// CUfunction_attribute values, numbered apart from the device's.
	funcAttrMaxThreadsPerBlock = 0

	hostAllocDeviceMap = 0x2

	poolAttrReleaseThreshold = 4
)

// Loaded functions of the driver API, named as in cuda.h.
var (
	driverOnce sync.Once
	driverErr  error
	driverLib  uintptr

	cuInit                    func(flags uint32) result
	cuGetErrorName            func(r result, name **byte) result
	cuDeviceGetCount          func(n *int32) result
	cuDeviceGet               func(dev *int32, ordinal int32) result
	cuDeviceGetName           func(name *byte, n int32, dev int32) result
	cuDeviceGetAttribute      func(v *int32, attr int32, dev int32) result
	cuDevicePrimaryCtxRetain  func(ctx *uintptr, dev int32) result
	cuCtxSetCurrent           func(ctx uintptr) result
	cuStreamCreate            func(s *uintptr, flags uint32) result
	cuStreamSynchronize       func(s uintptr) result
	cuModuleLoadData          func(m *uintptr, image *byte) result
	cuModuleGetFunction       func(f *uintptr, m uintptr, name string) result
	cuFuncGetAttribute        func(v *int32, attr int32, f uintptr) result
	cuLaunchKernel            func(f uintptr, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, params *unsafe.Pointer, extra uintptr) result
	cuMemAllocAsync           func(p *uint64, n uintptr, stream uintptr) result
	cuMemFreeAsync            func(p uint64, stream uintptr) result
	cuMemAlloc                func(p *uint64, n uintptr) result
	cuMemFree                 func(p uint64) result
	cuMemcpyHtoDAsync         func(dst uint64, src unsafe.Pointer, n uintptr, stream uintptr) result
	cuMemcpyDtoHAsync         func(dst unsafe.Pointer, src uint64, n uintptr, stream uintptr) result
	cuMemHostAlloc            func(p *unsafe.Pointer, n uintptr, flags uint32) result
	cuMemHostGetDevicePointer func(p *uint64, host unsafe.Pointer, flags uint32) result
	cuDeviceGetDefaultMemPool func(pool *uintptr, dev int32) result
	cuMemPoolSetAttribute     func(pool uintptr, attr int32, value unsafe.Pointer) result
)

// loadDriver loads libcuda and initializes it once.
func loadDriver() error {
	driverOnce.Do(func() {
		lib, err := purego.Dlopen("libcuda.so.1", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			driverErr = err
			return
		}
		driverLib = lib
		purego.RegisterLibFunc(&cuInit, lib, "cuInit")
		purego.RegisterLibFunc(&cuGetErrorName, lib, "cuGetErrorName")
		purego.RegisterLibFunc(&cuDeviceGetCount, lib, "cuDeviceGetCount")
		purego.RegisterLibFunc(&cuDeviceGet, lib, "cuDeviceGet")
		purego.RegisterLibFunc(&cuDeviceGetName, lib, "cuDeviceGetName")
		purego.RegisterLibFunc(&cuDeviceGetAttribute, lib, "cuDeviceGetAttribute")
		purego.RegisterLibFunc(&cuDevicePrimaryCtxRetain, lib, "cuDevicePrimaryCtxRetain")
		purego.RegisterLibFunc(&cuCtxSetCurrent, lib, "cuCtxSetCurrent")
		purego.RegisterLibFunc(&cuStreamCreate, lib, "cuStreamCreate")
		purego.RegisterLibFunc(&cuStreamSynchronize, lib, "cuStreamSynchronize")
		purego.RegisterLibFunc(&cuModuleLoadData, lib, "cuModuleLoadData")
		purego.RegisterLibFunc(&cuModuleGetFunction, lib, "cuModuleGetFunction")
		purego.RegisterLibFunc(&cuFuncGetAttribute, lib, "cuFuncGetAttribute")
		purego.RegisterLibFunc(&cuLaunchKernel, lib, "cuLaunchKernel")
		purego.RegisterLibFunc(&cuMemAllocAsync, lib, "cuMemAllocAsync")
		purego.RegisterLibFunc(&cuMemFreeAsync, lib, "cuMemFreeAsync")
		purego.RegisterLibFunc(&cuMemAlloc, lib, "cuMemAlloc_v2")
		purego.RegisterLibFunc(&cuMemFree, lib, "cuMemFree_v2")
		purego.RegisterLibFunc(&cuMemcpyHtoDAsync, lib, "cuMemcpyHtoDAsync_v2")
		purego.RegisterLibFunc(&cuMemcpyDtoHAsync, lib, "cuMemcpyDtoHAsync_v2")
		purego.RegisterLibFunc(&cuMemHostAlloc, lib, "cuMemHostAlloc")
		purego.RegisterLibFunc(&cuMemHostGetDevicePointer, lib, "cuMemHostGetDevicePointer_v2")
		purego.RegisterLibFunc(&cuDeviceGetDefaultMemPool, lib, "cuDeviceGetDefaultMemPool")
		purego.RegisterLibFunc(&cuMemPoolSetAttribute, lib, "cuMemPoolSetAttribute")
		if r := cuInit(0); r != cudaSuccess {
			driverErr = fmt.Errorf("cuInit: %s", errName(r))
		}
	})
	return driverErr
}

// CUDA graph constants.
const (
	captureModeThreadLocal = 1 // CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
	captureStatusActive    = 1 // CU_STREAM_CAPTURE_STATUS_ACTIVE
)

// kernelNodeParams is a CUDA_KERNEL_NODE_PARAMS_v2.
type kernelNodeParams struct {
	fn                     uintptr
	gridX, gridY, gridZ    uint32
	blockX, blockY, blockZ uint32
	_                      uint32 // sharedMemBytes
	params                 *unsafe.Pointer
	_                      uintptr    // extra
	_                      [2]uintptr // kern, ctx
}

// Loaded functions of the driver's CUDA graph API, which a driver for
// CUDA 12 has. Kernels launch one by one if any is missing.
var (
	graphOnce sync.Once
	graphErr  error

	cuStreamBeginCapture           func(s uintptr, mode int32) result
	cuStreamEndCapture             func(s uintptr, graph *uintptr) result
	cuStreamGetCaptureInfo         func(s uintptr, status *int32, id *uint64, graph *uintptr, deps **uintptr, n *uintptr) result
	cuGraphInstantiate             func(exec *uintptr, graph uintptr, flags uint64) result
	cuGraphExecKernelNodeSetParams func(exec, node uintptr, params *kernelNodeParams) result
	cuGraphLaunch                  func(exec, s uintptr) result
	cuGraphExecDestroy             func(exec uintptr) result
	cuGraphDestroy                 func(graph uintptr) result
)

// loadGraphs loads the CUDA graph API once. loadDriver must have
// succeeded.
func loadGraphs() error {
	graphOnce.Do(func() {
		for _, f := range []struct {
			fptr any
			name string
		}{
			{&cuStreamBeginCapture, "cuStreamBeginCapture_v2"},
			{&cuStreamEndCapture, "cuStreamEndCapture"},
			{&cuStreamGetCaptureInfo, "cuStreamGetCaptureInfo_v2"},
			{&cuGraphInstantiate, "cuGraphInstantiateWithFlags"},
			{&cuGraphExecKernelNodeSetParams, "cuGraphExecKernelNodeSetParams_v2"},
			{&cuGraphLaunch, "cuGraphLaunch"},
			{&cuGraphExecDestroy, "cuGraphExecDestroy"},
			{&cuGraphDestroy, "cuGraphDestroy"},
		} {
			sym, err := purego.Dlsym(driverLib, f.name)
			if err != nil {
				graphErr = err
				return
			}
			purego.RegisterFunc(f.fptr, sym)
		}
	})
	return graphErr
}

// errName returns the name of a driver result, such as
// CUDA_ERROR_OUT_OF_MEMORY.
func errName(r result) string {
	var name *byte
	if cuGetErrorName(r, &name) != cudaSuccess || name == nil {
		return fmt.Sprintf("CUDA error %d", r)
	}
	return goString(name)
}

// goString returns the NUL-terminated string at p.
func goString(p *byte) string {
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

// nvrtcResult is an nvrtcResult.
type nvrtcResult uint32

const nvrtcSuccess nvrtcResult = 0

// Loaded functions of NVRTC, named as in nvrtc.h.
var (
	nvrtcOnce sync.Once
	nvrtcErr  error

	nvrtcVersion              func(major, minor *int32) nvrtcResult
	nvrtcGetNumSupportedArchs func(n *int32) nvrtcResult
	nvrtcGetSupportedArchs    func(archs *int32) nvrtcResult
	nvrtcGetErrorString       func(r nvrtcResult) string
	nvrtcCreateProgram        func(prog *uintptr, src, name string, headers int32, sources, names uintptr) nvrtcResult
	nvrtcDestroyProgram       func(prog *uintptr) nvrtcResult
	nvrtcCompileProgram       func(prog uintptr, n int32, options **byte) nvrtcResult
	nvrtcGetProgramLogSize    func(prog uintptr, n *uintptr) nvrtcResult
	nvrtcGetProgramLog        func(prog uintptr, log *byte) nvrtcResult
	nvrtcGetPTXSize           func(prog uintptr, n *uintptr) nvrtcResult
	nvrtcGetPTX               func(prog uintptr, ptx *byte) nvrtcResult
	nvrtcGetCUBINSize         func(prog uintptr, n *uintptr) nvrtcResult
	nvrtcGetCUBIN             func(prog uintptr, cubin *byte) nvrtcResult
)

// loadNVRTC loads libnvrtc once. It needs no device, so kernels compile
// on machines without one.
func loadNVRTC() error {
	nvrtcOnce.Do(func() {
		lib, err := purego.Dlopen("libnvrtc.so.12", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			nvrtcErr = err
			return
		}
		purego.RegisterLibFunc(&nvrtcVersion, lib, "nvrtcVersion")
		purego.RegisterLibFunc(&nvrtcGetNumSupportedArchs, lib, "nvrtcGetNumSupportedArchs")
		purego.RegisterLibFunc(&nvrtcGetSupportedArchs, lib, "nvrtcGetSupportedArchs")
		purego.RegisterLibFunc(&nvrtcGetErrorString, lib, "nvrtcGetErrorString")
		purego.RegisterLibFunc(&nvrtcCreateProgram, lib, "nvrtcCreateProgram")
		purego.RegisterLibFunc(&nvrtcDestroyProgram, lib, "nvrtcDestroyProgram")
		purego.RegisterLibFunc(&nvrtcCompileProgram, lib, "nvrtcCompileProgram")
		purego.RegisterLibFunc(&nvrtcGetProgramLogSize, lib, "nvrtcGetProgramLogSize")
		purego.RegisterLibFunc(&nvrtcGetProgramLog, lib, "nvrtcGetProgramLog")
		purego.RegisterLibFunc(&nvrtcGetPTXSize, lib, "nvrtcGetPTXSize")
		purego.RegisterLibFunc(&nvrtcGetPTX, lib, "nvrtcGetPTX")
		purego.RegisterLibFunc(&nvrtcGetCUBINSize, lib, "nvrtcGetCUBINSize")
		purego.RegisterLibFunc(&nvrtcGetCUBIN, lib, "nvrtcGetCUBIN")
	})
	return nvrtcErr
}

// blasStatus is a cublasStatus_t.
type blasStatus uint32

const (
	blasSuccess blasStatus = 0

	blasOpN int32 = 0 // CUBLAS_OP_N
	blasOpT int32 = 1 // CUBLAS_OP_T
)

// Loaded functions of cuBLAS, named as in cublas_v2.h.
var (
	blasOnce sync.Once
	blasErr  error

	cublasCreate              func(h *uintptr) blasStatus
	cublasSetStream           func(h, stream uintptr) blasStatus
	cublasSgemmStridedBatched func(h uintptr, transA, transB, m, n, k int32, alpha *float32,
		a uint64, lda int32, strideA int64, b uint64, ldb int32, strideB int64,
		beta *float32, c uint64, ldc int32, strideC int64, batch int32) blasStatus
)

// loadBLAS loads libcublas once.
func loadBLAS() error {
	blasOnce.Do(func() {
		lib, err := purego.Dlopen("libcublas.so.12", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			blasErr = err
			return
		}
		purego.RegisterLibFunc(&cublasCreate, lib, "cublasCreate_v2")
		purego.RegisterLibFunc(&cublasSetStream, lib, "cublasSetStream_v2")
		purego.RegisterLibFunc(&cublasSgemmStridedBatched, lib, "cublasSgemmStridedBatched")
	})
	return blasErr
}
