// Package rocm implements the mimir backend for AMD GPUs through ROCm's
// HIP, without cgo.
//
// It calls the HIP runtime, hiprtc and hipBLAS through purego. Every
// kernel runs as HIP C++ source that the shared kernel generator writes,
// for the wavefront size of the device, and hiprtc compiles at runtime
// for the device's architecture, once per kernel structure, except
// float32 matrix products, which run on hipBLAS when it is installed.
// Compiled kernels are cached on disk, under the user cache directory or
// MIMIR_CACHE_DIR. Kernels run in order on one stream, and device memory
// comes from the runtime's stream-ordered memory pool, which keeps freed
// memory for the next allocation, so a training step allocates nothing
// from the runtime once it has run.
//
// The backend needs ROCm 5.4 or later: libamdhip64 and hiprtc, which
// ROCm 6 and later ship as libhiprtc, where the dynamic loader finds
// them, as the rocm/dev images and the ROCm packages install them, or
// where MIMIR_HIP and MIMIR_HIPRTC name them. libhipblas, or the library
// MIMIR_HIPBLAS names, is optional. It runs on linux/amd64 and
// linux/arm64; elsewhere New reports ErrUnavailable.
//
// Loading those libraries needs the C library, so on Linux a program
// that imports this package, directly or through the tensor package, is
// linked dynamically even with CGO_ENABLED=0. The norocm build tag leaves
// the backend out, which makes New report ErrUnavailable.
package rocm

import "errors"

// ErrUnavailable reports that no ROCm device can be used.
var ErrUnavailable = errors.New("rocm unavailable")
