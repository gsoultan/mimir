// Package cuda implements the mimir backend for NVIDIA GPUs through CUDA,
// without cgo.
//
// It calls the CUDA driver, NVRTC and cuBLAS through purego. Every kernel
// runs as CUDA C++ source that the shared kernel generator writes and
// NVRTC compiles at runtime, once per kernel structure, except float32
// matrix products, which run on cuBLAS. Compiled kernels are cached on
// disk, under the user cache directory or MIMIR_CACHE_DIR. Kernels run in
// order on one stream, and device memory comes from the driver's
// stream-ordered memory pool, which keeps freed memory for the next
// allocation, so a training step allocates nothing from the driver once
// it has run.
//
// The backend needs an NVIDIA driver for CUDA 12 and the libraries
// libcuda.so.1, libnvrtc.so.12 and libcublas.so.12 where the dynamic
// loader finds them, as the nvidia/cuda runtime images and the CUDA
// toolkit install them. It runs on linux/amd64 and linux/arm64; elsewhere
// New reports ErrUnavailable.
//
// Loading those libraries needs the C library, so on Linux a program
// that imports this package, directly or through the tensor package, is
// linked dynamically even with CGO_ENABLED=0. The nocuda build tag leaves
// the backend out, which makes New report ErrUnavailable and the program
// a static binary again.
package cuda

import "errors"

// ErrUnavailable reports that no CUDA device can be used.
var ErrUnavailable = errors.New("cuda unavailable")
