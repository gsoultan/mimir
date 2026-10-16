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

import (
	"errors"

	"github.com/gsoultan/mimir/backend/internal/gen"
)

// ErrUnavailable reports that no CUDA device can be used.
var ErrUnavailable = errors.New("cuda unavailable")

// sgemm is a call of cublasSgemmStridedBatched: batch products of an m ×
// k and a k × n matrix, each read in column-major order and transposed if
// the operand says, into m × n column-major results ldc elements per
// column and strideC elements apart.
//
// cuBLAS reads matrices in column-major order, which is row-major order
// transposed, so the backend computes the transpose of the row-major
// result C = A·B as Cᵀ = Bᵀ·Aᵀ: cuBLAS's first operand is the product's
// right one, input 1, and its second the left one, input 0.
type sgemm struct {
	m, n, k      int
	a, b         operand
	ldc, strideC int
	batch        int
}

// operand is how cuBLAS reads one operand: the matrices of input in,
// column-major with ld elements per column and stride elements apart,
// from element offset, transposed if trans is set.
type operand struct {
	in                 int
	offset, ld, stride int
	trans              bool
}

// planSgemm returns the cuBLAS call that computes the product g.
func planSgemm(g gen.GEMM) sgemm {
	// A row-major matrix stored with ld elements per row is, read in
	// column-major order with ld elements per column, its transpose. So
	// the right operand as stored is Bᵀ, which cuBLAS reads as it is,
	// unless the operand is stored transposed: then it is B, and cuBLAS
	// transposes it. Likewise the left one.
	return sgemm{
		m: g.N, n: g.M, k: g.K,
		a:       operand{in: 1, offset: g.B.Offset, ld: g.B.LD, stride: g.B.Stride, trans: g.B.Trans},
		b:       operand{in: 0, offset: g.A.Offset, ld: g.A.LD, stride: g.A.Stride, trans: g.A.Trans},
		ldc:     g.N,
		strideC: g.ResultSize,
		batch:   g.Batch,
	}
}

// at returns the element of the buffer that holds element (i, j) of
// matrix b of the operand o stands for, as cuBLAS reads it.
func (o operand) at(b, i, j int) int {
	if o.trans {
		i, j = j, i
	}
	return o.offset + b*o.stride + j*o.ld + i
}
