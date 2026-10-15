package cpu

import (
	"context"

	"github.com/gsoultan/mimir/backend"
)

// matmul computes [..., M, K] x [..., K, N] -> [..., M, N]. Batch dimensions
// are equal in shape but may have any strides, including 0 for broadcast.
// Float32 products run on the packed GEMM in gemm.go. Int32 products split
// work by output rows; each row accumulates in i-k-j order so the inner
// loop streams a row of B and a row of the output.
func matmul[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, x, y []T) error {
	if len(out) == 0 {
		return nil
	}
	vx, vy := k.In[0], k.In[1]
	r := len(k.Shape)
	m, n, kk := k.Shape[r-2], k.Shape[r-1], vx.Shape[r-1]
	batch := k.Shape[:r-2]
	offX := batchOffsets(batch, vx)
	offY := batchOffsets(batch, vy)
	sxm, sxk := vx.Strides[r-2], vx.Strides[r-1]
	syk, syn := vy.Strides[r-2], vy.Strides[r-1]
	if out32, ok := any(out).([]float32); ok {
		a := operand{data: any(x).([]float32), rs: sxm, cs: sxk}
		bm := operand{data: any(y).([]float32), rs: syk, cs: syn}
		return b.sgemm(ctx, gemmKernels[0], out32, a, offX, bm, offY, m, n, kk)
	}

	grain := max(1, elementGrain/max(1, kk*n))
	return b.parallelFor(ctx, len(offX)*m, grain, func(lo, hi int) error {
		for row := lo; row < hi; row++ {
			bi, i := row/m, row%m
			dst := out[row*n : row*n+n]
			clear(dst)
			xOff := offX[bi] + i*sxm
			for p := range kk {
				a := x[xOff+p*sxk]
				yOff := offY[bi] + p*syk
				if syn == 1 {
					for j, v := range y[yOff : yOff+n] {
						dst[j] += a * v
					}
					continue
				}
				for j := range dst {
					dst[j] += a * y[yOff+j*syn]
				}
			}
		}
		return nil
	})
}

// batchOffsets returns the buffer offset of each matrix in the batch
// dimensions of v, in row-major batch order.
func batchOffsets(batch []int, v backend.View) []int {
	n, _ := backend.NumElements(batch)
	offs := make([]int, n)
	c := newCursor(batch, []backend.View{{Shape: batch, Strides: v.Strides[:len(batch)], Offset: v.Offset}}, 0)
	for i := range offs {
		offs[i] = c.offs[0]
		c.next()
	}
	return offs
}
