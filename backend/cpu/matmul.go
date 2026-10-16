package cpu

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// runMatMul runs a matmul kernel whose operands have the output's dtype.
func runMatMul[T number](ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, ins []backend.Buffer, epi *evaluator) error {
	out, err := data[T](outBuf)
	if err != nil {
		return err
	}
	x, err := data[T](ins[0])
	if err != nil {
		return err
	}
	y, err := data[T](ins[1])
	if err != nil {
		return err
	}
	return matmul(ctx, b, k, out, x, y, epi)
}

// epilogue returns the evaluator of a matmul's epilogue program, or nil if
// it has none. The program reads the product as input 0 and ins[2:],
// through k.In[2:], as inputs 1 on. Their views share a shape with as many
// elements as the product, read in row-major order alongside it, so the
// program can rewrite the product in place.
func epilogue(k *backend.Kernel, out backend.Buffer, ins []backend.Buffer) (*evaluator, error) {
	if k.Attrs.Program == nil {
		return nil, nil
	}
	shape := k.Shape
	if len(k.In) > 2 {
		shape = k.In[2].Shape
	}
	views := []backend.View{backend.Contiguous(shape)}
	for _, v := range k.In[2:] {
		if !slices.Equal(v.Shape, shape) {
			return nil, fmt.Errorf("epilogue inputs of shapes %v and %v: %w", shape, v.Shape, backend.ErrInternal)
		}
		views = append(views, v)
	}
	if n, _ := backend.NumElements(shape); n != out.Len() {
		return nil, fmt.Errorf("epilogue of shape %v for %d elements: %w", shape, out.Len(), backend.ErrInternal)
	}
	return newEvaluator(shape, views, k.Attrs.Program, nil, []backend.Buffer{out}, append([]backend.Buffer{out}, ins[2:]...))
}

// matmul computes [..., M, K] x [..., K, N] -> [..., M, N], summing the
// products over the batch axes k.Attrs.Axes. Batch dimensions are equal in
// shape but may have any strides, including 0 for broadcast. Float32
// products run on the packed GEMM in gemm.go. Int32 products split work by
// output rows; each row accumulates in i-k-j order so the inner loop
// streams a row of B and a row of the output. If epi is not nil, it
// rewrites each part of the output once that part is computed, while it is
// still in cache.
func matmul[T number](ctx context.Context, b *Backend, k *backend.Kernel, out, x, y []T, epi *evaluator) error {
	if len(out) == 0 {
		return nil
	}
	vx, vy, sums := summed(k)
	r := len(k.Shape)
	m, n, kk := k.Shape[r-2], k.Shape[r-1], vx.Shape[r-1]
	batch := vx.Shape[:r-2]
	offX := batchOffsets(batch, vx)
	offY := batchOffsets(batch, vy)
	sxm, sxk := vx.Strides[r-2], vx.Strides[r-1]
	syk, syn := vy.Strides[r-2], vy.Strides[r-1]
	if out32, ok := any(out).([]float32); ok {
		a := operand{data: any(x).([]float32), rs: sxm, cs: sxk}
		bm := operand{data: any(y).([]float32), rs: syk, cs: syn}
		return b.sgemm(ctx, gemmKernels[0], out32, a, offX, bm, offY, m, n, kk, sums, epi)
	}

	grain := max(1, elementGrain/max(1, kk*n*sums))
	return b.parallelFor(ctx, len(out)/n, grain, func(lo, hi int) error {
		for row := lo; row < hi; row++ {
			o, i := row/m, row%m
			dst := out[row*n : row*n+n]
			clear(dst)
			for t := o * sums; t < (o+1)*sums; t++ {
				xOff := offX[t] + i*sxm
				for p := range kk {
					a := x[xOff+p*sxk]
					yOff := offY[t] + p*syk
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
		}
		return epi.apply(lo*n, hi*n)
	})
}

// summed returns the operand views of a matmul kernel with the batch axes
// that it sums over, k.Attrs.Axes, moved after the others, so that the
// products each output sums are consecutive in row-major batch order, and
// the number of products each output sums.
func summed(k *backend.Kernel) (vx, vy backend.View, sums int) {
	vx, vy = k.In[0], k.In[1]
	if len(k.Attrs.Axes) == 0 {
		return vx, vy, 1
	}
	r := len(vx.Shape)
	perm := make([]int, 0, r)
	for d := range r - 2 {
		if !slices.Contains(k.Attrs.Axes, d) {
			perm = append(perm, d)
		}
	}
	sums = 1
	for _, d := range k.Attrs.Axes {
		perm = append(perm, d)
		sums *= vx.Shape[d]
	}
	perm = append(perm, r-2, r-1)
	return vx.Permute(perm), vy.Permute(perm), sums
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
