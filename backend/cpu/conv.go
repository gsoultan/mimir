package cpu

import (
	"context"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// runConv runs the convolution kernels OpConv, OpConvWeight and
// OpConvInput. It copies out the windows, or folds the product, as the
// scheduler's unfold, product and fold would, and multiplies with plain
// loops that sum in order, one output row per task. The CPU does not ask
// the scheduler for these kernels: they are the reference that the GPU
// kernels are tested against.
func runConv(ctx context.Context, b *Backend, k *backend.Kernel, outs, ins []backend.Buffer) error {
	c, err := backend.ConvOf(k)
	if err != nil {
		return err
	}
	out, err := data[float32](outs[0])
	if err != nil {
		return err
	}
	a, err := data[float32](ins[0])
	if err != nil {
		return err
	}
	y, err := data[float32](ins[1])
	if err != nil {
		return err
	}
	va := k.In[0]
	at := func(i0, i1, i2, i3 int) float32 {
		return a[va.Offset+i0*va.Strides[0]+i1*va.Strides[1]+i2*va.Strides[2]+i3*va.Strides[3]]
	}
	rows, cols := c.Rows(), c.Cols()
	// win holds the windows, [N, G, C*K, O] row-major, or the product
	// that OpConvInput folds, of the same shape.
	win := make([]float32, c.N*c.G*rows*cols)
	shape := slices.Concat([]int{c.N, c.G * c.C}, k.Attrs.Kernel, c.Positions)
	window := backend.Attrs{Kernel: k.Attrs.Kernel, Stride: k.Attrs.Stride, Dilation: k.Attrs.Dilation, Pads: k.Attrs.Pads}
	if k.Op != backend.OpConvInput {
		u := &backend.Kernel{Op: backend.OpUnfold, DType: backend.Float32, Shape: shape, In: k.In[1:2], Attrs: window}
		if err := unfold(ctx, b, u, win, y); err != nil {
			return err
		}
	}
	grain := func(work int) int { return max(1, elementGrain/max(work, 1)) }
	switch k.Op {
	case backend.OpConv:
		epi, err := epilogue(k, outs[0], ins)
		if err != nil {
			return err
		}
		return b.parallelFor(ctx, c.N*c.G*c.M, grain(rows*cols), func(lo, hi int) error {
			for r := lo; r < hi; r++ {
				n, g, m := r/(c.G*c.M), r/c.M%c.G, r%c.M
				dst := out[r*cols:][:cols]
				clear(dst)
				for j := range rows {
					w := at(n, g, m, j)
					for o, v := range win[((n*c.G+g)*rows+j)*cols:][:cols] {
						dst[o] += w * v
					}
				}
			}
			return epi.apply(lo*cols, hi*cols)
		})
	case backend.OpConvWeight:
		return b.parallelFor(ctx, c.Parts*c.G*c.M, grain(rows*c.N*cols/c.Parts), func(lo, hi int) error {
			for r := lo; r < hi; r++ {
				s, g, m := r/(c.G*c.M), r/c.M%c.G, r%c.M
				p0, p1 := c.Part(s)
				dst := out[r*rows:][:rows]
				for j := range dst {
					var acc float32
					for p := p0; p < p1; p++ {
						n, o := p/cols, p%cols
						acc += at(n, g, m, o) * win[((n*c.G+g)*rows+j)*cols+o]
					}
					dst[j] = acc
				}
			}
			return nil
		})
	}
	// OpConvInput: the product, then its fold.
	vy := k.In[1]
	err = b.parallelFor(ctx, c.N*c.G*rows, grain(c.M*cols), func(lo, hi int) error {
		for r := lo; r < hi; r++ {
			n, g, j := r/(c.G*rows), r/rows%c.G, r%rows
			dst := win[r*cols:][:cols]
			clear(dst)
			for m := range c.M {
				w := at(n, g, j, m)
				base := vy.Offset + n*vy.Strides[0] + g*vy.Strides[1] + m*vy.Strides[2]
				for o := range dst {
					dst[o] += w * y[base+o*vy.Strides[3]]
				}
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	f := &backend.Kernel{Op: backend.OpFold, DType: backend.Float32, Shape: k.Shape, In: []backend.View{backend.Contiguous(shape)}, Attrs: window}
	return fold(ctx, b, f, out, win)
}
