package cpu

import (
	"context"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// The CPU runs the convolution kernels (see backend.Convolver), so that a
// convolution's windows never take a buffer of the scheduler's: 3×3
// convolutions of stride and dilation 1 run as Winograd's F(2×2, 3×3)
// (see winograd.go), those of stride 1 with few channels or positions as
// direct convolutions (see conv_direct.go), and the others as products of
// windows that the kernel copies out, or a product that it folds, in a
// buffer of its own.
var _ backend.Convolver = (*Backend)(nil)

// ConvParts returns 1: a weight gradient sums its parts itself, in a
// number of parts that depends only on its shape.
func (*Backend) ConvParts(*backend.Kernel) int { return 1 }

// convScratch holds the windows that a convolution kernel copies out.
var convScratch = sync.Pool{New: func() any { return new([]float32) }}

// runConv runs the convolution kernels OpConv, OpConvWeight and
// OpConvInput.
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
	if c.Parts > 1 {
		return convReference(ctx, b, k, c, out, a, y)
	}
	dc, isDirect := direct(k, c)
	if wc, ok := winograd(k, c); ok && !isDirect {
		return runWinograd(ctx, b, k, c, wc, outs, ins, out, a, y)
	}
	if isDirect {
		vx := k.In[1]
		x := winoImage{data: y, off: vx.Offset, sn: vx.Strides[0], sg: c.C * vx.Strides[1], sc: vx.Strides[1], sh: vx.Strides[2], sw: vx.Strides[3]}
		if k.Op == backend.OpConvWeight {
			return b.directWeightGrad(ctx, dc, x, k.In[0], a, out)
		}
		epi, err := epilogue(k, outs[0], ins)
		if err != nil {
			return err
		}
		return b.directForward(ctx, dc, x, k.In[0], a, out, epi)
	}
	rows, cols := c.Rows(), c.Cols()
	shape := slices.Concat([]int{c.N, c.G * c.C}, k.Attrs.Kernel, c.Positions)
	window := backend.Attrs{Kernel: k.Attrs.Kernel, Stride: k.Attrs.Stride, Dilation: k.Attrs.Dilation, Pads: k.Attrs.Pads}
	// win holds the windows, [N, G, C*K, O] row-major, or the product
	// that OpConvInput folds, of the same shape.
	scratch := convScratch.Get().(*[]float32)
	defer convScratch.Put(scratch)
	*scratch = grow(*scratch, c.N*c.G*rows*cols)
	win := *scratch
	windows := backend.Contiguous([]int{c.N, c.G, rows, cols})
	if k.Op != backend.OpConvInput {
		u := &backend.Kernel{Op: backend.OpUnfold, DType: backend.Float32, Shape: shape, In: k.In[1:2], Attrs: window}
		if err := unfold(ctx, b, u, win, y); err != nil {
			return err
		}
	}
	switch k.Op {
	case backend.OpConv:
		epi, err := epilogue(k, outs[0], ins)
		if err != nil {
			return err
		}
		mm := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: k.Shape, In: []backend.View{k.In[0], windows}}
		return matmul(ctx, b, mm, out, a, win, epi)
	case backend.OpConvWeight:
		// The product of the gradient and the transposed windows, summed
		// over the batch.
		mm := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: k.Shape,
			In: []backend.View{k.In[0], windows.Permute([]int{0, 1, 3, 2})}, Attrs: backend.Attrs{Axes: []int{0}}}
		return matmul(ctx, b, mm, out, a, win, nil)
	}
	// OpConvInput: the product, then its fold.
	mm := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: windows.Shape, In: k.In[:2]}
	if err := matmul(ctx, b, mm, win, a, y, nil); err != nil {
		return err
	}
	f := &backend.Kernel{Op: backend.OpFold, DType: backend.Float32, Shape: k.Shape, In: []backend.View{backend.Contiguous(shape)}, Attrs: window}
	return fold(ctx, b, f, out, win)
}

// winograd reports whether the convolution kernel k, of geometry c, runs
// as Winograd's F(2×2, 3×3), and returns the forward convolution it
// runs: the layer's for OpConv and OpConvWeight, and for OpConvInput the
// convolution of the output's gradient with the flipped weights, padded
// by 2 less the layer's padding. The layer must convolve 3×3 windows of
// stride and dilation 1, padded by at most 2, over a 2-D image, with
// weights that every item shares, and enough channels in and out to
// fill the micro-kernel's panels.
func winograd(k *backend.Kernel, c backend.Conv) (winoConv, bool) {
	a := k.Attrs
	if len(c.Image) != 2 || !slices.Equal(a.Kernel, []int{3, 3}) || !slices.Equal(a.Stride, []int{1, 1}) || !slices.Equal(a.Dilation, []int{1, 1}) {
		return winoConv{}, false
	}
	for _, p := range a.Pads {
		if p[0] > 2 || p[1] > 2 {
			return winoConv{}, false
		}
	}
	if k.Op != backend.OpConvWeight && c.N > 1 && k.In[0].Strides[0] != 0 {
		return winoConv{}, false // weights of every item's own
	}
	// Few channels leave most of a panel empty, and the windows' product
	// is then the faster.
	if c.C < 8 || c.M < 8 {
		return winoConv{}, false
	}
	wc := winoConv{n: c.N, g: c.G, cin: c.C, cout: c.M, h: c.Image[0], w: c.Image[1],
		oh: c.Positions[0], ow: c.Positions[1], pt: a.Pads[0][0], pl: a.Pads[1][0]}
	if k.Op == backend.OpConvInput {
		wc.cin, wc.cout = c.M, c.C
		wc.h, wc.w, wc.oh, wc.ow = c.Positions[0], c.Positions[1], c.Image[0], c.Image[1]
		wc.pt, wc.pl = 2-a.Pads[0][0], 2-a.Pads[1][0]
	}
	return wc, true
}

// runWinograd runs the convolution kernel k as the Winograd convolution
// wc (see winograd).
func runWinograd(ctx context.Context, b *Backend, k *backend.Kernel, c backend.Conv, wc winoConv, outs, ins []backend.Buffer, out, a, y []float32) error {
	va, vy := k.In[0], k.In[1]
	// The image [N, G*C, H, W] of OpConv and OpConvWeight.
	image := func(v backend.View, d []float32) winoImage {
		return winoImage{data: d, off: v.Offset, sn: v.Strides[0], sg: c.C * v.Strides[1], sc: v.Strides[1], sh: v.Strides[2], sw: v.Strides[3]}
	}
	// A product's [N, G, M, O] as an image [N, G*M, OH, OW].
	product := func(v backend.View, d []float32) winoImage {
		return winoImage{data: d, off: v.Offset, sn: v.Strides[0], sg: v.Strides[1], sc: v.Strides[2], sh: c.Positions[1] * v.Strides[3], sw: v.Strides[3]}
	}
	switch k.Op {
	case backend.OpConv:
		epi, err := epilogue(k, outs[0], ins)
		if err != nil {
			return err
		}
		// In[0] is the weights [N, G, M, C*9].
		s1, s2, s3 := va.Strides[1], va.Strides[2], va.Strides[3]
		w := winoFilters{data: a, off: va.Offset, sg: s1, so: s2, si: 9 * s3, sh: 3 * s3, sw: s3}
		return b.winoForward(ctx, wc, image(vy, y), w, out, epi)
	case backend.OpConvInput:
		// In[0] is the transposed weights [N, G, C*9, M], whose windows
		// the convolution of the gradient reads flipped.
		s1, s2, s3 := va.Strides[1], va.Strides[2], va.Strides[3]
		w := winoFilters{data: a, off: va.Offset + 8*s2, sg: s1, so: 9 * s2, si: s3, sh: -3 * s2, sw: -s2}
		return b.winoForward(ctx, wc, product(vy, y), w, out, nil)
	}
	return b.winoWeightGrad(ctx, wc, image(vy, y), product(va, a), out)
}

// convReference runs the convolution kernel k, of geometry c, with plain
// loops that sum in order, one output row per task, after copying out the
// windows, or before folding the product, as the scheduler's unfold,
// product and fold would. It runs weight gradients in several parts.
func convReference(ctx context.Context, b *Backend, k *backend.Kernel, c backend.Conv, out, a, y []float32) error {
	va := k.In[0]
	at := func(i0, i1, i2, i3 int) float32 {
		return a[va.Offset+i0*va.Strides[0]+i1*va.Strides[1]+i2*va.Strides[2]+i3*va.Strides[3]]
	}
	rows, cols := c.Rows(), c.Cols()
	win := make([]float32, c.N*c.G*rows*cols)
	shape := slices.Concat([]int{c.N, c.G * c.C}, k.Attrs.Kernel, c.Positions)
	window := backend.Attrs{Kernel: k.Attrs.Kernel, Stride: k.Attrs.Stride, Dilation: k.Attrs.Dilation, Pads: k.Attrs.Pads}
	u := &backend.Kernel{Op: backend.OpUnfold, DType: backend.Float32, Shape: shape, In: k.In[1:2], Attrs: window}
	if err := unfold(ctx, b, u, win, y); err != nil {
		return err
	}
	grain := max(1, elementGrain/max(rows*c.N*cols/c.Parts, 1))
	return b.parallelFor(ctx, c.Parts*c.G*c.M, grain, func(lo, hi int) error {
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
