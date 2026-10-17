package sched

import (
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// On a backend that runs convolution kernels (see backend.Convolver), a
// convolution runs as one of them instead of as the unfold, product and
// fold that the tensor package builds it from, so that its windows never
// take a buffer:
//
//   - a product of a matrix and the windows of an unfold, reshaped to
//     [N, G, C*K, O], runs as OpConv, with the product's epilogue, if it
//     has one: the forward pass;
//   - such a product with the windows transposed, summed over the batch,
//     runs as OpConvWeight: the gradient of the weights. When the backend
//     asks for it, the kernel sums parts of the batch separately into a
//     buffer of its own and a second kernel adds the parts together;
//   - a fold of a product that nothing else reads, reshaped to the
//     windows, runs as OpConvInput: the gradient of the image.
//
// The windows are read where the unfold reads them, from its input, by
// every product of them, whether or not the unfold is shared, as the
// forward pass and the weight gradient share it. An unfold that nothing
// else reads then needs no kernel.

// conv is how a planned node runs as a convolution kernel.
type conv struct {
	op     backend.Op
	window backend.Attrs // the window, without a fill value
	parts  int           // OpConvWeight: the parts it sums separately, if more than 1
}

// windowAttrs returns the window of the unfold or fold n.
func windowAttrs(n *graph.Node) backend.Attrs {
	a := n.Attrs
	return backend.Attrs{Kernel: a.Kernel, Stride: a.Stride, Dilation: a.Dilation, Pads: a.Pads}
}

// windowsOf returns the unfold whose windows x is, through unrealized
// view nodes, if its fill value is 0, and the view of the unfold's
// output in row-major order that x is.
func windowsOf(x *graph.Node) (*graph.Node, backend.View, bool) {
	var views []*graph.Node
	n := x
	for ; !n.Realized() && n.Op.IsView(); n = n.Inputs[0] {
		views = append(views, n)
	}
	if n.Realized() || n.Op != backend.OpUnfold || n.Attrs.Value != 0 || n.DType != backend.Float32 {
		return nil, backend.View{}, false
	}
	v := backend.Contiguous(n.Shape)
	for _, view := range slices.Backward(views) {
		var ok bool
		if v, ok = viewOf(v, view); !ok {
			return nil, backend.View{}, false
		}
	}
	return n, v, true
}

// sameView reports whether a and b read the same elements in the same
// order.
func sameView(a, b backend.View) bool {
	if !slices.Equal(a.Shape, b.Shape) || a.Offset != b.Offset {
		return false
	}
	for d, size := range a.Shape {
		if size > 1 && a.Strides[d] != b.Strides[d] {
			return false
		}
	}
	return true
}

// convWindows returns the unfold of an image [N, G*C, D...] whose
// windows x reads as the [N, G, C*K, O] matrices of G groups, transposed
// if transposed is set, which the product mm, of rank 4, multiplies.
func convWindows(mm, x *graph.Node, transposed bool) *graph.Node {
	u, v, ok := windowsOf(x)
	if !ok || len(mm.Shape) != 4 {
		return nil
	}
	s := len(u.Attrs.Kernel)
	image := u.Inputs[0].Shape
	g := mm.Shape[1]
	if len(image) != 2+s || g == 0 || image[1]%g != 0 {
		return nil
	}
	k, _ := backend.NumElements(u.Shape[2 : 2+s])
	o, _ := backend.NumElements(u.Shape[2+s:])
	want := backend.Contiguous([]int{image[0], g, image[1] / g * k, o})
	if transposed {
		want = want.Permute([]int{0, 1, 3, 2})
	}
	if !sameView(v, want) {
		return nil
	}
	return u
}

// convolve rewrites w, the work of a product, gemm, or of a fold of one,
// n, to run as a convolution kernel, if it is one. It reports whether it
// did.
func (p *plan) convolve(n *graph.Node, w *work, private func(*graph.Node) bool) bool {
	cv, ok := p.be.(backend.Convolver)
	if !ok {
		return false
	}
	switch {
	case w.gemm != nil:
		// A product, or one summed over batch axes, which summedProduct
		// has checked are those of a private matmul.
		mm, summed := w.gemm, w.gemm.Op == backend.OpReduceSum
		if summed {
			mm = mm.Inputs[0]
		} else if u := convWindows(mm, mm.Inputs[1], false); u != nil {
			if windowed(cv, backend.OpConv, mm.Shape, windowAttrs(u), mm.Inputs[0].Shape, u.Inputs[0].Shape) {
				return false
			}
			w.conv = &conv{op: backend.OpConv, window: windowAttrs(u)}
			w.deps[1] = u.Inputs[0]
			return true
		}
		// A weight gradient sums over the batch, unless it has one item.
		// Its kernel takes no epilogue: a product of the second order,
		// such as the derivative of a weight gradient, has one, and runs
		// as a product.
		u := convWindows(mm, mm.Inputs[1], true)
		if u == nil || w.prog != nil || summed && !slices.Equal(w.gemm.Attrs.Axes, []int{0}) || !summed && mm.Shape[0] != 1 {
			return false
		}
		w.conv = &conv{op: backend.OpConvWeight, window: windowAttrs(u), parts: 1}
		w.deps[1] = u.Inputs[0]
		k := &backend.Kernel{Op: backend.OpConvWeight, DType: n.DType, Shape: n.Shape, Attrs: w.conv.window,
			In: []backend.View{backend.Contiguous(w.deps[0].Shape), backend.Contiguous(w.deps[1].Shape)}}
		if _, err := backend.ConvOf(k); err == nil {
			w.conv.parts = max(1, cv.ConvParts(k))
		}
		return true
	case n.Op == backend.OpFold && n.DType == backend.Float32:
		// The product must be private, as must the views between, or it
		// would be computed twice.
		var views []*graph.Node
		x := n.Inputs[0]
		for ; x.Op.IsView() && private(x); x = x.Inputs[0] {
			views = append(views, x)
		}
		if x.Op != backend.OpMatMul || !private(x) || len(x.Shape) != 4 {
			return false
		}
		v := backend.Contiguous(x.Shape)
		for _, view := range slices.Backward(views) {
			if v, ok = viewOf(v, view); !ok {
				return false
			}
		}
		s := len(n.Attrs.Kernel)
		image, g := n.Shape, x.Shape[1]
		if !sameView(v, backend.Contiguous(n.Inputs[0].Shape)) || len(image) != 2+s || g == 0 || image[1]%g != 0 ||
			x.Shape[0] != image[0] || x.Shape[2]*g != image[1]*windowSize(n) {
			return false
		}
		if windowed(cv, backend.OpConvInput, n.Shape, windowAttrs(n), x.Inputs[0].Shape, x.Inputs[1].Shape) {
			return false
		}
		w.deps = slices.Clone(x.Inputs)
		w.conv = &conv{op: backend.OpConvInput, window: windowAttrs(n)}
		return true
	}
	return false
}

// windowed reports whether the backend cv chooses to run the convolution
// kernel of op over the window a, from inputs of shapes x and y into
// shape, as the product of its windows (see backend.WindowChooser). It
// asks about the kernel of a layer, whose weights, input 0, every item
// shares.
func windowed(cv backend.Convolver, op backend.Op, shape []int, a backend.Attrs, x, y []int) bool {
	wc, ok := cv.(backend.WindowChooser)
	if !ok || len(x) == 0 {
		return false
	}
	one := slices.Clone(x)
	one[0] = 1
	vx := backend.Contiguous(one).Expand(x)
	k := &backend.Kernel{Op: op, DType: backend.Float32, Shape: shape, Attrs: a, In: []backend.View{vx, backend.Contiguous(y)}}
	if _, err := backend.ConvOf(k); err != nil {
		return false
	}
	return wc.Windowed(k)
}

// windowSize returns the elements of a window of the unfold or fold n.
func windowSize(n *graph.Node) int {
	k, _ := backend.NumElements(n.Attrs.Kernel)
	return k
}

// kernel returns the convolution kernel that w runs for n, reading the
// results of w.deps through views. For a weight gradient in parts, it
// computes the parts, which the kernel of sum adds together.
func (c *conv) kernel(n *graph.Node, w *work, views []backend.View) *backend.Kernel {
	k := &backend.Kernel{Op: c.op, DType: n.DType, Shape: n.Shape, In: slices.Clone(views), Attrs: c.window}
	if c.op != backend.OpConvInput {
		// A product's shape, which its epilogue, if any, reads in
		// row-major order.
		k.Shape = slices.Clone(w.gemm.Shape)
		k.Attrs.Program = w.prog
	}
	if c.parts > 1 {
		k.Shape[0] = c.parts
	}
	return k
}

// sum returns the kernel that adds the parts of the weight gradient n,
// which v reads.
func (c *conv) sum(n *graph.Node, v backend.View) *backend.Kernel {
	return &backend.Kernel{Op: backend.OpReduceSum, DType: n.DType, Shape: n.Shape, In: []backend.View{v}, Attrs: backend.Attrs{Axes: []int{0}}}
}
