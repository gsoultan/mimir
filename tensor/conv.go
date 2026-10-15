package tensor

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/internal/graph"
)

// ConvOptions configures Conv2d, MaxPool2d and AvgPool2d. Zero fields take
// their defaults: stride 1 (the kernel size for pooling), no padding,
// dilation 1 and one group.
type ConvOptions struct {
	Stride   int
	Padding  int // added on every side of both spatial dimensions
	Dilation int // spacing between kernel elements
	Groups   int // Conv2d only: channels are split into this many groups
}

func (o ConvOptions) resolve(stride int) (ConvOptions, error) {
	if o.Stride < 0 || o.Padding < 0 || o.Dilation < 0 || o.Groups < 0 {
		return o, fmt.Errorf("tensor: negative convolution option in %+v: %w", o, ErrShape)
	}
	if o.Stride == 0 {
		o.Stride = stride
	}
	o.Dilation = max(o.Dilation, 1)
	o.Groups = max(o.Groups, 1)
	return o, nil
}

// window returns the k×k window over the spatial dimensions of an
// [N, C, H, W] tensor that o describes.
func (o ConvOptions) window(k int) graph.Window {
	p := [2]int{o.Padding, o.Padding}
	return graph.Window{
		Kernel:   []int{k, k},
		Stride:   []int{o.Stride, o.Stride},
		Dilation: []int{o.Dilation, o.Dilation},
		Pads:     [][2]int{p, p},
	}
}

// windows returns the k×k sliding windows over the spatial dimensions of
// an [N, C, H, W] tensor as [N, C, K, K, OH, OW], padding with pad.
func (t *Tensor) windows(k int, o ConvOptions, pad float64) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if len(t.node.Shape) != 4 {
		return fail(fmt.Errorf("tensor: %d×%d windows of %v, want [N, C, H, W]: %w", k, k, t.node.Shape, ErrShape))
	}
	return t.unfold(o.window(k), pad)
}

// unfold returns the sliding windows w over the trailing dimensions of t,
// filling the padding with value: [B..., D...] becomes [B..., K..., O...].
func (t *Tensor) unfold(w graph.Window, value float64) *Tensor {
	out := wrap(graph.Unfold(t.node, w, value))
	if !tracked(t) || out.err != nil {
		return out
	}
	spatial := t.node.Shape[len(t.node.Shape)-len(w.Kernel):]
	return track(out, func(g *Tensor) []*Tensor { return grads(g.fold(spatial, w)) }, t)
}

// fold sums the windows w in t, shaped [B..., K..., O...], into a
// [B..., spatial...] tensor: the adjoint of unfold.
func (t *Tensor) fold(spatial []int, w graph.Window) *Tensor {
	out := wrap(graph.Fold(t.node, spatial, w))
	if !tracked(t) || out.err != nil {
		return out
	}
	return track(out, func(g *Tensor) []*Tensor { return grads(g.unfold(w, 0)) }, t)
}

// Conv2d returns the 2-D convolution (cross-correlation, like
// torch.nn.functional.conv2d) of t, shaped [N, C, H, W], with the kernel w,
// shaped [Cout, C/Groups, KH, KW] with KH == KW. b is the [Cout] bias, or
// nil for none.
func (t *Tensor) Conv2d(w, b *Tensor, opts ConvOptions) *Tensor {
	if err := firstErr(t, w); err != nil {
		return fail(err)
	}
	o, err := opts.resolve(1)
	if err != nil {
		return fail(err)
	}
	xs, ws := t.node.Shape, w.node.Shape
	if len(xs) != 4 || len(ws) != 4 || ws[2] != ws[3] || ws[0]%o.Groups != 0 || ws[1]*o.Groups != xs[1] {
		return fail(fmt.Errorf("tensor: conv2d of %v with kernel %v in %d groups: %w", xs, ws, o.Groups, ErrShape))
	}
	n, g, cout, k := xs[0], o.Groups, ws[0], ws[2]
	win := t.windows(k, o, 0) // [N, C, K, K, OH, OW]
	if win.err != nil {
		return win
	}
	oh, ow := win.node.Shape[4], win.node.Shape[5]
	cols := win.Reshape(n, g, xs[1]/g*k*k, oh*ow)
	out := w.Reshape(g, cout/g, ws[1]*k*k).MatMul(cols).Reshape(n, cout, oh, ow)
	if b == nil {
		return out
	}
	if err := firstErr(b); err != nil {
		return fail(err)
	}
	if b.Rank() != 1 || b.node.Shape[0] != cout {
		return fail(fmt.Errorf("tensor: conv2d bias %v for %d channels: %w", b.node.Shape, cout, ErrShape))
	}
	return out.Add(b.Reshape(cout, 1, 1))
}

// MaxPool2d returns the maximum of each k×k window of t, shaped
// [N, C, H, W], like torch.nn.functional.max_pool2d. The stride defaults
// to k, and padding counts as negative infinity.
func (t *Tensor) MaxPool2d(k int, opts ConvOptions) *Tensor {
	o, err := opts.resolve(k)
	if err != nil {
		return fail(err)
	}
	if o.Padding > k/2 {
		return fail(fmt.Errorf("tensor: max pool padding %d for kernel %d: %w", o.Padding, k, ErrShape))
	}
	return t.windows(k, o, math.Inf(-1)).Max(2, 3)
}

// AvgPool2d returns the mean of each k×k window of t, shaped [N, C, H, W],
// like torch.nn.functional.avg_pool2d. The stride defaults to k, and
// padding counts as zeros in the mean.
func (t *Tensor) AvgPool2d(k int, opts ConvOptions) *Tensor {
	o, err := opts.resolve(k)
	if err != nil {
		return fail(err)
	}
	if o.Padding > k/2 {
		return fail(fmt.Errorf("tensor: average pool padding %d for kernel %d: %w", o.Padding, k, ErrShape))
	}
	return t.windows(k, o, 0).Mean(2, 3)
}
