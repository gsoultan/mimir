package tensor

import (
	"fmt"
	"math"
	"slices"

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
	if err := t.checkWindows(k, o); err != nil {
		return fail(err)
	}
	return t.unfold(o.window(k), pad)
}

// checkWindows reports whether t is an [N, C, H, W] tensor with k×k
// windows o.
func (t *Tensor) checkWindows(k int, o ConvOptions) error {
	if err := firstErr(t); err != nil {
		return err
	}
	if len(t.node.Shape) != 4 {
		return fmt.Errorf("tensor: %d×%d windows of %v, want [N, C, H, W]: %w", k, k, t.node.Shape, ErrShape)
	}
	_, err := graph.Unfold(t.node, o.window(k), 0)
	return err
}

// pool combines each k×k window of t, an [N, C, H, W] tensor, with reduce
// over the window's two axes, padding with pad.
func (t *Tensor) pool(k int, o ConvOptions, pad float64, reduce func(t *Tensor, axes ...int) *Tensor) *Tensor {
	if err := t.checkWindows(k, o); err != nil {
		return fail(err)
	}
	if w := o.window(k); tiles(w) {
		// Combine the windows where they lie: [N, C, OH, K, OW, K].
		return reduce(t.tiles(w), 3, 5)
	}
	return reduce(t.unfold(o.window(k), pad), 2, 3)
}

// unfold returns the sliding windows w over the trailing dimensions of t,
// filling the padding with value: [B..., D...] becomes [B..., K..., O...].
func (t *Tensor) unfold(w graph.Window, value float64) *Tensor {
	out := wrap(graph.Unfold(t.node, w, value))
	if out.err == nil && tiles(w) {
		// [B..., O0, K0, O1, K1, ...] -> [B..., K..., O...]
		s := len(w.Kernel)
		lead := len(t.node.Shape) - s
		perm := make([]int, 0, lead+2*s)
		for d := range lead {
			perm = append(perm, d)
		}
		for d := range s {
			perm = append(perm, lead+2*d+1)
		}
		for d := range s {
			perm = append(perm, lead+2*d)
		}
		return t.tiles(w).permute(perm)
	}
	if !tracked(t) || out.err != nil {
		return out
	}
	spatial := t.node.Shape[len(t.node.Shape)-len(w.Kernel):]
	return track(out, func(g *Tensor) []*Tensor { return grads(g.fold(spatial, w)) }, t)
}

// tiles reports whether the valid windows w are apart, unpadded and
// undilated, as those of pooling with the stride its kernel size, or of a
// 1×1 convolution, so they can be read as a view of their input.
func tiles(w graph.Window) bool {
	for d, k := range w.Kernel {
		if w.Pads[d] != [2]int{} || k > w.Stride[d] || k > 1 && w.Dilation[d] > 1 {
			return false
		}
	}
	return true
}

// tiles returns the windows w of t, which tiles(w) accepts, as a view
// shaped [B..., O0, K0, O1, K1, ...]: each windowed dimension is cut into
// chunks of the stride, whose first kernel-size elements are a window.
func (t *Tensor) tiles(w graph.Window) *Tensor {
	lead := len(t.node.Shape) - len(w.Kernel)
	cut := slices.Clone(t.node.Shape)           // each windowed dimension a whole number of chunks
	chunks := slices.Clone(t.node.Shape[:lead]) // [B..., O0, S0, O1, S1, ...]
	window := slices.Clone(chunks)              // [B..., O0, K0, O1, K1, ...]
	for d, k := range w.Kernel {
		st := w.Stride[d]
		o := (t.node.Shape[lead+d]-k)/st + 1
		cut[lead+d] = o * st
		chunks = append(chunks, o, st)
		window = append(window, o, k)
	}
	// Padding the last chunk adds elements no window reads: it ends within
	// the dimension.
	pads, keep := make([][2]int, len(cut)), whole(t.node.Shape)
	for d, n := range cut {
		pads[d][1] = max(n-t.node.Shape[d], 0)
		keep[d][1] = min(n, t.node.Shape[d])
	}
	x := t
	if !slices.Equal(keep, whole(t.node.Shape)) {
		x = x.shrink(keep)
	}
	if !slices.Equal(x.node.Shape, cut) {
		x = x.pad(pads, 0)
	}
	x = x.reshape(chunks)
	if !slices.Equal(window, chunks) {
		x = x.shrink(whole(window))
	}
	return x
}

// whole returns shrink ranges that keep every element of shape.
func whole(shape []int) [][2]int {
	ranges := make([][2]int, len(shape))
	for i, d := range shape {
		ranges[i] = [2]int{0, d}
	}
	return ranges
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
	return t.pool(k, o, math.Inf(-1), (*Tensor).Max)
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
	return t.pool(k, o, 0, (*Tensor).Mean)
}
