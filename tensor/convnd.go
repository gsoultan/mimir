package tensor

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/internal/graph"
)

// windowN returns the window over s spatial dimensions with the kernel
// sizes k that o describes.
func (o ConvOptions) windowN(k []int) graph.Window {
	s := len(k)
	p := [2]int{o.Padding, o.Padding}
	return graph.Window{
		Kernel:   slices.Clone(k),
		Stride:   slices.Repeat([]int{o.Stride}, s),
		Dilation: slices.Repeat([]int{o.Dilation}, s),
		Pads:     slices.Repeat([][2]int{p}, s),
	}
}

// Conv1d returns the 1-D convolution of t, shaped [N, C, L], with the
// kernel w, shaped [Cout, C/Groups, K], like torch.nn.functional.conv1d.
// b is the [Cout] bias, or nil for none.
func (t *Tensor) Conv1d(w, b *Tensor, opts ConvOptions) *Tensor { return t.convN(1, w, b, opts) }

// Conv3d returns the 3-D convolution of t, shaped [N, C, D, H, W], with
// the kernel w, shaped [Cout, C/Groups, KD, KH, KW], like
// torch.nn.functional.conv3d. b is the [Cout] bias, or nil for none.
func (t *Tensor) Conv3d(w, b *Tensor, opts ConvOptions) *Tensor { return t.convN(3, w, b, opts) }

// convN is the convolution over s spatial dimensions.
func (t *Tensor) convN(s int, w, b *Tensor, opts ConvOptions) *Tensor {
	if err := firstErr(t, w); err != nil {
		return fail(err)
	}
	o, err := opts.resolve(1)
	if err != nil {
		return fail(err)
	}
	xs, ws := t.node.Shape, w.node.Shape
	if len(xs) != s+2 || len(ws) != s+2 || ws[0]%o.Groups != 0 || ws[1]*o.Groups != xs[1] {
		return fail(fmt.Errorf("tensor: conv%dd of %v with kernel %v in %d groups: %w", s, xs, ws, o.Groups, ErrShape))
	}
	n, g, cout, k := xs[0], o.Groups, ws[0], ws[2:]
	win := t.unfold(o.windowN(k), 0) // [N, C, K..., O...]
	if win.err != nil {
		return win
	}
	out := win.node.Shape[2+s:]
	cols := win.Reshape(n, g, xs[1]/g*prod(k), prod(out))
	y := w.Reshape(g, cout/g, ws[1]*prod(k)).MatMul(cols).Reshape(append([]int{n, cout}, out...)...)
	return addBias(y, b, s, "conv")
}

// addBias adds b, a [C] bias, to y, shaped [N, C, spatial...] with s
// spatial dimensions, or returns y when b is nil.
func addBias(y, b *Tensor, s int, what string) *Tensor {
	if b == nil {
		return y
	}
	if err := firstErr(y, b); err != nil {
		return fail(err)
	}
	c := y.node.Shape[1]
	if b.Rank() != 1 || b.node.Shape[0] != c {
		return fail(fmt.Errorf("tensor: %s bias %v for %d channels: %w", what, b.node.Shape, c, ErrShape))
	}
	return y.Add(b.Reshape(append([]int{c}, slices.Repeat([]int{1}, s)...)...))
}

// ConvTranspose1d returns the transposed 1-D convolution of t, shaped
// [N, C, L], with the kernel w, shaped [C, Cout/Groups, K], like
// torch.nn.functional.conv_transpose1d: the gradient of Conv1d with
// respect to its input. outPad adds to one side of the output, to tell
// apart the sizes a strided convolution maps to one.
func (t *Tensor) ConvTranspose1d(w, b *Tensor, opts ConvOptions, outPad int) *Tensor {
	return t.convTransposeN(1, w, b, opts, outPad)
}

// ConvTranspose2d is ConvTranspose1d for [N, C, H, W] inputs and
// [C, Cout/Groups, KH, KW] kernels, like
// torch.nn.functional.conv_transpose2d.
func (t *Tensor) ConvTranspose2d(w, b *Tensor, opts ConvOptions, outPad int) *Tensor {
	return t.convTransposeN(2, w, b, opts, outPad)
}

func (t *Tensor) convTransposeN(s int, w, b *Tensor, opts ConvOptions, outPad int) *Tensor {
	if err := firstErr(t, w); err != nil {
		return fail(err)
	}
	o, err := opts.resolve(1)
	if err != nil {
		return fail(err)
	}
	xs, ws := t.node.Shape, w.node.Shape
	if len(xs) != s+2 || len(ws) != s+2 || ws[0] != xs[1] || xs[1]%o.Groups != 0 || outPad < 0 || outPad >= o.Stride {
		return fail(fmt.Errorf("tensor: transposed conv%dd of %v with kernel %v in %d groups, output padding %d: %w", s, xs, ws, o.Groups, outPad, ErrShape))
	}
	n, g, cin, k := xs[0], o.Groups, xs[1], ws[2:]
	cout := ws[1] * g
	in := xs[2:]
	size := make([]int, s)
	for d := range s {
		size[d] = (in[d]-1)*o.Stride - 2*o.Padding + o.Dilation*(k[d]-1) + 1 + outPad
		if size[d] < 1 {
			return fail(fmt.Errorf("tensor: transposed conv%dd of %v with kernel %v makes no output: %w", s, xs, ws, ErrShape))
		}
	}
	// Each input position spreads its kernel's worth of outputs, which
	// fold sums where they overlap: [g, Cout/g·K, Cin/g] · [N, g, Cin/g, I].
	wt := w.Reshape(g, cin/g, ws[1]*prod(k)).Transpose(1, 2)
	cols := wt.MatMul(t.Reshape(n, g, cin/g, prod(in)))
	cols = cols.Reshape(append(append([]int{n, cout}, k...), in...)...)
	y := cols.fold(size, o.windowN(k))
	return addBias(y, b, s, "transposed conv")
}
