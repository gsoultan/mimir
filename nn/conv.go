package nn

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Conv1d is a 1-D convolution over [N, C, L] inputs, like torch.nn.Conv1d,
// for sequences such as audio or sensor readings.
type Conv1d struct {
	Weight  *tensor.Param // [out, in/groups, k]
	Bias    *tensor.Param // [out], or nil
	Options tensor.ConvOptions
}

// NewConv1d returns a convolution from in to out channels with a kernel
// of k. Stride, Padding, Dilation, Groups and NoBias configure it.
func NewConv1d(in, out, k int, opts ...Option) *Conv1d {
	o := resolve(opts, options{})
	w, b := convParams(o, in, out, k)
	return &Conv1d{Weight: w, Bias: b, Options: o.conv}
}

// convParams draws a convolution's weight [out, in/groups, k...] and bias
// from U(-1/√fan_in, 1/√fan_in), PyTorch's default.
func convParams(o options, in, out int, k ...int) (w, b *tensor.Param) {
	g := max(o.conv.Groups, 1)
	fan := in / g
	for _, d := range k {
		fan *= d
	}
	bound := 1 / math.Sqrt(float64(max(fan, 1)))
	w = uniform(o, bound, append([]int{out, in / g}, k...)...)
	if !o.noBias {
		b = uniform(o, bound, out)
	}
	return w, b
}

func value(p *tensor.Param) *tensor.Tensor {
	if p == nil {
		return nil
	}
	return p.Value()
}

// Forward maps [N, in, L] to [N, out, L'].
func (c *Conv1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Conv1d(c.Weight.Value(), value(c.Bias), c.Options)
}

// Conv3d is a 3-D convolution over [N, C, D, H, W] inputs, like
// torch.nn.Conv3d with a cubic kernel, for video and volumes such as
// scans.
type Conv3d struct {
	Weight  *tensor.Param // [out, in/groups, k, k, k]
	Bias    *tensor.Param // [out], or nil
	Options tensor.ConvOptions
}

// NewConv3d returns a convolution from in to out channels with a k×k×k
// kernel. Stride, Padding, Dilation, Groups and NoBias configure it.
func NewConv3d(in, out, k int, opts ...Option) *Conv3d {
	o := resolve(opts, options{})
	w, b := convParams(o, in, out, k, k, k)
	return &Conv3d{Weight: w, Bias: b, Options: o.conv}
}

// Forward maps [N, in, D, H, W] to [N, out, D', H', W'].
func (c *Conv3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Conv3d(c.Weight.Value(), value(c.Bias), c.Options)
}

// NewDepthwiseConv2d returns a convolution that filters each of c
// channels on its own with mult k×k kernels, as MobileNet does: a Conv2d
// in c groups.
func NewDepthwiseConv2d(c, mult, k int, opts ...Option) *Conv2d {
	return NewConv2d(c, c*mult, k, append(opts, Groups(c))...)
}

// ConvTranspose2d is a transposed 2-D convolution over [N, C, H, W]
// inputs, like torch.nn.ConvTranspose2d with a square kernel: it grows
// what a strided Conv2d shrinks, as decoders and segmentation heads do.
type ConvTranspose2d struct {
	Weight  *tensor.Param // [in, out/groups, k, k]
	Bias    *tensor.Param // [out], or nil
	Options tensor.ConvOptions
	// OutputPadding adds to one side of the output, below the stride.
	OutputPadding int
}

// NewConvTranspose2d returns a transposed convolution from in to out
// channels with a k×k kernel. Stride, Padding, Dilation, Groups and
// NoBias configure it.
func NewConvTranspose2d(in, out, k int, opts ...Option) *ConvTranspose2d {
	o := resolve(opts, options{})
	g := max(o.conv.Groups, 1)
	// PyTorch's fan in for a transposed kernel is its second dimension.
	bound := 1 / math.Sqrt(float64(max(out/g*k*k, 1)))
	c := &ConvTranspose2d{Weight: uniform(o, bound, in, out/g, k, k), Options: o.conv}
	if !o.noBias {
		c.Bias = uniform(o, bound, out)
	}
	return c
}

// Forward maps [N, in, H, W] to [N, out, H', W'].
func (c *ConvTranspose2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.ConvTranspose2d(c.Weight.Value(), value(c.Bias), c.Options, c.OutputPadding)
}

// ConvTranspose1d is ConvTranspose2d for [N, C, L] inputs.
type ConvTranspose1d struct {
	Weight        *tensor.Param // [in, out/groups, k]
	Bias          *tensor.Param // [out], or nil
	Options       tensor.ConvOptions
	OutputPadding int
}

// NewConvTranspose1d returns a transposed convolution from in to out
// channels with a kernel of k.
func NewConvTranspose1d(in, out, k int, opts ...Option) *ConvTranspose1d {
	o := resolve(opts, options{})
	g := max(o.conv.Groups, 1)
	bound := 1 / math.Sqrt(float64(max(out/g*k, 1)))
	c := &ConvTranspose1d{Weight: uniform(o, bound, in, out/g, k), Options: o.conv}
	if !o.noBias {
		c.Bias = uniform(o, bound, out)
	}
	return c
}

// Forward maps [N, in, L] to [N, out, L'].
func (c *ConvTranspose1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.ConvTranspose1d(c.Weight.Value(), value(c.Bias), c.Options, c.OutputPadding)
}

// GroupNorm normalizes [N, C, ...] inputs over groups of channels and
// every position, like torch.nn.GroupNorm: unlike BatchNorm it reads each
// example on its own, so it works the same with batches of one and in
// training and inference.
type GroupNorm struct {
	Groups int
	Weight *tensor.Param // [C], or nil
	Bias   *tensor.Param // [C], or nil
	Eps    float64
}

// NewGroupNorm returns a GroupNorm of c channels in groups groups, with
// scale 1, shift 0 and Eps 1e-5. NoBias leaves out the scale and shift.
// One group is layer normalization over channels and positions; c groups
// is instance normalization.
func NewGroupNorm(groups, c int, opts ...Option) *GroupNorm {
	o := resolve(opts, options{eps: 1e-5})
	g := &GroupNorm{Groups: groups, Eps: o.eps}
	if !o.noBias {
		g.Weight = tensor.NewParam(tensor.Ones(tensor.Float32, c))
		g.Bias = tensor.NewParam(tensor.Zeros(tensor.Float32, c))
	}
	return g
}

// Forward normalizes x.
func (g *GroupNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	shape := x.Shape()
	if len(shape) < 2 || g.Groups < 1 || shape[1]%g.Groups != 0 {
		return tensor.Failed(fmt.Errorf("nn: group norm of %v in %d groups: %w", shape, g.Groups, tensor.ErrShape))
	}
	z := x.Reshape(shape[0], g.Groups, -1)
	m := z.Mean(-1).Unsqueeze(-1)
	d := z.Sub(m)
	v := d.Mul(d).Mean(-1).Unsqueeze(-1)
	y := d.Div(v.Add(scalar(g.Eps)).Sqrt()).Reshape(shape...)
	if g.Weight == nil {
		return y
	}
	bcast := slices.Repeat([]int{1}, len(shape))
	bcast[1] = shape[1]
	return y.Mul(g.Weight.Value().Reshape(bcast...)).Add(g.Bias.Value().Reshape(bcast...))
}

// GlobalAvgPool averages every position of [N, C, ...] inputs into
// [N, C], as the end of ResNet and MobileNet does, whatever the size of
// the picture.
func GlobalAvgPool() Func {
	return func(x *tensor.Tensor) *tensor.Tensor {
		if err := x.Err(); err != nil {
			return x
		}
		if x.Rank() < 3 {
			return tensor.Failed(fmt.Errorf("nn: global average pool of %v, want [N, C, ...]: %w", x.Shape(), tensor.ErrShape))
		}
		return x.Flatten(2, -1).Mean(-1)
	}
}

// Upsample repeats each position of [N, C, H, W] inputs factor times
// along both spatial dimensions, like torch.nn.Upsample with mode
// "nearest".
func Upsample(factor int) Func {
	return func(x *tensor.Tensor) *tensor.Tensor {
		if err := x.Err(); err != nil {
			return x
		}
		s := x.Shape()
		if len(s) != 4 || factor < 1 {
			return tensor.Failed(fmt.Errorf("nn: upsample of %v by %d, want [N, C, H, W]: %w", s, factor, tensor.ErrShape))
		}
		n, c, h, w := s[0], s[1], s[2], s[3]
		return x.Reshape(n, c, h, 1, w, 1).Expand(n, c, h, factor, w, factor).Reshape(n, c, h*factor, w*factor)
	}
}
