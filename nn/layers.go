package nn

import (
	"fmt"
	"math"
	"slices"
	"sync/atomic"

	"github.com/gsoultan/mimir/tensor"
)

// Linear applies y = x Wᵀ + b over the last dimension, like
// torch.nn.Linear. For inference with weights quantized as a GGUF file
// holds them, Weight may be a buffer (tensor.NewBuffer) of a
// block-quantized dtype, which the product reads without dequantizing
// (see tensor.Tensor.MatMulT).
type Linear struct {
	Weight *tensor.Param // [out, in]
	Bias   *tensor.Param // [out], or nil
}

// NewLinear returns a Linear layer from in to out features. Weights and
// bias are drawn from U(-1/√in, 1/√in), PyTorch's default.
func NewLinear(in, out int, opts ...Option) *Linear {
	o := resolve(opts, options{})
	bound := 1 / math.Sqrt(float64(max(in, 1)))
	l := &Linear{Weight: uniform(o, bound, out, in)}
	if !o.noBias {
		l.Bias = uniform(o, bound, out)
	}
	return l
}

// Forward maps [..., in] to [..., out].
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	y := x.MatMulT(l.Weight.Value())
	if l.Bias != nil {
		y = y.Add(l.Bias.Value())
	}
	return y
}

// Conv2d is a 2-D convolution over [N, C, H, W] inputs, like
// torch.nn.Conv2d with a square kernel.
type Conv2d struct {
	Weight  *tensor.Param // [out, in/groups, k, k]
	Bias    *tensor.Param // [out], or nil
	Options tensor.ConvOptions
}

// NewConv2d returns a convolution from in to out channels with a k×k
// kernel. Stride, Padding, Dilation, Groups and NoBias configure it.
// Weights and bias are drawn from U(-1/√fan_in, 1/√fan_in), PyTorch's
// default.
func NewConv2d(in, out, k int, opts ...Option) *Conv2d {
	o := resolve(opts, options{})
	g := max(o.conv.Groups, 1)
	bound := 1 / math.Sqrt(float64(max(in/g*k*k, 1)))
	c := &Conv2d{Weight: uniform(o, bound, out, in/g, k, k), Options: o.conv}
	if !o.noBias {
		c.Bias = uniform(o, bound, out)
	}
	return c
}

// Forward maps [N, in, H, W] to [N, out, H', W'].
func (c *Conv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	var b *tensor.Tensor
	if c.Bias != nil {
		b = c.Bias.Value()
	}
	return x.Conv2d(c.Weight.Value(), b, c.Options)
}

// Embedding looks up rows of a table by Int32 index, like
// torch.nn.Embedding.
type Embedding struct {
	Weight *tensor.Param // [num, dim]
}

// NewEmbedding returns a table of num rows of dim values drawn from N(0, 1).
func NewEmbedding(num, dim int, opts ...Option) *Embedding {
	o := resolve(opts, options{})
	return &Embedding{Weight: tensor.NewParam(tensor.Randn(o.nextSeed(), num, dim))}
}

// Forward maps Int32 indices of any shape to [..., dim]. An index outside
// the table fails with ErrIndex when computed.
func (e *Embedding) Forward(idx *tensor.Tensor) *tensor.Tensor {
	w := e.Weight.Value()
	switch {
	case idx.Err() != nil:
		return idx
	case w.Err() != nil:
		return w
	case idx.DType() != tensor.Int32:
		return tensor.Failed(fmt.Errorf("nn: embedding of %s indices: %w", idx.DType(), tensor.ErrDType))
	}
	shape, dim := idx.Shape(), w.Shape()[1]
	rows := w.Gather(0, idx.Reshape(-1, 1).Expand(-1, dim))
	return rows.Reshape(append(shape, dim)...)
}

// LayerNorm normalizes the last dimension to zero mean and unit variance,
// then scales and shifts it, like torch.nn.LayerNorm.
type LayerNorm struct {
	Weight *tensor.Param // [dim], or nil
	Bias   *tensor.Param // [dim], or nil
	Eps    float64
}

// NewLayerNorm returns a LayerNorm over a last dimension of size dim, with
// scale 1, shift 0 and Eps 1e-5. NoBias leaves out the scale and shift.
func NewLayerNorm(dim int, opts ...Option) *LayerNorm {
	o := resolve(opts, options{eps: 1e-5})
	l := &LayerNorm{Eps: o.eps}
	if !o.noBias {
		l.Weight = tensor.NewParam(tensor.Ones(tensor.Float32, dim))
		l.Bias = tensor.NewParam(tensor.Zeros(tensor.Float32, dim))
	}
	return l
}

// Forward normalizes x over its last dimension.
func (l *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	m := x.Mean(-1).Unsqueeze(-1)
	d := x.Sub(m)
	v := d.Mul(d).Mean(-1).Unsqueeze(-1)
	y := d.Div(v.Add(scalar(l.Eps)).Sqrt())
	if l.Weight != nil {
		y = y.Mul(l.Weight.Value()).Add(l.Bias.Value())
	}
	return y
}

// BatchNorm normalizes each channel of [N, C, ...] inputs over the batch,
// like torch.nn.BatchNorm1d and BatchNorm2d. In training mode it uses the
// batch's statistics and updates the running statistics, which Backward
// computes along with the gradients; in inference mode it uses the running
// statistics.
type BatchNorm struct {
	Weight      *tensor.Param // [C], or nil
	Bias        *tensor.Param // [C], or nil
	RunningMean *tensor.Param // [C] buffer
	RunningVar  *tensor.Param // [C] buffer
	Eps         float64
	Momentum    float64

	ranks    []int // accepted input ranks
	training atomic.Bool
}

// NewBatchNorm1d returns batch normalization of c channels for [N, C] or
// [N, C, L] inputs, with Eps 1e-5 and Momentum 0.1.
func NewBatchNorm1d(c int, opts ...Option) *BatchNorm { return newBatchNorm(c, []int{2, 3}, opts) }

// NewBatchNorm2d returns batch normalization of c channels for
// [N, C, H, W] inputs, with Eps 1e-5 and Momentum 0.1.
func NewBatchNorm2d(c int, opts ...Option) *BatchNorm { return newBatchNorm(c, []int{4}, opts) }

func newBatchNorm(c int, ranks []int, opts []Option) *BatchNorm {
	o := resolve(opts, options{eps: 1e-5, momentum: 0.1})
	b := &BatchNorm{
		RunningMean: tensor.NewBuffer(tensor.Zeros(tensor.Float32, c)),
		RunningVar:  tensor.NewBuffer(tensor.Ones(tensor.Float32, c)),
		Eps:         o.eps,
		Momentum:    o.momentum,
		ranks:       ranks,
	}
	if !o.noBias {
		b.Weight = tensor.NewParam(tensor.Ones(tensor.Float32, c))
		b.Bias = tensor.NewParam(tensor.Zeros(tensor.Float32, c))
	}
	b.training.Store(true)
	return b
}

// SetTraining switches between batch and running statistics.
func (b *BatchNorm) SetTraining(on bool) { b.training.Store(on) }

// Forward normalizes x.
func (b *BatchNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	shape := x.Shape()
	if !slices.Contains(b.ranks, len(shape)) {
		return tensor.Failed(fmt.Errorf("nn: batch norm of %v, want rank %v: %w", shape, b.ranks, tensor.ErrShape))
	}
	c := shape[1]
	axes := []int{0}
	for d := 2; d < len(shape); d++ {
		axes = append(axes, d)
	}
	bcast := slices.Repeat([]int{1}, len(shape))
	bcast[1] = c
	perChannel := func(t *tensor.Tensor) *tensor.Tensor { return t.Reshape(bcast...) }

	var y *tensor.Tensor
	var effects []*tensor.Tensor
	if b.training.Load() {
		n := x.Size() / max(c, 1)
		if n < 2 {
			return tensor.Failed(fmt.Errorf("nn: batch norm in training needs more than one value per channel, got %v: %w", shape, tensor.ErrShape))
		}
		mean := x.Mean(axes...)
		d := x.Sub(perChannel(mean))
		v := d.Mul(d).Mean(axes...)
		y = d.Div(perChannel(v.Add(scalar(b.Eps)).Sqrt()))
		m := b.Momentum
		unbiased := v.Detach().Mul(scalar(float64(n) / float64(n-1)))
		newMean := b.RunningMean.Value().Mul(scalar(1 - m)).Add(mean.Detach().Mul(scalar(m)))
		newVar := b.RunningVar.Value().Mul(scalar(1 - m)).Add(unbiased.Mul(scalar(m)))
		if err := b.RunningMean.Set(newMean); err != nil {
			return tensor.Failed(err)
		}
		if err := b.RunningVar.Set(newVar); err != nil {
			return tensor.Failed(err)
		}
		effects = []*tensor.Tensor{newMean, newVar}
	} else {
		mean, v := b.RunningMean.Value(), b.RunningVar.Value()
		y = x.Sub(perChannel(mean)).Div(perChannel(v.Add(scalar(b.Eps)).Sqrt()))
	}
	if b.Weight != nil {
		y = y.Mul(perChannel(b.Weight.Value())).Add(perChannel(b.Bias.Value()))
	}
	return y.WithEffects(effects...)
}

// Dropout zeroes each element with probability P during training and
// scales the rest by 1/(1-P), like torch.nn.Dropout. In inference mode it
// passes inputs through.
type Dropout struct {
	P float64

	seed     uint64
	calls    atomic.Uint64
	training atomic.Bool
}

// NewDropout returns a Dropout layer that drops with probability p. Each
// call draws a new mask from a seed sequence fixed when the layer is made.
func NewDropout(p float64, opts ...Option) *Dropout {
	o := resolve(opts, options{})
	d := &Dropout{P: p, seed: o.nextSeed()}
	d.training.Store(true)
	return d
}

// SetTraining switches dropping on or off.
func (d *Dropout) SetTraining(on bool) { d.training.Store(on) }

// Forward drops elements of x.
func (d *Dropout) Forward(x *tensor.Tensor) *tensor.Tensor {
	switch {
	case d.P < 0 || d.P > 1 || math.IsNaN(d.P):
		return tensor.Failed(fmt.Errorf("nn: dropout probability %v: %w", d.P, tensor.ErrShape))
	case !d.training.Load() || d.P == 0 || x.Err() != nil:
		return x
	case d.P == 1:
		return x.Mul(scalar(0))
	}
	keep := 1 - d.P
	mask := tensor.Rand(splitmix(d.seed^d.calls.Add(1)), x.Shape()...).Less(scalar(keep))
	return x.Mul(mask).Mul(scalar(1 / keep))
}

// Sequential applies layers in order, like torch.nn.Sequential.
type Sequential struct {
	Layers []Module
}

// NewSequential returns the composition of layers.
func NewSequential(layers ...Module) *Sequential { return &Sequential{Layers: layers} }

// Forward applies each layer to the previous one's output.
func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}

// ReLU returns the max(x, 0) activation.
func ReLU() Func { return (*tensor.Tensor).ReLU }

// GELU returns the GELU activation (tanh approximation).
func GELU() Func { return (*tensor.Tensor).GELU }

// Sigmoid returns the logistic activation.
func Sigmoid() Func { return (*tensor.Tensor).Sigmoid }

// Tanh returns the hyperbolic tangent activation.
func Tanh() Func { return (*tensor.Tensor).Tanh }

// SiLU returns the x·sigmoid(x) activation.
func SiLU() Func { return (*tensor.Tensor).SiLU }

// Softmax returns softmax along axis.
func Softmax(axis int) Func {
	return func(x *tensor.Tensor) *tensor.Tensor { return x.Softmax(axis) }
}

// LogSoftmax returns log-softmax along axis.
func LogSoftmax(axis int) Func {
	return func(x *tensor.Tensor) *tensor.Tensor { return x.LogSoftmax(axis) }
}

// Flatten merges every dimension after the first, like torch.nn.Flatten.
func Flatten() Func {
	return func(x *tensor.Tensor) *tensor.Tensor { return x.Flatten(1, -1) }
}

// MaxPool2d returns k×k max pooling, with stride k unless Stride says
// otherwise, like torch.nn.MaxPool2d.
func MaxPool2d(k int, opts ...Option) Func {
	o := resolve(opts, options{})
	return func(x *tensor.Tensor) *tensor.Tensor { return x.MaxPool2d(k, o.conv) }
}

// AvgPool2d returns k×k average pooling, with stride k unless Stride says
// otherwise, like torch.nn.AvgPool2d.
func AvgPool2d(k int, opts ...Option) Func {
	o := resolve(opts, options{})
	return func(x *tensor.Tensor) *tensor.Tensor { return x.AvgPool2d(k, o.conv) }
}
