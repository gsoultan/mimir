// Package distributions provides probability distributions over tensors,
// modeled on torch.distributions: each samples, scores values by their log
// density or mass, and reports its entropy, mean and variance, and KL
// returns the Kullback-Leibler divergence between two of a family.
//
// A distribution holds a batch of distributions of one family, whose
// parameters broadcast to its batch shape, over events of its event
// shape: a scalar for most families, a vector for MultivariateNormal and
// Dirichlet. Every result is a tensor of the batch shape, or of the
// shape of a sample: the sample shape, then the batch shape, then the
// event shape. Gradients flow to every parameter that requires them.
//
// Sampling is like mimir's other random operations: the caller passes a
// seed, and the same seed draws the same sample on every device. A
// distribution with RSample draws reparameterized samples, through which
// gradients flow to the parameters; Sample draws the same values without
// gradients. Gamma, Beta and Dirichlet are reparameterized by the
// implicit gradients of their gamma samples (see tensor.RandGamma).
//
// Parameters are not validated: a scale that is not positive, say, gives
// NaN wherever it is used, rather than an error. Bad shapes are errors,
// carried by every tensor a distribution returns, like tensor's own.
package distributions

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Distribution is a batch of probability distributions of one family.
type Distribution interface {
	// BatchShape returns the shape of the batch of distributions.
	BatchShape() []int
	// EventShape returns the shape of one value: [] for a scalar.
	EventShape() []int
	// Sample draws values without gradients, shaped as shape followed by
	// the batch and event shapes, determined by seed.
	Sample(seed uint64, shape ...int) *tensor.Tensor
	// LogProb returns the log density or log mass of value, which
	// broadcasts against the batch shape followed by the event shape.
	// The result has value's shape without the event dimensions.
	LogProb(value *tensor.Tensor) *tensor.Tensor
	// Entropy returns the entropy of each distribution, in nats.
	Entropy() *tensor.Tensor
	// Mean returns the mean of each distribution: the batch shape
	// followed by the event shape.
	Mean() *tensor.Tensor
	// Variance returns the variance of each element of an event.
	Variance() *tensor.Tensor
}

// Reparameterized is a distribution that draws samples as a
// differentiable function of its parameters, like PyTorch's rsample.
type Reparameterized interface {
	Distribution
	// RSample draws values as Sample does, with gradients that flow to
	// the parameters.
	RSample(seed uint64, shape ...int) *tensor.Tensor
}

// ErrNoKL reports a pair of distributions KL has no divergence for.
var ErrNoKL = errors.New("distributions: no KL divergence for this pair")

// base holds what every distribution shares: its shapes and any error
// from its construction.
type base struct {
	batch, event []int
	err          error
}

// BatchShape returns the shape of the batch of distributions.
func (b *base) BatchShape() []int { return slices.Clone(b.batch) }

// EventShape returns the shape of one value.
func (b *base) EventShape() []int { return slices.Clone(b.event) }

// failed returns a tensor carrying the construction error, or nil if
// there is none.
func (b *base) failed() *tensor.Tensor {
	if b.err != nil {
		return tensor.Failed(b.err)
	}
	return nil
}

// sampleShape returns shape followed by the batch and event shapes.
func (b *base) sampleShape(shape []int) []int { return slices.Concat(shape, b.batch, b.event) }

// newBase returns the shapes of a family whose parameters broadcast to
// the batch shape after dropping the trailing dims[i] dimensions of
// params[i], which hold an event, of event shape.
func newBase(family string, event []int, params []*tensor.Tensor, dims []int) base {
	shapes := make([][]int, len(params))
	for i, p := range params {
		if err := p.Err(); err != nil {
			return base{err: err}
		}
		s := p.Shape()
		if len(s) < dims[i] {
			return base{err: fmt.Errorf("distributions: %s parameter %d of shape %v has fewer than %d dimensions: %w", family, i, s, dims[i], tensor.ErrShape)}
		}
		if !p.DType().IsFloat() {
			return base{err: fmt.Errorf("distributions: %s parameter %d of %s, want a float type: %w", family, i, p.DType(), tensor.ErrDType)}
		}
		shapes[i] = s[:len(s)-dims[i]]
	}
	batch, err := broadcastShapes(shapes...)
	if err != nil {
		return base{err: fmt.Errorf("distributions: %s parameters: %w", family, err)}
	}
	return base{batch: batch, event: event}
}

// broadcastShapes returns the NumPy broadcast of shapes.
func broadcastShapes(shapes ...[]int) ([]int, error) {
	rank := 0
	for _, s := range shapes {
		rank = max(rank, len(s))
	}
	out := slices.Repeat([]int{1}, rank)
	for _, s := range shapes {
		for i, d := range s {
			o := &out[rank-len(s)+i]
			switch {
			case d == *o || d == 1:
			case *o == 1:
				*o = d
			default:
				return nil, fmt.Errorf("shapes %v do not broadcast: %w", shapes, tensor.ErrShape)
			}
		}
	}
	return out, nil
}

// expandTo broadcasts t to shape, adding leading dimensions.
func expandTo(t *tensor.Tensor, shape []int) *tensor.Tensor {
	if err := t.Err(); err != nil {
		return t
	}
	s := t.Shape()
	if extra := len(shape) - len(s); extra > 0 {
		t = t.Reshape(slices.Concat(slices.Repeat([]int{1}, extra), s)...)
	}
	return t.Expand(shape...)
}

// scalar returns v as a Float32 scalar.
func scalar(v float64) *tensor.Tensor { return tensor.Scalar(float32(v)) }

// subSeed returns the seed of the k-th random tensor a sample draws from
// seed, by SplitMix64, so that the tensors are independent.
func subSeed(seed uint64, k uint64) uint64 {
	z := seed + (k+1)*0x9e3779b97f4a7c15
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}

// tiny is the smallest normal float32, the least value gamma samples and
// their ratios take, as in PyTorch.
const tiny = 0x1p-126

// atLeastTiny returns x raised to tiny where it is smaller, with the
// gradient of x unchanged, as PyTorch clamps samples in place.
func atLeastTiny(x *tensor.Tensor) *tensor.Tensor {
	d := x.Detach()
	return x.Add(d.Maximum(scalar(tiny)).Sub(d))
}

// xlogy returns x·log(y), and 0 where x is 0, like torch.xlogy.
func xlogy(x, y *tensor.Tensor) *tensor.Tensor {
	return tensor.Where(x.Equal(scalar(0)), scalar(0), x.Mul(y.Log()))
}

// softplus returns log(1 + e^x), computed stably.
func softplus(x *tensor.Tensor) *tensor.Tensor {
	return x.Maximum(scalar(0)).Add(x.Abs().Neg().Exp().Log1p())
}

// eps is the spacing of float32 above 1, by which PyTorch keeps
// probabilities away from 0 and 1.
const eps = 0x1p-23

// clampProbs returns p within [eps, 1-eps].
func clampProbs(p *tensor.Tensor) *tensor.Tensor {
	return p.Maximum(scalar(eps)).Minimum(scalar(1 - eps))
}

// halfLog2Pi is log(2π)/2.
var halfLog2Pi = 0.5 * math.Log(2*math.Pi)
