package distributions

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// Bernoulli is the distribution of a coin that lands 1 with probability
// Probs and 0 otherwise, like torch.distributions.Bernoulli. Its values
// are floats.
type Bernoulli struct {
	base
	// Probs and Logits are the probability of 1 and its log-odds; a
	// distribution made from one computes the other from it.
	Probs, Logits *tensor.Tensor
}

// NewBernoulli returns Bernoulli distributions with probabilities probs.
// Their log-odds keep the probabilities within float32's spacing of 0 and
// 1, as PyTorch's do.
func NewBernoulli(probs *tensor.Tensor) *Bernoulli {
	p := clampProbs(probs)
	return &Bernoulli{base: newBase("bernoulli", nil, []*tensor.Tensor{probs}, []int{0}), Probs: probs, Logits: p.Log().Sub(p.Neg().Log1p())}
}

// NewBernoulliLogits returns Bernoulli distributions with log-odds logits.
func NewBernoulliLogits(logits *tensor.Tensor) *Bernoulli {
	return &Bernoulli{base: newBase("bernoulli", nil, []*tensor.Tensor{logits}, []int{0}), Probs: logits.Sigmoid(), Logits: logits}
}

// Sample draws 1 with probability Probs and 0 otherwise.
func (d *Bernoulli) Sample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return tensor.Rand(seed, d.sampleShape(shape)...).Less(d.Probs.Detach())
}

// LogProb returns the log mass of value, from the log-odds: minus the
// binary cross-entropy.
func (d *Bernoulli) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return d.Logits.Mul(value).Sub(softplus(d.Logits))
}

// Entropy returns -p·log p - (1-p)·log(1-p).
func (d *Bernoulli) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(softplus(d.Logits).Sub(d.Logits.Mul(d.Probs)), d.batch)
}

// Mean returns Probs.
func (d *Bernoulli) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Probs, d.batch)
}

// Variance returns Probs·(1 - Probs).
func (d *Bernoulli) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Probs.Mul(scalar(1).Sub(d.Probs)), d.batch)
}

// Categorical is the distribution over the categories 0 to K-1 along the
// last axis of its parameters, like torch.distributions.Categorical. Its
// values are Int32 categories.
type Categorical struct {
	base
	// Probs are the probabilities of the categories and Logits their
	// logarithms, both normalized; a distribution made from one computes
	// the other from it.
	Probs, Logits *tensor.Tensor
}

// NewCategorical returns categorical distributions with probabilities
// probs, [..., K], normalized along the last axis: they need not sum to
// 1.
func NewCategorical(probs *tensor.Tensor) *Categorical {
	b := newCategoricalBase(probs)
	p := probs.Div(probs.Sum(-1).Unsqueeze(-1))
	return &Categorical{base: b, Probs: p, Logits: clampProbs(p).Log()}
}

// NewCategoricalLogits returns categorical distributions with
// unnormalized log-probabilities logits, [..., K].
func NewCategoricalLogits(logits *tensor.Tensor) *Categorical {
	b := newCategoricalBase(logits)
	l := logits.LogSoftmax(-1)
	return &Categorical{base: b, Probs: l.Exp(), Logits: l}
}

func newCategoricalBase(p *tensor.Tensor) base {
	b := newBase("categorical", nil, []*tensor.Tensor{p}, []int{1})
	if b.err == nil && p.Shape()[p.Rank()-1] == 0 {
		b.err = fmt.Errorf("distributions: categorical of no categories: %w", tensor.ErrShape)
	}
	return b
}

// categories returns K.
func (d *Categorical) categories() int {
	s := d.Logits.Shape()
	return s[len(s)-1]
}

// Sample draws categories by the Gumbel-max trick: the position of the
// largest of the log-probabilities plus Gumbel noise.
func (d *Categorical) Sample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	k := d.categories()
	u := tensor.Rand(seed, append(d.sampleShape(shape), k)...)
	gumbel := u.Log().Neg().Log().Neg()
	return d.Logits.Detach().Add(gumbel).Argmax(-1)
}

// LogProb returns the log-probabilities of the categories value, Int32 or
// float, which broadcast against the batch shape.
func (d *Categorical) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	if err := value.Err(); err != nil {
		return value
	}
	if value.DType() != tensor.Int32 {
		value = value.Cast(tensor.Int32)
	}
	shape, err := broadcastShapes(value.Shape(), d.batch)
	if err != nil {
		return tensor.Failed(fmt.Errorf("distributions: categorical value of shape %v: %w", value.Shape(), err))
	}
	logits := expandTo(d.Logits, append(shape, d.categories()))
	return logits.Gather(-1, expandTo(value, shape).Unsqueeze(-1)).Squeeze(-1)
}

// minLogit stands in for the log of a zero probability in sums of
// p·log p, so that it counts 0 rather than NaN.
const minLogit = -math.MaxFloat32

// Entropy returns -Σ p·log p over the categories.
func (d *Categorical) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return d.Probs.Mul(d.Logits.Maximum(scalar(minLogit))).Sum(-1).Neg()
}

// Mean is NaN, as in PyTorch: categories have no order.
func (d *Categorical) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return tensor.Full(tensor.Float32, math.NaN(), d.batch...)
}

// Variance is NaN, as in PyTorch.
func (d *Categorical) Variance() *tensor.Tensor { return d.Mean() }

// Poisson is the distribution of counts with mean Rate, like
// torch.distributions.Poisson. Its values are float counts.
type Poisson struct {
	base
	Rate *tensor.Tensor
}

// NewPoisson returns Poisson distributions with rates rate.
func NewPoisson(rate *tensor.Tensor) *Poisson {
	return &Poisson{base: newBase("poisson", nil, []*tensor.Tensor{rate}, []int{0}), Rate: rate}
}

// Sample draws counts, as tensor.RandPoisson does.
func (d *Poisson) Sample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return tensor.RandPoisson(seed, expandTo(d.Rate.Detach(), d.sampleShape(shape)))
}

// LogProb returns k·log λ - λ - log k! for counts k.
func (d *Poisson) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return xlogy(value, d.Rate).Sub(d.Rate).Sub(value.Add(scalar(1)).Lgamma())
}

// poissonTerms is the number of counts the entropy of a small rate sums
// over, and poissonSmall the rate from which it uses the asymptotic
// series instead. The terms the sum leaves out, and the error of the
// series, are below float32's precision.
const (
	poissonTerms = 96
	poissonSmall = 24
)

// Entropy returns -Σ p(k)·log p(k): summed over the first counts for
// rates below 24, and from its asymptotic series, ½·log(2πeλ) - 1/(12λ) -
// 1/(24λ²) - 19/(360λ³), above. PyTorch has none.
func (d *Poisson) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	rate := expandTo(d.Rate, d.batch)
	small := rate.Less(scalar(poissonSmall))
	// Each branch sees a rate in its own range, so the other's NaN or
	// infinity never reaches the gradient.
	lo := tensor.Where(small, rate, scalar(1))
	k := tensor.Arange[float32](0, poissonTerms, 1).To(rate.Device()).Reshape(onesThen(len(d.batch), poissonTerms)...)
	lp := xlogy(k, lo.Unsqueeze(-1)).Sub(lo.Unsqueeze(-1)).Sub(k.Add(scalar(1)).Lgamma())
	sum := lp.Exp().Mul(lp).Sum(-1).Neg()
	hi := tensor.Where(small, scalar(poissonSmall), rate)
	r := hi.Recip()
	series := hi.Mul(scalar(2 * math.Pi * math.E)).Log().Mul(scalar(0.5)).
		Sub(r.Mul(scalar(1.0 / 12).Add(r.Mul(scalar(1.0 / 24).Add(r.Mul(scalar(19.0 / 360)))))))
	return tensor.Where(small, sum, series)
}

// onesThen returns n ones followed by k.
func onesThen(n, k int) []int {
	s := make([]int, n+1)
	for i := range n {
		s[i] = 1
	}
	s[n] = k
	return s
}

// Mean returns Rate.
func (d *Poisson) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Rate, d.batch)
}

// Variance returns Rate.
func (d *Poisson) Variance() *tensor.Tensor { return d.Mean() }
