package distributions

import (
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// Uniform is the uniform distribution on [Low, High), like
// torch.distributions.Uniform.
type Uniform struct {
	base
	Low, High *tensor.Tensor
}

// NewUniform returns uniform distributions on [low, high).
func NewUniform(low, high *tensor.Tensor) *Uniform {
	return &Uniform{base: newBase("uniform", nil, []*tensor.Tensor{low, high}, []int{0, 0}), Low: low, High: high}
}

// RSample draws low + U·(high - low) for U uniform on [0, 1).
func (d *Uniform) RSample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return tensor.Rand(seed, d.sampleShape(shape)...).Mul(d.High.Sub(d.Low)).Add(d.Low)
}

// Sample draws as RSample does, without gradients.
func (d *Uniform) Sample(seed uint64, shape ...int) *tensor.Tensor {
	return d.RSample(seed, shape...).Detach()
}

// LogProb returns -log(high - low) inside the support and -Inf outside.
func (d *Uniform) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	inside := d.Low.Greater(value).Add(value.Less(d.High).Neg()).Equal(scalar(-1))
	return tensor.Where(inside, d.High.Sub(d.Low).Log().Neg(), scalar(math.Inf(-1)))
}

// Entropy returns log(high - low).
func (d *Uniform) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.High.Sub(d.Low).Log(), d.batch)
}

// Mean returns (low + high)/2.
func (d *Uniform) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Low.Add(d.High).Mul(scalar(0.5)), d.batch)
}

// Variance returns (high - low)²/12.
func (d *Uniform) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	w := d.High.Sub(d.Low)
	return expandTo(w.Mul(w).Mul(scalar(1.0/12)), d.batch)
}

// Exponential is the exponential distribution with rate Rate, mean
// 1/Rate, like torch.distributions.Exponential.
type Exponential struct {
	base
	Rate *tensor.Tensor
}

// NewExponential returns exponential distributions with rates rate.
func NewExponential(rate *tensor.Tensor) *Exponential {
	return &Exponential{base: newBase("exponential", nil, []*tensor.Tensor{rate}, []int{0}), Rate: rate}
}

// RSample draws -log(1 - U)/rate for U uniform on [0, 1).
func (d *Exponential) RSample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return tensor.Rand(seed, d.sampleShape(shape)...).Neg().Log1p().Neg().Div(d.Rate)
}

// Sample draws as RSample does, without gradients.
func (d *Exponential) Sample(seed uint64, shape ...int) *tensor.Tensor {
	return d.RSample(seed, shape...).Detach()
}

// LogProb returns log(rate) - rate·value.
func (d *Exponential) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return d.Rate.Log().Sub(d.Rate.Mul(value))
}

// Entropy returns 1 - log(rate).
func (d *Exponential) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(scalar(1).Sub(d.Rate.Log()), d.batch)
}

// Mean returns 1/rate.
func (d *Exponential) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Rate.Recip(), d.batch)
}

// Variance returns 1/rate².
func (d *Exponential) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Rate.Mul(d.Rate).Recip(), d.batch)
}

// Laplace is the Laplace distribution with location Loc and scale Scale,
// like torch.distributions.Laplace.
type Laplace struct {
	base
	Loc, Scale *tensor.Tensor
}

// NewLaplace returns Laplace distributions with locations loc and scales
// scale.
func NewLaplace(loc, scale *tensor.Tensor) *Laplace {
	return &Laplace{base: newBase("laplace", nil, []*tensor.Tensor{loc, scale}, []int{0, 0}), Loc: loc, Scale: scale}
}

// RSample draws loc - scale·sign(u)·log(1 - |u|) for u uniform on
// [-1 + ε, 1 - ε), with ε float32's spacing above 1, as PyTorch keeps u
// off -1.
func (d *Laplace) RSample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	u := tensor.Rand(seed, d.sampleShape(shape)...).Mul(scalar(2 - 2*eps)).Add(scalar(eps - 1))
	sign := u.Greater(scalar(0)).Sub(u.Less(scalar(0)))
	return d.Loc.Sub(d.Scale.Mul(sign).Mul(u.Abs().Neg().Log1p()))
}

// Sample draws as RSample does, without gradients.
func (d *Laplace) Sample(seed uint64, shape ...int) *tensor.Tensor {
	return d.RSample(seed, shape...).Detach()
}

// LogProb returns -log(2·scale) - |value - loc|/scale.
func (d *Laplace) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return d.Scale.Mul(scalar(2)).Log().Neg().Sub(value.Sub(d.Loc).Abs().Div(d.Scale))
}

// Entropy returns 1 + log(2·scale).
func (d *Laplace) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Scale.Mul(scalar(2)).Log().Add(scalar(1)), d.batch)
}

// Mean returns loc.
func (d *Laplace) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Loc, d.batch)
}

// Variance returns 2·scale².
func (d *Laplace) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Scale.Mul(d.Scale).Mul(scalar(2)), d.batch)
}
