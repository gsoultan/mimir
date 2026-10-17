package distributions

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// Normal is the normal distribution N(loc, scale²), like
// torch.distributions.Normal.
type Normal struct {
	base
	Loc, Scale *tensor.Tensor
}

// NewNormal returns normal distributions with mean loc and standard
// deviation scale, which broadcast to the batch shape.
func NewNormal(loc, scale *tensor.Tensor) *Normal {
	return &Normal{base: newBase("normal", nil, []*tensor.Tensor{loc, scale}, []int{0, 0}), Loc: loc, Scale: scale}
}

// RSample draws loc + scale·ε for standard normal ε.
func (d *Normal) RSample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return tensor.Randn(seed, d.sampleShape(shape)...).Mul(d.Scale).Add(d.Loc)
}

// Sample draws as RSample does, without gradients.
func (d *Normal) Sample(seed uint64, shape ...int) *tensor.Tensor {
	return d.RSample(seed, shape...).Detach()
}

// LogProb returns the log density of value.
func (d *Normal) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	z := value.Sub(d.Loc).Div(d.Scale)
	return z.Mul(z).Mul(scalar(-0.5)).Sub(d.Scale.Log()).Sub(scalar(halfLog2Pi))
}

// Entropy returns 1/2 + log(√(2π)·scale).
func (d *Normal) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Scale.Log().Add(scalar(0.5+halfLog2Pi)), d.batch)
}

// Mean returns loc.
func (d *Normal) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Loc, d.batch)
}

// Variance returns scale².
func (d *Normal) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Scale.Mul(d.Scale), d.batch)
}

// CDF returns the probability of a value at most value.
func (d *Normal) CDF(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	z := value.Sub(d.Loc).Div(d.Scale.Mul(scalar(math.Sqrt2)))
	return z.Erf().Add(scalar(1)).Mul(scalar(0.5))
}

// ICDF returns the value whose CDF is p, the quantile function.
func (d *Normal) ICDF(p *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	z := p.Mul(scalar(2)).Sub(scalar(1)).Erfinv().Mul(scalar(math.Sqrt2))
	return z.Mul(d.Scale).Add(d.Loc)
}

// MultivariateNormal is the normal distribution over vectors with mean
// Loc and covariance L·Lᵀ for the lower triangular ScaleTril L with a
// positive diagonal, like torch.distributions.MultivariateNormal with
// scale_tril.
type MultivariateNormal struct {
	base
	Loc, ScaleTril *tensor.Tensor
}

// NewMultivariateNormal returns multivariate normal distributions with
// means loc, [..., n], and scale_tril, [..., n, n], whose upper triangle
// is ignored. Their leading dimensions broadcast to the batch shape.
func NewMultivariateNormal(loc, scaleTril *tensor.Tensor) *MultivariateNormal {
	b := newBase("multivariate normal", nil, []*tensor.Tensor{loc, scaleTril}, []int{1, 2})
	if b.err == nil {
		ls, ss := loc.Shape(), scaleTril.Shape()
		n := ls[len(ls)-1]
		if ss[len(ss)-1] != n || ss[len(ss)-2] != n {
			b.err = fmt.Errorf("distributions: multivariate normal of mean %v and scale %v: %w", ls, ss, tensor.ErrShape)
		}
		b.event = []int{n}
	}
	return &MultivariateNormal{base: b, Loc: loc, ScaleTril: tril(scaleTril)}
}

// NewMultivariateNormalCovariance returns multivariate normal
// distributions with means loc and symmetric positive definite
// covariances cov, [..., n, n], read from their lower triangle, by their
// Cholesky factors.
func NewMultivariateNormalCovariance(loc, cov *tensor.Tensor) *MultivariateNormal {
	return NewMultivariateNormal(loc, cov.Cholesky())
}

// tril returns the lower triangle of the matrices m, zero above.
func tril(m *tensor.Tensor) *tensor.Tensor {
	if m.Err() != nil || m.Rank() < 2 {
		return m
	}
	s := m.Shape()
	n, k := s[len(s)-2], s[len(s)-1]
	mask := make([]float32, n*k)
	for i := range n {
		for j := range min(i+1, k) {
			mask[i*k+j] = 1
		}
	}
	return m.Mul(tensor.FromSlice(mask, n, k).Cast(m.DType()))
}

// diagonal returns the diagonals of the square matrices m.
func diagonal(m *tensor.Tensor) *tensor.Tensor {
	s := m.Shape()
	n := s[len(s)-1]
	eye := make([]float32, n*n)
	for i := range n {
		eye[i*n+i] = 1
	}
	return m.Mul(tensor.FromSlice(eye, n, n).Cast(m.DType())).Sum(-1)
}

// halfLogDet returns log|L| = Σ log Lᵢᵢ, half the log determinant of the
// covariance.
func (d *MultivariateNormal) halfLogDet() *tensor.Tensor { return diagonal(d.ScaleTril).Log().Sum(-1) }

// RSample draws loc + L·ε for a vector ε of standard normals.
func (d *MultivariateNormal) RSample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	e := tensor.Randn(seed, d.sampleShape(shape)...)
	return d.ScaleTril.MatMul(e.Unsqueeze(-1)).Squeeze(-1).Add(d.Loc)
}

// Sample draws as RSample does, without gradients.
func (d *MultivariateNormal) Sample(seed uint64, shape ...int) *tensor.Tensor {
	return d.RSample(seed, shape...).Detach()
}

// LogProb returns the log density of the vectors value, [..., n]: the
// squared Mahalanobis distance by a triangular solve with L.
func (d *MultivariateNormal) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	m := d.mahalanobis(value.Sub(d.Loc))
	n := float64(d.event[0])
	return m.Mul(scalar(-0.5)).Sub(d.halfLogDet()).Sub(scalar(n * halfLog2Pi))
}

// mahalanobis returns |L⁻¹·x|² for the vectors x, whose leading
// dimensions broadcast against L's.
func (d *MultivariateNormal) mahalanobis(x *tensor.Tensor) *tensor.Tensor {
	y := tensor.TriangularSolve(d.ScaleTril, x.Unsqueeze(-1), false)
	return y.Mul(y).Sum(-2, -1)
}

// Entropy returns n/2·(1 + log 2π) + log|L|.
func (d *MultivariateNormal) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	n := float64(d.event[0])
	return expandTo(d.halfLogDet().Add(scalar(n*(0.5+halfLog2Pi))), d.batch)
}

// Mean returns loc.
func (d *MultivariateNormal) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Loc, d.sampleShape(nil))
}

// Variance returns the diagonal of the covariance, the squared norms of
// the rows of L.
func (d *MultivariateNormal) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.ScaleTril.Mul(d.ScaleTril).Sum(-1), d.sampleShape(nil))
}
