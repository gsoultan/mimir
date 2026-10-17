package distributions

import (
	"fmt"

	"github.com/gsoultan/mimir/tensor"
)

// Gamma is the gamma distribution with shape Concentration α and rate
// Rate β, density β^α·x^(α-1)·e^(-βx)/Γ(α), like
// torch.distributions.Gamma.
type Gamma struct {
	base
	Concentration, Rate *tensor.Tensor
}

// NewGamma returns gamma distributions with concentrations concentration
// and rates rate.
func NewGamma(concentration, rate *tensor.Tensor) *Gamma {
	return &Gamma{base: newBase("gamma", nil, []*tensor.Tensor{concentration, rate}, []int{0, 0}), Concentration: concentration, Rate: rate}
}

// RSample draws samples of Gamma(α, 1), divided by β: see
// tensor.RandGamma for how and for their gradients. Samples are at least
// the smallest normal float32.
func (d *Gamma) RSample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	s := d.sampleShape(shape)
	return atLeastTiny(tensor.RandGamma(seed, expandTo(d.Concentration, s)).Div(d.Rate))
}

// Sample draws as RSample does, without gradients.
func (d *Gamma) Sample(seed uint64, shape ...int) *tensor.Tensor {
	return d.RSample(seed, shape...).Detach()
}

// LogProb returns α·log β + (α-1)·log x - βx - log Γ(α).
func (d *Gamma) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	a, b := d.Concentration, d.Rate
	return xlogy(a, b).Add(xlogy(a.Sub(scalar(1)), value)).Sub(b.Mul(value)).Sub(a.Lgamma())
}

// Entropy returns α - log β + log Γ(α) + (1-α)·ψ(α).
func (d *Gamma) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	a := d.Concentration
	return expandTo(a.Sub(d.Rate.Log()).Add(a.Lgamma()).Add(scalar(1).Sub(a).Mul(a.Digamma())), d.batch)
}

// Mean returns α/β.
func (d *Gamma) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Concentration.Div(d.Rate), d.batch)
}

// Variance returns α/β².
func (d *Gamma) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return expandTo(d.Concentration.Div(d.Rate.Mul(d.Rate)), d.batch)
}

// Dirichlet is the Dirichlet distribution over the probability vectors of
// K categories, with concentrations Concentration along the last axis,
// like torch.distributions.Dirichlet.
type Dirichlet struct {
	base
	Concentration *tensor.Tensor
}

// NewDirichlet returns Dirichlet distributions with concentrations
// concentration, [..., K].
func NewDirichlet(concentration *tensor.Tensor) *Dirichlet {
	b := newBase("dirichlet", nil, []*tensor.Tensor{concentration}, []int{1})
	if b.err == nil {
		b.event = []int{concentration.Shape()[concentration.Rank()-1]}
		if b.event[0] == 0 {
			b.err = fmt.Errorf("distributions: dirichlet of no categories: %w", tensor.ErrShape)
		}
	}
	return &Dirichlet{base: b, Concentration: concentration}
}

// RSample draws gamma samples of the concentrations and divides each by
// their sum, which reparameterizes the sample by their implicit
// gradients.
func (d *Dirichlet) RSample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return normalized(tensor.RandGamma(seed, expandTo(d.Concentration, d.sampleShape(shape))))
}

// normalized returns the gamma samples g divided by their sums along the
// last axis, at least tiny.
func normalized(g *tensor.Tensor) *tensor.Tensor {
	return atLeastTiny(g.Div(g.Sum(-1).Unsqueeze(-1)))
}

// Sample draws as RSample does, without gradients.
func (d *Dirichlet) Sample(seed uint64, shape ...int) *tensor.Tensor {
	return d.RSample(seed, shape...).Detach()
}

// LogProb returns Σ (αᵢ-1)·log xᵢ + log Γ(Σ αᵢ) - Σ log Γ(αᵢ).
func (d *Dirichlet) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return dirichletLogProb(d.Concentration, value)
}

func dirichletLogProb(a, x *tensor.Tensor) *tensor.Tensor {
	return xlogy(a.Sub(scalar(1)), x).Sum(-1).Add(a.Sum(-1).Lgamma()).Sub(a.Lgamma().Sum(-1))
}

// Entropy returns log B(α) + (α₀ - K)·ψ(α₀) - Σ (αᵢ-1)·ψ(αᵢ), with α₀
// = Σ αᵢ and B the multivariate beta function.
func (d *Dirichlet) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	return dirichletEntropy(d.Concentration)
}

func dirichletEntropy(a *tensor.Tensor) *tensor.Tensor {
	k := float64(a.Shape()[a.Rank()-1])
	a0 := a.Sum(-1)
	return a.Lgamma().Sum(-1).Sub(a0.Lgamma()).
		Sub(scalar(k).Sub(a0).Mul(a0.Digamma())).
		Sub(a.Sub(scalar(1)).Mul(a.Digamma()).Sum(-1))
}

// Mean returns α/α₀.
func (d *Dirichlet) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	a := d.Concentration
	return expandTo(a.Div(a.Sum(-1).Unsqueeze(-1)), d.sampleShape(nil))
}

// Variance returns αᵢ·(α₀ - αᵢ)/(α₀²·(α₀ + 1)).
func (d *Dirichlet) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	a := d.Concentration
	a0 := a.Sum(-1).Unsqueeze(-1)
	return expandTo(a.Mul(a0.Sub(a)).Div(a0.Mul(a0).Mul(a0.Add(scalar(1)))), d.sampleShape(nil))
}

// Beta is the beta distribution on [0, 1] with concentrations
// Concentration1 α of 1 and Concentration0 β of 0, density
// x^(α-1)·(1-x)^(β-1)/B(α, β), like torch.distributions.Beta.
type Beta struct {
	base
	Concentration1, Concentration0 *tensor.Tensor
}

// NewBeta returns beta distributions with concentrations concentration1
// and concentration0.
func NewBeta(concentration1, concentration0 *tensor.Tensor) *Beta {
	return &Beta{base: newBase("beta", nil, []*tensor.Tensor{concentration1, concentration0}, []int{0, 0}), Concentration1: concentration1, Concentration0: concentration0}
}

// RSample draws X/(X+Y) for X and Y gamma samples of the two
// concentrations, as Dirichlet.RSample draws a pair.
func (d *Beta) RSample(seed uint64, shape ...int) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	s := d.sampleShape(shape)
	x := tensor.RandGamma(subSeed(seed, 0), expandTo(d.Concentration1, s))
	y := tensor.RandGamma(subSeed(seed, 1), expandTo(d.Concentration0, s))
	return atLeastTiny(x.Div(x.Add(y)))
}

// Sample draws as RSample does, without gradients.
func (d *Beta) Sample(seed uint64, shape ...int) *tensor.Tensor {
	return d.RSample(seed, shape...).Detach()
}

// LogProb returns (α-1)·log x + (β-1)·log(1-x) - log B(α, β).
func (d *Beta) LogProb(value *tensor.Tensor) *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	a, b := d.Concentration1, d.Concentration0
	return xlogy(a.Sub(scalar(1)), value).Add(xlogy(b.Sub(scalar(1)), scalar(1).Sub(value))).Sub(logBeta(a, b))
}

// logBeta returns log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a + b).
func logBeta(a, b *tensor.Tensor) *tensor.Tensor {
	return a.Lgamma().Add(b.Lgamma()).Sub(a.Add(b).Lgamma())
}

// Entropy returns log B(α, β) - (α-1)·ψ(α) - (β-1)·ψ(β) + (α+β-2)·ψ(α+β).
func (d *Beta) Entropy() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	a, b := d.Concentration1, d.Concentration0
	one := scalar(1)
	t := a.Add(b)
	return expandTo(logBeta(a, b).Sub(a.Sub(one).Mul(a.Digamma())).Sub(b.Sub(one).Mul(b.Digamma())).
		Add(t.Sub(scalar(2)).Mul(t.Digamma())), d.batch)
}

// Mean returns α/(α + β).
func (d *Beta) Mean() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	a, b := d.Concentration1, d.Concentration0
	return expandTo(a.Div(a.Add(b)), d.batch)
}

// Variance returns αβ/((α + β)²·(α + β + 1)).
func (d *Beta) Variance() *tensor.Tensor {
	if f := d.failed(); f != nil {
		return f
	}
	a, b := d.Concentration1, d.Concentration0
	t := a.Add(b)
	return expandTo(a.Mul(b).Div(t.Mul(t).Mul(t.Add(scalar(1)))), d.batch)
}
