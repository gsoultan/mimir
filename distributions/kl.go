package distributions

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// KL returns the Kullback-Leibler divergence KL(p ‖ q) = E_p[log p - log
// q] of each pair of distributions of p's and q's batches, which
// broadcast, like torch.distributions.kl_divergence. It is defined in
// closed form between two distributions of one family: Normal,
// MultivariateNormal, Bernoulli, Categorical, Uniform, Exponential,
// Laplace, Gamma, Beta, Dirichlet or Poisson. Other pairs fail with
// ErrNoKL. Where p puts mass that q does not, the divergence is +Inf.
func KL(p, q Distribution) *tensor.Tensor {
	switch p := p.(type) {
	case *Normal:
		if q, ok := q.(*Normal); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klNormal(p, q) })
		}
	case *MultivariateNormal:
		if q, ok := q.(*MultivariateNormal); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klMultivariateNormal(p, q) })
		}
	case *Bernoulli:
		if q, ok := q.(*Bernoulli); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klBernoulli(p, q) })
		}
	case *Categorical:
		if q, ok := q.(*Categorical); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klCategorical(p, q) })
		}
	case *Uniform:
		if q, ok := q.(*Uniform); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klUniform(p, q) })
		}
	case *Exponential:
		if q, ok := q.(*Exponential); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor {
				r := q.Rate.Div(p.Rate)
				return r.Sub(r.Log()).Sub(scalar(1))
			})
		}
	case *Laplace:
		if q, ok := q.(*Laplace); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klLaplace(p, q) })
		}
	case *Gamma:
		if q, ok := q.(*Gamma); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klGamma(p, q) })
		}
	case *Beta:
		if q, ok := q.(*Beta); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klBeta(p, q) })
		}
	case *Dirichlet:
		if q, ok := q.(*Dirichlet); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor { return klDirichlet(p, q) })
		}
	case *Poisson:
		if q, ok := q.(*Poisson); ok {
			return kl2(p.err, q.err, func() *tensor.Tensor {
				return xlogy(p.Rate, p.Rate).Sub(xlogy(p.Rate, q.Rate)).Sub(p.Rate).Add(q.Rate)
			})
		}
	}
	return tensor.Failed(fmt.Errorf("distributions: KL of %T and %T: %w", p, q, ErrNoKL))
}

// kl2 returns the divergence f computes, or a tensor carrying the error
// of either distribution's construction.
func kl2(errP, errQ error, f func() *tensor.Tensor) *tensor.Tensor {
	switch {
	case errP != nil:
		return tensor.Failed(errP)
	case errQ != nil:
		return tensor.Failed(errQ)
	}
	return f()
}

// klNormal is ½·(σp²/σq² + (μp - μq)²/σq² - 1 - log(σp²/σq²)).
func klNormal(p, q *Normal) *tensor.Tensor {
	r := p.Scale.Div(q.Scale)
	r = r.Mul(r)
	t := p.Loc.Sub(q.Loc).Div(q.Scale)
	return r.Add(t.Mul(t)).Sub(scalar(1)).Sub(r.Log()).Mul(scalar(0.5))
}

// klMultivariateNormal is log|Lq| - log|Lp| + ½·(|Lq⁻¹·Lp|²_F + |Lq⁻¹·(μq
// - μp)|² - n).
func klMultivariateNormal(p, q *MultivariateNormal) *tensor.Tensor {
	if p.event[0] != q.event[0] {
		return tensor.Failed(fmt.Errorf("distributions: KL of multivariate normals of %d and %d dimensions: %w", p.event[0], q.event[0], tensor.ErrShape))
	}
	m := tensor.TriangularSolve(q.ScaleTril, p.ScaleTril, false)
	frob := m.Mul(m).Sum(-2, -1)
	return q.halfLogDet().Sub(p.halfLogDet()).
		Add(frob.Add(q.mahalanobis(q.Loc.Sub(p.Loc))).Sub(scalar(float64(p.event[0]))).Mul(scalar(0.5)))
}

// klBernoulli is p·log(p/q) + (1-p)·log((1-p)/(1-q)), from the log-odds.
func klBernoulli(p, q *Bernoulli) *tensor.Tensor {
	t1 := p.Probs.Mul(softplus(q.Logits.Neg()).Sub(softplus(p.Logits.Neg())))
	t2 := scalar(1).Sub(p.Probs).Mul(softplus(q.Logits).Sub(softplus(p.Logits)))
	return t1.Add(t2)
}

// klCategorical is Σ p·(log p - log q): +Inf where q is 0 and p is not,
// and 0 where p is 0.
func klCategorical(p, q *Categorical) *tensor.Tensor {
	if p.categories() != q.categories() {
		return tensor.Failed(fmt.Errorf("distributions: KL of categoricals of %d and %d categories: %w", p.categories(), q.categories(), tensor.ErrShape))
	}
	t := p.Probs.Mul(p.Logits.Sub(q.Logits))
	zero := scalar(0)
	t = tensor.Where(q.Probs.Equal(zero), scalar(math.Inf(1)), t)
	return tensor.Where(p.Probs.Equal(zero), zero, t).Sum(-1)
}

// klUniform is log((high_q - low_q)/(high_p - low_p)) where q's support
// covers p's, and +Inf where it does not.
func klUniform(p, q *Uniform) *tensor.Tensor {
	r := q.High.Sub(q.Low).Div(p.High.Sub(p.Low)).Log()
	outside := q.Low.Greater(p.Low).Add(q.High.Less(p.High)).Greater(scalar(0))
	return tensor.Where(outside, scalar(math.Inf(1)), r)
}

// klLaplace is -log(bp/bq) + |μp - μq|/bq + (bp/bq)·e^(-|μp - μq|/bp) - 1.
func klLaplace(p, q *Laplace) *tensor.Tensor {
	r := p.Scale.Div(q.Scale)
	d := p.Loc.Sub(q.Loc).Abs()
	return r.Log().Neg().Add(d.Div(q.Scale)).Add(r.Mul(d.Div(p.Scale).Neg().Exp())).Sub(scalar(1))
}

// klGamma is αq·log(βp/βq) + log Γ(αq) - log Γ(αp) + (αp - αq)·ψ(αp) +
// (βq - βp)·αp/βp.
func klGamma(p, q *Gamma) *tensor.Tensor {
	ap, bp, aq, bq := p.Concentration, p.Rate, q.Concentration, q.Rate
	return aq.Mul(bp.Div(bq).Log()).Add(aq.Lgamma()).Sub(ap.Lgamma()).
		Add(ap.Sub(aq).Mul(ap.Digamma())).Add(bq.Sub(bp).Mul(ap.Div(bp)))
}

// klBeta is log B(αq, βq) - log B(αp, βp) + (αp - αq)·ψ(αp) + (βp -
// βq)·ψ(βp) + (αq - αp + βq - βp)·ψ(αp + βp).
func klBeta(p, q *Beta) *tensor.Tensor {
	ap, bp, aq, bq := p.Concentration1, p.Concentration0, q.Concentration1, q.Concentration0
	return logBeta(aq, bq).Sub(logBeta(ap, bp)).
		Add(ap.Sub(aq).Mul(ap.Digamma())).Add(bp.Sub(bq).Mul(bp.Digamma())).
		Add(aq.Sub(ap).Add(bq).Sub(bp).Mul(ap.Add(bp).Digamma()))
}

// klDirichlet is log Γ(Σ αp) - log Γ(Σ αq) - Σ (log Γ(αp) - log Γ(αq)) +
// Σ (αp - αq)·(ψ(αp) - ψ(Σ αp)).
func klDirichlet(p, q *Dirichlet) *tensor.Tensor {
	if p.event[0] != q.event[0] {
		return tensor.Failed(fmt.Errorf("distributions: KL of dirichlets of %d and %d categories: %w", p.event[0], q.event[0], tensor.ErrShape))
	}
	ap, aq := p.Concentration, q.Concentration
	sp := ap.Sum(-1)
	return sp.Lgamma().Sub(aq.Sum(-1).Lgamma()).
		Sub(ap.Lgamma().Sub(aq.Lgamma()).Sum(-1)).
		Add(ap.Sub(aq).Mul(ap.Digamma().Sub(sp.Digamma().Unsqueeze(-1))).Sum(-1))
}
