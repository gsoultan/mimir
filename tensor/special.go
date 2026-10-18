package tensor

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// The special functions of torch.special that probability distributions
// need. Each is one kernel on every device, accurate where composing it
// from other operations would not be, such as log(1 + t) for small t.
// They take float tensors only.

// Log1p returns log(1 + t), accurate for t near zero, like torch.log1p.
func (t *Tensor) Log1p() *Tensor { return t.unary(backend.OpLog1p) }

// Expm1 returns e^t - 1, accurate for t near zero, like torch.expm1.
func (t *Tensor) Expm1() *Tensor { return t.unary(backend.OpExpm1) }

// Erf returns the error function of t, like torch.erf.
func (t *Tensor) Erf() *Tensor { return t.unary(backend.OpErf) }

// Erfinv returns the inverse of the error function, like torch.erfinv:
// ±Inf at ±1 and NaN outside [-1, 1].
func (t *Tensor) Erfinv() *Tensor { return t.unary(backend.OpErfinv) }

// Lgamma returns the natural logarithm of the absolute value of the gamma
// function, like torch.lgamma: +Inf at zero and the negative integers.
func (t *Tensor) Lgamma() *Tensor { return t.unary(backend.OpLgamma) }

// Digamma returns ψ(t), the derivative of Lgamma, like torch.digamma: -Inf
// at +0 and NaN at the negative integers. Its gradient has no derivative
// in turn, unlike Lgamma's.
func (t *Tensor) Digamma() *Tensor { return t.unary(backend.OpDigamma) }

// trigamma returns ψ'(t), the derivative of Digamma, which has none.
func (t *Tensor) trigamma() *Tensor { return t.unary(backend.OpTrigamma) }

// specialRule returns the rule of a special function: the gradient, or
// the tangent, times its derivative at the input x, whose result is y.
func specialRule(op backend.Op) rule {
	var d func(x, y *Tensor) *Tensor
	switch op {
	case backend.OpLog1p:
		d = func(x, _ *Tensor) *Tensor { return x.Add(x.scalar(1)).Recip() }
	case backend.OpExpm1:
		d = func(_, y *Tensor) *Tensor { return y.Add(y.scalar(1)) }
	case backend.OpErf:
		d = func(x, _ *Tensor) *Tensor { return x.Mul(x).Neg().Exp().Mul(x.scalar(2 / math.SqrtPi)) }
	case backend.OpErfinv:
		d = func(_, y *Tensor) *Tensor { return y.Mul(y).Exp().Mul(y.scalar(math.SqrtPi / 2)) }
	case backend.OpLgamma:
		d = func(x, _ *Tensor) *Tensor { return x.Digamma() }
	case backend.OpDigamma:
		d = func(x, _ *Tensor) *Tensor { return x.trigamma() }
	default:
		return failing(fmt.Errorf("tensor: derivative of %s: %w", op, ErrNotDifferentiable))
	}
	return rule{
		vjp: func(g *Tensor, in []*Tensor, y *Tensor) []*Tensor { return grads(g.Mul(d(in[0], y))) },
		jvp: func(t, in []*Tensor, y *Tensor) *Tensor { return t[0].Mul(d(in[0], y)) },
	}
}

// RandGamma returns a sample of Gamma(α, 1) for each element α of alpha,
// determined by seed, like torch._standard_gamma: samples of Gamma(α, β)
// are these divided by β. A parameter that is not positive gives NaN,
// and samples are never below the smallest normal float32.
//
// The gradient with respect to alpha is the implicit reparameterization
// gradient, as PyTorch computes it: the derivative of the sample that
// keeps its quantile fixed. It has no derivative in turn.
//
// The samples and their gradients are computed on the CPU in float64, and
// alpha on a GPU is copied there and back, as the factorizations are. A
// half alpha gives half samples.
func RandGamma(seed uint64, alpha *Tensor) *Tensor {
	return sample(backend.OpRandGamma, seed, alpha)
}

// RandPoisson returns a sample of Poisson(λ) for each element λ of rate,
// determined by seed, like torch.poisson: a count, held as a float. A
// negative rate gives NaN. Samples have no gradient, as in PyTorch, and
// are computed on the CPU like RandGamma's.
func RandPoisson(seed uint64, rate *Tensor) *Tensor {
	return sample(backend.OpRandPoisson, seed, rate)
}

// sample draws a sample of op for each element of the float tensor p.
func sample(op backend.Op, seed uint64, p *Tensor) *Tensor {
	if err := firstErr(p); err != nil {
		return fail(err)
	}
	dt := p.node.DType
	if !dt.IsFloat() {
		return fail(fmt.Errorf("tensor: %s of %s: %w", op, dt, ErrDType))
	}
	if batchLevel(p) != 0 {
		return fail(fmt.Errorf("tensor: %s: %w", op, ErrNotBatchable))
	}
	a := p.Cast(Float32)
	x, d := a.onCPU()
	if err := firstErr(x); err != nil {
		return fail(err)
	}
	z := wrap(graph.Sample(op, x.node, seed))
	out := z.To(d)
	if op == backend.OpRandGamma {
		out = track(out, func(g *Tensor) []*Tensor {
			x, _ := a.onCPU()
			if err := firstErr(x); err != nil {
				return grads(fail(err))
			}
			return grads(g.Mul(wrap(graph.GammaGrad(x.node, z.node)).To(d)))
		}, a)
	}
	return out.Cast(dt)
}
