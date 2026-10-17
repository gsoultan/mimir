package cpu

import (
	"math"

	"github.com/gsoultan/mimir/backend"
)

// The special functions compute in float64 and round once, so each
// float32 result is the correctly rounded value but in rare ties: the
// error of math.Lgamma, math.Erf and the series below is far smaller
// than float32's spacing.

// specialLoop sets dst[i] = op(a[i]) for a special function op.
func specialLoop[T number](op backend.Op, dst, a []T) {
	a = a[:len(dst)]
	var f func(float64) float64
	switch op {
	case backend.OpLog1p:
		f = math.Log1p
	case backend.OpExpm1:
		f = math.Expm1
	case backend.OpErf:
		f = math.Erf
	case backend.OpErfinv:
		f = math.Erfinv
	case backend.OpLgamma:
		f = lgamma
	case backend.OpDigamma:
		f = digamma
	case backend.OpTrigamma:
		f = trigamma
	default:
		return
	}
	for i, x := range a {
		dst[i] = T(f(float64(x)))
	}
}

// lgamma returns log|Γ(x)|, +Inf at the poles and at ±Inf.
func lgamma(x float64) float64 {
	if math.IsInf(x, -1) {
		return math.Inf(1)
	}
	v, _ := math.Lgamma(x)
	return v
}

// sinCosPi returns sin(πx) and cos(πx), exact at the integers and half
// integers: x is reduced to [-1, 1] first, which is exact in floats.
func sinCosPi(x float64) (s, c float64) {
	r := x - 2*math.Round(x/2)
	switch r {
	case 0:
		return 0, 1
	case 0.5:
		return 1, 0
	case -0.5:
		return -1, 0
	case 1, -1:
		return 0, -1
	}
	return math.Sin(math.Pi * r), math.Cos(math.Pi * r)
}

// digamma returns ψ(x), the derivative of log Γ(x), like torch.digamma:
// -Inf at +0, +Inf at -0, NaN at the negative integers and at -Inf. It
// reflects negative x to 1 - x, steps x up to 10 by ψ(x) = ψ(x+1) - 1/x
// and sums the asymptotic series there, whose error is below 1e-14.
func digamma(x float64) float64 {
	switch {
	case x == 0:
		return math.Copysign(math.Inf(1), -x)
	case math.IsInf(x, 1):
		return x
	case x < 0:
		if x == math.Floor(x) {
			return math.NaN()
		}
		s, c := sinCosPi(x)
		return digamma(1-x) - math.Pi*c/s
	}
	r := 0.0
	for ; x < 10; x++ {
		r -= 1 / x
	}
	x2 := 1 / (x * x)
	return r + math.Log(x) - 0.5/x - x2*(1.0/12-x2*(1.0/120-x2*(1.0/252-x2*(1.0/240-x2/132))))
}

// trigamma returns ψ'(x), +Inf at the non-positive integers, like
// torch.polygamma(1, x), by the same reflection, recurrence and
// asymptotic series as digamma.
func trigamma(x float64) float64 {
	switch {
	case x <= 0 && x == math.Floor(x):
		return math.Inf(1)
	case math.IsInf(x, 1):
		return 0
	case x < 0:
		s, _ := sinCosPi(x)
		return math.Pi*math.Pi/(s*s) - trigamma(1-x)
	}
	r := 0.0
	for ; x < 10; x++ {
		r += 1 / (x * x)
	}
	x2 := 1 / (x * x)
	return r + 1/x + x2/2 + x2/x*(1.0/6-x2*(1.0/30-x2*(1.0/42-x2*(1.0/30-x2*5/66))))
}
