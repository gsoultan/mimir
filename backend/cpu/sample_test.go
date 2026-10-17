package cpu

import (
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// samples runs the sampler op over n copies of the parameter p.
func samples(t *testing.T, op backend.Op, p float32, n int, seed uint64) []float32 {
	t.Helper()
	k := &backend.Kernel{Op: op, DType: backend.Float32, Shape: []int{n},
		In: []backend.View{backend.Contiguous([]int{1}).Expand([]int{n})}, Attrs: backend.Attrs{Seed: seed}}
	got, err := run[float32](t, k, f32buf(p))
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// moments returns the mean and variance of v.
func moments(v []float32) (mean, variance float64) {
	for _, x := range v {
		mean += float64(x)
	}
	mean /= float64(len(v))
	for _, x := range v {
		variance += (float64(x) - mean) * (float64(x) - mean)
	}
	return mean, variance / float64(len(v)-1)
}

// TestSamplersHaveTheirMoments checks the mean and variance of many
// samples of Gamma(α, 1) and Poisson(λ), both α and λ, within five
// standard errors, and that a seed gives the same samples every time
// whatever the number of workers.
func TestSamplersHaveTheirMoments(t *testing.T) {
	const n = 1 << 17
	for _, c := range []struct {
		op backend.Op
		p  float32
	}{
		{backend.OpRandGamma, 0.05}, {backend.OpRandGamma, 0.6}, {backend.OpRandGamma, 1}, {backend.OpRandGamma, 4.5}, {backend.OpRandGamma, 300},
		{backend.OpRandPoisson, 0.3}, {backend.OpRandPoisson, 6}, {backend.OpRandPoisson, 10}, {backend.OpRandPoisson, 77.5}, {backend.OpRandPoisson, 5000},
	} {
		v := samples(t, c.op, c.p, n, 9)
		mean, variance := moments(v)
		p := float64(c.p)
		// The variance of the sample variance is 2σ⁴/n plus the fourth
		// cumulant over n: 6α for a gamma, λ for a Poisson.
		k4 := p
		if c.op == backend.OpRandGamma {
			k4 = 6 * p
		}
		seMean, seVar := math.Sqrt(p/n), math.Sqrt((2*p*p+k4)/n)
		if math.Abs(mean-p) > 5*seMean || math.Abs(variance-p) > 5*seVar {
			t.Errorf("%s(%v): mean %v and variance %v, want %v ± %.3g and ± %.3g", c.op, p, mean, variance, p, 5*seMean, 5*seVar)
		}
		for _, x := range v {
			if !(x > 0) && !(c.op == backend.OpRandPoisson && x == 0) || c.op == backend.OpRandPoisson && x != float32(math.Floor(float64(x))) {
				t.Fatalf("%s(%v) drew %v", c.op, p, x)
			}
		}
		if again := samples(t, c.op, c.p, n, 9); !slices.Equal(v, again) {
			t.Errorf("%s(%v) drew different samples from one seed", c.op, p)
		}
	}
}

// TestSamplersRefuseBadParameters checks the samples of parameters
// outside the samplers' domains.
func TestSamplersRefuseBadParameters(t *testing.T) {
	for _, c := range []struct {
		op   backend.Op
		p    float32
		want float32
	}{
		{backend.OpRandGamma, 0, float32(math.NaN())}, {backend.OpRandGamma, -1, float32(math.NaN())},
		{backend.OpRandGamma, float32(math.NaN()), float32(math.NaN())}, {backend.OpRandGamma, float32(math.Inf(1)), float32(math.Inf(1))},
		{backend.OpRandPoisson, 0, 0}, {backend.OpRandPoisson, -1, float32(math.NaN())},
	} {
		got := samples(t, c.op, c.p, 3, 1)[2]
		if got != c.want && !(math.IsNaN(float64(got)) && math.IsNaN(float64(c.want))) {
			t.Errorf("%s(%v) = %v, want %v", c.op, c.p, got, c.want)
		}
	}
}

// TestGammaGradKeepsTheCDF checks the implicit gradient dx/dα of gamma
// samples: moving α by δ and x by δ·dx/dα leaves P(α, x) as it was, to
// second order in δ.
func TestGammaGradKeepsTheCDF(t *testing.T) {
	cdf := func(a, x float64) float64 {
		if x < a+1 {
			return gammaP(a, x)
		}
		return 1 - gammaQ(a, x)
	}
	for _, alpha := range []float64{0.02, 0.3, 1, 2.5, 17, 400, 3e4, 3e5} {
		for _, u := range []float64{0.001, 0.1, 0.5, 0.9, 0.999} {
			// The quantile x of u, by bisection on the CDF.
			lo, hi := 0.0, alpha+40*math.Sqrt(alpha)+40
			for range 200 {
				if mid := (lo + hi) / 2; cdf(alpha, mid) < u {
					lo = mid
				} else {
					hi = mid
				}
			}
			x := (lo + hi) / 2
			if x == 0 {
				continue
			}
			g := gammaGrad(alpha, x)
			d := 1e-4 * alpha
			want := (quantileShift(cdf, alpha+d, x, u) - quantileShift(cdf, alpha-d, x, u)) / (2 * d)
			if math.Abs(g-want) > 1e-4*math.Max(math.Abs(want), x/alpha) {
				t.Errorf("α %v, x %v: dx/dα %v, finite difference of the quantile %v", alpha, x, g, want)
			}
		}
	}
	for _, c := range [][2]float64{{0, 1}, {-1, 1}, {1, -1}, {math.NaN(), 1}} {
		if g := gammaGrad(c[0], c[1]); !math.IsNaN(g) {
			t.Errorf("gammaGrad(%v, %v) = %v, want NaN", c[0], c[1], g)
		}
	}
}

// quantileShift returns the quantile of u under Gamma(a), found by
// Newton's method from x, near it.
func quantileShift(cdf func(a, x float64) float64, a, x, u float64) float64 {
	for range 50 {
		lg, _ := math.Lgamma(a)
		p := math.Exp((a-1)*math.Log(x) - x - lg)
		step := (cdf(a, x) - u) / p
		x -= step
		if math.Abs(step) < 1e-15*x {
			break
		}
	}
	return x
}

// BenchmarkSamplers measures a million samples and implicit gradients.
func BenchmarkSamplers(b *testing.B) {
	const n = 1 << 20
	be := New()
	for _, c := range []struct {
		name string
		op   backend.Op
		p    float32
	}{{"gamma/0.5", backend.OpRandGamma, 0.5}, {"gamma/4", backend.OpRandGamma, 4}, {"poisson/3", backend.OpRandPoisson, 3}, {"poisson/50", backend.OpRandPoisson, 50}, {"gamma_grad/4", backend.OpGammaGrad, 4}} {
		b.Run(c.name, func(b *testing.B) {
			k := &backend.Kernel{Op: c.op, DType: backend.Float32, Shape: []int{n}, In: []backend.View{backend.Contiguous([]int{1}).Expand([]int{n})}, Attrs: backend.Attrs{Seed: 3}}
			ins := []backend.Buffer{f32buf(c.p)}
			if c.op == backend.OpGammaGrad {
				x := make([]float32, n)
				for i := range x {
					x[i] = 1 + 6*float32(i)/n
				}
				k.In = append(k.In, backend.Contiguous([]int{n}))
				ins = append(ins, f32buf(x...))
			}
			out, _ := be.Alloc(backend.Float32, n)
			b.SetBytes(4 * n)
			for b.Loop() {
				if err := be.Run(b.Context(), k, []backend.Buffer{out}, ins); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
