package tensor_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// TestSpecialFunctionGradientsMatchFiniteDifferences checks the gradient
// of every special function against central differences.
func TestSpecialFunctionGradientsMatchFiniteDifferences(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, c := range []struct {
		name   string
		lo, hi float32
		f      func(x *tensor.Tensor) *tensor.Tensor
	}{
		{"log1p", -0.5, 3, (*tensor.Tensor).Log1p},
		{"expm1", -2, 2, (*tensor.Tensor).Expm1},
		{"erf", -2, 2, (*tensor.Tensor).Erf},
		{"erfinv", -0.8, 0.8, (*tensor.Tensor).Erfinv},
		{"lgamma", 0.3, 6, (*tensor.Tensor).Lgamma},
		{"lgamma_negative", -2.9, -2.1, (*tensor.Tensor).Lgamma},
		{"digamma", 0.6, 6, (*tensor.Tensor).Digamma},
	} {
		t.Run(c.name, func(t *testing.T) {
			checkFiniteDifferences(t, gradCase{
				name: c.name,
				in:   [][]float32{vals(r, 12, c.lo, c.hi)},
				dims: [][]int{{3, 4}},
				f:    func(x []*tensor.Tensor) *tensor.Tensor { return c.f(x[0]) },
			})
		})
	}
}

// TestLgammaHasSecondDerivatives checks that the gradient of Lgamma,
// Digamma, has a derivative in turn, ψ', and that Digamma's has none.
func TestLgammaHasSecondDerivatives(t *testing.T) {
	x := tensor.FromSlice([]float32{0.5, 1, 1.5}, 3).RequiresGrad()
	d1 := tensor.Gradients(x.Lgamma().Sum(), x)[0]
	d2 := tensor.Gradients(d1.Sum(), x)[0]
	euler := 0.5772156649015329
	want1 := []float64{-euler - 2*math.Ln2, -euler, 2 - euler - 2*math.Ln2}
	want2 := []float64{math.Pi * math.Pi / 2, math.Pi * math.Pi / 6, math.Pi*math.Pi/2 - 4}
	for i, g := range data[float32](t, d1) {
		if math.Abs(float64(g)-want1[i]) > 1e-6 {
			t.Errorf("lgamma'(%v) = %v, want %v", []float32{0.5, 1, 1.5}[i], g, want1[i])
		}
	}
	for i, g := range data[float32](t, d2) {
		if math.Abs(float64(g)-want2[i]) > 1e-5 {
			t.Errorf("lgamma''(%v) = %v, want %v", []float32{0.5, 1, 1.5}[i], g, want2[i])
		}
	}
	d3 := tensor.Gradients(tensor.Gradients(x.Digamma().Sum(), x)[0].Sum(), x)[0]
	if err := d3.Err(); !errors.Is(err, tensor.ErrNotDifferentiable) {
		t.Errorf("second derivative of Digamma: %v, want ErrNotDifferentiable", err)
	}
}

// TestRandGammaAndPoisson checks the samplers' means, the implicit
// gradient of gamma samples, d E[z]/dα = 1, and the samplers' errors.
func TestRandGammaAndPoisson(t *testing.T) {
	const n = 1 << 16
	ctx := t.Context()
	alpha := tensor.FromSlice([]float32{0.4, 3}, 2, 1).RequiresGrad()
	z := tensor.RandGamma(5, alpha.Expand(2, n))
	m := z.Mean(1)
	if err := m.Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	means, g := data[float32](t, m), data[float32](t, alpha.Grad())
	for i, a := range []float64{0.4, 3} {
		se := math.Sqrt(a / n)
		if math.Abs(float64(means[i])-a) > 5*se {
			t.Errorf("mean of Gamma(%v) samples %v", a, means[i])
		}
		// The gradient of the mean is the mean of dz/dα, whose
		// expectation is 1 and spread about that of z/α.
		if math.Abs(float64(g[i])-1) > 5*math.Sqrt(1/(a*n))+1e-3 {
			t.Errorf("d mean / dα of Gamma(%v) samples %v, want 1", a, g[i])
		}
	}
	p := data[float32](t, tensor.RandPoisson(5, tensor.Full(tensor.BFloat16, 7, n)).Cast(tensor.Float32).Mean())
	if math.Abs(float64(p[0])-7) > 5*math.Sqrt(7.0/n) {
		t.Errorf("mean of Poisson(7) samples %v", p[0])
	}
	again := data[float32](t, tensor.RandGamma(5, alpha.Detach().Expand(2, n)).Mean(1))
	if again[0] != means[0] || again[1] != means[1] {
		t.Errorf("RandGamma drew %v, then %v from one seed", means, again)
	}
	if err := tensor.RandGamma(1, tensor.Ones(tensor.Int32, 3)).Err(); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("RandGamma of Int32: %v, want ErrDType", err)
	}
	if err := tensor.Ones(tensor.Int32, 3).Lgamma().Err(); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("Lgamma of Int32: %v, want ErrDType", err)
	}
}

// BenchmarkSpecialFunctions measures Lgamma and Digamma over a million
// elements.
func BenchmarkSpecialFunctions(b *testing.B) {
	x := tensor.Rand(1, 1<<20).Mul(tensor.Scalar[float32](20)).Add(tensor.Scalar[float32](0.1))
	if err := x.Realize(b.Context()); err != nil {
		b.Fatal(err)
	}
	for _, c := range []struct {
		name string
		f    func(*tensor.Tensor) *tensor.Tensor
	}{{"lgamma", (*tensor.Tensor).Lgamma}, {"digamma", (*tensor.Tensor).Digamma}, {"erf", (*tensor.Tensor).Erf}, {"log1p", (*tensor.Tensor).Log1p}} {
		b.Run(c.name, func(b *testing.B) {
			b.SetBytes(4 << 20)
			for b.Loop() {
				if err := c.f(x).Realize(b.Context()); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
