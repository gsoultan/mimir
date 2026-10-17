package distributions_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/distributions"
	"github.com/gsoultan/mimir/tensor"
)

func data(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, err := x.Cast(tensor.Float32).Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func vec(v ...float32) *tensor.Tensor { return tensor.FromSlice(v, len(v)) }

// samples is the number of samples moment checks draw.
const samples = 1 << 15

// checkMoments compares the mean and variance of each element of the
// samples x, [samples, ...], with those d reports, within six standard
// errors: of the mean from the reported variance, and of the variance
// from the samples' fourth moment.
func checkMoments(t *testing.T, d distributions.Distribution, x *tensor.Tensor) {
	t.Helper()
	want := slices.Concat([]int{samples}, d.BatchShape(), d.EventShape())
	if !slices.Equal(x.Shape(), want) {
		t.Fatalf("samples of shape %v, want %v", x.Shape(), want)
	}
	v := data(t, x)
	mean, variance := data(t, d.Mean()), data(t, d.Variance())
	b := len(mean)
	for j := range b {
		var m, m2, m4 float64
		for i := range samples {
			m += float64(v[i*b+j])
		}
		m /= samples
		for i := range samples {
			e := float64(v[i*b+j]) - m
			m2 += e * e
			m4 += e * e * e * e
		}
		m2 /= samples - 1
		m4 /= samples
		seMean := math.Sqrt(float64(variance[j]) / samples)
		seVar := math.Sqrt(max(m4-m2*m2, 0) / samples)
		if math.Abs(m-float64(mean[j])) > 6*seMean+1e-6 || math.Abs(m2-float64(variance[j])) > 6*seVar+1e-6 {
			t.Errorf("element %d: sample mean %.5g and variance %.5g, want %.5g and %.5g (standard errors %.2g, %.2g)",
				j, m, m2, mean[j], variance[j], seMean, seVar)
		}
	}
}

// TestSamplesHaveTheirMoments draws many samples of every family and
// compares their means and variances with Mean and Variance.
func TestSamplesHaveTheirMoments(t *testing.T) {
	l := tensor.FromSlice([]float32{1, 0, 0, 0.5, 2, 0, -0.3, 0.2, 0.7}, 3, 3)
	for name, d := range map[string]distributions.Distribution{
		"normal":              distributions.NewNormal(vec(-1, 2), vec(0.5, 3)),
		"multivariate normal": distributions.NewMultivariateNormal(vec(1, -2, 0.5), l),
		"bernoulli":           distributions.NewBernoulli(vec(0.1, 0.5, 0.97)),
		"uniform":             distributions.NewUniform(vec(-1, 5), vec(3, 5.5)),
		"exponential":         distributions.NewExponential(vec(0.5, 4)),
		"laplace":             distributions.NewLaplace(vec(1, -2), vec(0.5, 2)),
		"gamma":               distributions.NewGamma(vec(0.3, 1, 7.5), vec(2, 1, 0.5)),
		"beta":                distributions.NewBeta(vec(0.5, 2, 9), vec(0.5, 5, 1)),
		"dirichlet":           distributions.NewDirichlet(vec(0.4, 1, 3, 6)),
		"poisson":             distributions.NewPoisson(vec(0.2, 4, 15, 120)),
	} {
		t.Run(name, func(t *testing.T) { checkMoments(t, d, d.Sample(11, samples)) })
	}
}

// TestCategoricalSamplesFollowTheProbabilities checks the frequency of
// each category of many samples.
func TestCategoricalSamplesFollowTheProbabilities(t *testing.T) {
	p := []float64{0.05, 0.6, 0, 0.35}
	d := distributions.NewCategorical(tensor.FromSlice([]float32{1, 12, 0, 7}, 4))
	x := d.Sample(3, samples)
	if x.DType() != tensor.Int32 || !slices.Equal(x.Shape(), []int{samples}) {
		t.Fatalf("samples %s%v, want Int32[%d]", x.DType(), x.Shape(), samples)
	}
	v, err := x.Data[int32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	count := make([]float64, 4)
	for _, k := range v {
		count[k]++
	}
	for k, c := range count {
		f, se := c/samples, math.Sqrt(p[k]*(1-p[k])/samples)
		if math.Abs(f-p[k]) > 6*se {
			t.Errorf("category %d drawn %.4f of the time, want %.4f", k, f, p[k])
		}
	}
}

// TestRSampleGradients checks the gradients of the means of
// reparameterized samples with respect to the parameters against the
// derivatives of the means in closed form.
func TestRSampleGradients(t *testing.T) {
	ctx := t.Context()
	cases := []struct {
		name   string
		params []float32
		draw   func(p []*tensor.Tensor) *tensor.Tensor
		want   func(p []float64) []float64
	}{
		{"normal", []float32{0.3, 1.5}, func(p []*tensor.Tensor) *tensor.Tensor {
			x := distributions.NewNormal(p[0], p[1]).RSample(1, samples)
			return x.Mul(x).Mean() // E[x²] = μ² + σ²
		}, func(p []float64) []float64 { return []float64{2 * p[0], 2 * p[1]} }},
		{"gamma", []float32{2.5, 1.5}, func(p []*tensor.Tensor) *tensor.Tensor {
			return distributions.NewGamma(p[0], p[1]).RSample(2, samples).Mean() // α/β
		}, func(p []float64) []float64 { return []float64{1 / p[1], -p[0] / (p[1] * p[1])} }},
		{"gamma below 1", []float32{0.4, 1}, func(p []*tensor.Tensor) *tensor.Tensor {
			return distributions.NewGamma(p[0], p[1]).RSample(3, samples).Mean()
		}, func(p []float64) []float64 { return []float64{1 / p[1], -p[0] / (p[1] * p[1])} }},
		{"beta", []float32{2, 3}, func(p []*tensor.Tensor) *tensor.Tensor {
			return distributions.NewBeta(p[0], p[1]).RSample(4, samples).Mean() // a/(a+b)
		}, func(p []float64) []float64 {
			s := (p[0] + p[1]) * (p[0] + p[1])
			return []float64{p[1] / s, -p[0] / s}
		}},
		{"dirichlet", []float32{1.5, 0.7, 2}, func(p []*tensor.Tensor) *tensor.Tensor {
			x := distributions.NewDirichlet(tensor.Stack(0, p...)).RSample(5, samples)
			return x.Shrink([][2]int{{0, samples}, {0, 1}}).Mean() // a₀/Σa
		}, func(p []float64) []float64 {
			s := p[0] + p[1] + p[2]
			return []float64{(s - p[0]) / (s * s), -p[0] / (s * s), -p[0] / (s * s)}
		}},
		{"exponential", []float32{2}, func(p []*tensor.Tensor) *tensor.Tensor {
			return distributions.NewExponential(p[0]).RSample(6, samples).Mean()
		}, func(p []float64) []float64 { return []float64{-1 / (p[0] * p[0])} }},
		{"laplace", []float32{0.5, 2}, func(p []*tensor.Tensor) *tensor.Tensor {
			x := distributions.NewLaplace(p[0], p[1]).RSample(7, samples)
			return x.Mul(x).Mean() // μ² + 2b²
		}, func(p []float64) []float64 { return []float64{2 * p[0], 4 * p[1]} }},
		{"uniform", []float32{-1, 3}, func(p []*tensor.Tensor) *tensor.Tensor {
			return distributions.NewUniform(p[0], p[1]).RSample(8, samples).Mean()
		}, func([]float64) []float64 { return []float64{0.5, 0.5} }},
		{"multivariate normal", []float32{0.5, 2}, func(p []*tensor.Tensor) *tensor.Tensor {
			// Scale [[p0, 0], [p1, 1]]: E|x|² = p0² + p1² + 1.
			l := tensor.Stack(0, p[0], tensor.Scalar[float32](0), p[1], tensor.Scalar[float32](1)).Reshape(2, 2)
			x := distributions.NewMultivariateNormal(tensor.Zeros(tensor.Float32, 2), l).RSample(9, samples)
			return x.Mul(x).Sum(-1).Mean()
		}, func(p []float64) []float64 { return []float64{2 * p[0], 2 * p[1]} }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := make([]*tensor.Tensor, len(c.params))
			wide := make([]float64, len(c.params))
			for i, v := range c.params {
				p[i] = tensor.Scalar(v).RequiresGrad()
				wide[i] = float64(v)
			}
			if err := c.draw(p).Backward(ctx); err != nil {
				t.Fatal(err)
			}
			for i, w := range c.want(wide) {
				g := data(t, p[i].Grad())[0]
				// The gradient is a mean over the samples; its spread is
				// within a few percent at this many samples.
				if math.Abs(float64(g)-w) > 0.05*math.Max(1, math.Abs(w)) {
					t.Errorf("parameter %d: gradient %v, want about %v", i, g, w)
				}
			}
		})
	}
}

// TestSampleShapesAndSeeds checks the shapes of samples, that a seed
// draws the same values every time, and that Sample has no gradient.
func TestSampleShapesAndSeeds(t *testing.T) {
	loc := tensor.Zeros(tensor.Float32, 3, 1).RequiresGrad()
	d := distributions.NewNormal(loc, vec(1, 2))
	if b := d.BatchShape(); !slices.Equal(b, []int{3, 2}) {
		t.Errorf("batch shape %v, want [3 2]", b)
	}
	x := d.Sample(5, 4, 2)
	if !slices.Equal(x.Shape(), []int{4, 2, 3, 2}) {
		t.Errorf("sample shape %v, want [4 2 3 2]", x.Shape())
	}
	if x.Tracked() {
		t.Error("Sample requires gradients")
	}
	if !d.RSample(5).Tracked() {
		t.Error("RSample does not require gradients")
	}
	if a, b := data(t, d.Sample(5, 4, 2)), data(t, x); !slices.Equal(a, b) {
		t.Error("one seed drew different samples")
	}
	if a, b := data(t, d.Sample(6, 4, 2)), data(t, x); slices.Equal(a, b) {
		t.Error("two seeds drew the same samples")
	}
	m := distributions.NewDirichlet(tensor.Ones(tensor.Float32, 2, 5))
	if s := m.Sample(1, 3).Shape(); !slices.Equal(s, []int{3, 2, 5}) {
		t.Errorf("dirichlet sample shape %v, want [3 2 5]", s)
	}
	if s := m.LogProb(m.Sample(1, 3)).Shape(); !slices.Equal(s, []int{3, 2}) {
		t.Errorf("dirichlet log-density shape %v, want [3 2]", s)
	}
}

// TestSupportAndLimits checks values outside a support and divergences
// that are infinite.
func TestSupportAndLimits(t *testing.T) {
	inf := math.Inf(1)
	u := distributions.NewUniform(vec(0), vec(2))
	if got := data(t, u.LogProb(vec(-1, 0, 1, 2))); got[0] != float32(-inf) || got[3] != float32(-inf) || math.Abs(float64(got[1])+math.Ln2) > 1e-6 {
		t.Errorf("uniform log-densities %v, want [-Inf -ln2 -ln2 -Inf]", got)
	}
	if got := data(t, distributions.KL(u, distributions.NewUniform(vec(0.5), vec(3)))); got[0] != float32(inf) {
		t.Errorf("KL of uniforms whose supports do not nest %v, want +Inf", got)
	}
	p := distributions.NewCategorical(tensor.FromSlice([]float32{0, 1, 1}, 3))
	q := distributions.NewCategorical(tensor.FromSlice([]float32{1, 0, 1}, 3))
	if got := data(t, distributions.KL(p, q)); got[0] != float32(inf) {
		t.Errorf("KL of categoricals with a zero of q under p %v, want +Inf", got)
	}
	r := distributions.NewCategorical(tensor.Ones(tensor.Float32, 3))
	if got := data(t, distributions.KL(p, r)); math.Abs(float64(got[0])-math.Log(1.5)) > 1e-6 {
		t.Errorf("KL of a categorical with a zero probability %v, want ln 1.5", got)
	}
	if got := data(t, p.Entropy()); math.Abs(float64(got[0])-math.Ln2) > 1e-6 {
		t.Errorf("entropy of a categorical with a zero probability %v, want ln 2", got)
	}
	if got := data(t, distributions.NewBernoulli(vec(0, 1)).Entropy()); got[0] > 1e-5 || got[1] > 1e-5 {
		t.Errorf("entropy of certain coins %v, want about 0", got)
	}
}

// TestErrors checks that bad shapes, dtypes and pairs give errors rather
// than panics.
func TestErrors(t *testing.T) {
	for name, c := range map[string]struct {
		got  *tensor.Tensor
		want error
	}{
		"normal shapes":     {distributions.NewNormal(vec(1, 2), vec(1, 2, 3)).Sample(1), tensor.ErrShape},
		"normal dtype":      {distributions.NewNormal(tensor.Ones(tensor.Int32, 2), vec(1, 2)).Entropy(), tensor.ErrDType},
		"mvn shapes":        {distributions.NewMultivariateNormal(vec(1, 2), tensor.Ones(tensor.Float32, 3, 3)).Mean(), tensor.ErrShape},
		"mvn rank":          {distributions.NewMultivariateNormal(vec(1, 2), vec(1, 2)).LogProb(vec(0, 0)), tensor.ErrShape},
		"categorical empty": {distributions.NewCategorical(tensor.Ones(tensor.Float32, 2, 0)).Sample(1), tensor.ErrShape},
		"categorical sizes": {distributions.KL(distributions.NewCategorical(vec(1, 1)), distributions.NewCategorical(vec(1, 1, 1))), tensor.ErrShape},
		"no KL":             {distributions.KL(distributions.NewNormal(vec(0), vec(1)), distributions.NewLaplace(vec(0), vec(1))), distributions.ErrNoKL},
		"failed parameter":  {distributions.NewGamma(tensor.Failed(tensor.ErrIndex), vec(1)).RSample(1), tensor.ErrIndex},
	} {
		if err := c.got.Err(); !errors.Is(err, c.want) {
			t.Errorf("%s: %v, want %v", name, err, c.want)
		}
	}
}

// BenchmarkSampleAndLogProb measures drawing a batch of samples of a
// family and scoring them.
func BenchmarkSampleAndLogProb(b *testing.B) {
	for name, d := range map[string]distributions.Distribution{
		"normal":    distributions.NewNormal(tensor.Zeros(tensor.Float32, 256), tensor.Ones(tensor.Float32, 256)),
		"gamma":     distributions.NewGamma(tensor.Full(tensor.Float32, 2.5, 256), tensor.Ones(tensor.Float32, 256)),
		"dirichlet": distributions.NewDirichlet(tensor.Full(tensor.Float32, 0.8, 16, 16)),
		"poisson":   distributions.NewPoisson(tensor.Full(tensor.Float32, 30, 256)),
	} {
		b.Run(name, func(b *testing.B) {
			seed := uint64(0)
			for b.Loop() {
				seed++
				if err := d.LogProb(d.Sample(seed, 1024)).Realize(b.Context()); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
