package cluster

import (
	"context"
	"math"
	"runtime"

	"github.com/gsoultan/mimir/tabular"
)

// Mixture is a Gaussian mixture: each group is a normal distribution with
// its own mean and a variance per feature, and a weight, as
// scikit-learn's GaussianMixture with covariance_type="diag".
type Mixture struct {
	Features int
	Scaler   tabular.Scaler
	Weights  []float32 // K weights that sum to 1
	Means    []float32 // K×Features, standardized
	Vars     []float32 // K×Features, standardized
	// LogLikelihood is the mean log-likelihood of the training rows.
	LogLikelihood float64
	Iterations    int
}

// K returns the number of groups.
func (m *Mixture) K() int { return len(m.Weights) }

// minVar is added to every variance, as scikit-learn's reg_covar, so a
// group of one value has a density.
const minVar = 1e-6

// GaussianMixture fits a mixture of k normal distributions to the rows of
// x, features numbers each, by expectation-maximization from a k-means
// grouping.
func GaussianMixture(ctx context.Context, x []float32, features, k int, opts ...Option) (*Mixture, error) {
	c, n, err := options(x, features, k, opts)
	if err != nil {
		return nil, err
	}
	km, err := KMeans(ctx, x, features, k, append(opts, Inits(1))...)
	if err != nil {
		return nil, err
	}
	f := features
	z := km.Scaler.Transform(x)
	m := &Mixture{Features: f, Scaler: km.Scaler, Weights: make([]float32, k), Means: append([]float32(nil), km.Centers...), Vars: make([]float32, k*f)}
	// Start each group from the rows k-means gave it.
	assign := make([]int, n)
	nearestAll(z, km.Centers, f, assign, c.workers)
	resp := make([]float64, n*k)
	for i, a := range assign {
		resp[i*k+a] = 1
	}
	m.maximize(z, resp)
	prev := math.Inf(-1)
	for it := range c.iters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		m.Iterations = it + 1
		ll := m.expect(z, resp, c.workers)
		m.maximize(z, resp)
		m.LogLikelihood = ll
		if ll-prev < c.tol {
			break
		}
		prev = ll
	}
	m.LogLikelihood = m.expect(z, resp, c.workers)
	return m, nil
}

// expect sets resp to each row's probability of each group and returns
// the rows' mean log-likelihood.
func (m *Mixture) expect(z []float32, resp []float64, workers int) float64 {
	f, k := m.Features, m.K()
	n := len(z) / f
	parts := make([]float64, workers)
	rows(n, workers, func(w, lo, hi int) {
		for i := lo; i < hi; i++ {
			r := resp[i*k : (i+1)*k]
			parts[w] += m.logProbs(z[i*f:(i+1)*f], r)
		}
	})
	var s float64
	for _, v := range parts {
		s += v
	}
	return s / float64(n)
}

// logProbs sets r to the row's probability of each group and returns the
// row's log-likelihood.
func (m *Mixture) logProbs(row []float32, r []float64) float64 {
	f := m.Features
	hi := math.Inf(-1)
	for a := range r {
		l := math.Log(math.Max(float64(m.Weights[a]), 1e-300))
		for j, v := range row {
			vr := float64(m.Vars[a*f+j])
			d := float64(v - m.Means[a*f+j])
			l -= 0.5 * (math.Log(2*math.Pi*vr) + d*d/vr)
		}
		r[a] = l
		hi = math.Max(hi, l)
	}
	var sum float64
	for a, l := range r {
		r[a] = math.Exp(l - hi)
		sum += r[a]
	}
	for a := range r {
		r[a] /= sum
	}
	return hi + math.Log(sum)
}

// maximize sets m's weights, means and variances from the
// responsibilities resp.
func (m *Mixture) maximize(z []float32, resp []float64) {
	f, k := m.Features, m.K()
	n := len(z) / f
	nk := make([]float64, k)
	mean := make([]float64, k*f)
	sq := make([]float64, k*f)
	for i := range n {
		row := z[i*f : (i+1)*f]
		for a := range k {
			p := resp[i*k+a]
			if p == 0 {
				continue
			}
			nk[a] += p
			for j, v := range row {
				mean[a*f+j] += p * float64(v)
				sq[a*f+j] += p * float64(v) * float64(v)
			}
		}
	}
	for a := range k {
		w := nk[a] + 10*math.SmallestNonzeroFloat64
		m.Weights[a] = float32(nk[a] / float64(n))
		for j := range f {
			mu := mean[a*f+j] / w
			m.Means[a*f+j] = float32(mu)
			m.Vars[a*f+j] = float32(math.Max(sq[a*f+j]/w-mu*mu, 0) + minVar)
		}
	}
}

// Probabilities returns K probabilities per row of x: how likely each
// group is to have drawn it.
func (m *Mixture) Probabilities(x []float32) []float32 {
	z := m.Scaler.Transform(x)
	f, k := m.Features, m.K()
	out := make([]float32, len(z)/f*k)
	rows(len(z)/f, runtime.GOMAXPROCS(0), func(_, lo, hi int) {
		r := make([]float64, k)
		for i := lo; i < hi; i++ {
			m.logProbs(z[i*f:(i+1)*f], r)
			for a, p := range r {
				out[i*k+a] = float32(p)
			}
		}
	})
	return out
}

// Predict is Probabilities, so a Mixture is a tabular.Model whose classes
// are groups.
func (m *Mixture) Predict(x []float32) []float32 { return m.Probabilities(x) }

// Assign returns the most likely group of each row of x.
func (m *Mixture) Assign(x []float32) []int {
	p, k := m.Probabilities(x), m.K()
	out := make([]int, len(p)/k)
	for i := range out {
		for a := range k {
			if p[i*k+a] > p[i*k+out[i]] {
				out[i] = a
			}
		}
	}
	return out
}

// LogDensity returns the log-likelihood of each row of x under m, in
// standardized units: low for rows unlike any group.
func (m *Mixture) LogDensity(x []float32) []float64 {
	z := m.Scaler.Transform(x)
	f, k := m.Features, m.K()
	out := make([]float64, len(z)/f)
	r := make([]float64, k)
	for i := range out {
		out[i] = m.logProbs(z[i*f:(i+1)*f], r)
	}
	return out
}
