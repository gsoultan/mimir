// Package bayes fits naive Bayes classifiers to tables of numbers, like
// scikit-learn's GaussianNB, MultinomialNB and BernoulliNB: each class's
// features are taken as independent, so a class's likelihood is the
// product of one simple distribution per feature, fit in one pass.
//
//	m, err := bayes.Gaussian(ctx, d)
//	if err != nil { ... }
//	p := m.Predict(rows) // a probability for each class of each row
//
// Gaussian suits measurements; Multinomial counts, such as how often each
// word appears in a text; Bernoulli whether each word appears at all.
// They fit in one pass, need few rows, and make a strong first model for
// text. A missing value, NaN, is left out of a row's likelihood. A Model
// holds only exported fields, so encoding/json saves and loads it; Module
// serves it as an nn.Module.
package bayes

import (
	"context"
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Model is a fitted naive Bayes classifier. Every kind's log-likelihood
// of class c for a row x is
//
//	Prior[c] + Σ_j (Square[c,j]·x_j² + Linear[c,j]·x_j + Const[c,j])
//
// over the features x_j that aren't missing, x_j first binarized for
// Bernoulli.
type Model struct {
	Kind     string // "gaussian", "multinomial" or "bernoulli"
	Features int
	Classes  int
	// Prior is each class's log prior.
	Prior []float32
	// Square, Linear and Const hold Features numbers for each class,
	// class by class.
	Square, Linear, Const []float32
	// Binarize is the value above which a Bernoulli feature counts as
	// present.
	Binarize float32
}

// Option configures a fit.
type Option func(*config)

type config struct {
	smoothing, alpha float64
	binarize         float64
	classes          int
	priors           []float64
}

// VarSmoothing adds this share of the largest feature variance to every
// variance of a Gaussian fit, for stability. The default is 1e-9,
// scikit-learn's.
func VarSmoothing(s float64) Option { return func(c *config) { c.smoothing = s } }

// Alpha sets the additive (Laplace) smoothing of Multinomial and
// Bernoulli counts. The default is 1.
func Alpha(a float64) Option { return func(c *config) { c.alpha = a } }

// Binarize sets the value above which a Bernoulli feature counts as
// present. The default is 0.
func Binarize(t float64) Option { return func(c *config) { c.binarize = t } }

// Classes sets the number of classes, for when the last ones are missing
// from the training rows. The default counts them in d.Y.
func Classes(k int) Option { return func(c *config) { c.classes = k } }

// Priors sets each class's prior probability instead of its share of the
// rows.
func Priors(p ...float64) Option { return func(c *config) { c.priors = p } }

func setup(ctx context.Context, d tabular.Data, opts []Option, c config) (config, *Model, error) {
	for _, o := range opts {
		o(&c)
	}
	if err := d.Check(tabular.Classification); err != nil {
		return c, nil, err
	}
	if err := ctx.Err(); err != nil {
		return c, nil, err
	}
	if c.smoothing < 0 || c.alpha < 0 || math.IsNaN(c.smoothing+c.alpha+c.binarize) {
		return c, nil, fmt.Errorf("bayes: smoothing %v, alpha %v: %w", c.smoothing, c.alpha, tensor.ErrShape)
	}
	m := &Model{Features: d.Features, Classes: d.Classes()}
	if c.classes > 0 {
		if c.classes < m.Classes || c.classes > tabular.MaxClasses {
			return c, nil, fmt.Errorf("bayes: %d classes for answers up to %d: %w", c.classes, m.Classes-1, tensor.ErrShape)
		}
		m.Classes = c.classes
	}
	k := m.Classes
	counts := make([]float64, k)
	for _, y := range d.Y {
		counts[int(y)]++
	}
	m.Prior = make([]float32, k)
	if c.priors != nil {
		if len(c.priors) != k {
			return c, nil, fmt.Errorf("bayes: %d priors for %d classes: %w", len(c.priors), k, tensor.ErrShape)
		}
		var sum float64
		for _, p := range c.priors {
			if !(p >= 0) {
				return c, nil, fmt.Errorf("bayes: prior %v: %w", p, tensor.ErrShape)
			}
			sum += p
		}
		for i, p := range c.priors {
			m.Prior[i] = float32(math.Log(p / sum))
		}
	} else {
		for i, n := range counts {
			// A class with no rows can't be predicted.
			m.Prior[i] = float32(math.Log(n / float64(len(d.Y))))
		}
	}
	f := d.Features
	m.Square, m.Linear, m.Const = make([]float32, k*f), make([]float32, k*f), make([]float32, k*f)
	return c, m, nil
}

// Gaussian fits a normal distribution to each feature of each class.
func Gaussian(ctx context.Context, d tabular.Data, opts ...Option) (*Model, error) {
	c, m, err := setup(ctx, d, opts, config{smoothing: 1e-9})
	if err != nil {
		return nil, err
	}
	m.Kind = "gaussian"
	k, f := m.Classes, m.Features
	sum, sq, cnt := make([]float64, k*f), make([]float64, k*f), make([]float64, k*f)
	all, allSq, allCnt := make([]float64, f), make([]float64, f), make([]float64, f)
	for i := range d.Rows() {
		y := int(d.Y[i])
		for j, v := range d.Row(i) {
			if v != v {
				continue
			}
			x := float64(v)
			sum[y*f+j] += x
			sq[y*f+j] += x * x
			cnt[y*f+j]++
			all[j] += x
			allSq[j] += x * x
			allCnt[j]++
		}
	}
	var widest float64
	for j := range f {
		if allCnt[j] > 0 {
			mean := all[j] / allCnt[j]
			widest = max(widest, allSq[j]/allCnt[j]-mean*mean)
		}
	}
	eps := c.smoothing * max(widest, 0)
	for i := range k * f {
		mean, v := 0.0, 1.0
		if cnt[i] > 0 {
			mean = sum[i] / cnt[i]
			v = max(sq[i]/cnt[i]-mean*mean, 0)
		}
		v += eps
		if v == 0 {
			v = 1e-12
		}
		m.Square[i] = float32(-0.5 / v)
		m.Linear[i] = float32(mean / v)
		m.Const[i] = float32(-0.5*mean*mean/v - 0.5*math.Log(2*math.Pi*v))
	}
	return m, nil
}

// Multinomial fits each class's share of the total of each feature, for
// counts that are 0 or more.
func Multinomial(ctx context.Context, d tabular.Data, opts ...Option) (*Model, error) {
	c, m, err := setup(ctx, d, opts, config{alpha: 1})
	if err != nil {
		return nil, err
	}
	m.Kind = "multinomial"
	k, f := m.Classes, m.Features
	cnt := make([]float64, k*f)
	for i := range d.Rows() {
		y := int(d.Y[i])
		for j, v := range d.Row(i) {
			if v != v {
				continue
			}
			if v < 0 {
				return nil, fmt.Errorf("bayes: multinomial count %v at row %d: %w", v, i, tensor.ErrShape)
			}
			cnt[y*f+j] += float64(v)
		}
	}
	for y := range k {
		var total float64
		for j := range f {
			total += cnt[y*f+j] + c.alpha
		}
		for j := range f {
			m.Linear[y*f+j] = float32(math.Log((cnt[y*f+j] + c.alpha) / max(total, 1e-300)))
		}
	}
	return m, nil
}

// Bernoulli fits each class's chance that each feature is present, above
// the Binarize threshold.
func Bernoulli(ctx context.Context, d tabular.Data, opts ...Option) (*Model, error) {
	c, m, err := setup(ctx, d, opts, config{alpha: 1})
	if err != nil {
		return nil, err
	}
	m.Kind, m.Binarize = "bernoulli", float32(c.binarize)
	k, f := m.Classes, m.Features
	on, rows := make([]float64, k*f), make([]float64, k)
	for i := range d.Rows() {
		y := int(d.Y[i])
		rows[y]++
		for j, v := range d.Row(i) {
			if v == v && v > m.Binarize {
				on[y*f+j]++
			}
		}
	}
	for y := range k {
		for j := range f {
			p := (on[y*f+j] + c.alpha) / (rows[y] + 2*c.alpha)
			p = min(max(p, 1e-12), 1-1e-12)
			m.Linear[y*f+j] = float32(math.Log(p) - math.Log(1-p))
			m.Const[y*f+j] = float32(math.Log(1 - p))
		}
	}
	return m, nil
}

// LogLikelihood returns each row's joint log-likelihood of each class,
// Classes per row.
func (m *Model) LogLikelihood(x []float32) []float64 {
	f, k := m.Features, m.Classes
	if f <= 0 || len(x)%f != 0 {
		panic(fmt.Sprintf("bayes: %d numbers for rows of %d features", len(x), f))
	}
	n := len(x) / f
	out := make([]float64, n*k)
	for i := range n {
		row := x[i*f : (i+1)*f]
		for c := range k {
			s := float64(m.Prior[c])
			for j, v := range row {
				if v != v {
					continue
				}
				if m.Kind == "bernoulli" {
					v = 0
					if row[j] > m.Binarize {
						v = 1
					}
				}
				at := c*f + j
				xv := float64(v)
				s += float64(m.Square[at])*xv*xv + float64(m.Linear[at])*xv + float64(m.Const[at])
			}
			out[i*k+c] = s
		}
	}
	return out
}

// Predict answers each row of x, rows of m.Features numbers one after
// another, with m.Classes probabilities per row. It panics if x is not a
// whole number of rows.
func (m *Model) Predict(x []float32) []float32 {
	ll := m.LogLikelihood(x)
	k := m.Classes
	out := make([]float32, len(ll))
	for i := 0; i < len(ll); i += k {
		top := math.Inf(-1)
		for _, v := range ll[i : i+k] {
			top = max(top, v)
		}
		var sum float64
		for _, v := range ll[i : i+k] {
			sum += math.Exp(v - top)
		}
		for c, v := range ll[i : i+k] {
			out[i+c] = float32(math.Exp(v-top) / sum)
		}
	}
	return out
}

var _ tabular.Model = (*Model)(nil)
