// Package forecast predicts what comes next in series of numbers measured
// at equal steps, such as daily sales of each store, by gradient-boosted
// trees over each step's recent values, as the global models that win
// forecasting contests do: one model learns from every series at once.
//
//	m, err := forecast.Fit(ctx, series, forecast.Season(7))
//	if err != nil { ... }
//	next := m.Forecast(series[0], 14) // the next two weeks of series 0
//
// Each series is scaled by the mean size of its values, so a model learns
// the shape of series in thousands and in tens alike. A value of NaN is
// missing. It forecasts one step at a time, feeding each forecast back as
// the next step's latest value.
package forecast

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/tree"
)

// Option configures Fit.
type Option func(*config)

type config struct {
	lags, season, holdout int
	trees                 []tree.Option
}

// Lags sets how many of the latest values each step reads. The default
// is twice the season, and at least 14.
func Lags(n int) Option { return func(c *config) { c.lags = n } }

// Season sets the length of the series' cycle in steps, such as 7 for
// days of a week or 12 for months of a year. The default is 0, none.
func Season(n int) Option { return func(c *config) { c.season = n } }

// Holdout sets how many of each series' last steps Fit keeps aside to
// know when more trees stop helping. The default is 0: a tenth of the
// steps, up to the season or 14.
func Holdout(n int) Option { return func(c *config) { c.holdout = n } }

// Trees passes options to the boosted trees, such as tree.Trees.
func Trees(opts ...tree.Option) Option {
	return func(c *config) { c.trees = append(c.trees, opts...) }
}

// Model forecasts series.
type Model struct {
	Lags, Season int
	// Trees answers a step's value, scaled, from its features.
	Trees *tree.Model
	// Error is the mean absolute error of one-step forecasts of the held
	// out steps, scaled, and Naive that of repeating the value a season
	// back, or the last value without a season: below 1, Error/Naive says
	// the model beats the naive forecast.
	Error, Naive float64
}

// Features returns the number of features a step reads.
func (m *Model) Features() int { return features(m.Lags, m.Season) }

func features(lags, season int) int {
	f := lags + 2 // the lags, the mean of the lags, how many are present
	if season > 0 {
		f += 2 // the position in the season, as a sine and a cosine
	}
	return f
}

// scale returns the mean absolute value of the values of s that are not
// NaN, or 1 when there are none or all are 0.
func scale(s []float32) float32 {
	var sum float64
	n := 0
	for _, v := range s {
		if v == v {
			sum += math.Abs(float64(v))
			n++
		}
	}
	if n == 0 || sum == 0 {
		return 1
	}
	return float32(sum / float64(n))
}

// row appends the features of the step at t of the scaled series s, whose
// earlier steps are s[:t], to dst.
func row(dst []float32, s []float32, t, lags, season int) []float32 {
	var sum float64
	n := 0
	for l := 1; l <= lags; l++ {
		v := float32(math.NaN())
		if t-l >= 0 {
			v = s[t-l]
		}
		dst = append(dst, v)
		if v == v {
			sum += float64(v)
			n++
		}
	}
	mean := float32(math.NaN())
	if n > 0 {
		mean = float32(sum / float64(n))
	}
	dst = append(dst, mean, float32(n))
	if season > 0 {
		a := 2 * math.Pi * float64(t%season) / float64(season)
		dst = append(dst, float32(math.Sin(a)), float32(math.Cos(a)))
	}
	return dst
}

// Fit learns to forecast the series, each at least Lags+2 steps long.
func Fit(ctx context.Context, series [][]float32, opts ...Option) (*Model, error) {
	c := &config{}
	for _, o := range opts {
		o(c)
	}
	if c.season < 0 || c.lags < 0 || c.holdout < 0 {
		return nil, fmt.Errorf("forecast: lags %d, season %d, holdout %d: %w", c.lags, c.season, c.holdout, tensor.ErrShape)
	}
	if c.lags == 0 {
		c.lags = max(2*c.season, 14)
	}
	longest := 0
	for _, s := range series {
		longest = max(longest, len(s))
	}
	if len(series) == 0 || longest < 3 {
		return nil, fmt.Errorf("forecast: %d series of at most %d steps: %w", len(series), longest, tensor.ErrShape)
	}
	c.lags = min(c.lags, longest-2)
	hold := c.holdout
	if hold == 0 {
		hold = max(1, min(longest/10, max(c.season, 14)))
	}
	f := features(c.lags, c.season)
	var train, val tabular.Data
	train.Features, val.Features = f, f
	var naive, naiveN float64
	for _, s := range series {
		k := scale(s)
		z := make([]float32, len(s))
		for i, v := range s {
			z[i] = v / k
		}
		cut := max(len(z)-hold, 1)
		for t := 1; t < len(z); t++ {
			if z[t] != z[t] {
				continue // no answer to learn
			}
			if t < cut {
				train.X = row(train.X, z, t, c.lags, c.season)
				train.Y = append(train.Y, z[t])
				continue
			}
			val.X = row(val.X, z, t, c.lags, c.season)
			val.Y = append(val.Y, z[t])
			back := 1
			if c.season > 0 && t >= c.season {
				back = c.season
			}
			if p := z[t-back]; p == p {
				naive += math.Abs(float64(p - z[t]))
				naiveN++
			}
		}
	}
	if train.Rows() < 2 {
		return nil, fmt.Errorf("forecast: %d steps to learn from: %w", train.Rows(), tensor.ErrShape)
	}
	topts := []tree.Option{tree.Trees(1000), tree.LearningRate(0.05), tree.MinLeaf(5)}
	if val.Rows() > 0 {
		topts = append(topts, tree.Validation(val, 50))
	}
	topts = append(topts, c.trees...)
	tm, err := tree.GradientBoosting(ctx, train, tabular.Regression, topts...)
	if err != nil {
		return nil, err
	}
	m := &Model{Lags: c.lags, Season: c.season, Trees: tm}
	if val.Rows() > 0 {
		p := tm.Predict(val.X)
		for i, y := range val.Y {
			m.Error += math.Abs(float64(p[i] - y))
		}
		m.Error /= float64(val.Rows())
		if naiveN > 0 {
			m.Naive = naive / naiveN
		}
	}
	return m, nil
}

// Forecast returns the next steps values of the series whose values so
// far are history.
func (m *Model) Forecast(history []float32, steps int) []float32 {
	k := scale(history)
	z := make([]float32, len(history), len(history)+steps)
	for i, v := range history {
		z[i] = v / k
	}
	out := make([]float32, steps)
	var x []float32
	for s := range steps {
		t := len(z)
		x = row(x[:0], z, t, m.Lags, m.Season)
		v := m.Trees.Predict(x)[0]
		z = append(z, v)
		out[s] = v * k
	}
	return out
}

// Rows returns, for the step after the history, the features Trees reads
// and the scale its answer is in, for serving one step with Trees'
// Module.
func (m *Model) Rows(history []float32) (x []float32, size float32) {
	k := scale(history)
	z := slices.Clone(history)
	for i := range z {
		z[i] /= k
	}
	return row(nil, z, len(z), m.Lags, m.Season), k
}
