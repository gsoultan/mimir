// Package linear fits linear regression and logistic regression to tables
// of numbers, with L1 and L2 penalties, like scikit-learn's ElasticNet and
// LogisticRegression.
//
//	m, err := linear.Logistic(ctx, d, linear.L2(1e-3))
//	if err != nil { ... }
//	p := m.Predict(row) // a probability for each class
//
// Both standardize each feature before they fit, so the penalties weigh
// every feature alike whatever its units, and report weights in the
// features' own units. A missing value, NaN, counts as the feature's
// mean over the training rows. A Model holds only exported fields, so
// encoding/json saves and loads it, and it is a tabular.Model.
package linear

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Model is a fitted linear model: for each output, a weight per feature
// and a bias, passed through Link.
type Model struct {
	Task     tabular.Task
	Features int
	// Classes is the number of classes of a Classification model, and 0
	// for Regression.
	Classes int
	// Outputs is 1 for Regression and two classes, Classes otherwise.
	Outputs int
	// Link is "identity" for Regression, "sigmoid" for two classes, where
	// the output is the log-odds of class 1, and "softmax" for more.
	Link string
	// Weights holds Features weights for each output, output by output.
	Weights []float32
	Bias    []float32
	// Fill is the value each feature takes when it is missing.
	Fill []float32
	// Iterations is how many passes the fit took.
	Iterations int
}

// Weight returns output o's weight of feature j.
func (m *Model) Weight(o, j int) float32 { return m.Weights[o*m.Features+j] }

// Predict answers each row of x, which holds rows of m.Features numbers
// one after another. For Classification it returns m.Classes
// probabilities per row; for Regression, one value per row. It panics if
// x is not a whole number of rows.
func (m *Model) Predict(x []float32) []float32 {
	f, k := m.Features, m.Outputs
	if f <= 0 || len(x)%f != 0 {
		panic(fmt.Sprintf("linear: %d numbers for rows of %d features", len(x), f))
	}
	n := len(x) / f
	out := make([]float32, n*max(k, m.Classes))
	raw := make([]float64, k)
	for i := range n {
		row := x[i*f : (i+1)*f]
		for o := range k {
			s := float64(m.Bias[o])
			w := m.Weights[o*f : (o+1)*f]
			for j, v := range row {
				if v != v {
					v = m.Fill[j]
				}
				s += float64(w[j]) * float64(v)
			}
			raw[o] = s
		}
		switch m.Link {
		case "sigmoid":
			p := 1 / (1 + math.Exp(-raw[0]))
			out[2*i], out[2*i+1] = float32(1-p), float32(p)
		case "softmax":
			softmax(raw)
			for o, p := range raw {
				out[i*k+o] = float32(p)
			}
		default:
			out[i] = float32(raw[0])
		}
	}
	return out
}

// Option configures Regression and Logistic.
type Option func(*config)

type config struct {
	l1, l2  float64
	iters   int
	tol     float64
	workers int
	classes int
}

// L1 adds l times the sum of the weights' absolute values to the loss, as
// the lasso does, driving the weights of unhelpful features to zero.
func L1(l float64) Option { return func(c *config) { c.l1 = l } }

// L2 adds l/2 times the sum of the squared weights to the loss, as ridge
// regression does. The default is 0 for Regression, and for Logistic one
// over the number of rows, the penalty of scikit-learn's C = 1.
func L2(l float64) Option { return func(c *config) { c.l2 = l } }

// MaxIter caps the passes over the rows. The default is 1000.
func MaxIter(n int) Option { return func(c *config) { c.iters = n } }

// Tolerance stops the fit once a pass changes no standardized weight by
// more than tol, or, for Logistic without L1, once no part of the
// gradient exceeds tol. The default is 1e-6.
func Tolerance(tol float64) Option { return func(c *config) { c.tol = tol } }

// Workers sets how many goroutines Logistic uses. The default is
// GOMAXPROCS.
func Workers(n int) Option { return func(c *config) { c.workers = n } }

// Classes sets the number of classes, for when the last ones are missing
// from the training rows.
func Classes(n int) Option { return func(c *config) { c.classes = n } }

func options(task tabular.Task, d tabular.Data, l2 float64, opts []Option) (*config, error) {
	c := &config{l2: l2, iters: 1000, tol: 1e-6, workers: runtime.GOMAXPROCS(0)}
	for _, o := range opts {
		o(c)
	}
	if err := d.Check(task); err != nil {
		return nil, err
	}
	bad := func(name string, v any) error { return fmt.Errorf("linear: %s %v: %w", name, v, tensor.ErrShape) }
	switch {
	case !(c.l1 >= 0) || math.IsInf(c.l1, 0):
		return nil, bad("L1", c.l1)
	case !(c.l2 >= 0) || math.IsInf(c.l2, 0):
		return nil, bad("L2", c.l2)
	case c.iters < 1:
		return nil, bad("iterations", c.iters)
	case !(c.tol > 0):
		return nil, bad("tolerance", c.tol)
	case c.classes < 0 || c.classes > tabular.MaxClasses:
		return nil, bad("classes", c.classes)
	}
	c.workers = max(c.workers, 1)
	return c, nil
}

// standard holds a table's features standardized, column by column.
type standard struct {
	n, f      int
	cols      []float32 // cols[j*n+i] is row i's feature j, standardized
	mean, std []float64 // std is 0 for a constant feature
}

func standardize(d tabular.Data) *standard {
	n, f := d.Rows(), d.Features
	s := &standard{n: n, f: f, cols: make([]float32, n*f), mean: make([]float64, f), std: make([]float64, f)}
	for j := range f {
		var sum, sq, cnt float64
		for i := range n {
			if v := float64(d.X[i*f+j]); v == v {
				sum += v
				cnt++
			}
		}
		if cnt > 0 {
			s.mean[j] = sum / cnt
		}
		for i := range n {
			if v := float64(d.X[i*f+j]); v == v {
				sq += (v - s.mean[j]) * (v - s.mean[j])
			}
		}
		s.std[j] = math.Sqrt(sq / float64(n))
		if s.std[j] < 1e-12*math.Max(1, math.Abs(s.mean[j])) {
			s.std[j] = 0
			continue
		}
		col := s.cols[j*n : (j+1)*n]
		for i := range n {
			if v := float64(d.X[i*f+j]); v == v {
				col[i] = float32((v - s.mean[j]) / s.std[j])
			}
		}
	}
	return s
}

// model turns standardized weights w, k outputs of f, and biases b into a
// Model in the features' own units.
func (s *standard) model(task tabular.Task, k int, w, b []float64) *Model {
	m := &Model{Task: task, Features: s.f, Outputs: k, Weights: make([]float32, k*s.f), Bias: make([]float32, k), Fill: make([]float32, s.f)}
	for j, v := range s.mean {
		m.Fill[j] = float32(v)
	}
	for o := range k {
		bias := b[o]
		for j := range s.f {
			if s.std[j] == 0 {
				continue
			}
			wj := w[o*s.f+j] / s.std[j]
			m.Weights[o*s.f+j] = float32(wj)
			bias -= wj * s.mean[j]
		}
		m.Bias[o] = float32(bias)
	}
	return m
}

func softmax(r []float64) {
	hi := r[0]
	for _, v := range r {
		hi = math.Max(hi, v)
	}
	var sum float64
	for j, v := range r {
		r[j] = math.Exp(v - hi)
		sum += r[j]
	}
	for j := range r {
		r[j] /= sum
	}
}

// soft is the soft-thresholding operator, the proximal step of an L1
// penalty.
func soft(v, l float64) float64 {
	switch {
	case v > l:
		return v - l
	case v < -l:
		return v + l
	}
	return 0
}

// rows calls f over [0, n) in chunks on up to workers goroutines, each
// with its own index, below workers.
func rows(n, workers int, f func(w, lo, hi int)) {
	workers = min(workers, max(n/1024, 1))
	if workers <= 1 {
		f(0, 0, n)
		return
	}
	var wg sync.WaitGroup
	step := (n + workers - 1) / workers
	for w := range workers {
		lo, hi := w*step, min((w+1)*step, n)
		wg.Go(func() { f(w, lo, hi) })
	}
	wg.Wait()
}

func canceled(ctx context.Context, it int) error {
	if it%16 == 0 {
		return ctx.Err()
	}
	return nil
}
