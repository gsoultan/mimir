// Package neighbors answers each row from the rows of the table nearest
// it, like scikit-learn's KNeighborsClassifier and KNeighborsRegressor:
// the share of each class among the k nearest, or their mean answer.
//
//	m, err := neighbors.Fit(ctx, d, tabular.Classification, neighbors.K(7), neighbors.ByDistance())
//	if err != nil { ... }
//	p := m.Predict(rows) // a probability for each class of each row
//
// Fitting keeps the rows; Predict compares each row with every one of
// them, in parallel, which suits tables of up to a few hundred thousand
// rows. Each feature is standardized first unless Raw is given, so no
// feature counts more for its units; a missing value, NaN, counts as the
// feature's mean. A Model holds only exported fields, so encoding/json
// saves and loads it; Module serves it as an nn.Module.
package neighbors

import (
	"container/heap"
	"context"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// The distances, by the names Model.Metric holds.
const (
	Euclidean = "euclidean"
	Manhattan = "manhattan"
	Cosine    = "cosine" // 1 - the cosine of the angle between the rows
)

// Model is a fitted nearest-neighbours model.
type Model struct {
	Task     tabular.Task
	Features int
	// Classes is the number of classes of a Classification model, and 0
	// for Regression.
	Classes int
	K       int
	// Distance weighs each neighbour by 1/distance instead of alike.
	Distance bool
	Metric   string
	// Mean and Std standardize each feature; Std is all ones when Raw.
	Mean, Std []float32
	// X holds the rows, standardized, and Y their answers.
	X, Y []float32
}

// Option configures Fit.
type Option func(*config)

type config struct {
	k        int
	distance bool
	metric   string
	raw      bool
	classes  int
}

// K sets how many neighbours answer. The default is 5.
func K(k int) Option { return func(c *config) { c.k = k } }

// ByDistance weighs each neighbour by the inverse of its distance, so
// nearer ones count more, as scikit-learn's weights="distance".
func ByDistance() Option { return func(c *config) { c.distance = true } }

// Metric sets the distance: Euclidean (the default), Manhattan or Cosine.
func Metric(m string) Option { return func(c *config) { c.metric = m } }

// Raw compares the features in their own units, without standardizing.
func Raw() Option { return func(c *config) { c.raw = true } }

// Classes sets the number of classes, for when the last ones are missing
// from the training rows. The default counts them in d.Y.
func Classes(k int) Option { return func(c *config) { c.classes = k } }

// Fit keeps the rows of d to answer task.
func Fit(ctx context.Context, d tabular.Data, task tabular.Task, opts ...Option) (*Model, error) {
	c := config{k: 5, metric: Euclidean}
	for _, o := range opts {
		o(&c)
	}
	if err := d.Check(task); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c.k < 1 || !slices.Contains([]string{Euclidean, Manhattan, Cosine}, c.metric) {
		return nil, fmt.Errorf("neighbors: k %d, metric %q: %w", c.k, c.metric, tensor.ErrShape)
	}
	n, f := d.Rows(), d.Features
	m := &Model{Task: task, Features: f, K: min(c.k, n), Distance: c.distance, Metric: c.metric, Y: slices.Clone(d.Y)}
	if task == tabular.Classification {
		m.Classes = d.Classes()
		if c.classes > 0 {
			if c.classes < m.Classes || c.classes > tabular.MaxClasses {
				return nil, fmt.Errorf("neighbors: %d classes for answers up to %d: %w", c.classes, m.Classes-1, tensor.ErrShape)
			}
			m.Classes = c.classes
		}
	}
	m.Mean, m.Std = make([]float32, f), make([]float32, f)
	sum, sq, cnt := make([]float64, f), make([]float64, f), make([]float64, f)
	for i := range n {
		for j, v := range d.Row(i) {
			if v == v {
				sum[j] += float64(v)
				sq[j] += float64(v) * float64(v)
				cnt[j]++
			}
		}
	}
	for j := range f {
		mean := sum[j] / max(cnt[j], 1)
		std := math.Sqrt(max(sq[j]/max(cnt[j], 1)-mean*mean, 0))
		if c.raw || std == 0 {
			std = 1
		}
		m.Mean[j], m.Std[j] = float32(mean), float32(std)
	}
	m.X = make([]float32, len(d.X))
	m.scale(d.X, m.X)
	return m, nil
}

// scale writes src standardized to dst, a missing value 0.
func (m *Model) scale(src, dst []float32) {
	f := m.Features
	for i, v := range src {
		j := i % f
		if v != v {
			dst[i] = 0
		} else {
			dst[i] = (v - m.Mean[j]) / m.Std[j]
		}
	}
}

func (m *Model) dist(a, b []float32) float64 {
	switch m.Metric {
	case Manhattan:
		var s float64
		for j, v := range a {
			s += math.Abs(float64(v) - float64(b[j]))
		}
		return s
	case Cosine:
		var dot, na, nb float64
		for j, v := range a {
			dot += float64(v) * float64(b[j])
			na += float64(v) * float64(v)
			nb += float64(b[j]) * float64(b[j])
		}
		if na == 0 || nb == 0 {
			return 1
		}
		return 1 - dot/math.Sqrt(na*nb)
	}
	var s float64
	for j, v := range a {
		t := float64(v) - float64(b[j])
		s += t * t
	}
	return math.Sqrt(s)
}

// Neighbor is one of the rows nearest a row.
type Neighbor struct {
	Row      int
	Distance float64
}

// worst is a max-heap of the nearest so far, the farthest on top.
type worst []Neighbor

func (h worst) Len() int { return len(h) }
func (h worst) Less(i, j int) bool {
	if h[i].Distance != h[j].Distance {
		return h[i].Distance > h[j].Distance
	}
	return h[i].Row > h[j].Row
}
func (h worst) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *worst) Push(x any)   { *h = append(*h, x.(Neighbor)) }
func (h *worst) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// nearest returns the K rows nearest the standardized row z, nearest
// first, ties to the earlier row.
func (m *Model) nearest(z []float32, h *worst) []Neighbor {
	f := m.Features
	*h = (*h)[:0]
	n := len(m.Y)
	for i := range n {
		d := m.dist(z, m.X[i*f:(i+1)*f])
		if h.Len() < m.K {
			heap.Push(h, Neighbor{i, d})
		} else if top := (*h)[0]; d < top.Distance || d == top.Distance && i < top.Row {
			(*h)[0] = Neighbor{i, d}
			heap.Fix(h, 0)
		}
	}
	out := slices.Clone(*h)
	slices.SortFunc(out, func(a, b Neighbor) int {
		if a.Distance != b.Distance {
			if a.Distance < b.Distance {
				return -1
			}
			return 1
		}
		return a.Row - b.Row
	})
	return out
}

// Neighbors returns the K training rows nearest each row of x, nearest
// first.
func (m *Model) Neighbors(x []float32) [][]Neighbor {
	f := m.Features
	if f <= 0 || len(x)%f != 0 {
		panic(fmt.Sprintf("neighbors: %d numbers for rows of %d features", len(x), f))
	}
	n := len(x) / f
	z := make([]float32, len(x))
	m.scale(x, z)
	out := make([][]Neighbor, n)
	workers := min(runtime.GOMAXPROCS(0), max(n, 1))
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() {
			var h worst
			for i := w; i < n; i += workers {
				out[i] = m.nearest(z[i*f:(i+1)*f], &h)
			}
		})
	}
	wg.Wait()
	return out
}

// weight is a neighbour's weight.
func (m *Model) weight(d float64) float64 {
	if !m.Distance {
		return 1
	}
	return 1 / max(d, 1e-9)
}

// Predict answers each row of x, rows of m.Features numbers one after
// another: for Classification m.Classes probabilities per row, the
// weighted share of each class among the neighbours, for Regression their
// weighted mean answer. It panics if x is not a whole number of rows.
func (m *Model) Predict(x []float32) []float32 {
	near := m.Neighbors(x)
	k := max(m.Classes, 1)
	out := make([]float32, len(near)*k)
	for i, ns := range near {
		var total float64
		acc := make([]float64, k)
		for _, nb := range ns {
			w := m.weight(nb.Distance)
			total += w
			if m.Task == tabular.Classification {
				acc[int(m.Y[nb.Row])] += w
			} else {
				acc[0] += w * float64(m.Y[nb.Row])
			}
		}
		for c := range k {
			out[i*k+c] = float32(acc[c] / total)
		}
	}
	return out
}

var _ tabular.Model = (*Model)(nil)
