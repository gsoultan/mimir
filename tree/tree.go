// Package tree trains decision trees, random forests and gradient-boosted
// trees on tables of numbers, like scikit-learn's trees and XGBoost.
//
//	d := tabular.Data{X: x, Features: 8, Y: y} // x holds len(y) rows of 8 numbers
//	m, err := tree.GradientBoosting(ctx, d, tabular.Classification,
//		tree.Trees(300), tree.LearningRate(0.05), tree.Validation(val, 20))
//	if err != nil { ... }
//	p := m.Predict(row) // a probability for each class
//
// Every model sorts each feature into at most 255 bins by its quantiles
// before it learns, as LightGBM and XGBoost's hist method do, so a split
// costs the same however many distinct values a feature has. A NaN is a
// missing value: each split learns which side missing values go to.
//
// A Model holds only exported fields, so encoding/json saves and loads
// it, and it is a tabular.Model.
package tree

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/gsoultan/mimir/tabular"
)

// Link turns a model's summed tree outputs into its answer.
type Link string

const (
	Identity Link = "identity" // the sum is the answer
	Sigmoid  Link = "sigmoid"  // the sum is the log-odds of class 1
	Softmax  Link = "softmax"  // the sums are each class's logit
	// Isolation is declared with IsolationForest.
)

// Model is a trained forest or boosted ensemble: Base plus Scale times the
// sum of every tree's leaf values, passed through Link.
type Model struct {
	Task     tabular.Task
	Features int
	// Classes is the number of classes of a Classification model, and 0
	// for Regression.
	Classes int
	// Outputs is the number of values each leaf holds: 1 for Regression
	// and a boosted two-class model, Classes otherwise.
	Outputs int
	Link    Link
	Base    []float32 // Outputs starting values
	Scale   float32   // the weight of each tree
	Trees   []Tree
	// Importance is the share of the gain from all splits on each
	// feature; it sums to 1 unless no tree split.
	Importance []float64
}

// Tree is one tree, its nodes in flat slices. Node 0 is the root.
type Tree struct {
	// Feature is the column a node splits on, or -1 for a leaf.
	Feature []int32
	// Threshold sends a row left when its feature is at most it.
	Threshold []float32
	// MissingLeft sends a row whose feature is NaN left, and right if false.
	MissingLeft []bool
	Left, Right []int32
	// Value holds Outputs numbers for each node, the leaves' answers.
	Value []float32
}

// Nodes returns the number of nodes in t.
func (t *Tree) Nodes() int { return len(t.Feature) }

// leaf returns the leaf that row reaches.
func (t *Tree) leaf(row []float32) int32 {
	n := int32(0)
	for t.Feature[n] >= 0 {
		v := row[t.Feature[n]]
		switch {
		case v != v:
			if t.MissingLeft[n] {
				n = t.Left[n]
			} else {
				n = t.Right[n]
			}
		case v <= t.Threshold[n]:
			n = t.Left[n]
		default:
			n = t.Right[n]
		}
	}
	return n
}

// Predict answers each row of x, which holds rows of m.Features numbers
// one after another. For Classification it returns m.Classes
// probabilities per row; for Regression, one value per row. It panics if
// x is not a whole number of rows.
func (m *Model) Predict(x []float32) []float32 {
	if m.Features <= 0 || len(x)%m.Features != 0 {
		panic(fmt.Sprintf("tree: %d numbers for rows of %d features", len(x), m.Features))
	}
	n := len(x) / m.Features
	k := m.Outputs
	raw := make([]float32, n*k)
	parallel(n, 0, 256, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			r := raw[i*k : i*k+k]
			copy(r, m.Base)
			row := x[i*m.Features : (i+1)*m.Features]
			for t := range m.Trees {
				tr := &m.Trees[t]
				v := tr.Value[int(tr.leaf(row))*k:]
				for j := range r {
					r[j] += m.Scale * v[j]
				}
			}
		}
	})
	return m.answer(raw, n)
}

// answer applies m's link to raw, n rows of Outputs sums.
func (m *Model) answer(raw []float32, n int) []float32 {
	switch m.Link {
	case Sigmoid:
		out := make([]float32, 2*n)
		for i, v := range raw {
			p := sigmoid(v)
			out[2*i], out[2*i+1] = 1-p, p
		}
		return out
	case Softmax:
		for i := range n {
			softmax(raw[i*m.Outputs : (i+1)*m.Outputs])
		}
	case Isolation:
		for i, v := range raw {
			raw[i] = float32(math.Exp2(-float64(v)))
		}
	}
	return raw
}

// Class returns the most likely class of each row of a Classification
// model's Predict output p.
func (m *Model) Class(p []float32) []int {
	c := make([]int, len(p)/max(m.Classes, 1))
	for i := range c {
		row := p[i*m.Classes : (i+1)*m.Classes]
		for j, v := range row {
			if v > row[c[i]] {
				c[i] = j
			}
		}
	}
	return c
}

func sigmoid(v float32) float32 { return float32(1 / (1 + math.Exp(-float64(v)))) }

func softmax(r []float32) {
	hi := r[0]
	for _, v := range r {
		hi = max(hi, v)
	}
	var sum float64
	for j, v := range r {
		e := math.Exp(float64(v - hi))
		r[j] = float32(e)
		sum += e
	}
	for j := range r {
		r[j] = float32(float64(r[j]) / sum)
	}
}

// parallel calls f over [0, n) in contiguous chunks of at least grain on
// up to workers goroutines, or GOMAXPROCS when workers is 0.
func parallel(n, workers, grain int, f func(lo, hi int)) {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	workers = min(workers, max(n/grain, 1))
	if workers <= 1 {
		f(0, n)
		return
	}
	var wg sync.WaitGroup
	step := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += step {
		wg.Go(func() { f(lo, min(lo+step, n)) })
	}
	wg.Wait()
}
