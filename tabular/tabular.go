// Package tabular holds what the models that learn from tables of numbers
// share: the table they learn from, what they answer, and the folds that
// score them, as in scikit-learn's model_selection.
//
// Package tree grows decision trees, random forests and gradient-boosted
// trees, package linear fits linear and logistic regression, and
// package metrics scores what any of them answer:
//
//	d := tabular.Data{X: x, Features: 8, Y: y}
//	train, test := d.Split(0.2, 1)
//	m, err := tree.GradientBoosting(ctx, train, tabular.Classification)
//	if err != nil { ... }
//	p := m.Predict(test.X) // a probability for each class of each row
//
// These models read and answer plain float32 slices rather than tensors:
// they do not learn by gradient descent through package tensor.
package tabular

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/gsoultan/mimir/tensor"
)

// Task is what a model answers.
type Task int

const (
	// Regression answers a number: Y holds the value of each row.
	Regression Task = iota
	// Classification answers a class: Y holds the class index of each
	// row, 0, 1, 2 and so on, and the model a probability for each class.
	Classification
)

func (t Task) String() string {
	switch t {
	case Regression:
		return "regression"
	case Classification:
		return "classification"
	}
	return fmt.Sprintf("Task(%d)", int(t))
}

// MaxClasses is the most classes a Classification model may have.
const MaxClasses = 1 << 16

// Data is a table to learn from.
type Data struct {
	// X holds the rows one after another, Features numbers each. NaN is
	// a missing value.
	X []float32
	// Features is the number of columns of X.
	Features int
	// Y is the answer for each row: a class index for Classification, a
	// value for Regression.
	Y []float32
}

// Rows returns the number of rows in d.
func (d Data) Rows() int {
	if d.Features <= 0 {
		return 0
	}
	return len(d.X) / d.Features
}

// Row returns row i of d.X.
func (d Data) Row(i int) []float32 { return d.X[i*d.Features : (i+1)*d.Features] }

// Check reports an error wrapping tensor.ErrShape unless d holds at least
// one whole row, one answer per row, and answers fit for task: finite,
// and class indices for Classification.
func (d Data) Check(task Task) error {
	n := d.Rows()
	switch {
	case d.Features <= 0:
		return fmt.Errorf("tabular: %d features: %w", d.Features, tensor.ErrShape)
	case n == 0 || len(d.X) != n*d.Features:
		return fmt.Errorf("tabular: %d numbers for rows of %d features: %w", len(d.X), d.Features, tensor.ErrShape)
	case len(d.Y) != n:
		return fmt.Errorf("tabular: %d answers for %d rows: %w", len(d.Y), n, tensor.ErrShape)
	}
	for i, v := range d.Y {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("tabular: answer %d is %v: %w", i, v, tensor.ErrShape)
		}
		if task == Classification && (v < 0 || v != float32(math.Trunc(float64(v))) || v >= MaxClasses) {
			return fmt.Errorf("tabular: class %v of row %d is not an index: %w", v, i, tensor.ErrShape)
		}
	}
	return nil
}

// Classes returns one more than the largest class index in d.Y, and at
// least 2.
func (d Data) Classes() int {
	k := 2
	for _, y := range d.Y {
		k = max(k, int(y)+1)
	}
	return k
}

// Take returns the rows of d at the indices in rows, in that order.
func (d Data) Take(rows []int) Data {
	t := Data{X: make([]float32, 0, len(rows)*d.Features), Features: d.Features, Y: make([]float32, 0, len(rows))}
	for _, i := range rows {
		t.X = append(t.X, d.Row(i)...)
		t.Y = append(t.Y, d.Y[i])
	}
	return t
}

// Split shuffles d's rows with seed and returns them in two parts, the
// second holding the fraction test of them, rounded, and at least one row
// when d has two.
func (d Data) Split(test float64, seed uint64) (train, held Data) {
	n := d.Rows()
	perm := rand.New(rand.NewPCG(seed, 0x73706c6974)).Perm(n)
	k := int(math.Round(test * float64(n)))
	if n >= 2 {
		k = min(max(k, 1), n-1)
	}
	return d.Take(perm[k:]), d.Take(perm[:k])
}

// Model is a trained model of a table.
type Model interface {
	// Predict answers each row of x, rows of the model's features one
	// after another: for Classification a probability for each class of
	// each row, for Regression one value per row.
	Predict(x []float32) []float32
}

// Fit trains a model of d.
type Fit func(ctx context.Context, d Data) (Model, error)

// Score scores a model's answers p for the rows whose true answers are y.
type Score func(p, y []float32) float64

// CrossValidate shuffles d's rows with seed into k folds of about equal
// size and, for each fold, fits a model of the other folds and scores it
// on that one. It returns the k scores.
func CrossValidate(ctx context.Context, d Data, k int, seed uint64, fit Fit, score Score) ([]float64, error) {
	n := d.Rows()
	if k < 2 || k > n {
		return nil, fmt.Errorf("tabular: %d folds of %d rows: %w", k, n, tensor.ErrShape)
	}
	perm := rand.New(rand.NewPCG(seed, 0x6376)).Perm(n)
	scores := make([]float64, k)
	for f := range k {
		lo, hi := f*n/k, (f+1)*n/k
		rest := append(append([]int(nil), perm[:lo]...), perm[hi:]...)
		m, err := fit(ctx, d.Take(rest))
		if err != nil {
			return nil, fmt.Errorf("tabular: fold %d: %w", f, err)
		}
		held := d.Take(perm[lo:hi])
		scores[f] = score(m.Predict(held.X), held.Y)
	}
	return scores, nil
}
