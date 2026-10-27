// Package decompose finds the main directions a table's rows vary in,
// by principal component analysis, as scikit-learn's PCA does, to draw a
// table in two dimensions or shrink it before another model learns.
//
//	p, err := decompose.PCA(ctx, x, features, 2)
//	if err != nil { ... }
//	xy := p.Transform(rows) // two numbers per row
//
// PCA standardizes the features first, so it finds the directions of the
// features' correlations rather than of their units, and counts a missing
// value, NaN, as the feature's mean.
package decompose

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Components is a fitted PCA: K directions, each Features numbers long,
// in standardized units, from the most varied.
type Components struct {
	Features int
	Scaler   tabular.Scaler
	// Directions holds K unit vectors of Features numbers each.
	Directions []float32
	// Variance is the variance of the training rows along each direction,
	// and Explained its share of the total variance.
	Variance, Explained []float64
}

// K returns the number of components.
func (c *Components) K() int { return len(c.Variance) }

// PCA returns the k directions the rows of x, features numbers each,
// vary in most.
func PCA(ctx context.Context, x []float32, features, k int) (*Components, error) {
	if features <= 0 || len(x) == 0 || len(x)%features != 0 {
		return nil, fmt.Errorf("decompose: %d numbers for rows of %d features: %w", len(x), features, tensor.ErrShape)
	}
	if k < 1 || k > features {
		return nil, fmt.Errorf("decompose: %d components of %d features: %w", k, features, tensor.ErrShape)
	}
	f, n := features, len(x)/features
	sc := tabular.FitScaler(x, f)
	z := sc.Transform(x)
	// The covariance of the standardized rows, in float64.
	cov := make([]float64, f*f)
	for i := range n {
		row := z[i*f : (i+1)*f]
		for a, va := range row {
			if va == 0 {
				continue
			}
			for b := range a + 1 {
				cov[a*f+b] += float64(va) * float64(row[b])
			}
		}
	}
	c32 := make([]float32, f*f)
	var total float64
	for a := range f {
		for b := range a + 1 {
			v := cov[a*f+b] / float64(n)
			c32[a*f+b], c32[b*f+a] = float32(v), float32(v)
		}
		total += cov[a*f+a] / float64(n)
	}
	w, v := tensor.FromSlice(c32, f, f).Eigh()
	vals, err := w.Data[float32](ctx)
	if err != nil {
		return nil, err
	}
	vecs, err := v.Data[float32](ctx)
	if err != nil {
		return nil, err
	}
	out := &Components{Features: f, Scaler: sc, Directions: make([]float32, 0, k*f)}
	// Eigenvalues come in ascending order: take the last k, largest first.
	for c := f - 1; c >= f-k; c-- {
		dir := make([]float32, f)
		for j := range f {
			dir[j] = vecs[j*f+c]
		}
		// Fix each direction's sign: its largest entry is positive.
		big := 0
		for j, d := range dir {
			if abs(d) > abs(dir[big]) {
				big = j
			}
		}
		if dir[big] < 0 {
			for j := range dir {
				dir[j] = -dir[j]
			}
		}
		out.Directions = append(out.Directions, dir...)
		out.Variance = append(out.Variance, max(float64(vals[c]), 0))
	}
	for _, v := range out.Variance {
		share := 0.0
		if total > 0 {
			share = v / total
		}
		out.Explained = append(out.Explained, share)
	}
	return out, nil
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// Transform returns each row of x as K numbers: its standardized values
// along each direction.
func (c *Components) Transform(x []float32) []float32 {
	z := c.Scaler.Transform(x)
	f, k := c.Features, c.K()
	n := len(z) / f
	out := make([]float32, n*k)
	for i := range n {
		row := z[i*f : (i+1)*f]
		for d := range k {
			var s float64
			for j, v := range row {
				s += float64(v) * float64(c.Directions[d*f+j])
			}
			out[i*k+d] = float32(s)
		}
	}
	return out
}

// Inverse returns each row of y, K numbers, back as Features numbers in
// the features' own units: the nearest row along the directions.
func (c *Components) Inverse(y []float32) []float32 {
	f, k := c.Features, c.K()
	n := len(y) / k
	out := make([]float32, n*f)
	for i := range n {
		for j := range f {
			var s float64
			for d := range k {
				s += float64(y[i*k+d]) * float64(c.Directions[d*f+j])
			}
			out[i*f+j] = float32(s)*c.Scaler.Std[j] + c.Scaler.Mean[j]
		}
	}
	return out
}

// Ranked returns the features by how much they weigh in direction d, the
// most first.
func (c *Components) Ranked(d int) []int {
	f := c.Features
	idx := make([]int, f)
	for j := range idx {
		idx[j] = j
	}
	dir := c.Directions[d*f : (d+1)*f]
	slices.SortStableFunc(idx, func(a, b int) int {
		switch {
		case abs(dir[a]) > abs(dir[b]):
			return -1
		case abs(dir[a]) < abs(dir[b]):
			return 1
		}
		return 0
	})
	return idx
}
