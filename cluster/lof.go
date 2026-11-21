// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mimir, a pure-Go machine learning framework,
// created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

package cluster

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"slices"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Outliers is a local outlier factor, as scikit-learn's
// LocalOutlierFactor with novelty: a row's score compares how crowded
// its K nearest neighbors are with how crowded the row itself is. About
// 1 is a row as crowded as its neighbors; well above 1, a row in a gap
// next to a crowd. Unlike an isolation forest, which looks at the whole
// table, it finds an odd row next to a dense group even when a sparse
// group elsewhere holds rows as far apart.
type Outliers struct {
	Features int
	Scaler   tabular.Scaler
	K        int
	// Rows holds the training rows, standardized, which new rows are
	// compared with.
	Rows []float32
	// Reach is each training row's distance to its K-th neighbor, and
	// Density its local reachability density.
	Reach, Density []float64
	// Scores is the factor of each training row.
	Scores []float64
}

// LocalOutlier learns the local outlier factor of the rows of x,
// features numbers each, over k neighbors (20 is usual) from fewer than
// the rows. It compares every row with every other, so it suits tables
// up to some tens of thousands of rows.
func LocalOutlier(ctx context.Context, x []float32, features, k int, opts ...Option) (*Outliers, error) {
	c, n, err := options(x, features, 1, opts)
	if err != nil {
		return nil, err
	}
	if k < 1 || k >= n {
		return nil, fmt.Errorf("cluster: %d neighbors of %d rows: %w", k, n, tensor.ErrShape)
	}
	f := features
	sc := tabular.FitScaler(x, f)
	z := sc.Transform(x)
	o := &Outliers{Features: f, Scaler: sc, K: k, Rows: z,
		Reach: make([]float64, n), Density: make([]float64, n), Scores: make([]float64, n)}
	near := make([][]neighbor, n)
	rows(n, c.workers, func(_, lo, hi int) {
		for i := lo; i < hi; i++ {
			if i%256 == 0 && ctx.Err() != nil {
				return
			}
			near[i] = o.nearest(z[i*f:(i+1)*f], i)
			o.Reach[i] = near[i][k-1].d
		}
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for i := range n {
		o.Density[i] = o.density(near[i])
	}
	for i := range n {
		o.Scores[i] = o.factor(near[i], o.Density[i])
	}
	return o, nil
}

type neighbor struct {
	i int
	d float64
}

// nearest returns the K training rows nearest row, nearest first,
// leaving out the training row skip, or none when skip is -1.
func (o *Outliers) nearest(row []float32, skip int) []neighbor {
	f, k := o.Features, o.K
	best := make([]neighbor, 0, k+1)
	for j := range len(o.Rows) / f {
		if j == skip {
			continue
		}
		d := math.Sqrt(dist(row, o.Rows[j*f:(j+1)*f]))
		if len(best) == k && d >= best[k-1].d {
			continue
		}
		at, _ := slices.BinarySearchFunc(best, d, func(a neighbor, d float64) int {
			if a.d <= d {
				return -1
			}
			return 1
		})
		best = slices.Insert(best, at, neighbor{j, d})
		if len(best) > k {
			best = best[:k]
		}
	}
	return best
}

// density returns the local reachability density of a row with
// neighbors near: one over its mean reach distance to them.
func (o *Outliers) density(near []neighbor) float64 {
	var s float64
	for _, nb := range near {
		s += max(o.Reach[nb.i], nb.d)
	}
	return float64(len(near)) / max(s, 1e-10)
}

// factor returns the outlier factor of a row with neighbors near and
// density own: its neighbors' mean density over its own.
func (o *Outliers) factor(near []neighbor, own float64) float64 {
	var s float64
	for _, nb := range near {
		s += o.Density[nb.i]
	}
	return s / float64(len(near)) / own
}

// Score returns the outlier factor of each row of x, rows of Features
// numbers, against the training rows.
func (o *Outliers) Score(x []float32) []float64 {
	z := o.Scaler.Transform(x)
	f := o.Features
	out := make([]float64, len(z)/f)
	rows(len(out), runtime.GOMAXPROCS(0), func(_, lo, hi int) {
		for i := lo; i < hi; i++ {
			near := o.nearest(z[i*f:(i+1)*f], -1)
			out[i] = o.factor(near, o.density(near))
		}
	})
	return out
}
