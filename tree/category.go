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

package tree

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// MaxCategories is the most categories a categorical column may have.
const MaxCategories = 254

// Category marks a column that holds categories, such as a region or a
// plan, rather than an amount. A tree splits it on a set of its
// categories, as LightGBM does, rather than on a threshold, which finds in
// one split what one-hot columns take many to find.
type Category struct {
	// Feature is the column of whole-number codes 0, 1, 2 and so on, or
	// the first of Width one-hot columns. NaN, or no one-hot column set,
	// is a missing value.
	Feature int `json:"feature"`
	// Width, when above 0, is the number of one-hot columns from Feature
	// on, column Feature+c set (above 0.5) for category c.
	Width int `json:"width,omitempty"`
}

// Categorical reads the columns of cats as categories. A tree model reads
// rows as they are given, so the columns of a one-hot run stay where they
// are; only the first is split on.
func Categorical(cats ...Category) Option {
	return func(c *config) { c.cats = append(c.cats, cats...) }
}

// checkCategories reports an error unless cats name distinct columns of
// rows of f features.
func checkCategories(cats []Category, f int) error {
	used := make([]bool, f)
	for _, ct := range cats {
		w := max(ct.Width, 1)
		if ct.Feature < 0 || ct.Width < 0 || ct.Width > MaxCategories || ct.Feature+w > f {
			return fmt.Errorf("tree: category column %d of width %d for rows of %d features: %w", ct.Feature, ct.Width, f, tensor.ErrShape)
		}
		for j := ct.Feature; j < ct.Feature+w; j++ {
			if used[j] {
				return fmt.Errorf("tree: feature %d is in two category columns: %w", j, tensor.ErrShape)
			}
			used[j] = true
		}
	}
	return nil
}

// codes sets the first column of each one-hot run of cats in row to its
// category's code, or NaN when none is set, and returns row.
func codes(row []float32, cats []Category) []float32 {
	for _, ct := range cats {
		if ct.Width == 0 {
			continue
		}
		code := float32(math.NaN())
		for c, v := range row[ct.Feature : ct.Feature+ct.Width] {
			if v > 0.5 {
				code = float32(c)
				break
			}
		}
		row[ct.Feature] = code
	}
	return row
}

// coded returns d with its one-hot runs read as codes, as codes does,
// and how many categories each feature has: 0 for an amount. It reports
// an error if a code column holds anything but whole numbers 0 to
// MaxCategories-1, or NaN.
func coded(d tabular.Data, cats []Category) (tabular.Data, []int, error) {
	if len(cats) == 0 {
		return d, nil, nil
	}
	if err := checkCategories(cats, d.Features); err != nil {
		return d, nil, err
	}
	f := d.Features
	n := d.Rows()
	x := slices.Clone(d.X)
	for i := range n {
		codes(x[i*f:(i+1)*f], cats)
	}
	count := make([]int, f)
	for _, ct := range cats {
		if ct.Width > 0 {
			count[ct.Feature] = ct.Width
			continue
		}
		top := 0
		for i := range n {
			v := x[i*f+ct.Feature]
			if v != v {
				continue
			}
			if v < 0 || v >= MaxCategories || v != float32(math.Trunc(float64(v))) {
				return d, nil, fmt.Errorf("tree: category column %d holds %v; codes are whole numbers 0 to %d: %w", ct.Feature, v, MaxCategories-1, tensor.ErrShape)
			}
			top = max(top, int(v)+1)
		}
		count[ct.Feature] = max(top, 1)
	}
	d.X = x
	return d, count, nil
}

// splittable returns the features a tree may split on: all of f but the
// one-hot columns after the first of each run, which codes leaves unread.
func splittable(f int, cats []Category) []int {
	skip := make([]bool, f)
	for _, ct := range cats {
		for j := ct.Feature + 1; j < ct.Feature+ct.Width; j++ {
			skip[j] = true
		}
	}
	feats := make([]int, 0, f)
	for j := range f {
		if !skip[j] {
			feats = append(feats, j)
		}
	}
	return feats
}

// catSmooth is added to each category's hessian before the categories are
// ordered by their gradients, so a rare one does not sort to an end on a
// few rows, as LightGBM's cat_smooth does.
const catSmooth = 10

// bestCat returns the best split of categorical feature j into a set of
// categories and the rest. It orders the categories by gradient over
// hessian and tries every cut of that order, which for one output finds
// the best of all sets (Fisher, 1958); for several, it does so for each
// output's order.
func (gr *grower) bestCat(h []float64, j int, total []float64, parent float64) split {
	k, s, cfg := gr.k, gr.stride(), gr.cfg
	l2, minLeaf, minWeight := cfg.l2, float64(cfg.minLeaf), cfg.minWeight
	miss := h[:s:s]
	var present []int // codes with rows here
	for c := range len(h)/s - 1 {
		if h[(c+1)*s+2*k] > 0 {
			present = append(present, c)
		}
	}
	best := split{feature: j}
	if len(present) < 2 && miss[2*k] == 0 {
		return best
	}
	cat := func(c int) []float64 { return h[(c+1)*s : (c+2)*s] }
	// order sorts the categories by output o's gradient over hessian.
	order := func(o int) []int {
		ord := slices.Clone(present)
		slices.SortStableFunc(ord, func(a, b int) int {
			ea, eb := cat(a), cat(b)
			ra, rb := ea[o]/(ea[k+o]+catSmooth), eb[o]/(eb[k+o]+catSmooth)
			switch {
			case ra < rb:
				return -1
			case ra > rb:
				return 1
			}
			return 0
		})
		return ord
	}
	left, right, cum := make([]float64, s), make([]float64, s), make([]float64, s)
	var bestLeft []float64
	var bestSet []int
	try := func(set []int, sums []float64, missLeft bool) {
		for i := range s {
			left[i] = sums[i]
			if missLeft {
				left[i] += miss[i]
			}
			right[i] = total[i] - left[i]
		}
		if left[2*k] < minLeaf || right[2*k] < minLeaf {
			return
		}
		var wl, wr, sc float64
		for o := range k {
			wl += left[k+o]
			wr += right[k+o]
			sc += left[o]*left[o]/(left[k+o]+l2) + right[o]*right[o]/(right[k+o]+l2)
		}
		if wl < minWeight || wr < minWeight {
			return
		}
		if g := sc - parent; g > best.gain {
			best = split{gain: g, feature: j, missLeft: missLeft}
			bestLeft = append(bestLeft[:0], left[:2*k]...)
			bestSet = append(bestSet[:0], set...)
		}
	}
	both := func(set []int, sums []float64) {
		if miss[2*k] > 0 {
			try(set, sums, true)
			try(set, sums, false)
		} else {
			try(set, sums, false)
		}
	}
	for o := range k {
		ord := order(o)
		clear(cum)
		for i, c := range ord[:max(len(ord)-1, 0)] {
			for q, v := range cat(c) {
				cum[q] += v
			}
			both(ord[:i+1], cum)
		}
	}
	if len(present) > 0 && miss[2*k] > 0 {
		// Every present category left, the missing rows right.
		clear(cum)
		for _, c := range present {
			for q, v := range cat(c) {
				cum[q] += v
			}
		}
		try(present, cum, false)
	}
	if best.gain > 0 {
		best.left = bestLeft
		best.cats = make([]uint64, (len(h)/s-1+63)/64)
		for _, c := range bestSet {
			best.cats[c>>6] |= 1 << (c & 63)
		}
	}
	return best
}
