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

	"github.com/gsoultan/mimir/tensor"
)

// Exp is the link of a count or amount model: the sum is the log of the
// answer, so every answer is above zero.
const Exp Link = "exp"

// Loss is what GradientBoosting fits for Regression. The default is
// SquaredError.
type Loss struct {
	kind string
	p    float64 // Tweedie's power, or Quantile's level
}

const (
	squaredLoss  = ""
	poissonLoss  = "poisson"
	tweedieLoss  = "tweedie"
	quantileLoss = "quantile"
)

// SquaredError fits the mean: the usual answer for a number.
func SquaredError() Loss { return Loss{} }

// Poisson fits counts, such as orders or calls a day, as LightGBM's
// poisson objective: a log link and the Poisson deviance. Answers are
// above zero, and the answers taught from must be at least zero.
func Poisson() Loss { return Loss{kind: poissonLoss} }

// Tweedie fits amounts with many zeros, such as insurance claims or the
// spend of each customer in a month, as LightGBM's tweedie objective
// with variance power p, from 1 (like Poisson) to 2 (like Gamma); 1.5
// is usual. Answers are above zero, and the answers taught from must be
// at least zero.
func Tweedie(p float64) Loss { return Loss{kind: tweedieLoss, p: p} }

// Quantile fits the level-th quantile, from 0 to 1: 0.5 is the median,
// and 0.1 and 0.9 together give a range that holds about 8 rows in 10.
// It minimizes the pinball loss, and sets each new leaf to the quantile
// of the rows that reach it, as LightGBM does.
func Quantile(level float64) Loss { return Loss{kind: quantileLoss, p: level} }

// String names l, such as "tweedie(1.5)".
func (l Loss) String() string {
	switch l.kind {
	case squaredLoss:
		return "squared error"
	case poissonLoss:
		return "poisson"
	}
	return fmt.Sprintf("%s(%g)", l.kind, l.p)
}

// Objective sets the loss GradientBoosting fits for Regression.
func Objective(l Loss) Option { return func(c *config) { c.loss = l } }

// check reports whether l can learn y.
func (l Loss) check(y []float32) error {
	bad := func(what string, v any) error { return fmt.Errorf("tree: %s %v: %w", what, v, tensor.ErrShape) }
	switch l.kind {
	case squaredLoss:
		return nil
	case tweedieLoss:
		if !(l.p >= 1 && l.p < 2) {
			return bad("tweedie power", l.p)
		}
	case quantileLoss:
		if !(l.p > 0 && l.p < 1) {
			return bad("quantile", l.p)
		}
		return nil
	case poissonLoss:
	default:
		return bad("loss", l.kind)
	}
	for _, v := range y {
		if v < 0 {
			return bad(l.String()+" answer", v)
		}
	}
	return nil
}

// link returns the link l's model answers through.
func (l Loss) link() Link {
	if l.kind == poissonLoss || l.kind == tweedieLoss {
		return Exp
	}
	return Identity
}

// base returns the constant that best fits y under l.
func (l Loss) base(y []float32) float32 {
	switch l.kind {
	case poissonLoss, tweedieLoss:
		var s float64
		for _, v := range y {
			s += float64(v)
		}
		return float32(math.Log(math.Max(s/float64(len(y)), 1e-9)))
	case quantileLoss:
		return quantile(slices.Clone(y), l.p)
	}
	var s float64
	for _, v := range y {
		s += float64(v)
	}
	return float32(s / float64(len(y)))
}

// grad returns the gradient and hessian of l at raw for answer y.
func (l Loss) grad(raw, y float32) (float32, float32) {
	switch l.kind {
	case poissonLoss:
		mu := math.Exp(math.Min(float64(raw), 80))
		return float32(mu - float64(y)), float32(math.Max(mu, 1e-6))
	case tweedieLoss:
		r := math.Min(float64(raw), 80)
		a, b := math.Exp((1-l.p)*r), math.Exp((2-l.p)*r)
		g := -float64(y)*a + b
		h := -float64(y)*(1-l.p)*a + (2-l.p)*b
		return float32(g), float32(math.Max(h, 1e-6))
	case quantileLoss:
		if raw >= y {
			return float32(1 - l.p), 1
		}
		return float32(-l.p), 1
	}
	return raw - y, 1
}

// loss returns l at raw for answer y: the deviance, up to a constant,
// for Poisson and Tweedie.
func (l Loss) loss(raw, y float32) float64 {
	r, t := float64(raw), float64(y)
	switch l.kind {
	case poissonLoss:
		return math.Exp(math.Min(r, 80)) - t*r
	case tweedieLoss:
		r = math.Min(r, 80)
		return -t*math.Exp((1-l.p)*r)/(1-l.p) + math.Exp((2-l.p)*r)/(2-l.p)
	case quantileLoss:
		if t >= r {
			return l.p * (t - r)
		}
		return (1 - l.p) * (r - t)
	}
	return (r - t) * (r - t)
}

// renew sets each leaf of t that rows reach to the quantile of what the
// rows reaching it still lack, divided by scale, as LightGBM refits a
// quantile model's leaves: a mean of gradients moves too slowly.
func (l Loss) renew(t *Tree, x []float32, f int, rows []int32, raw, y []float32, scale float32) {
	if l.kind != quantileLoss {
		return
	}
	lack := map[int32][]float32{}
	for _, r := range rows {
		leaf := t.leaf(x[int(r)*f : int(r+1)*f])
		lack[leaf] = append(lack[leaf], y[r]-raw[r])
	}
	for leaf, v := range lack {
		t.Value[leaf] = quantile(v, l.p) / scale
	}
}

// quantile returns the q-th quantile of v, which it sorts, between the
// nearest two values.
func quantile(v []float32, q float64) float32 {
	slices.Sort(v)
	at := q * float64(len(v)-1)
	i := int(at)
	if i+1 >= len(v) {
		return v[len(v)-1]
	}
	w := float32(at - float64(i))
	return v[i]*(1-w) + v[i+1]*w
}
