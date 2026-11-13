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
	"strconv"

	"github.com/gsoultan/mimir/tensor"
)

// Objective is the loss a Regression model is boosted on, in place of
// squared error. The zero Objective is squared error.
type Objective struct {
	// Name is "poisson", "tweedie" or "quantile", or empty for squared
	// error.
	Name string `json:"name,omitempty"`
	// Power is Tweedie's variance power, between 1 and 2.
	Power float64 `json:"power,omitempty"`
	// Alpha is the share of answers a Quantile model's answer should be
	// above, between 0 and 1.
	Alpha float64 `json:"alpha,omitempty"`
}

// Poisson fits counts, such as orders a week, by their log: the answer is
// never negative, and a count many times another is as easy to fit as one
// a few more. Answers must be 0 or more, and not all 0.
func Poisson() Objective { return Objective{Name: "poisson"} }

// Tweedie fits amounts that are often exactly 0 and otherwise spread out,
// such as insurance claims or a customer's spend in a week, by their log,
// as LightGBM's tweedie loss does. power, between 1 and 2, sets how much
// of the spread comes from the zeros: near 1 like Poisson counts, near 2
// like amounts that are never 0; 1.5 suits most. Answers must be 0 or
// more, and not all 0.
func Tweedie(power float64) Objective { return Objective{Name: "tweedie", Power: power} }

// Quantile fits the answer that a share alpha of the answers falls below,
// such as 0.9 for a number that 9 in 10 stay under, with the pinball
// loss. Two such models, at 0.1 and 0.9, give a range 8 in 10 answers fall
// in, fitted directly. Each leaf answers the quantile of its rows'
// remaining errors, as LightGBM's quantile loss does.
func Quantile(alpha float64) Objective { return Objective{Name: "quantile", Alpha: alpha} }

// Loss boosts a Regression model on o in place of squared error.
func Loss(o Objective) Option { return func(c *config) { c.loss = o } }

func (o Objective) check() error {
	switch o.Name {
	case "", "poisson":
		return nil
	case "tweedie":
		if !(o.Power > 1 && o.Power < 2) {
			return fmt.Errorf("tree: Tweedie power %v, want between 1 and 2: %w", o.Power, tensor.ErrShape)
		}
		return nil
	case "quantile":
		if !(o.Alpha > 0 && o.Alpha < 1) {
			return fmt.Errorf("tree: quantile %v, want between 0 and 1: %w", o.Alpha, tensor.ErrShape)
		}
		return nil
	}
	return fmt.Errorf("tree: loss %q: %w", o.Name, tensor.ErrShape)
}

// String names o as a Model's Loss does: "poisson", "tweedie:1.5" or
// "quantile:0.9", or "" for squared error.
func (o Objective) String() string {
	switch o.Name {
	case "tweedie":
		return "tweedie:" + strconv.FormatFloat(o.Power, 'g', -1, 64)
	case "quantile":
		return "quantile:" + strconv.FormatFloat(o.Alpha, 'g', -1, 64)
	}
	return o.Name
}

// logLinked reports whether o fits the log of the answer.
func (o Objective) logLinked() bool { return o.Name == "poisson" || o.Name == "tweedie" }

// checkAnswers reports an error unless y suits o.
func (o Objective) checkAnswers(y []float32) error {
	if !o.logLinked() {
		return nil
	}
	any := false
	for _, v := range y {
		if v < 0 {
			return fmt.Errorf("tree: %s loss with answer %v; answers must be 0 or more: %w", o.Name, v, tensor.ErrShape)
		}
		any = any || v > 0
	}
	if !any {
		return fmt.Errorf("tree: %s loss with every answer 0: %w", o.Name, tensor.ErrShape)
	}
	return nil
}

// poissonStep is added to the log answer in the Poisson hessian, so a
// leaf's step stays small where a few rows have tiny answers, as
// LightGBM's poisson_max_delta_step does.
const poissonStep = 0.7

// objGradients sets the gradients and hessians of o at raw, for a model
// of one output, in gh with stride sz.
func objGradients(o Objective, raw, y, gh []float32, sz, workers int) {
	parallel(len(y), workers, 16384, func(lo, hi int) {
		switch o.Name {
		case "poisson":
			for i := lo; i < hi; i++ {
				r := float64(raw[i])
				gh[i*sz] = float32(math.Exp(r) - float64(y[i]))
				gh[i*sz+1] = float32(math.Exp(r + poissonStep))
			}
		case "tweedie":
			p := o.Power
			for i := lo; i < hi; i++ {
				r, t := float64(raw[i]), float64(y[i])
				a, b := math.Exp((1-p)*r), math.Exp((2-p)*r)
				gh[i*sz] = float32(-t*a + b)
				gh[i*sz+1] = float32(max(-t*(1-p)*a+(2-p)*b, 1e-6))
			}
		case "quantile":
			a := float32(o.Alpha)
			for i := lo; i < hi; i++ {
				if y[i] > raw[i] {
					gh[i*sz] = -a
				} else {
					gh[i*sz] = 1 - a
				}
				gh[i*sz+1] = 1
			}
		}
	})
}

// objBase returns the constant raw answer that best fits y under o.
func objBase(o Objective, y []float32) float32 {
	if o.Name == "quantile" {
		return quantileOf(slices.Clone(y), o.Alpha)
	}
	var s float64
	for _, v := range y {
		s += float64(v)
	}
	return float32(math.Log(s / float64(len(y))))
}

// objLoss returns the mean loss of o at raw for answers y.
func objLoss(o Objective, raw, y []float32) float64 {
	var s float64
	for i, v := range raw {
		r, t := float64(v), float64(y[i])
		switch o.Name {
		case "poisson":
			s += math.Exp(r) - t*r
		case "tweedie":
			p := o.Power
			s += -t*math.Exp((1-p)*r)/(1-p) + math.Exp((2-p)*r)/(2-p)
		case "quantile":
			if t > r {
				s += o.Alpha * (t - r)
			} else {
				s += (1 - o.Alpha) * (r - t)
			}
		}
	}
	return s / float64(len(y))
}

// quantileOf returns the alpha quantile of v, which it sorts.
func quantileOf(v []float32, alpha float64) float32 {
	if len(v) == 0 {
		return 0
	}
	slices.Sort(v)
	pos := alpha * float64(len(v)-1)
	i := int(pos)
	if i+1 >= len(v) {
		return v[len(v)-1]
	}
	f := float32(pos - float64(i))
	return v[i] + f*(v[i+1]-v[i])
}

// renewQuantiles sets each leaf of t that rows reached to the alpha
// quantile of the rows' errors y-raw, held within the leaf's bounds when
// there are any. leafOf holds each row's leaf.
func renewQuantiles(t *Tree, rows []int32, leafOf func(r int32) int32, raw, y []float32, alpha float64, bounds [][2]float64) {
	byLeaf := map[int32][]float32{}
	for _, r := range rows {
		l := leafOf(r)
		byLeaf[l] = append(byLeaf[l], y[r]-raw[r])
	}
	for l, e := range byLeaf {
		v := float64(quantileOf(e, alpha))
		if bounds != nil {
			v = min(max(v, bounds[l][0]), bounds[l][1])
		}
		t.Value[l] = float32(v)
	}
}
