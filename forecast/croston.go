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

package forecast

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// Intermittent forecasts a series that is 0 at many steps, such as sales
// of a spare part or a slow seller, where smoothing and trees chase the
// last few sales and forecast too much or too little. It smooths the size
// of the demand when there is some apart from how often there is any, as
// Croston (1972) did, in one of two forms:
//
//   - "sba", Croston's method with the bias correction of Syntetos and
//     Boylan (2005): the size over the steps between demands, times
//     1-Beta/2.
//   - "tsb", the method of Teunter, Syntetos and Babai (2011): the size
//     times the smoothed chance of any demand at a step, which falls
//     every step without one, so an item no one buys any more is
//     forecast to fade out.
//
// Croston fits both and keeps the one with the smaller error. The forecast
// is flat: the demand per step to plan stock for.
type Intermittent struct {
	// Method is "sba" or "tsb".
	Method string
	// Alpha is how much of each new demand size the size takes, and
	// Beta how much of each new gap ("sba") or of each step's demand or
	// none ("tsb") the gap or chance takes, 0 to 1.
	Alpha, Beta float64
	// Size is the smoothed size of a demand, Interval the smoothed steps
	// from one demand to the next ("sba"), and Chance the smoothed
	// chance of any demand at a step ("tsb"), after the last step.
	Size, Interval, Chance float64
	// Rate is the forecast demand per step.
	Rate float32
	// Zeros is the share of the steps with a value that are 0.
	Zeros float64
	Steps int
	// Error is the mean absolute error of one-step forecasts of the held
	// out steps, and Naive that of repeating the last value, both in
	// units of the series' scale, as Smoothing's are.
	Error, Naive float64

	gap int // steps since the last demand
}

// Croston fits an intermittent demand forecast to series, which must be
// 0 or more, NaN for an unknown step, with demand at 2 steps at least.
// The last Holdout steps (by default a tenth, between 1 and 14) are
// scored; the rest fit the smoothing.
func Croston(series []float32, opts ...Option) (*Intermittent, error) {
	c := &config{}
	for _, o := range opts {
		o(c)
	}
	if c.holdout < 0 {
		return nil, fmt.Errorf("forecast: holdout %d: %w", c.holdout, tensor.ErrShape)
	}
	n := len(series)
	z := make([]float64, n)
	present, demands := 0, 0
	for i, v := range series {
		if v < 0 {
			return nil, fmt.Errorf("forecast: demand %v at step %d; intermittent demand is 0 or more: %w", v, i, tensor.ErrShape)
		}
		z[i] = float64(v)
		if v == v {
			present++
			if v > 0 {
				demands++
			}
		}
	}
	if demands < 2 || n < 3 {
		return nil, fmt.Errorf("forecast: %d steps with demand of %d: %w", demands, n, tensor.ErrShape)
	}
	hold := c.holdout
	if hold == 0 {
		hold = max(1, min(n/10, 14))
	}
	hold = min(hold, n-2)
	fit := z[:n-hold]
	if count(fit) < 2 {
		fit = z // too few demands to hold any out
	}
	var best *Intermittent
	bestErr := math.Inf(1)
	for _, method := range []string{"sba", "tsb"} {
		m := &Intermittent{Method: method, Steps: n}
		e := m.search(fit)
		if e < bestErr {
			best, bestErr = m, e
		}
	}
	m := best
	k := float64(scale(series))
	m.Zeros = float64(present-demands) / float64(max(present, 1))
	// Run over every step, scoring the held out ones.
	m.init(z)
	var errN, naiveN float64
	last := math.NaN()
	for t, v := range z {
		if t >= n-hold && v == v {
			m.Error += math.Abs(m.rate()-v) / k
			errN++
			if last == last {
				m.Naive += math.Abs(last-v) / k
				naiveN++
			}
		}
		if v == v {
			last = v
		}
		m.update(v)
	}
	m.Error /= max(errN, 1)
	m.Naive /= max(naiveN, 1)
	m.Rate = float32(m.rate())
	return m, nil
}

// count returns how many of z are above 0.
func count(z []float64) int {
	n := 0
	for _, v := range z {
		if v > 0 {
			n++
		}
	}
	return n
}

// search sets m's Alpha and Beta to those of a grid with the least
// squared one-step error over z, and returns that error.
func (m *Intermittent) search(z []float64) float64 {
	grid := []float64{0.02, 0.05, 0.1, 0.15, 0.2, 0.3}
	best, ba, bb := math.Inf(1), 0.1, 0.1
	for _, a := range grid {
		for _, b := range grid {
			m.Alpha, m.Beta = a, b
			if e := m.sse(z); e < best {
				best, ba, bb = e, a, b
			}
		}
	}
	m.Alpha, m.Beta = ba, bb
	return best
}

// sse returns the squared one-step error of m over z, from the step after
// its first demand on.
func (m *Intermittent) sse(z []float64) float64 {
	m.init(z)
	var s float64
	started := false
	for _, v := range z {
		if started && v == v {
			e := m.rate() - v
			s += e * e
		}
		started = started || v > 0
		m.update(v)
	}
	return s
}

// init starts m's state from z: the mean demand size, the mean gap
// between demands and the share of steps with any.
func (m *Intermittent) init(z []float64) {
	var size float64
	demands, present, first, last := 0, 0, -1, -1
	for t, v := range z {
		if v != v {
			continue
		}
		present++
		if v > 0 {
			size += v
			demands++
			if first < 0 {
				first = t
			}
			last = t
		}
	}
	m.Size = size / float64(max(demands, 1))
	m.Interval = 1
	if demands > 1 {
		m.Interval = float64(last-first) / float64(demands-1)
	}
	m.Chance = float64(demands) / float64(max(present, 1))
	m.gap = 0
}

// rate returns the demand per step m forecasts now.
func (m *Intermittent) rate() float64 {
	if m.Method == "tsb" {
		return m.Chance * m.Size
	}
	return m.Size / m.Interval * (1 - m.Beta/2)
}

// update moves m's state past a step of value v, NaN when unknown.
func (m *Intermittent) update(v float64) {
	m.gap++
	if v != v {
		return
	}
	if v > 0 {
		m.Size += m.Alpha * (v - m.Size)
		m.Interval += m.Beta * (float64(m.gap) - m.Interval)
		m.Chance += m.Beta * (1 - m.Chance)
		m.gap = 0
		return
	}
	m.Chance -= m.Beta * m.Chance
}

// Forecast returns the next steps steps' demand: Rate at each.
func (m *Intermittent) Forecast(steps int) []float32 {
	out := make([]float32, steps)
	for i := range out {
		out[i] = m.Rate
	}
	return out
}

// Sparse reports whether series has demand at fewer than 3 in 4 of its
// known steps, and none below 0: the rule of thumb (Syntetos, Boylan and
// Croston, 2005, a mean gap above 1.32 steps) for when Croston forecasts
// better than smoothing.
func Sparse(series []float32) bool {
	present, demands := 0, 0
	for _, v := range series {
		if v < 0 {
			return false
		}
		if v == v {
			present++
			if v > 0 {
				demands++
			}
		}
	}
	return present > 0 && demands >= 2 && float64(present)/float64(demands) > 1.32
}
