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

// Intermittent is Croston's method for demand that is zero on most
// steps, such as a spare part sold a few times a month: it smooths the
// size of the steps with demand and the number of steps between them
// apart, and forecasts their ratio, the demand per step. Exponential
// smoothing and boosted trees both chase the zeros on such a series and
// forecast too little right after a sale and too much after a gap.
//
// It uses the Syntetos-Boylan correction, (1 - Alpha/2) times the ratio,
// which takes out Croston's known bias upward.
type Intermittent struct {
	// Alpha is how much of each new demand and gap the smoothing takes,
	// 0 to 1.
	Alpha float64
	// Size is the smoothed demand of the steps with demand, and Interval
	// the smoothed steps from one to the next, after the last step.
	Size, Interval float64
	// Zeros is the share of present steps without demand.
	Zeros float64
	// Error is the mean absolute error of one-step forecasts of the held
	// out steps, and Naive that of forecasting the mean of the steps
	// before each, both in the series' units divided by its mean
	// absolute value, as Smoothing's.
	Error, Naive float64
	Scale        float32
}

// Croston fits Croston's method to a series of at least 3 steps, of
// which at least 2 have demand above zero, picking Alpha by the error of
// its one-step forecasts. A value of NaN is missing; a value below zero
// is an error.
func Croston(series []float32, opts ...Option) (*Intermittent, error) {
	c := &config{}
	for _, o := range opts {
		o(c)
	}
	if c.holdout < 0 {
		return nil, fmt.Errorf("forecast: holdout %d: %w", c.holdout, tensor.ErrShape)
	}
	n := len(series)
	k := scale(series)
	z := make([]float64, n)
	demand, present := 0, 0
	for i, v := range series {
		if v < 0 {
			return nil, fmt.Errorf("forecast: demand %v at step %d: %w", v, i, tensor.ErrShape)
		}
		z[i] = float64(v / k)
		if v == v {
			present++
			if v > 0 {
				demand++
			}
		}
	}
	if n < 3 || demand < 2 {
		return nil, fmt.Errorf("forecast: %d steps, %d with demand: %w", n, demand, tensor.ErrShape)
	}
	hold := c.holdout
	if hold == 0 {
		hold = max(1, min(n/5, 14))
	}
	hold = min(hold, n-2)
	m := &Intermittent{Scale: k, Zeros: 1 - float64(demand)/float64(present)}
	// The error of the fitted steps is flat in Alpha over wide ranges,
	// so a grid is enough.
	best := math.Inf(1)
	for a := 0.05; a <= 0.5+1e-9; a += 0.05 {
		if e, _ := m.run(z[:n-hold], a, 0); e < best {
			best, m.Alpha = e, a
		}
	}
	var naive float64
	m.Error, naive = m.run(z, m.Alpha, n-hold)
	m.Naive = naive
	return m, nil
}

// run smooths z with alpha and returns the mean absolute one-step error
// over the steps from from on, and that of forecasting the mean of the
// steps before each, leaving the state after the last step.
func (m *Intermittent) run(z []float64, alpha float64, from int) (float64, float64) {
	m.Size, m.Interval = 0, 0
	gap, started := 1.0, false
	var err, naive, sum float64
	errN, seen := 0, 0
	for t, v := range z {
		if v != v {
			gap++
			continue
		}
		if t >= from && started {
			err += math.Abs(m.rate(alpha) - v)
			naive += math.Abs(sum/float64(max(seen, 1)) - v)
			errN++
		}
		sum += v
		seen++
		if v == 0 {
			gap++
			continue
		}
		if !started {
			// The first demand sets the state; its gap is the steps
			// up to it.
			m.Size, m.Interval, started = v, gap, true
		} else {
			m.Size += alpha * (v - m.Size)
			m.Interval += alpha * (gap - m.Interval)
		}
		gap = 1
	}
	return err / float64(max(errN, 1)), naive / float64(max(errN, 1))
}

// rate returns the corrected demand per step of the state.
func (m *Intermittent) rate(alpha float64) float64 {
	if m.Interval == 0 {
		return 0
	}
	return (1 - alpha/2) * m.Size / m.Interval
}

// Forecast returns the next steps values after the series: the same
// demand per step for each, as Croston's method has no trend or season.
func (m *Intermittent) Forecast(steps int) []float32 {
	out := make([]float32, steps)
	v := float32(m.rate(m.Alpha)) * m.Scale
	for i := range out {
		out[i] = v
	}
	return out
}
