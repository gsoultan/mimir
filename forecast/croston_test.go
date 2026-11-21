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
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// lumpy returns n steps of demand that comes at a step with chance p, of
// a size around size.
func lumpy(n int, p, size float64, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 7))
	s := make([]float32, n)
	for i := range s {
		if r.Float64() < p {
			s[i] = float32(math.Max(1, math.Round(size+size/3*r.NormFloat64())))
		}
	}
	return s
}

func TestCrostonForecastsTheMeanDemand(t *testing.T) {
	for _, tc := range []struct{ p, size float64 }{{0.2, 6}, {0.1, 20}, {0.35, 3}} {
		var sumF, sumHW, sumTrue float64
		trials := 20
		for k := range trials {
			s := lumpy(260, tc.p, tc.size, uint64(k+1))
			if !Sparse(s) {
				t.Fatalf("p %v: a series with demand at %v of steps is not sparse", tc.p, tc.p)
			}
			m, err := Croston(s[:200])
			if err != nil {
				t.Fatal(err)
			}
			f := m.Forecast(60)
			if f[0] != f[59] || f[0] != m.Rate {
				t.Fatalf("the forecast is not flat: %v", f[:3])
			}
			// The forecast's error against the next 60 steps' total, as
			// stock would be planned.
			var want float64
			for _, v := range s[200:] {
				want += float64(v)
			}
			sumF += math.Abs(float64(m.Rate)*60-want) / want
			sumTrue += math.Abs(tc.p*tc.size*60-want) / want
			hw, err := Smooth(s[:200])
			if err != nil {
				t.Fatal(err)
			}
			var got float64
			for _, v := range hw.Forecast(60) {
				got += float64(v)
			}
			sumHW += math.Abs(got-want) / want
		}
		// Demand this lumpy misses its own mean by a lot: compare with
		// knowing the true mean.
		if sumF > 2*sumTrue {
			t.Errorf("p %v: Croston misses the next 60 steps' demand by %v on average, the true mean by %v", tc.p, sumF/float64(trials), sumTrue/float64(trials))
		}
		if sumF > sumHW {
			t.Errorf("p %v: Croston misses by %v, smoothing by %v", tc.p, sumF/float64(trials), sumHW/float64(trials))
		}
	}
}

func TestCrostonFadesADeadItem(t *testing.T) {
	// Sold every few steps, then never again.
	s := append(lumpy(120, 0.3, 5, 3), make([]float32, 80)...)
	m, err := Croston(s)
	if err != nil {
		t.Fatal(err)
	}
	if m.Method != "tsb" {
		t.Errorf("method %q for an item that stopped selling, want tsb", m.Method)
	}
	if m.Rate > 0.3 {
		t.Errorf("rate %v after 80 steps without a sale", m.Rate)
	}
}

func TestCrostonInput(t *testing.T) {
	s := lumpy(100, 0.3, 4, 5)
	s[10], s[11] = float32(math.NaN()), float32(math.NaN())
	m, err := Croston(s, Holdout(10))
	if err != nil {
		t.Fatal(err)
	}
	if !(m.Rate > 0) || m.Zeros < 0.5 || m.Error <= 0 {
		t.Errorf("rate %v, zeros %v, error %v", m.Rate, m.Zeros, m.Error)
	}
	for name, bad := range map[string][]float32{
		"negative":  {1, 0, -2, 0, 3},
		"one sale":  {0, 0, 4, 0, 0, 0},
		"all zeros": make([]float32, 30),
	} {
		if _, err := Croston(bad); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
	if Sparse([]float32{3, 4, 5, 6, 0, 7}) {
		t.Error("a series with demand most steps is sparse")
	}
}
