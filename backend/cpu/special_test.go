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

package cpu

import (
	"math"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// TestSpecialFunctions checks each special function kernel at values
// known in closed form, at its poles and limits.
func TestSpecialFunctions(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	gamma := 0.5772156649015329 // Euler's constant
	cases := []struct {
		op   backend.Op
		x    []float32
		want []float64
	}{
		{backend.OpLog1p, []float32{1e-10, -0.5, 1, -1, -2, float32(inf)}, []float64{1e-10, math.Log(0.5), math.Ln2, -inf, nan, inf}},
		{backend.OpExpm1, []float32{1e-10, -1, 1, float32(-inf), 100}, []float64{1e-10, math.Exp(-1) - 1, math.E - 1, -1, inf}},
		{backend.OpErf, []float32{0, 0.5, -2, float32(inf)}, []float64{0, 0.5204998778130465, -0.9953222650189527, 1}},
		{backend.OpErfinv, []float32{0, 0.5, -0.75, 1, -1, 1.5}, []float64{0, 0.4769362762044699, -0.8134198475976185, inf, -inf, nan}},
		{backend.OpLgamma, []float32{0.5, -0.5, 1, 2, 0, -2, 10, float32(-inf)}, []float64{0.5 * math.Log(math.Pi), math.Log(2 * math.Sqrt(math.Pi)), 0, 0, inf, inf, math.Log(362880), inf}},
		{backend.OpDigamma, []float32{1, 0.5, -0.5, 0, float32(math.Copysign(0, -1)), -1, 100, float32(inf)},
			[]float64{-gamma, -gamma - 2*math.Ln2, 0.03648997397857652, -inf, inf, nan, 4.600161852738087, inf}},
		{backend.OpTrigamma, []float32{1, 0.5, 1.5, -0.5, 0, -3, float32(inf)},
			[]float64{math.Pi * math.Pi / 6, math.Pi * math.Pi / 2, math.Pi*math.Pi/2 - 4, math.Pi*math.Pi/2 + 4, inf, inf, 0}},
	}
	for _, c := range cases {
		t.Run(c.op.String(), func(t *testing.T) {
			k := &backend.Kernel{Op: c.op, DType: backend.Float32, Shape: []int{len(c.x)}, In: []backend.View{backend.Contiguous([]int{len(c.x)})}}
			got, err := run[float32](t, k, f32buf(c.x...))
			if err != nil {
				t.Fatal(err)
			}
			for i, w := range c.want {
				g := float64(got[i])
				if math.IsNaN(w) != math.IsNaN(g) || !math.IsNaN(w) && math.Abs(g-w) > 2e-7*math.Max(1, math.Abs(w)) && g != w {
					t.Errorf("%s(%v) = %v, want %v", c.op, c.x[i], g, w)
				}
			}
		})
	}
}

// TestPolygammaAreDerivatives checks digamma against central differences
// of lgamma, and trigamma against those of digamma, on both sides of
// zero.
func TestPolygammaAreDerivatives(t *testing.T) {
	const h = 1e-5
	for x := -4.75; x < 30; x += 0.37 {
		if d := (lgamma(x+h) - lgamma(x-h)) / (2 * h); math.Abs(digamma(x)-d) > 1e-6*math.Max(1, math.Abs(d)) {
			// lgamma is log|Γ|, whose derivative is ψ on both sides.
			t.Errorf("digamma(%v) = %v, lgamma' = %v", x, digamma(x), d)
		}
		if d := (digamma(x+h) - digamma(x-h)) / (2 * h); math.Abs(trigamma(x)-d) > 1e-5*math.Max(1, math.Abs(d)) {
			t.Errorf("trigamma(%v) = %v, digamma' = %v", x, trigamma(x), d)
		}
	}
}
