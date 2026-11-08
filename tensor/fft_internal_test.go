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

package tensor

import (
	"math"
	"testing"
)

// TestBluesteinMatchesTheTransform checks the transform GPUs build for
// lengths that are not powers of two, run here on the CPU, against the
// CPU's kernel, values and gradients.
func TestBluesteinMatchesTheTransform(t *testing.T) {
	ctx := t.Context()
	for _, inverse := range []bool{false, true} {
		x := Randn(3, 2, 100, 3, 2)
		w := Randn(4, 2, 100, 3, 2)
		var vals, gs [2][]float32
		for i, f := range []func(*Tensor) *Tensor{
			func(x *Tensor) *Tensor { return x.bluestein(1, inverse) },
			func(x *Tensor) *Tensor { return x.dft(1, inverse) },
		} {
			p := x.RequiresGrad()
			y := f(p)
			if err := y.Mul(w).Sum().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			var err error
			if vals[i], err = y.Detach().Data[float32](ctx); err != nil {
				t.Fatal(err)
			}
			if gs[i], err = p.Grad().Data[float32](ctx); err != nil {
				t.Fatal(err)
			}
		}
		for i := range vals[0] {
			if d := math.Abs(float64(vals[0][i] - vals[1][i])); d > 1e-4*math.Max(1, math.Abs(float64(vals[1][i]))) {
				t.Fatalf("inverse %v: element %d is %v, want %v", inverse, i, vals[0][i], vals[1][i])
			}
			if d := math.Abs(float64(gs[0][i] - gs[1][i])); d > 1e-4*math.Max(1, math.Abs(float64(gs[1][i]))) {
				t.Fatalf("inverse %v: gradient %d is %v, want %v", inverse, i, gs[0][i], gs[1][i])
			}
		}
	}
}
