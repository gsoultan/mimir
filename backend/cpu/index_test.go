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
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

func TestScatterAddAddsInRowMajorOrder(t *testing.T) {
	// Random floats, so that a different order of the updates of an
	// element changes its sum.
	rng := rand.New(rand.NewPCG(5, 6))
	shape := []int{30, 40, 20} // several chunks of lines for the workers
	for axis := range shape {
		upd := slices.Clone(shape)
		upd[axis] = 300
		m, _ := backend.NumElements(upd)
		n, _ := backend.NumElements(shape)
		dst, src, idx := make([]float32, n), make([]float32, m), make([]int32, m)
		for i := range dst {
			dst[i] = rng.Float32()
		}
		for i := range src {
			src[i], idx[i] = rng.Float32()*2-1, int32(rng.IntN(shape[axis]))
		}
		// Updates read transposed, from a buffer laid out in reverse.
		rev := slices.Clone(upd)
		slices.Reverse(rev)
		vu := backend.Contiguous(rev).Permute([]int{2, 1, 0})
		want := slices.Clone(dst)
		eachIndex(upd, func(m []int) {
			at := slices.Clone(m)
			at[axis] = int(idx[offsetOf(backend.Contiguous(upd), m)])
			want[offsetOf(backend.Contiguous(shape), at)] += src[offsetOf(vu, m)]
		})
		k := &backend.Kernel{
			Op: backend.OpScatterAdd, DType: backend.Float32, Shape: shape,
			In:    []backend.View{backend.Contiguous(shape), backend.Contiguous(upd), vu},
			Attrs: backend.Attrs{Axis: axis},
		}
		for _, workers := range []int{1, 3} {
			got := runOn[float32](t, workers, k, f32buf(dst...), i32buf(idx...), f32buf(src...))
			if !slices.Equal(got, want) {
				t.Fatalf("axis %d, %d workers: sums differ from row-major order", axis, workers)
			}
		}
	}
}
