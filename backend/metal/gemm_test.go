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

package metal

import (
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// TestGEMMPacked checks which batched products MPS may run: those whose
// matrices lie rows × ld apart, and no others.
func TestGEMMPacked(t *testing.T) {
	f := backend.Float32
	for name, c := range map[string]struct {
		x    backend.View
		n    int
		want bool
	}{
		"contiguous": {backend.Contiguous([]int{2, 4, 6}), 48, true},
		"padded":     {backend.View{Shape: []int{2, 4, 6}, Strides: []int{48, 8, 1}, Offset: 9}, 4 * 48, false},
	} {
		k := &backend.Kernel{Op: backend.OpMatMul, DType: f, Shape: []int{2, 4, 5},
			In: []backend.View{c.x, backend.Contiguous([]int{2, 6, 5})}}
		g, ok := gen.PlanGEMM(k, []int{c.n, 60})
		if !ok {
			t.Fatalf("%s: not planned", name)
		}
		if packed(g) != c.want {
			t.Errorf("%s: packed %t, want %t", name, packed(g), c.want)
		}
	}
}
