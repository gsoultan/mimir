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

package tensor_test

import (
	"context"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/tensor"
)

// namedGPU is a CPU backend that calls itself a GPU, so that the tensor
// package treats it as one.
type namedGPU struct{ *cpu.Backend }

func (namedGPU) Name() string { return "cuda" }

// TestFactorizationsStayOnTheGPU checks that matrices on a GPU are
// factorized there, with no copies to the CPU and back, unless they are
// too large, and that the results are right.
func TestFactorizationsStayOnTheGPU(t *testing.T) {
	gpu := tensor.DeviceOn(namedGPU{cpu.New()})
	ops := func(n int, f func(x *tensor.Tensor) *tensor.Tensor) []backend.Op {
		t.Helper()
		g := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{f(x[0])} }
		tg, err := tensor.Trace(g, []*tensor.Tensor{tensor.Zeros(tensor.Float32, n, n).To(gpu)}, nil)
		if err != nil {
			t.Fatal(err)
		}
		var out []backend.Op
		for n := tg.Nodes[tg.Outputs[0]]; len(n.Inputs) > 0; n = tg.Nodes[n.Inputs[0]] {
			out = append(out, n.Op)
		}
		return out
	}
	chol := func(x *tensor.Tensor) *tensor.Tensor { return x.Cholesky() }
	if got, want := ops(4, chol), []backend.Op{backend.OpCholesky}; !slices.Equal(got, want) {
		t.Errorf("Cholesky of 4×4 computed by %v, want %v", got, want)
	}
	if got := ops(2000, chol); !slices.Contains(got, backend.OpReshape) {
		t.Errorf("Cholesky of 2000×2000 computed by %v, want copies to the CPU and back", got)
	}
	solve := func(x *tensor.Tensor) *tensor.Tensor { return tensor.TriangularSolve(x, x, false) }
	if got, want := ops(4, solve), []backend.Op{backend.OpTriangularSolve}; !slices.Equal(got, want) {
		t.Errorf("triangular solve of 4×4 computed by %v, want %v", got, want)
	}

	a := tensor.FromSlice([]float32{4, 2, 2, 3}, 2, 2).To(gpu)
	l, err := a.Cholesky().Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 0, 1, 1.4142135}
	for i := range want {
		if d := l[i] - want[i]; d > 1e-6 || d < -1e-6 {
			t.Fatalf("Cholesky on the GPU = %v, want %v", l, want)
		}
	}
	if d := a.Cholesky().Device(); !d.Equal(gpu) {
		t.Errorf("Cholesky of a GPU's matrix is on %v", d)
	}
}

// TestVectorSignsLeadPositive checks the sign convention that devices
// share: each eigenvector and each left singular vector leads with a
// positive element, the first whose magnitude is at least half the
// largest's.
func TestVectorSignsLeadPositive(t *testing.T) {
	a := tensor.FromSlice([]float32{2, -1, 0.5, -1, 3, 0.25, 0.5, 0.25, 1}, 3, 3)
	leads := func(name string, v []float32, rows, cols int) {
		for c := range cols {
			big := float32(0)
			for r := range rows {
				big = max(big, abs32(v[r*cols+c]))
			}
			for r := range rows {
				if x := v[r*cols+c]; abs32(x) >= big/2 {
					if x < 0 {
						t.Errorf("%s column %d leads with %v", name, c, x)
					}
					break
				}
			}
		}
	}
	_, v := a.Eigh()
	vs, err := v.Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	leads("eigenvectors", vs, 3, 3)
	u, _, _ := tensor.FromSlice([]float32{1, -2, 3, -4, 5, -6}, 3, 2).SVD()
	us, err := u.Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	leads("left singular vectors", us, 3, 2)
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
