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

package kerneltest

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// Linalg returns cases of factorizations and triangular solves of
// single matrices and batches, read in place and through other views, of
// well-conditioned matrices whose results do not hang on rounding, and
// of matrices that fail: one that is not positive definite and a
// triangular one with a zero on its diagonal. Results are compared within
// a tolerance, since GPU kernels compute in float and by other
// algorithms.
func Linalg(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(17, 18))
	f := backend.Float32
	var cases []Case
	views := func(shape []int, keep ...string) []string {
		var out []string
		for name := range Views(shape) {
			if len(keep) == 0 || slices.Contains(keep, name) {
				out = append(out, name)
			}
		}
		slices.Sort(out)
		return out
	}
	factor := func(op backend.Op, shape []int, tol float64, keep []string, gen func(rows, cols int) []float64) {
		for _, name := range views(shape, keep...) {
			if name == "broadcast" && len(shape) == 2 && op != backend.OpCholesky && op != backend.OpLU {
				continue // equal rows: a matrix of rank 1, whose vectors are not unique
			}
			x := Views(shape)[name]
			in := Fill(t, x, shape, matrices(shape, gen))
			out := shape
			if op != backend.OpCholesky {
				_, size, err := backend.FactorParts(op, shape)
				if err != nil {
					t.Fatal(err)
				}
				out = []int{size}
			}
			cases = append(cases, Case{
				Name: fmt.Sprintf("%s/%v/%s", op, shape, name),
				K:    backend.Kernel{Op: op, DType: f, Shape: out, In: []backend.View{x.V}},
				Ins:  []backend.Buffer{in},
				Tol:  tol,
			})
		}
	}
	random := func(rows, cols int) []float64 {
		a := make([]float64, rows*cols)
		for i := range a {
			a[i] = rng.Float64()*4 - 2
		}
		return a
	}
	spd := func(n, _ int) []float64 {
		m := random(n, n)
		a := make([]float64, n*n)
		for i := range n {
			for j := range n {
				s := 0.0
				for q := range n {
					s += m[i*n+q] * m[j*n+q]
				}
				a[i*n+j] = s / float64(n)
			}
			a[i*n+i]++
		}
		return a
	}
	notPD := func(n, _ int) []float64 {
		a := spd(n, n)
		a[(n/2)*n+n/2] = -1
		return a
	}
	for _, s := range [][]int{{1, 1}, {5, 5}, {2, 3, 9, 9}, {40, 40}} {
		factor(backend.OpCholesky, s, 1e-4, nil, spd)
		factor(backend.OpLU, s, 1e-4, nil, random)
		factor(backend.OpEigh, s, 2e-3, nil, random)
	}
	factor(backend.OpCholesky, []int{3, 6, 6}, 0, []string{"contiguous"}, notPD)
	for _, s := range [][]int{{1, 1}, {6, 4}, {4, 6}, {5, 5}, {2, 3, 7, 3}, {2, 3, 7}, {33, 20}} {
		factor(backend.OpQR, s, 2e-4, nil, random)
		factor(backend.OpSVD, s, 2e-3, nil, random)
	}
	// A zero matrix and an LU of a singular one, whose zero pivot stays.
	zero := func(rows, cols int) []float64 { return make([]float64, rows*cols) }
	factor(backend.OpLU, []int{4, 4}, 1e-6, []string{"contiguous"}, zero)
	factor(backend.OpQR, []int{4, 3}, 1e-6, []string{"contiguous"}, zero)

	// Triangular systems with diagonals far from zero, and one with a
	// zero on it.
	tri := func(zeroAt int) func(n, _ int) []float64 {
		return func(n, _ int) []float64 {
			a := random(n, n)
			for i := range n {
				for j := range n {
					a[i*n+j] /= float64(n)
				}
				a[i*n+i] = math.Copysign(2+rng.Float64(), a[i*n+i])
			}
			if zeroAt >= 0 {
				a[zeroAt*n+zeroAt] = 0
			}
			return a
		}
	}
	for _, s := range []struct{ a, b []int }{
		{[]int{1, 1}, []int{1, 3}},
		{[]int{6, 6}, []int{6, 1}},
		{[]int{2, 9, 9}, []int{2, 9, 4}},
		{[]int{30, 30}, []int{30, 17}},
	} {
		for _, name := range views(s.a, "contiguous", "transposed", "padded") {
			x := Views(s.a)[name]
			y := Views(s.b)["flipped"]
			a := Fill(t, x, s.a, matrices(s.a, tri(-1)))
			b := Fill(t, y, s.b, matrices(s.b, random))
			for _, upper := range []bool{false, true} {
				for _, unit := range []bool{false, true} {
					cases = append(cases, Case{
						Name: fmt.Sprintf("triangular_solve/%v/%v/%s/upper=%v/unit=%v", s.a, s.b, name, upper, unit),
						K: backend.Kernel{Op: backend.OpTriangularSolve, DType: f, Shape: s.b, In: []backend.View{x.V, y.V},
							Attrs: backend.Attrs{Upper: upper, Unit: unit}},
						Ins: []backend.Buffer{a, b},
						Tol: 1e-4,
					})
				}
			}
		}
	}
	a := Fill(t, Views([]int{5, 5})["contiguous"], []int{5, 5}, matrices([]int{5, 5}, tri(3)))
	b := Fill(t, Views([]int{5, 2})["contiguous"], []int{5, 2}, matrices([]int{5, 2}, random))
	cases = append(cases, Case{
		Name: "triangular_solve/singular",
		K: backend.Kernel{Op: backend.OpTriangularSolve, DType: f, Shape: []int{5, 2},
			In: []backend.View{backend.Contiguous([]int{5, 5}), backend.Contiguous([]int{5, 2})}},
		Ins: []backend.Buffer{a, b},
	})
	return cases
}

// matrices returns the elements, in row-major order, of a batch of
// matrices of shape, each from gen.
func matrices(shape []int, gen func(rows, cols int) []float64) []float64 {
	r := len(shape)
	items, _ := backend.NumElements(shape[:r-2])
	var out []float64
	for range items {
		out = append(out, gen(shape[r-2], shape[r-1])...)
	}
	return out
}

// Fill returns a Host buffer of x.N floats that x reads as the tensor of
// shape whose elements, in row-major order, are vals; the rest of the
// buffer is zero. A view that reads an element more than once, as a
// broadcast does, reads the last value written there.
func Fill(t *testing.T, x View, shape []int, vals []float64) backend.Buffer {
	t.Helper()
	buf := make([]float32, x.N)
	idx := make([]int, len(shape))
	for _, v := range vals {
		off := x.V.Offset
		for d, i := range idx {
			off += i * x.V.Strides[d]
		}
		buf[off] = float32(v)
		for d := len(idx) - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	b, err := Host.Alloc(backend.Float32, x.N)
	if err != nil {
		t.Fatal(err)
	}
	if err := Host.Upload(t.Context(), b, Bytes(buf)); err != nil {
		t.Fatal(err)
	}
	return b
}
