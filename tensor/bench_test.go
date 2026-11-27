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
	"fmt"
	"testing"
	"time"

	"github.com/gsoultan/mimir/tensor"
)

// realized returns a computed random tensor so benchmarks time only the
// operation under test.
func realized(b *testing.B, seed uint64, shape ...int) *tensor.Tensor {
	b.Helper()
	x := tensor.Randn(seed, shape...)
	if err := x.Realize(b.Context()); err != nil {
		b.Fatal(err)
	}
	return x
}

// warmup is the number of times a benchmark runs its operation before it
// starts timing: the first run compiles the GPU kernels, plans the
// schedule and autotunes, the second replays the cached schedule and, on
// CUDA, captures its graph, and the third replays it as the rest will.
// Each of these happens once per program, so timing them would report a
// step's speed wrong by seconds when b.N is small.
const warmup = 3

// warm runs step warmup times, and then on until warmupTime has passed
// since it began: a GPU that idled, as it does while a benchmark binary
// builds, runs at low clocks until it has been busy for a while, a few
// hundred milliseconds on a T4, so a short benchmark would otherwise time
// some steps at the clocks of an idle GPU. tools/bench/attention.py warms
// PyTorch the same way.
func warm(step func()) {
	start := time.Now()
	for i := 0; i < warmup || time.Since(start) < warmupTime; i++ {
		step()
	}
}

// warmupTime is the least time warm runs steps for.
const warmupTime = 300 * time.Millisecond

// run times the realization of f's result, after warmup untimed ones.
func run(b *testing.B, f func() *tensor.Tensor) {
	b.Helper()
	realize := func() {
		if err := f().Realize(b.Context()); err != nil {
			b.Fatal(err)
		}
	}
	warm(realize)
	for b.Loop() {
		realize()
	}
}

func BenchmarkMatMul(b *testing.B) {
	for _, n := range []int{64, 256, 512} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			x, y := realized(b, 1, n, n), realized(b, 2, n, n)
			b.SetBytes(int64(3 * n * n * 4))
			run(b, func() *tensor.Tensor { return x.MatMul(y) })
			b.ReportMetric(2*float64(n*n*n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	const n = 1 << 22
	x, y := realized(b, 1, n), realized(b, 2, n)
	b.SetBytes(3 * n * 4)
	run(b, func() *tensor.Tensor { return x.Add(y) })
}

func BenchmarkAddBroadcast(b *testing.B) {
	x, y := realized(b, 1, 2048, 2048), realized(b, 2, 2048)
	b.SetBytes(2 * 2048 * 2048 * 4)
	run(b, func() *tensor.Tensor { return x.Add(y) })
}

func BenchmarkSum(b *testing.B) {
	x := realized(b, 1, 2048, 2048)
	b.SetBytes(2048 * 2048 * 4)
	run(b, func() *tensor.Tensor { return x.Sum(1) })
}

func BenchmarkSoftmax(b *testing.B) {
	x := realized(b, 1, 1024, 1024)
	b.SetBytes(1024 * 1024 * 4)
	run(b, func() *tensor.Tensor { return x.Softmax(-1) })
}

func BenchmarkRandn(b *testing.B) {
	const n = 1 << 22
	b.SetBytes(n * 4)
	run(b, func() *tensor.Tensor { return tensor.Randn(1, n) })
}

func BenchmarkMLPForward(b *testing.B) {
	x := realized(b, 1, 256, 784)
	w1, w2 := realized(b, 2, 784, 512), realized(b, 3, 512, 10)
	b1, b2 := realized(b, 4, 512), realized(b, 5, 10)
	run(b, func() *tensor.Tensor {
		return x.MatMul(w1).Add(b1).ReLU().MatMul(w2).Add(b2).LogSoftmax(-1)
	})
}
