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
	"math"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkAttention times causal attention, forward and backward, at the
// shapes of GPT-2 small (12 heads of 64 features) over a batch of 8, as
// one fused kernel each way and composed of a product, a softmax and a
// product.
func BenchmarkAttention(b *testing.B) {
	const batch, heads, dim = 8, 12, 64
	for _, seq := range []int{256, 1024} {
		for _, impl := range []string{"fused", "composed"} {
			b.Run(fmt.Sprintf("%s/seq%d", impl, seq), func(b *testing.B) {
				shape := []int{batch, heads, seq, dim}
				q, k, v := realized(b, 1, shape...), realized(b, 2, shape...), realized(b, 3, shape...)
				q, k, v = q.RequiresGrad(), k.RequiresGrad(), v.RequiresGrad()
				w := realized(b, 4, shape...)
				attend := func() *tensor.Tensor {
					if impl == "fused" {
						return tensor.ScaledDotProductAttention(q, k, v, tensor.Causal())
					}
					return composedAttention(q, k, v, nil, 1/math.Sqrt(dim), true, 0)
				}
				step := func() {
					if err := attend().Mul(w).Sum().Backward(b.Context()); err != nil {
						b.Fatal(err)
					}
					q.ZeroGrad()
					k.ZeroGrad()
					v.ZeroGrad()
				}
				warm(step)
				for b.Loop() {
					step()
				}
				// The causal forward pass multiplies half the scores twice;
				// the backward pass recomputes them and multiplies four
				// times more.
				flops := 7 * float64(batch*heads*seq*seq*dim)
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}

// BenchmarkAttentionForward times the forward pass alone, as inference
// runs it.
func BenchmarkAttentionForward(b *testing.B) {
	const batch, heads, dim = 8, 12, 64
	for _, seq := range []int{256, 1024} {
		for _, impl := range []string{"fused", "composed"} {
			b.Run(fmt.Sprintf("%s/seq%d", impl, seq), func(b *testing.B) {
				shape := []int{batch, heads, seq, dim}
				q, k, v := realized(b, 1, shape...), realized(b, 2, shape...), realized(b, 3, shape...)
				run(b, func() *tensor.Tensor {
					if impl == "fused" {
						return tensor.ScaledDotProductAttention(q, k, v, tensor.Causal())
					}
					return composedAttention(q, k, v, nil, 1/math.Sqrt(dim), true, 0)
				})
				flops := 2 * float64(batch*heads*seq*seq*dim)
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}
