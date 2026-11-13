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

package nn_test

import (
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkLSTM times a 2-layer LSTM of 256 hidden units over sequences
// of 128 steps of 64 features, in batches of 32: the forward pass alone,
// the forward and backward passes, and a whole training step with an
// AdamW update.
func BenchmarkLSTM(b *testing.B) {
	const steps, batch, in, hidden = 128, 32, 64, 256
	setup := func(b *testing.B, opts ...nn.Option) (*nn.LSTM, *tensor.Tensor, *tensor.Tensor) {
		nn.ManualSeed(1)
		m := nn.NewLSTM(in, hidden, append([]nn.Option{nn.NumLayers(2)}, opts...)...)
		x, y := tensor.Randn(1, steps, batch, in), tensor.Randn(2, steps, batch, hidden)
		if err := tensor.Realize(b.Context(), x, y); err != nil {
			b.Fatal(err)
		}
		return m, x, y
	}
	b.Run("forward", func(b *testing.B) {
		m, x, _ := setup(b)
		step := func() {
			if err := m.Forward(x).Realize(b.Context()); err != nil {
				b.Fatal(err)
			}
		}
		warm(step)
		for b.Loop() {
			step()
		}
	})
	backward := func(b *testing.B, opts ...nn.Option) {
		m, x, y := setup(b, opts...)
		step := func() {
			if err := loss.MSE(m.Forward(x), y).Backward(b.Context()); err != nil {
				b.Fatal(err)
			}
			for _, p := range nn.Parameters(m) {
				p.ZeroGrad()
			}
		}
		warm(step)
		for b.Loop() {
			step()
		}
	}
	b.Run("backward", func(b *testing.B) { backward(b) })
	b.Run("backward/checkpoint16", func(b *testing.B) { backward(b, nn.StepCheckpoint(16)) })
	b.Run("train", func(b *testing.B) {
		m, x, y := setup(b)
		opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
		step := func() {
			if err := loss.MSE(m.Forward(x), y).Backward(b.Context()); err != nil {
				b.Fatal(err)
			}
			if err := opt.Step(b.Context()); err != nil {
				b.Fatal(err)
			}
			opt.ZeroGrad()
		}
		warm(step)
		for b.Loop() {
			step()
		}
	})
}
