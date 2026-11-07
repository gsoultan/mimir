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

package nn

import (
	"errors"
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// LoRA is a Linear layer with a low-rank adapter, as in "LoRA: Low-Rank
// Adaptation of Large Language Models" and PEFT's LoraLayer: it computes
// Base(x) + Scale·(x·Aᵀ)·Bᵀ, which is a Linear layer of weight
// W + Scale·B·A. Only A and B train; the base layer's weights are
// buffers, so they may be block-quantized, as weights read from a Mimir
// Model file of a language model are.
//
// B starts at zero, so a fresh LoRA computes what its base does.
type LoRA struct {
	Base  *Linear
	A     *tensor.Param // [rank, in]
	B     *tensor.Param // [out, rank]
	Scale float64       `nn:"-"` // alpha / rank
}

// NewLoRA returns base with an adapter of the given rank and scale
// alpha/rank. A is drawn from U(-1/√in, 1/√in), PyTorch's default for a
// Linear layer and PEFT's for A, and B is zero. Trainable weights of base
// become buffers, which Backward computes no gradient for.
func NewLoRA(base *Linear, rank int, alpha float64, opts ...Option) *LoRA {
	o := resolve(opts, options{})
	w := base.Weight.Value()
	out, in := 0, 0
	if s := w.Shape(); len(s) == 2 {
		out, in = s[0], s[1]
	}
	l := &LoRA{Base: &Linear{Weight: frozen(base.Weight), Bias: frozen(base.Bias)}}
	if out == 0 || in == 0 || rank <= 0 || !(alpha > 0) || math.IsInf(alpha, 0) {
		err := fmt.Errorf("nn: LoRA of rank %d and alpha %v for a weight of shape %v: %w", rank, alpha, w.Shape(), tensor.ErrShape)
		if w.Err() != nil {
			err = w.Err()
		}
		l.A = tensor.NewParam(tensor.Failed(err))
		l.B = l.A
		return l
	}
	l.A = uniform(o, 1/math.Sqrt(float64(in)), rank, in)
	l.B = tensor.NewParam(tensor.Zeros(tensor.Float32, out, rank))
	l.Scale = alpha / float64(rank)
	if err := errors.Join(l.A.To(w.Device()), l.B.To(w.Device())); err != nil {
		l.A = tensor.NewParam(tensor.Failed(err))
		l.B = l.A
	}
	return l
}

// frozen returns p as a buffer: p itself if it isn't trainable.
func frozen(p *tensor.Param) *tensor.Param {
	if p == nil || !p.Trainable() {
		return p
	}
	return tensor.NewBuffer(p.Value())
}

// Forward maps [..., in] to [..., out].
func (l *LoRA) Forward(x *tensor.Tensor) *tensor.Tensor {
	y := l.Base.Forward(x)
	d := x.MatMulT(l.A.Value()).MatMulT(l.B.Value())
	if l.Scale != 1 {
		d = d.Mul(tensor.Full(d.DType(), l.Scale).To(d.Device()))
	}
	return y.Add(d)
}
