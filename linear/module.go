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

package linear

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Shape is what sizes a Module: enough to build an empty one that saved
// weights load into.
type Shape struct {
	Features, Outputs int
	Classes           int // 0 for Regression
	Link              string
}

// Module is a Model as an nn.Module, so it runs, saves and loads wherever
// a network does. Forward maps x, shape [batch, Features], to [batch,
// Classes] logits whose softmax is the model's probabilities, or to
// [batch, 1] values for Regression. Its weights are buffers, not trained
// parameters.
type Module struct {
	Weight *tensor.Param // Float32 [Outputs, Features]
	Bias   *tensor.Param // Float32 [1, Outputs]
	Fill   *tensor.Param // Float32 [1, Features], what a missing value counts as
	shape  Shape
}

// NewModule returns a Module of shape s whose weights are zero, for saved
// weights to be loaded into.
func NewModule(s Shape) *Module {
	f, k := max(s.Features, 1), max(s.Outputs, 1)
	return &Module{
		Weight: tensor.NewBuffer(tensor.Zeros(tensor.Float32, k, f)),
		Bias:   tensor.NewBuffer(tensor.Zeros(tensor.Float32, 1, k)),
		Fill:   tensor.NewBuffer(tensor.Zeros(tensor.Float32, 1, f)),
		shape:  s,
	}
}

// Shape returns the shape of m's Module.
func (m *Model) Shape() Shape {
	return Shape{Features: m.Features, Outputs: m.Outputs, Classes: m.Classes, Link: m.Link}
}

// Module returns m as a Module.
func (m *Model) Module() *Module {
	return &Module{
		Weight: tensor.NewBuffer(tensor.FromSlice(slices.Clone(m.Weights), m.Outputs, m.Features)),
		Bias:   tensor.NewBuffer(tensor.FromSlice(slices.Clone(m.Bias), 1, m.Outputs)),
		Fill:   tensor.NewBuffer(tensor.FromSlice(slices.Clone(m.Fill), 1, m.Features)),
		shape:  m.Shape(),
	}
}

// Shape returns the shape mod was built with.
func (mod *Module) Shape() Shape { return mod.shape }

// Forward answers each row of x, shape [batch, Features].
func (mod *Module) Forward(x *tensor.Tensor) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	s := mod.shape
	if x.Rank() != 2 || x.Shape()[1] != s.Features {
		return tensor.Failed(fmt.Errorf("linear: input %v for rows of %d features: %w", x.Shape(), s.Features, tensor.ErrShape))
	}
	x = tensor.Where(x.Equal(x), x, mod.Fill.Value())
	out := x.MatMulT(mod.Weight.Value()).Add(mod.Bias.Value())
	if s.Link == "sigmoid" {
		// Logits 0 and the log-odds have the sigmoid's probabilities.
		return tensor.Cat(1, tensor.Zeros(tensor.Float32, x.Shape()[0], 1), out)
	}
	return out
}
