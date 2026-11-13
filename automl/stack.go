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

package automl

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Stacked is the family of a Result whose Best blends several candidates.
const Stacked = "stack"

// Stack has Search blend up to n of its best candidates when their blend
// scores better than the best alone, as auto-sklearn and AutoGluon do:
// each candidate's answers for rows it didn't learn from, kept from
// scoring it, are averaged with weights chosen by greedy ensemble
// selection (Caruana et al., 2004), and the weights kept are those of the
// best blend. The blend's score is on the same rows its weights were
// chosen on, so it is a little hopeful; a blend is kept only when it beats
// the best candidate by StackMargin. n is 3 by default when Stack is
// given; 0 or 1 turns stacking off.
func Stack(n int) Option { return func(c *config) { c.stack = n } }

// StackMargin is the share by which a blend's score must beat the best
// candidate's for Search to keep it.
const StackMargin = 0.005

// stackRounds is how many picks greedy ensemble selection makes: a
// weight is the share of the picks that went to its candidate.
const stackRounds = 20

// Blend is a weighted average of models' answers: of the values for
// Regression, of the probabilities for Classification.
type Blend struct {
	// Models are the blended models, each fit to every row, and Members
	// their candidates.
	Models  []tabular.Model
	Members []Candidate
	// Weights are each model's share, summing to 1.
	Weights []float64
}

// Predict answers each row of x with the weighted average of the models'
// answers.
func (b *Blend) Predict(x []float32) []float32 {
	var out []float32
	for i, m := range b.Models {
		p := m.Predict(x)
		if out == nil {
			out = make([]float32, len(p))
		}
		w := float32(b.Weights[i])
		for j, v := range p {
			out[j] += w * v
		}
	}
	return out
}

// blend picks weights for the candidates' out-of-fold answers p, all for
// the rows whose answers are y, by greedy ensemble selection with
// replacement, starting from the best alone. It returns the weights and
// the blend's score.
func blend(p [][]float32, y []float32, score tabular.Score) ([]float64, float64) {
	counts := make([]int, len(p))
	sum := make([]float32, len(p[0]))
	cur := make([]float32, len(sum))
	best, bestScore := -1, math.Inf(1)
	for i := range p {
		if s := score(p[i], y); s < bestScore {
			best, bestScore = i, s
		}
	}
	counts[best] = 1
	copy(sum, p[best])
	weights := func() []float64 {
		total := 0
		for _, c := range counts {
			total += c
		}
		w := make([]float64, len(counts))
		for i, c := range counts {
			w[i] = float64(c) / float64(total)
		}
		return w
	}
	keep := weights()
	for round := 1; round < stackRounds; round++ {
		pick, pickScore := -1, math.Inf(1)
		for i := range p {
			inv := 1 / float32(round+1)
			for j, v := range sum {
				cur[j] = (v + p[i][j]) * inv
			}
			if s := score(cur, y); s < pickScore {
				pick, pickScore = i, s
			}
		}
		counts[pick]++
		for j, v := range p[pick] {
			sum[j] += v
		}
		if pickScore < bestScore {
			bestScore, keep = pickScore, weights()
		}
	}
	return keep, bestScore
}

// StackModule is a Blend as an nn.Module: Forward maps x to the weighted
// average of its parts' answers, the logs of the averaged probabilities
// for Classification, as each part's softmax is its probabilities.
type StackModule struct {
	Parts   []nn.Module
	Weights *tensor.Param // Float32 [len(Parts)], a buffer
	classes int
}

// NewStackModule returns a StackModule of parts whose weights are zero,
// for saved weights to be loaded into. classes is 0 for Regression.
func NewStackModule(parts []nn.Module, classes int) *StackModule {
	return &StackModule{Parts: parts, Weights: tensor.NewBuffer(tensor.Zeros(tensor.Float32, max(len(parts), 1))), classes: classes}
}

// StackOf returns a StackModule of parts with weights, as a Blend's.
func StackOf(parts []nn.Module, weights []float64, classes int) (*StackModule, error) {
	if len(parts) == 0 || len(parts) != len(weights) {
		return nil, fmt.Errorf("automl: %d parts and %d weights: %w", len(parts), len(weights), tensor.ErrShape)
	}
	w := make([]float32, len(weights))
	for i, v := range weights {
		w[i] = float32(v)
	}
	return &StackModule{Parts: parts, Weights: tensor.NewBuffer(tensor.FromSlice(w)), classes: classes}, nil
}

// Forward blends the parts' answers for x.
func (s *StackModule) Forward(x *tensor.Tensor) *tensor.Tensor {
	var out *tensor.Tensor
	w := s.Weights.Value()
	for i, p := range s.Parts {
		y := p.Forward(x)
		if s.classes > 0 {
			y = y.Softmax(1)
		}
		y = y.Mul(w.Shrink([][2]int{{i, i + 1}}))
		if out == nil {
			out = y
		} else {
			out = out.Add(y)
		}
	}
	if s.classes > 0 {
		out = out.Maximum(tensor.Scalar(float32(math.SmallestNonzeroFloat32))).Log()
	}
	return out
}
