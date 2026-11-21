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

package automl_test

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/automl"
	"github.com/gsoultan/mimir/linear"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/tree"
)

// mixed returns rows whose answer is a straight line of many features,
// which trees draw in steps, plus a bend a line misses: a blend of a line
// and trees beats either.
func mixed(n int, seed uint64) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 2))
	d := tabular.Data{Features: 6}
	for range n {
		var y float64
		for j := range 6 {
			v := r.Float64()*2 - 1
			d.X = append(d.X, float32(v))
			y += float64(j+1) * v
		}
		y += 3*math.Sin(4*float64(d.X[len(d.X)-6])) + 0.5*r.NormFloat64()
		d.Y = append(d.Y, float32(y))
	}
	return d
}

func TestStackBlendsWhenItHelps(t *testing.T) {
	ctx := context.Background()
	d := mixed(500, 3)
	fams := automl.Families(automl.Boost, automl.Linear, automl.Neighbors)
	r, err := automl.Search(ctx, d, tabular.Regression, automl.Trials(5), fams, automl.Stack(3))
	if err != nil {
		t.Fatal(err)
	}
	b, ok := r.Best.(*automl.Blend)
	if !ok {
		t.Fatalf("best is a %T scoring %v; the trials: %v", r.Best, r.Winner.Score, r.Trials)
	}
	if r.Winner.Family != automl.Stacked || !(r.Winner.Score < r.Trials[0].Score) {
		t.Errorf("winner %v scores %v, the best alone %v", r.Winner.Family, r.Winner.Score, r.Trials[0].Score)
	}
	var sum float64
	for _, w := range b.Weights {
		if !(w > 0) {
			t.Errorf("weight %v", w)
		}
		sum += w
	}
	if math.Abs(sum-1) > 1e-9 || len(b.Models) < 2 || len(b.Models) != len(b.Members) {
		t.Errorf("%d models, %d members, weights %v", len(b.Models), len(b.Members), b.Weights)
	}
	// Better on new rows than the best alone.
	test := mixed(1000, 4)
	single, err := automl.Search(ctx, d, tabular.Regression, automl.Trials(5), fams)
	if err != nil {
		t.Fatal(err)
	}
	rmse := func(m tabular.Model) float64 {
		var s float64
		for i, v := range m.Predict(test.X) {
			e := float64(v - test.Y[i])
			s += e * e
		}
		return math.Sqrt(s / float64(len(test.Y)))
	}
	if bs, ss := rmse(b), rmse(single.Best); bs > ss {
		t.Errorf("the blend's RMSE on new rows %v, the best alone %v", bs, ss)
	}
	// Without Stack, no blend.
	if _, ok := single.Best.(*automl.Blend); ok {
		t.Error("a blend without Stack")
	}
}

func TestStackModuleAnswersAsBlend(t *testing.T) {
	ctx := context.Background()
	for _, task := range []tabular.Task{tabular.Regression, tabular.Classification} {
		d := mixed(300, 5)
		classes := 0
		if task == tabular.Classification {
			classes = 2
			for i, y := range d.Y {
				d.Y[i] = float32(map[bool]int{false: 0, true: 1}[y > 0])
			}
		}
		var opts []tree.Option
		if classes > 0 {
			opts = append(opts, tree.Classes(classes))
		}
		tm, err := tree.GradientBoosting(ctx, d, task, append(opts, tree.Trees(20))...)
		if err != nil {
			t.Fatal(err)
		}
		var lm *linear.Model
		if task == tabular.Regression {
			lm, err = linear.Regression(ctx, d)
		} else {
			lm, err = linear.Logistic(ctx, d, linear.Classes(2))
		}
		if err != nil {
			t.Fatal(err)
		}
		b := &automl.Blend{Models: []tabular.Model{tm, lm}, Weights: []float64{0.7, 0.3}}
		want := b.Predict(d.X)
		mod, err := automl.StackOf([]nn.Module{tm.Module(), lm.Module()}, b.Weights, classes)
		if err != nil {
			t.Fatal(err)
		}
		out := mod.Forward(tensor.FromSlice(d.X, d.Rows(), d.Features))
		if classes > 0 {
			out = out.Softmax(1)
		}
		got, err := out.Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-4*math.Max(1, math.Abs(float64(want[i]))) {
				t.Fatalf("%v: answer %d is %v, want %v", task, i, got[i], want[i])
			}
		}
	}
}
