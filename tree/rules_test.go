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

package tree

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// regions returns n rows of a region code 0 to 11 and a uniform feature,
// whose class is 1 for regions 1, 4, 7 and 10, which no one threshold on
// the code separates. oneHot lays the region out as 12 one-hot columns
// before the feature instead.
func regions(n int, seed uint64, oneHot bool) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 2))
	f := 2
	if oneHot {
		f = 13
	}
	d := tabular.Data{X: make([]float32, n*f), Features: f, Y: make([]float32, n)}
	for i := range n {
		c := r.IntN(12)
		row := d.Row(i)
		if oneHot {
			row[c] = 1
		} else {
			row[0] = float32(c)
		}
		row[f-1] = 2*r.Float32() - 1
		if c%3 == 1 {
			d.Y[i] = 1
		}
	}
	return d
}

func TestCategoriesSplitOnASet(t *testing.T) {
	ctx := context.Background()
	for _, oneHot := range []bool{false, true} {
		d := regions(1200, 3, oneHot)
		cat := Category{Feature: 0}
		if oneHot {
			cat.Width = 12
		}
		m, err := DecisionTree(ctx, d, tabular.Classification, MaxDepth(1), Categorical(cat))
		if err != nil {
			t.Fatal(err)
		}
		if a := accuracy(m, d); a != 1 {
			t.Errorf("one-hot %v: a one-split tree on the region scores %v, want 1", oneHot, a)
		}
		if m.Trees[0].Cats == nil || m.Trees[0].Cats[0] == nil {
			t.Errorf("one-hot %v: the root does not split on a set", oneHot)
		}
		// Without the category rule one split cannot do it.
		plain, err := DecisionTree(ctx, d, tabular.Classification, MaxDepth(1))
		if err != nil {
			t.Fatal(err)
		}
		if a := accuracy(plain, d); a > 0.9 {
			t.Errorf("one-hot %v: a one-split tree on the code scores %v without the rule", oneHot, a)
		}
	}
}

func TestCategoriesInEveryModel(t *testing.T) {
	ctx := context.Background()
	d := regions(1500, 5, false)
	three := regions(1500, 5, false)
	for i, row := 0, three.X; i < three.Rows(); i++ {
		three.Y[i] = float32(int(row[i*2]) % 3)
	}
	for name, fit := range fitters {
		for _, tc := range []struct {
			d    tabular.Data
			task tabular.Task
		}{{d, tabular.Classification}, {three, tabular.Classification}} {
			m, err := fit(ctx, tc.d, tc.task, Trees(20), Categorical(Category{Feature: 0}))
			if err != nil {
				t.Fatal(err)
			}
			if a := accuracy(m, tc.d); a < 0.98 {
				t.Errorf("%s: accuracy %v with the region as a category", name, a)
			}
		}
	}
}

func TestCategoryModuleAnswersAsPredict(t *testing.T) {
	ctx := context.Background()
	for _, oneHot := range []bool{false, true} {
		d := regions(800, 7, oneHot)
		cat := Category{Feature: 0}
		if oneHot {
			cat.Width = 12
		}
		m, err := GradientBoosting(ctx, d, tabular.Classification, Trees(10), Categorical(cat))
		if err != nil {
			t.Fatal(err)
		}
		// Ask about missing, unknown and odd codes too.
		q := tabular.Data{X: append([]float32(nil), d.X...), Features: d.Features}
		if oneHot {
			clear(q.Row(0)[:12])
			q.Row(1)[3], q.Row(1)[5] = 1, 1
		} else {
			q.Row(0)[0] = float32(math.NaN())
			q.Row(1)[0] = 40
			q.Row(2)[0] = -3
			q.Row(3)[0] = 2.5
			q.Row(4)[0] = 1000
		}
		want := m.Predict(q.X)
		var buf bytes.Buffer
		if err := mim.Save(ctx, &buf, m.Module(), nil); err != nil {
			t.Fatal(err)
		}
		f, err := mim.Parse(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		mod := NewModule(m.Shape())
		if err := f.Load(mod); err != nil {
			t.Fatal(err)
		}
		f.Close()
		got, err := mod.Forward(tensor.FromSlice(q.X, q.Rows(), q.Features)).Softmax(1).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-4 {
				t.Fatalf("one-hot %v: answer %d is %v, want %v", oneHot, i, got[i], want[i])
			}
		}
		js, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		var back Model
		if err := json.Unmarshal(js, &back); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(back.Predict(q.X), want) {
			t.Errorf("one-hot %v: a model read back from JSON answers otherwise", oneHot)
		}
	}
}

// monotoneAlong reports whether m's answers, or its class-1
// probabilities, move only the way rule says as feature j sweeps from -1
// to 1, for each of rows other rows.
func monotoneAlong(m *Model, j, rule, rows int, seed uint64) bool {
	r := rand.New(rand.NewPCG(seed, 3))
	for range rows {
		base := make([]float32, m.Features)
		for i := range base {
			base[i] = 2*r.Float32() - 1
		}
		prev := math.Inf(-rule)
		for s := range 41 {
			row := append([]float32(nil), base...)
			row[j] = -1 + float32(s)/20
			p := m.Predict(row)
			v := float64(p[len(p)-1])
			if rule > 0 && v < prev-1e-6 || rule < 0 && v > prev+1e-6 {
				return false
			}
			prev = v
		}
	}
	return true
}

func TestMonotoneRulesHold(t *testing.T) {
	ctx := context.Background()
	// The answer rises with x0 overall, with wiggles and noise a free
	// model follows.
	r := rand.New(rand.NewPCG(9, 9))
	reg := table(2000, 3, 11, func(row []float32) float32 {
		return 2*row[0] + 0.6*float32(math.Sin(12*float64(row[0]))) + row[1] + 0.3*float32(r.NormFloat64())
	})
	cls := table(2000, 3, 12, func(row []float32) float32 {
		if row[0]+0.4*float32(math.Sin(10*float64(row[0])))+0.5*float32(r.NormFloat64()) > row[1] {
			return 1
		}
		return 0
	})
	for name, fit := range fitters {
		for _, tc := range []struct {
			d    tabular.Data
			task tabular.Task
		}{{reg, tabular.Regression}, {cls, tabular.Classification}} {
			free, err := fit(ctx, tc.d, tc.task, Trees(30))
			if err != nil {
				t.Fatal(err)
			}
			if name != "boosting" && tc.task == tabular.Classification {
				// A forest's two classes are two outputs: no rule.
				continue
			}
			ruled, err := fit(ctx, tc.d, tc.task, Trees(30), Monotone(map[int]int{0: 1}))
			if err != nil {
				t.Fatal(err)
			}
			if monotoneAlong(free, 0, 1, 20, 1) {
				t.Logf("%s %v: the free model happens to rise with x0", name, tc.task)
			}
			if !monotoneAlong(ruled, 0, 1, 50, 2) {
				t.Errorf("%s %v: the ruled model falls somewhere as x0 rises", name, tc.task)
			}
			down, err := fit(ctx, tc.d, tc.task, Trees(30), Monotone(map[int]int{0: -1}))
			if err != nil {
				t.Fatal(err)
			}
			if !monotoneAlong(down, 0, -1, 50, 3) {
				t.Errorf("%s %v: the falling rule rises somewhere", name, tc.task)
			}
		}
	}
	if m, err := GradientBoosting(ctx, reg, tabular.Regression, Monotone(map[int]int{0: 1})); err != nil {
		t.Fatal(err)
	} else if fit := r2(m, reg); fit < 0.8 {
		t.Errorf("a rising rule that holds in the data costs too much: R² %v", fit)
	}
}

// counts returns n rows of 3 features and a Poisson count of mean
// exp(0.3 + 1.2 x0 - 0.8 x1), mostly small, often 0.
func counts(n int, seed uint64) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 4))
	return table(n, 3, seed, func(row []float32) float32 {
		return float32(poisson(r, math.Exp(0.3+1.2*float64(row[0])-0.8*float64(row[1]))))
	})
}

func poisson(r *rand.Rand, mean float64) int {
	l, k, p := math.Exp(-mean), 0, 1.0
	for {
		p *= r.Float64()
		if p <= l {
			return k
		}
		k++
	}
}

func TestPoissonAndTweedieFitCounts(t *testing.T) {
	ctx := context.Background()
	d, test := counts(3000, 13), counts(1000, 14)
	// Tweedie answers: a count of claims, each of a gamma-ish size.
	r := rand.New(rand.NewPCG(15, 5))
	claims := counts(3000, 16)
	for i, c := range claims.Y {
		var amt float64
		for range int(c) {
			amt += 100 * r.ExpFloat64()
		}
		claims.Y[i] = float32(amt)
	}
	for _, tc := range []struct {
		name string
		d    tabular.Data
		loss Objective
	}{{"poisson", d, Poisson()}, {"tweedie", claims, Tweedie(1.5)}} {
		m, err := GradientBoosting(ctx, tc.d, tabular.Regression, Trees(150), MaxDepth(3), Loss(tc.loss))
		if err != nil {
			t.Fatal(err)
		}
		if m.Link != Exp || m.Loss != tc.loss.String() {
			t.Errorf("%s: link %q, loss %q", tc.name, m.Link, m.Loss)
		}
		p := m.Predict(tc.d.X)
		var sumP, sumY float64
		for i, v := range p {
			if !(v > 0) {
				t.Fatalf("%s: answer %d is %v, want above 0", tc.name, i, v)
			}
			sumP += float64(v)
			sumY += float64(tc.d.Y[i])
		}
		if math.Abs(sumP/sumY-1) > 0.05 {
			t.Errorf("%s: answers total %v, the answers %v", tc.name, sumP, sumY)
		}
		var buf bytes.Buffer
		if err := mim.Save(ctx, &buf, m.Module(), nil); err != nil {
			t.Fatal(err)
		}
		f, err := mim.Parse(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		mod := NewModule(m.Shape())
		if err := f.Load(mod); err != nil {
			t.Fatal(err)
		}
		f.Close()
		got, err := mod.Forward(tensor.FromSlice(tc.d.X, tc.d.Rows(), 3)).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i := range p {
			if math.Abs(float64(got[i]-p[i])) > 1e-4*math.Max(1, float64(p[i])) {
				t.Fatalf("%s: module answer %d is %v, want %v", tc.name, i, got[i], p[i])
			}
		}
	}
	// On counts, the Poisson model beats squared error at the Poisson
	// deviance on rows it didn't learn from.
	pm, _ := GradientBoosting(ctx, d, tabular.Regression, Trees(150), MaxDepth(3), Loss(Poisson()))
	sm, _ := GradientBoosting(ctx, d, tabular.Regression, Trees(150), MaxDepth(3))
	if dp, ds := deviance(pm.Predict(test.X), test.Y), deviance(sm.Predict(test.X), test.Y); dp >= ds {
		t.Errorf("Poisson deviance %v, squared error's %v", dp, ds)
	}
}

// deviance returns the mean Poisson deviance of answers p, clamped above
// 0, for counts y.
func deviance(p, y []float32) float64 {
	var s float64
	for i, v := range p {
		mu, c := math.Max(float64(v), 1e-3), float64(y[i])
		d := mu - c
		if c > 0 {
			d += c * math.Log(c/mu)
		}
		s += 2 * d
	}
	return s / float64(len(p))
}

func TestQuantilesCoverTheirShare(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewPCG(17, 6))
	gen := func(n int, seed uint64) tabular.Data {
		return table(n, 2, seed, func(row []float32) float32 {
			// The spread grows with x1, so a fixed band would be wrong.
			return 50 + 20*row[0] + (5+5*(row[1]+1))*float32(r.NormFloat64())
		})
	}
	d, test := gen(4000, 18), gen(2000, 19)
	for _, alpha := range []float64{0.1, 0.5, 0.9} {
		m, err := GradientBoosting(ctx, d, tabular.Regression, Trees(200), MaxDepth(3), MinLeaf(20), Loss(Quantile(alpha)))
		if err != nil {
			t.Fatal(err)
		}
		p := m.Predict(test.X)
		below := 0
		for i, v := range p {
			if test.Y[i] <= v {
				below++
			}
		}
		if got := float64(below) / float64(len(p)); math.Abs(got-alpha) > 0.05 {
			t.Errorf("quantile %v: %v of new answers fall below it", alpha, got)
		}
	}
	// Under a rising rule, quantiles keep it.
	m, err := GradientBoosting(ctx, d, tabular.Regression, Trees(100), Loss(Quantile(0.9)), Monotone(map[int]int{0: 1}))
	if err != nil {
		t.Fatal(err)
	}
	if !monotoneAlong(m, 0, 1, 30, 4) {
		t.Error("a ruled quantile model falls somewhere as x0 rises")
	}
}

func TestBadRulesAndLosses(t *testing.T) {
	ctx := context.Background()
	d := table(100, 3, 21, wave)
	cls := table(100, 3, 21, xor)
	neg := table(100, 3, 21, func(row []float32) float32 { return row[0] })
	codesBad := regions(100, 22, false)
	codesBad.X[0] = 1.5
	for name, run := range map[string]func() error{
		"loss for classes": func() error {
			_, err := GradientBoosting(ctx, cls, tabular.Classification, Loss(Poisson()))
			return err
		},
		"negative counts": func() error {
			_, err := GradientBoosting(ctx, neg, tabular.Regression, Loss(Tweedie(1.5)))
			return err
		},
		"tweedie power": func() error {
			_, err := GradientBoosting(ctx, d, tabular.Regression, Loss(Tweedie(2.5)))
			return err
		},
		"quantile 1": func() error {
			_, err := GradientBoosting(ctx, d, tabular.Regression, Loss(Quantile(1)))
			return err
		},
		"loss for a forest": func() error {
			_, err := RandomForest(ctx, d, tabular.Regression, Loss(Quantile(0.5)))
			return err
		},
		"rule of 2": func() error {
			_, err := GradientBoosting(ctx, d, tabular.Regression, Monotone(map[int]int{0: 2}))
			return err
		},
		"rule on a missing feature": func() error {
			_, err := GradientBoosting(ctx, d, tabular.Regression, Monotone(map[int]int{5: 1}))
			return err
		},
		"rule on a category": func() error {
			_, err := GradientBoosting(ctx, d, tabular.Regression, Categorical(Category{Feature: 0, Width: 2}), Monotone(map[int]int{1: 1}))
			return err
		},
		"rule for three classes": func() error {
			_, err := GradientBoosting(ctx, table(100, 3, 21, quadrant), tabular.Classification, Monotone(map[int]int{0: 1}))
			return err
		},
		"category past the row": func() error {
			_, err := GradientBoosting(ctx, d, tabular.Regression, Categorical(Category{Feature: 2, Width: 2}))
			return err
		},
		"overlapping categories": func() error {
			_, err := GradientBoosting(ctx, d, tabular.Regression, Categorical(Category{Feature: 0, Width: 2}, Category{Feature: 1}))
			return err
		},
		"half a code": func() error {
			_, err := GradientBoosting(ctx, codesBad, tabular.Classification, Categorical(Category{Feature: 0}))
			return err
		},
	} {
		if err := run(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
}

// TestCategoryModuleAfterTheLastSet checks a Module whose last trees
// split on no category, so their nodes come after every set.
func TestCategoryModuleAfterTheLastSet(t *testing.T) {
	ctx := context.Background()
	m := &Model{Task: tabular.Regression, Features: 2, Outputs: 1, Link: Identity, Scale: 1, Base: []float32{0},
		Categories: []Category{{Feature: 0}},
		Trees: []Tree{
			{Feature: []int32{0, -1, -1}, Threshold: []float32{0, 0, 0}, MissingLeft: []bool{false, false, false},
				Left: []int32{1, -1, -1}, Right: []int32{2, -1, -1}, Value: []float32{0, 1, 2},
				Cats: [][]uint64{{0b1010}, nil, nil}},
			{Feature: []int32{1, -1, -1}, Threshold: []float32{0.5, 0, 0}, MissingLeft: []bool{true, false, false},
				Left: []int32{1, -1, -1}, Right: []int32{2, -1, -1}, Value: []float32{0, 10, 20}},
		}}
	x := []float32{1, 0, 2, 1, 3, 0.2, 0, 0.9, float32(math.NaN()), 0.1}
	want := m.Predict(x)
	got, err := m.Module().Forward(tensor.FromSlice(x, 5, 2)).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("module answers %v, want %v", got, want)
	}
}
