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
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// poisson draws a count with mean mu.
func poisson(r *rand.Rand, mu float64) float32 {
	l, k, p := math.Exp(-mu), 0, 1.0
	for {
		p *= r.Float64()
		if p <= l {
			return float32(k)
		}
		k++
	}
}

// counts returns n rows of 3 features whose answer is a count with mean
// exp(0.5 + x0 - x1), and that mean.
func counts(n int, seed uint64) (tabular.Data, []float64) {
	r := rand.New(rand.NewPCG(seed, 2))
	mu := make([]float64, n)
	d := table(n, 3, seed, func(row []float32) float32 { return 0 })
	for i := range n {
		mu[i] = math.Exp(0.5 + float64(d.X[i*3]) - float64(d.X[i*3+1]))
		d.Y[i] = poisson(r, mu[i])
	}
	return d, mu
}

func TestPoissonLearnsTheRate(t *testing.T) {
	ctx := context.Background()
	train, _ := counts(4000, 30)
	test, mu := counts(1000, 31)
	m, err := GradientBoosting(ctx, train, tabular.Regression, Objective(Poisson()), Trees(150), MaxDepth(3))
	if err != nil {
		t.Fatal(err)
	}
	if m.Link != Exp {
		t.Fatalf("link %s", m.Link)
	}
	p := m.Predict(test.X)
	var off, mean float64
	for i, v := range p {
		if v <= 0 {
			t.Fatalf("answer %d is %v, want above zero", i, v)
		}
		off += math.Abs(float64(v) - mu[i])
		mean += mu[i]
	}
	if off/mean > 0.15 {
		t.Errorf("off the true rate by %.3f of it", off/mean)
	}
}

func TestTweedieFitsAmountsWithZeros(t *testing.T) {
	ctx := context.Background()
	var mus []float64
	amounts := func(n int, seed uint64) tabular.Data {
		r := rand.New(rand.NewPCG(seed, 3))
		d := table(n, 3, seed, func(row []float32) float32 { return 0 })
		mus = mus[:0]
		for i := range n {
			mu := math.Exp(1 + 1.5*float64(d.X[i*3]))
			mus = append(mus, mu)
			// A few claims of about 20 each: zero most of the time.
			for range int(poisson(r, mu/20)) {
				d.Y[i] += float32(20 * r.ExpFloat64())
			}
		}
		return d
	}
	train, test := amounts(6000, 32), amounts(2000, 33)
	m, err := GradientBoosting(ctx, train, tabular.Regression, Objective(Tweedie(1.5)), Trees(150), MaxDepth(3))
	if err != nil {
		t.Fatal(err)
	}
	p := m.Predict(test.X)
	var got, want, zeros float64
	for i, v := range p {
		if v <= 0 {
			t.Fatalf("answer %d is %v, want above zero", i, v)
		}
		got += float64(v)
		want += mus[i]
		if test.Y[i] == 0 {
			zeros++
		}
	}
	if zeros < 0.4*float64(len(p)) {
		t.Fatalf("only %v zeros", zeros)
	}
	if math.Abs(got-want)/want > 0.15 {
		t.Errorf("answers sum to %.0f, the true means to %.0f", got, want)
	}
	// It must rank big spenders above small ones better than chance.
	if r := r2(m, test); r < 0.05 {
		t.Errorf("R² %.3f", r)
	}
}

func TestQuantileCoversItsShare(t *testing.T) {
	ctx := context.Background()
	noisy := func(n int, seed uint64) tabular.Data {
		r := rand.New(rand.NewPCG(seed, 4))
		return table(n, 2, seed, func(row []float32) float32 {
			return 3*row[0] + float32(r.NormFloat64())*(1+row[1])
		})
	}
	train, test := noisy(5000, 34), noisy(2000, 35)
	for _, q := range []float64{0.1, 0.5, 0.9} {
		m, err := GradientBoosting(ctx, train, tabular.Regression, Objective(Quantile(q)), Trees(100), MaxDepth(3))
		if err != nil {
			t.Fatal(err)
		}
		below := 0
		for i, v := range m.Predict(test.X) {
			if test.Y[i] <= v {
				below++
			}
		}
		if share := float64(below) / float64(len(test.Y)); math.Abs(share-q) > 0.04 {
			t.Errorf("quantile %.1f: %.3f of answers at most it", q, share)
		}
	}
}

func TestLossBadInput(t *testing.T) {
	ctx := context.Background()
	d := table(100, 2, 36, wave) // has answers below zero
	for name, l := range map[string]Loss{
		"poisson below zero": Poisson(),
		"tweedie below zero": Tweedie(1.5),
		"tweedie power 2":    Tweedie(2),
		"quantile 0":         Quantile(0),
		"quantile 1.5":       Quantile(1.5),
	} {
		if _, err := GradientBoosting(ctx, d, tabular.Regression, Objective(l)); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v", name, err)
		}
	}
	c := table(100, 2, 37, xor)
	if _, err := GradientBoosting(ctx, c, tabular.Classification, Objective(Poisson())); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("poisson classification: %v", err)
	}
}

func TestExpModuleAnswersAsPredict(t *testing.T) {
	ctx := context.Background()
	d, _ := counts(600, 38)
	m, err := GradientBoosting(ctx, d, tabular.Regression, Objective(Poisson()), Trees(15),
		Validation(d, 5))
	if err != nil {
		t.Fatal(err)
	}
	want := m.Predict(d.X)
	var buf bytes.Buffer
	if err := mim.Save(ctx, &buf, m.Module(), nil); err != nil {
		t.Fatal(err)
	}
	f, err := mim.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	mod := NewModule(m.Shape())
	if err := f.Load(mod); err != nil {
		t.Fatal(err)
	}
	got, err := mod.Forward(tensor.FromSlice(d.X, d.Rows(), 3)).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4*math.Max(1, float64(want[i])) {
			t.Fatalf("answer %d is %v, want %v", i, got[i], want[i])
		}
	}
}
