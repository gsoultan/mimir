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
	"math"
	"testing"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// TestModuleAnswersAsPredict checks a Module's softmax, or its values,
// against Predict for every kind of model, missing values included, and
// after a save to a Mimir Model file and a load into NewModule.
func TestModuleAnswersAsPredict(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		name string
		task tabular.Task
		y    func([]float32) float32
	}{
		{"two classes", tabular.Classification, xor},
		{"three classes", tabular.Classification, quadrant},
		{"regression", tabular.Regression, wave},
	} {
		d := table(600, 3, 20, tc.y)
		for i := 0; i < len(d.X); i += 7 {
			d.X[i] = float32(math.NaN())
		}
		for name, fit := range fitters {
			m, err := fit(ctx, d, tc.task, Trees(15))
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
			mod := NewModule(m.Shape())
			if err := f.Load(mod); err != nil {
				t.Fatal(err)
			}
			f.Close()
			out := mod.Forward(tensor.FromSlice(d.X, d.Rows(), 3))
			if tc.task == tabular.Classification {
				out = out.Softmax(1)
			}
			got, err := out.Data[float32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("%s, %s: %d answers, want %d", tc.name, name, len(got), len(want))
			}
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 1e-4*math.Max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("%s, %s: answer %d is %v, want %v", tc.name, name, i, got[i], want[i])
				}
			}
		}
	}
}
