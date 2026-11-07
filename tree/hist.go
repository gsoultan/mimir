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

// addRowsGo is addRows in Go: it adds the sz numbers gh[r*sz:] of each
// row r of rows into hist at the row's bin of each feature of feats, whose
// bins start at offs.
func addRowsGo(hist []float64, b *binned, feats, offs []int, rows []int32, gh []float32, sz int) {
	f := b.f
	v := make([]float64, sz)
	for _, r := range rows {
		row := b.bins[int(r)*f : int(r)*f+f]
		for o, x := range gh[int(r)*sz : int(r)*sz+sz] {
			v[o] = float64(x)
		}
		for fi, j := range feats {
			e := (offs[fi] + int(row[j])) * sz
			c := hist[e : e+sz : e+sz]
			for o, x := range v {
				c[o] += x
			}
		}
	}
}
