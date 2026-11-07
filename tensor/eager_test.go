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

package tensor

import (
	"errors"
	"testing"
)

func TestEagerModeComputesAtTheCall(t *testing.T) {
	defer func(old bool) { eager = old }(eager)
	eager = true

	if y := FromSlice([]float32{1, 2}).Exp(); !y.node.Realized() {
		t.Error("eager operation was not computed")
	}
	g := FromSlice([]float32{1, 2, 3}).Gather(0, FromSlice([]int32{7}))
	if !errors.Is(g.Err(), ErrIndex) {
		t.Fatalf("Err() = %v, want ErrIndex at the gather", g.Err())
	}
	if !errors.Is(g.Add(g).Err(), ErrIndex) {
		t.Error("error did not propagate")
	}
}
