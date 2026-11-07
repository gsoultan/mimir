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

package vision_test

import (
	"context"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/vision"
)

func TestResNetWidth(t *testing.T) {
	r := vision.ResNet18(3, vision.Width(0.25))
	if r.Width() != 128 {
		t.Errorf("width %d, want 128", r.Width())
	}
	full, quarter := 0, 0
	for _, p := range nn.Parameters(vision.ResNet18(3)) {
		full += p.Value().Size()
	}
	for _, p := range nn.Parameters(r) {
		quarter += p.Value().Size()
	}
	if quarter*10 > full {
		t.Errorf("%d parameters at a quarter width, %d at full", quarter, full)
	}
	nn.Eval(r)
	out, err := r.Forward(tensor.Randn(1, 2, 3, 64, 64)).Data[float32](context.Background())
	if err != nil || len(out) != 6 {
		t.Errorf("%d outputs, %v", len(out), err)
	}
}
