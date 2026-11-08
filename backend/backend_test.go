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

package backend

import "testing"

func TestDTypes(t *testing.T) {
	cases := []struct {
		dt    DType
		size  int
		float bool
		name  string
	}{
		{Float32, 4, true, "float32"},
		{Int32, 4, false, "int32"},
		{BFloat16, 2, true, "bfloat16"},
		{Float16, 2, true, "float16"},
		{Invalid, 0, false, "DType(0)"},
	}
	for _, c := range cases {
		if c.dt.Size() != c.size || c.dt.IsFloat() != c.float || c.dt.String() != c.name {
			t.Errorf("%s: Size %d, IsFloat %v, want %d, %v, %q", c.dt, c.dt.Size(), c.dt.IsFloat(), c.size, c.float, c.name)
		}
	}
}
