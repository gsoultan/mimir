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

package data

import "testing"

func TestChunkRows(t *testing.T) {
	for _, tc := range []struct{ features, want int }{
		{1, 8192},
		{32, 8192},
		{127, 8192},
		{128, 8128},
		{784, 1335},
		{1024, 1023},
		{3072, 341},
		{4096, 255}, // below the batch of TestStreamBatchLargerThanChunk
		{1 << 20, 1},
		{1 << 24, 1},
	} {
		got := chunkRows(tc.features)
		if got != tc.want {
			t.Errorf("chunkRows(%d) = %d, want %d", tc.features, got, tc.want)
		}
		if size := 4 * (tc.features + 1) * got; size > chunkBytes && got > 1 {
			t.Errorf("chunkRows(%d) makes chunks of %d bytes, over %d", tc.features, size, chunkBytes)
		}
	}
}
