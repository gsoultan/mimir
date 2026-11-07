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

package metal

import "github.com/gsoultan/mimir/backend/internal/gen"

// packed reports whether each operand's matrices lie one after another,
// rows × ld elements apart, or there is one matrix. Metal Performance
// Shaders on an M5 under macOS 26 reads batched matrices that far apart
// whatever the descriptor's matrix stride says, so a product of batches
// spaced further apart runs elsewhere.
func packed(g gen.GEMM) bool {
	return g.Batch == 1 || g.A.Stride == g.A.Rows*g.A.LD && g.B.Stride == g.B.Rows*g.B.LD
}
