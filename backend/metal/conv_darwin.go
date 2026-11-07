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

//go:build darwin && arm64

package metal

import (
	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// The backend runs convolutions as the generator's implicit-GEMM kernels,
// which read the windows of their image where they lie instead of
// copying them out first.
var _ backend.Convolver = (*Backend)(nil)

// ConvParts returns the number of parts into which the weight gradient k
// splits so that the device sums them in parallel.
func (*Backend) ConvParts(k *backend.Kernel) int { return gen.ConvParts(k) }
