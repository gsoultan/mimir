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

// Pooler is a Backend that runs the pooling kernels OpMaxPool and
// OpMaxPoolGrad. Package tensor pools on such a backend with one kernel
// each way instead of a reduction over the windows that OpUnfold copies
// out, or a view of them, and the gradient of that reduction, which read
// the windows several times over.
type Pooler interface {
	Backend
	// Pools reports whether the backend runs the pooling kernels.
	Pools() bool
}
