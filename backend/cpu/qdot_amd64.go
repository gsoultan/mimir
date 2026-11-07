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

//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./qamd64 -out ../../backend/cpu/qdot_amd64.s -stubs ../../backend/cpu/qdot_stub_amd64.go -pkg cpu"

import "github.com/gsoultan/mimir/backend"

func init() {
	f := x86Features()
	if !f.avx2 || !f.f16c {
		return
	}
	// The K-quant kernels that prefetch their weights are faster on Intel
	// cores and slower on AMD's; see prefetchAhead in tools/asmgen/qamd64.
	q8 := [maxRows]qdot{qdotQ8_0AVX2x1, qdotQ8_0AVX2x2, qdotQ8_0AVX2x3, qdotQ8_0AVX2x4}
	avx2 := map[backend.DType]*[maxRows]qdot{
		backend.Q8_0: &q8,
		backend.Q4_K: {qdotQ4KAVX2x1, qdotQ4KAVX2x2, qdotQ4KAVX2x3, qdotQ4KAVX2x4},
		backend.Q6_K: {qdotQ6KAVX2x1, qdotQ6KAVX2x2, qdotQ6KAVX2x3, qdotQ6KAVX2x4},
	}
	prefetch := map[backend.DType]*[maxRows]qdot{
		backend.Q8_0: &q8,
		backend.Q4_K: {qdotQ4KAVX2Px1, qdotQ4KAVX2Px2, qdotQ4KAVX2Px3, qdotQ4KAVX2Px4},
		backend.Q6_K: {qdotQ6KAVX2Px1, qdotQ6KAVX2Px2, qdotQ6KAVX2Px3, qdotQ6KAVX2Px4},
	}
	qdotSets["avx2"] = avx2
	qdotSets["avx2-prefetch"] = prefetch
	qdotISA, qdots = "avx2", avx2
	if f.intel {
		qdotISA, qdots = "avx2-prefetch", prefetch
	}
	tiles := map[backend.DType]qtile{backend.Q4_K: qtileQ4KAVX2, backend.Q6_K: qtileQ6KAVX2}
	qtileSets["avx2"] = tiles
	qtiles = tiles
	qtileUnpacks[backend.Q4_K] = func(dst, w []byte, nb int) {
		if nb > 0 {
			_, _ = dst[(nb-1)*qtileBlockBytes+qtileRowBytes-1], w[nb*144-1]
			unpackQ4KAVX2(nb, &dst[0], &w[0])
		}
	}
	qtileUnpacks[backend.Q6_K] = func(dst, w []byte, nb int) {
		if nb > 0 {
			_, _ = dst[(nb-1)*qtileBlockBytes+qtileRowBytes-1], w[nb*210-1]
			unpackQ6KAVX2(nb, &dst[0], &w[0])
		}
	}
	if f.avx512vnni {
		tiles := map[backend.DType]qtile{backend.Q4_K: qtileQ4KAVX512, backend.Q6_K: qtileQ6KAVX512}
		qtileSets["avx512vnni"] = tiles
		qtiles = tiles
	}
}
