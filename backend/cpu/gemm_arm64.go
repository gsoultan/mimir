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

//go:generate sh -c "cd ../../tools/asmgen && go run ./arm64 -out ../../backend/cpu/gemm_arm64.s"

// Every arm64 CPU has NEON. The kernel's B panel of kc×nr stays well
// inside the 64 KB L1 cache of current cores, and nc is a multiple of nr.
var neonKernel = gemmKernel{name: "neon", mr: 8, nr: 12, kc: 256, mc: 128, nc: 1536, tile: tileNEON}

func init() {
	gemmKernels = append([]*gemmKernel{&neonKernel}, gemmKernels...)
}

// tileNEON computes one 8×12 tile; see gemmKernel.tile.
//
//go:noescape
func tileNEON(kc int, a, b, c *float32, ldc int, accumulate bool)
