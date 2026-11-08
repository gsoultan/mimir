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

//go:build linux

package cpu

import "syscall"

// madvFree is MADV_FREE, which the syscall package does not name on
// every architecture.
const madvFree = 8

// lazyFree lets the kernel take the pages of mem, a mapping that no
// buffer uses, when it runs short of memory, without unmapping it. A page
// it takes reads as zero afterwards; one it leaves keeps its contents and
// costs no fault when it is written again. A kernel older than Linux 4.5
// does not know MADV_FREE, and the pages stay resident.
func lazyFree(mem []byte) { _ = syscall.Madvise(mem, madvFree) }
