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

//go:build !unix

package cpu

import "errors"

// Windows, Plan 9 and WebAssembly have no syscall.Mmap, so every buffer
// lives on the Go heap and Free leaves it to the garbage collector.
const offHeap = false

func mapAnon(int) ([]byte, func() error, error) {
	return nil, nil, errors.ErrUnsupported
}
