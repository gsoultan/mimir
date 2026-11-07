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

package cpu

// NewWorkers returns a CPU backend that uses up to workers goroutines per
// kernel.
func NewWorkers(workers int) *Backend {
	b := New()
	b.workers = workers
	return b
}

// SetDirectPositions sets the output positions below which a forward
// convolution of many channels runs as a direct one, and returns a
// function that restores the setting.
func SetDirectPositions(n int) (restore func()) {
	old := directPositions
	directPositions = n
	return func() { directPositions = old }
}
