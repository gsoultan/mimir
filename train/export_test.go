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

package train

import "time"

// SetPhases makes Parallel.Backward report the host's time in each of its
// phases to f until restore is called.
func SetPhases(f func(phase string, rank int, d time.Duration)) (restore func()) {
	was := phases
	phases = f
	return func() { phases = was }
}

// SetWaitPhases makes Parallel.Backward wait for the devices after each
// phase and report the waits to the function SetPhases set, until restore
// is called.
func SetWaitPhases(on bool) (restore func()) {
	was := waitPhases
	waitPhases = on
	return func() { waitPhases = was }
}

// AllReduced reports whether p's last Backward reduced the gradients into
// every replica's.
func AllReduced(p *Parallel) bool { return p.allReduced }
