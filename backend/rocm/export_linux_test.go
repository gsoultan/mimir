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

//go:build linux && (amd64 || arm64) && !norocm

package rocm

// SetGraphs turns HIP graphs on or off for the tests of other packages
// until restore is called.
func SetGraphs(on bool) (restore func()) {
	was := graphsOn
	graphsOn = on
	return func() { graphsOn = was }
}

// GraphLaunches returns the number of batches launched as HIP graphs so
// far, the number launched one call at a time, and the graphs captured.
func GraphLaunches() (graphs, direct, captures int64) {
	return graphCounts.launches.Load(), graphCounts.direct.Load(), graphCounts.captures.Load()
}
