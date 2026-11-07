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

package sched

import (
	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// Bind realizes n, a node that no schedule of this package computes, as
// the tensor package realizes a sharded tensor's node from shards that it
// computes on several devices: n holds buf of dev as its result and drops
// its inputs. If over is a realized node, n takes over's memory, with buf
// as its new contents, and over is stale afterwards, as a destination of
// RealizeInto is.
func Bind(n *graph.Node, dev backend.Backend, buf backend.Buffer, over *graph.Node) {
	mu.Lock()
	defer mu.Unlock()
	if over != nil {
		waitUnclaimed(over)
	}
	if over != nil && over.Realized() {
		m := over.Mem
		m.Version++
		m.Buf, m.Refs = buf, 1
		n.Mem, n.Ver = m, m.Version
	} else {
		n.Mem, n.Ver = &graph.Storage{Buf: buf, Dev: dev, Refs: 1}, 0
	}
	n.View = backend.Contiguous(n.Shape)
	n.Inputs = nil
}

// Invalidate makes the realized node n, and every node realized into its
// memory, stale, as Release does, without giving the memory to a pool:
// for a node that Bind realized, whose buffer holds other nodes.
func Invalidate(n *graph.Node) {
	mu.Lock()
	defer mu.Unlock()
	waitUnclaimed(n)
	if n.Realized() && !n.Stale() {
		n.Mem.Version++
		n.Mem.Refs = 0
	}
}

// Inputs returns the inputs of n, which Bind and the scheduler drop once
// n is realized, and whether n is realized, safely while other goroutines
// realize graphs that n is part of.
func Inputs(n *graph.Node) ([]*graph.Node, bool) {
	mu.Lock()
	defer mu.Unlock()
	return n.Inputs, n.Realized()
}
