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
