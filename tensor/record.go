package tensor

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// A Recording is how a computation on one device was realized: the
// program its schedule ran, and what the program read. Replay runs the
// program again over new values of the computation's leaves, without
// building its graph or walking it, for a caller that computes the same
// graph over and over, such as a model that answers one request after
// another: it builds the graph once, records it, and replays it on each
// request whose inputs have the shapes the recording was made with.
type Recording struct {
	be backend.Backend
	t  *sched.Trace
	// For each leaf the trace reads, its place among the leaves given to
	// Record, or, for a realized tensor they do not name, which every
	// replay reads again, ^its place in fixed.
	leaves  []int
	fixed   []*graph.Node
	args    []*graph.Node // the nodes whose constants and seeds the program reads, by number
	targets []bare        // the nodes the program computes, in the trace's order
	at      []int         // for each tensor given to Record, its place in targets
}

// bare is what a replay makes a target's node from.
type bare struct {
	op    backend.Op
	dt    backend.DType
	shape []int
	dev   backend.Backend
}

// Record computes targets, as Realize does, and records how, for Replay
// to compute them again over new leaves. leaves are the tensors that a
// replay gives new values: the computation's inputs, and the state it
// reads and writes. Each is computed first if it is not yet. Any other
// realized tensor the computation reads, such as a weight or a table
// built while the graph was, is read again by every replay, with the
// values it holds then.
//
// It returns a nil recording, with targets computed, when the
// computation cannot be replayed: when it runs on several devices or on
// a mesh, or when a target is realized already or is a leaf. The leaves'
// values do not matter to what a replay computes, only their shapes,
// dtypes and layouts, so a computation whose graph depends on a leaf's
// values, as one built by reading them does, must not be recorded.
func Record(ctx context.Context, leaves, targets []*Tensor) (*Recording, error) {
	if err := firstErr(leaves...); err != nil {
		return nil, err
	}
	if err := firstErr(targets...); err != nil {
		return nil, err
	}
	if err := computable(targets...); err != nil {
		return nil, err
	}
	if err := Realize(ctx, leaves...); err != nil {
		return nil, err
	}
	rec := &Recording{at: make([]int, len(targets))}
	var nodes []*graph.Node
	for i, t := range targets {
		if j := slices.Index(nodes, t.node); j >= 0 {
			rec.at[i] = j
			continue
		}
		rec.at[i] = len(nodes)
		nodes = append(nodes, t.node)
	}
	var devices int
	for be := range byDevice(nodes) {
		devices++
		rec.be = be
	}
	if _, mesh := rec.be.(*meshDevice); devices != 1 || mesh || slices.ContainsFunc(nodes, sched.Realized) {
		return nil, Realize(ctx, targets...)
	}
	leafAt := make(map[*graph.Node]int, len(leaves))
	for i, l := range leaves {
		if _, ok := leafAt[l.node]; !ok {
			leafAt[l.node] = i
		}
	}
	rec.targets = make([]bare, len(nodes))
	for i, n := range nodes {
		rec.targets[i] = bare{op: n.Op, dt: n.DType, shape: slices.Clone(n.Shape), dev: n.Dev}
	}
	traced := false
	err := sched.RealizeEachTraced(ctx, []backend.Backend{rec.be}, [][]*graph.Node{nodes}, nil, lazy.values, func(_ int, t *sched.Trace, order, ls []*graph.Node) {
		traced = true
		rec.leaves = make([]int, t.Leaves())
		for i := range rec.leaves {
			at, ok := leafAt[ls[i]]
			if !ok {
				at = ^len(rec.fixed)
				rec.fixed = append(rec.fixed, ls[i])
			}
			rec.leaves[i] = at
		}
		if as := t.Args(); len(as) > 0 {
			rec.args = make([]*graph.Node, as[len(as)-1]+1)
			for _, a := range as {
				// The program reads only the constant and seed; the
				// node's inputs would keep the first run's graph alive.
				n := order[a]
				rec.args[a] = &graph.Node{Op: n.Op, DType: n.DType, Shape: n.Shape, Attrs: n.Attrs, Dev: n.Dev}
			}
		}
		rec.t = t
	})
	if err != nil {
		return nil, err
	}
	if !traced || rec.t == nil {
		return nil, nil
	}
	return rec, nil
}

// Replay computes the targets of the recording again, as Record did, over
// leaves, which stand for those Record was given, in the same order, with
// their shapes, dtypes and layouts, on the same device: each is computed
// first if it is not yet. It returns the targets as Record was given
// them, computed. It returns false, having computed nothing, when the
// leaves do not fit the recording, when a tensor the recording reads
// again has been written over, or when its program is busy with another
// replay: the caller then computes its graph otherwise.
func (r *Recording) Replay(ctx context.Context, leaves []*Tensor) ([]*Tensor, bool, error) {
	if err := firstErr(leaves...); err != nil {
		return nil, false, err
	}
	var unrealized []*Tensor
	for _, l := range leaves {
		if !sched.Realized(l.node) {
			unrealized = append(unrealized, l)
		}
	}
	if err := Realize(ctx, unrealized...); err != nil {
		return nil, false, err
	}
	ls := make([]*graph.Node, len(r.leaves))
	for i, at := range r.leaves {
		switch {
		case at < 0:
			ls[i] = r.fixed[^at]
		case at >= len(leaves):
			return nil, false, fmt.Errorf("tensor: %d leaves for a recording of %d: %w", len(leaves), at+1, ErrShape)
		default:
			ls[i] = leaves[at].node
		}
	}
	var made []*graph.Node
	missed, err := sched.Replay(ctx, r.t, ls, r.args, func() []*graph.Node {
		made = make([]*graph.Node, len(r.targets))
		for i, b := range r.targets {
			made[i] = &graph.Node{Op: b.op, DType: b.dt, Shape: b.shape, Dev: b.dev}
		}
		return made
	}, lazy.values)
	if err != nil {
		return nil, false, err
	}
	if missed {
		return nil, false, nil
	}
	ts := make([]*Tensor, len(made))
	for i, n := range made {
		ts[i] = &Tensor{node: n}
	}
	out := make([]*Tensor, len(r.at))
	for i, j := range r.at {
		out[i] = ts[j]
	}
	return out, true, nil
}
