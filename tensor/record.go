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
// a mesh, or when a target is realized already or is a leaf, as every
// target is in eager mode. The leaves'
// values do not matter to what a replay computes, only their shapes,
// dtypes and layouts, so a computation whose graph depends on a leaf's
// values, as one built by reading them does, must not be recorded.
func Record(ctx context.Context, leaves, targets []*Tensor) (*Recording, error) {
	return RecordInto(ctx, leaves, nil, targets)
}

// RecordInto is Record that writes targets[i] over the memory of dsts[i]
// where it can, as RealizeInto does, and records that too: a replay then
// writes each target over the destination ReplayInto gives it. dsts is
// nil or holds a destination, or nil, for each target, as for
// RealizeInto, whose caller must be done with each. A destination may be
// a leaf, as the state a computation reads and writes, such as a
// key-value cache, is.
func RecordInto(ctx context.Context, leaves, dsts, targets []*Tensor) (*Recording, error) {
	if err := firstErr(leaves...); err != nil {
		return nil, err
	}
	if err := firstErr(targets...); err != nil {
		return nil, err
	}
	if err := computable(targets...); err != nil {
		return nil, err
	}
	if err := checkDestinations(dsts, targets); err != nil {
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
	over := rec.destinations(dsts, len(nodes))
	var devices int
	for be := range byDevice(nodes) {
		devices++
		rec.be = be
	}
	if _, mesh := rec.be.(*meshDevice); devices != 1 || mesh || slices.ContainsFunc(nodes, sched.Realized) {
		if dsts != nil {
			return nil, RealizeInto(ctx, dsts, targets)
		}
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
	var each [][]*graph.Node
	if over != nil {
		each = [][]*graph.Node{over}
	}
	err := sched.RealizeEachTraced(ctx, []backend.Backend{rec.be}, [][]*graph.Node{nodes}, each, lazy.values, func(_ int, t *sched.Trace, order, ls []*graph.Node) {
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
	return r.ReplayInto(ctx, leaves, nil)
}

// ReplayInto is Replay that writes the targets over dsts, as RecordInto
// did over the destinations it was given: dsts is nil or holds a
// destination, or nil, for each target Record was given, which stands
// for the one RecordInto was given. It returns false, having computed
// nothing and left dsts alone, when the destinations do not fit the
// recording either: when one is given where RecordInto had none, or the
// other way around, or when one is not computed or stale.
func (r *Recording) ReplayInto(ctx context.Context, leaves, dsts []*Tensor) ([]*Tensor, bool, error) {
	if err := firstErr(leaves...); err != nil {
		return nil, false, err
	}
	if err := checkDestinations(dsts, nil); err != nil {
		return nil, false, err
	}
	if dsts != nil && len(dsts) != len(r.at) {
		return nil, false, fmt.Errorf("tensor: %d destinations for a recording of %d targets: %w", len(dsts), len(r.at), ErrShape)
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
	}, r.destinations(dsts, len(r.targets)), lazy.values)
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

// destinations returns the destination of each of the n nodes a recording
// computes, from dsts, given for the targets in Record's order, or nil if
// there are none.
func (r *Recording) destinations(dsts []*Tensor, n int) []*graph.Node {
	var over []*graph.Node
	for i, d := range dsts {
		if d == nil {
			continue
		}
		if over == nil {
			over = make([]*graph.Node, n)
		}
		if j := r.at[i]; over[j] == nil {
			over[j] = d.node
		}
	}
	return over
}

// checkDestinations checks that each destination of dsts is computable
// and, if targets is not nil, has the shape and dtype of its target, as
// RealizeInto requires.
func checkDestinations(dsts, targets []*Tensor) error {
	if dsts == nil {
		return nil
	}
	if targets != nil && len(dsts) != len(targets) {
		return fmt.Errorf("tensor: %d destinations for %d tensors: %w", len(dsts), len(targets), ErrShape)
	}
	for i, d := range dsts {
		if d == nil {
			continue
		}
		if err := firstErr(d); err != nil {
			return err
		}
		if err := computable(d); err != nil {
			return err
		}
		if targets == nil {
			continue
		}
		if t := targets[i].node; !slices.Equal(d.node.Shape, t.Shape) || d.node.DType != t.DType {
			return fmt.Errorf("tensor: %s%v into %s%v: %w", t.DType, t.Shape, d.node.DType, d.node.Shape, ErrShape)
		}
	}
	return nil
}
