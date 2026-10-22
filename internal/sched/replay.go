package sched

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// A caller that builds the same graph on a device over and over, as the
// partitioner of sharded tensors does for each step of training, can skip
// building it: RealizeEachTraced hands it each device's schedule as a
// Trace, and ReplayEach runs a trace's program again over new leaves.
// Replaying checks what a schedule's key says of the leaves, the targets
// and the destinations, as a walk would, and refuses a schedule that
// would not have found the same program; only the graph's own nodes go
// unchecked, which the caller vouches for: the nodes the trace walked,
// with the constants and seeds it reads from the nodes it is given.

// Trace is the schedule of one device's graph, which ReplayEach replays.
type Trace struct {
	be     backend.Backend
	prog   *program // nil if the schedule computed nothing
	key    []uint64
	leafAt []int32 // where each leaf the walk met before the targets starts in key
	tailAt int     // where the targets start in key
	nt     int     // the targets
	args   []int32 // the nodes, by number, whose Value and Seed the program reads
}

// Leaves returns how many leaves a replay of t reads.
func (t *Trace) Leaves() int { return len(t.leafAt) }

// Args returns the numbers, in the walk's order, of the nodes whose
// constants and seeds the program reads.
func (t *Trace) Args() []int32 { return t.args }

// RealizeEachTraced is RealizeEach, which also calls trace with each
// device's schedule, once it has found its program: with the schedule's
// trace, the unrealized nodes it computes, in its order, and the
// realized nodes it reads, its leaves. trace runs on the device's
// goroutine, and must not keep order or leaves. A device whose schedule
// cannot be replayed, as one that computes nodes that keep lists first,
// is not traced.
func RealizeEachTraced(ctx context.Context, bes []backend.Backend, targets, dsts [][]*graph.Node, keep func() []*graph.Node, trace func(r int, t *Trace, order, leaves []*graph.Node)) error {
	return realizeEach(ctx, bes, targets, dsts, keep, trace)
}

// tracing returns the trace of the schedule w walked, of nt targets,
// which found prog.
func (w *walker) tracing(prog *program, nt int) *Trace {
	t := &Trace{be: w.be, prog: prog, key: slices.Clone(w.key), leafAt: slices.Clone(w.leafAt[:w.body]), tailAt: w.tailAt, nt: nt}
	if prog != nil {
		for i := range prog.steps {
			s := &prog.steps[i]
			if s.argsOf >= 0 {
				t.args = append(t.args, s.argsOf)
			}
			for _, c := range s.consts {
				t.args = append(t.args, c.node)
			}
		}
		slices.Sort(t.args)
		t.args = slices.Compact(t.args)
	}
	return t
}

// ReplayEach replays on each device r the schedule traces[r], over the
// leaves leaves[r], in the order of the trace's walk, computing targets
// that newTargets(r) returns, which stand for the trace's, and writing them
// over dsts[r], as RealizeEach would compute the graph the trace walked.
// args[r] holds, at each number the trace's Args lists, the node whose
// Value and Seed the program reads. A device with a nil trace, or whose
// trace does not fit its leaves, targets and destinations, runs nothing,
// and ReplayEach reports it as missed, for the caller to compute its
// graph otherwise; newTargets is not called for it, and is called on the
// device's goroutine.
func ReplayEach(ctx context.Context, traces []*Trace, leaves, args [][]*graph.Node, newTargets func(r int) []*graph.Node, dsts [][]*graph.Node, keep func() []*graph.Node) ([]bool, error) {
	lockAll()
	defer mu.Unlock()
	var kept []*graph.Node
	if keep != nil {
		kept = keep()
	}
	missed := slices.Repeat([]bool{true}, len(traces))
	errs := make([]error, len(traces))
	var wg sync.WaitGroup
	for r, t := range traces {
		if t == nil {
			continue
		}
		poolOf(t.be)
		// A node that keep lists on the device may need computing first
		// (see protect), which a replay does not do.
		if slices.ContainsFunc(kept, func(k *graph.Node) bool { return !k.Realized() && k.Dev == t.be }) {
			continue
		}
		var ds []*graph.Node
		if dsts != nil {
			ds = dsts[r]
		}
		wg.Go(func() {
			missed[r], errs[r] = t.replay(ctx, leaves[r], args[r], func() []*graph.Node { return newTargets(r) }, ds, nil, false)
		})
	}
	wg.Wait()
	return missed, errors.Join(errs...)
}

// Replay replays t on its device over leaves, as ReplayEach does for one
// device without destinations, but as RealizeInto runs a schedule: it
// gives the scheduler's lock up while the kernels run, so that
// computations that share no memory with it go on at once. keep is as
// for RealizeInto. It reports the replay as missed, having run nothing,
// when the trace does not fit the leaves, when the memory that keep
// spares would change its program, or when the program is running
// already; newTargets may have been called, and the caller computes its
// graph otherwise.
func Replay(ctx context.Context, t *Trace, leaves, args []*graph.Node, newTargets func() []*graph.Node, keep func() []*graph.Node) (missed bool, err error) {
	mu.Lock()
	defer mu.Unlock()
	err = retry(func() error {
		missed, err = t.replay(ctx, leaves, args, newTargets, nil, keep, true)
		return err
	})
	return missed, err
}

// replay runs t over leaves, as schedule does, unless t does not fit
// them, which it reports. mu is held: by the caller of the goroutines
// that run replays, or, with unlock set, by the caller, which gives it
// up while the kernels run and reports errBusy, having changed nothing,
// when running schedules keep this one from starting.
func (t *Trace) replay(ctx context.Context, leaves, args []*graph.Node, newTargets func() []*graph.Node, dsts []*graph.Node, keep func() []*graph.Node, unlock bool) (missed bool, err error) {
	if len(leaves) != len(t.leafAt) {
		return true, fmt.Errorf("sched: %d leaves for a trace of %d: %w", len(leaves), len(t.leafAt), backend.ErrInternal)
	}
	m := poolOf(t.be)
	w := m.takeWalker()
	defer m.giveWalker(w)
	if !t.fits(w, leaves, dsts) {
		return true, nil
	}
	if unlock && t.prog != nil && t.prog.busy {
		// Another schedule runs the program without mu; it is not
		// reentrant.
		return true, nil
	}
	targets := newTargets()
	if len(targets) != t.nt || slices.ContainsFunc(targets, (*graph.Node).Realized) {
		return false, fmt.Errorf("sched: %d targets for a trace of %d unrealized ones: %w", len(targets), t.nt, backend.ErrInternal)
	}
	if keep != nil && len(w.gifts) > 0 {
		// The program takes the memory of the donated leaves; one that
		// keep spares would need another.
		if w.withhold(keep, targets); !slices.Equal(w.key, t.key) {
			return true, nil
		}
	}
	if unlock && w.contended(dsts) {
		return false, errBusy
	}
	generation.Add(1)
	if t.prog == nil {
		consume(targets, dsts)
		return false, nil
	}
	w.order = append(w.order[:0], args...)
	defer m.tick()
	defer func() {
		if err != nil {
			_ = t.be.Sync(context.WithoutCancel(ctx))
		}
	}()
	if err := t.prog.reserve(m); err != nil {
		// Memory that running schedules hold may be back in the pool
		// once they end.
		if unlock && running > 0 && errors.Is(err, backend.ErrMemoryBudget) {
			return false, errBusy
		}
		return false, err
	}
	return false, t.prog.run(ctx, m, w, targets, dsts, unlock)
}

// fits walks leaves as the leaves of t's graph, and dsts, writing their
// part of the key, and reports whether the key is t's. Unrealized
// targets are the trace's; the realized ones, which a replay reads, are
// the walk's leaves after leaves.
func (t *Trace) fits(w *walker, leaves, dsts []*graph.Node) bool {
	w.walk, w.be = walks.Add(1), t.be
	w.gifts = w.gifts[:0]
	w.order, w.leaves, w.mems = w.order[:0], w.leaves[:0], 0
	w.key = w.key[:0]
	w.leafAt = w.leafAt[:0]
	verbatim := func(to int) bool {
		if len(w.key) > to || to > len(t.key) {
			return false
		}
		w.key = append(w.key, t.key[len(w.key):to]...)
		return true
	}
	for i, n := range leaves {
		if n == nil || !n.Realized() || n.Stale() || n.Mark.Walk == w.walk || n.Mem.Dev != t.be {
			return false
		}
		if !verbatim(int(t.leafAt[i])) {
			return false
		}
		w.leaf(n)
	}
	if !verbatim(t.tailAt) || len(t.key) == t.tailAt {
		return false
	}
	// The targets are the trace's: each is a number in the walk's order,
	// the same as before, or a leaf, which only a realized target is, and
	// which a replay of the same graph reads again among leaves.
	nt := t.nt
	if len(t.key) < t.tailAt+1+nt || len(dsts) != 0 && len(dsts) != nt {
		return false
	}
	w.key = append(w.key, t.key[t.tailAt:t.tailAt+1+nt]...)
	for i := range nt {
		switch d := destination(dsts, i); {
		case d == nil || !d.Realized() || d.Stale():
			w.push(tagNone << 56)
		default:
			w.push(w.ref(d))
		}
	}
	return slices.Equal(w.key, t.key)
}
