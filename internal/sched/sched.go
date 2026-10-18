// Package sched realizes lazy graphs on a backend.
//
// The scheduler runs kernels in dependency order: one per node, except
// that chains of elementwise nodes fuse into one kernel (see newPlan). It
// turns movement ops into views of their input's buffer where the layout
// allows, and frees every intermediate buffer right after its last use.
// A target can also take over the memory of a realized node that the
// caller is done with (see RealizeInto). A schedule that would take its
// device past a memory budget is refused before it starts (see SetBudget).
package sched

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// mu serializes Realize and Download. Realizing writes results into nodes
// that other graphs may share, so concurrent schedules could race on them.
// Kernels themselves still use every core.
var mu sync.Mutex

// Realize computes every target on be. On success each target holds its
// result and has dropped its inputs; on error no node is changed.
func Realize(ctx context.Context, be backend.Backend, targets ...*graph.Node) error {
	mu.Lock()
	defer mu.Unlock()
	return realize(ctx, be, targets, nil, nil)
}

// RealizeInto computes every target on be like Realize, but writes
// targets[i] over the memory of dsts[i] where it can instead of into a new
// buffer. A nil dsts[i], or nil dsts, is ignored. A target goes into its
// destination's memory when the destination is realized, with the
// target's dtype and shape, into a buffer of its own, and when no kernel
// reads that memory after the one that writes the target, which may
// itself read it elementwise, as an optimizer's update reads the old
// value it replaces.
//
// Either way each destination, and every node realized into the same
// memory, is stale afterwards, unless its target shares that memory. Once
// a kernel has begun to write over a destination the schedule ignores
// cancellation, so that it does not stop with some targets written and
// others not. If the backend fails after that, the targets are unchanged
// and the destinations written over are stale.
//
// keep, if not nil, lists unrealized nodes that must stay computable,
// such as values that parameters hold as recipes. It is called with the
// scheduler's lock held, so it may read the state of nodes. Each node it
// lists that reads the memory of a destination, and could still be
// computed, is computed first, in a schedule of its own, so that the
// targets are written over their destinations as if it had been computed
// before. A node that fails to compute stays uncomputed, and is stale
// once the destinations it reads are written over. Nor does the schedule
// take the memory of a donated node that a node keep lists reads (see
// donate.go).
func RealizeInto(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node, keep func() []*graph.Node) error {
	if dsts != nil && len(dsts) != len(targets) {
		return fmt.Errorf("sched: %d destinations for %d targets: %w", len(dsts), len(targets), backend.ErrInternal)
	}
	mu.Lock()
	defer mu.Unlock()
	targets, dsts = protect(ctx, be, targets, dsts, keep)
	return realize(ctx, be, targets, dsts, keep)
}

// RealizeEach realizes targets[i] on bes[i], writing them over dsts[i],
// as RealizeInto does for each i, with dsts nil or dsts[i] nil for no
// destinations. The schedules of the devices run at once, each on its
// own goroutine, so that walking one device's graph and launching its
// kernels overlaps the others': the schedules of a sharded computation
// then start on every device together rather than one after another.
//
// The devices must differ, and each graph must read only nodes on its
// own device or computed from constants: a schedule that would copy an
// input from another device fails with backend.ErrDevice. Nor may the
// graphs share an unrealized node. keep is as for RealizeInto; each
// device's schedule spares only the nodes that keep lists on that
// device. It returns the errors of the schedules that failed, joined;
// the others' targets are computed.
func RealizeEach(ctx context.Context, bes []backend.Backend, targets, dsts [][]*graph.Node, keep func() []*graph.Node) error {
	return realizeEach(ctx, bes, targets, dsts, keep, nil)
}

func realizeEach(ctx context.Context, bes []backend.Backend, targets, dsts [][]*graph.Node, keep func() []*graph.Node, trace func(r int, t *Trace, order, leaves []*graph.Node)) error {
	if len(targets) != len(bes) || dsts != nil && len(dsts) != len(bes) {
		return fmt.Errorf("sched: %d target lists and %d destination lists for %d devices: %w", len(targets), len(dsts), len(bes), backend.ErrInternal)
	}
	for i, ts := range targets {
		if dsts != nil && dsts[i] != nil && len(dsts[i]) != len(ts) {
			return fmt.Errorf("sched: %d destinations for %d targets: %w", len(dsts[i]), len(ts), backend.ErrInternal)
		}
		for _, be := range bes[:i] {
			if be == bes[i] {
				return fmt.Errorf("sched: two schedules at once on %s: %w", be.Name(), backend.ErrInternal)
			}
		}
	}
	mu.Lock()
	defer mu.Unlock()
	ts, ds := slices.Clone(targets), make([][]*graph.Node, len(bes))
	keeps := make([]func() []*graph.Node, len(bes))
	traces := make([]func(*Trace, []*graph.Node, []*graph.Node), len(bes))
	var kept []*graph.Node
	if keep != nil {
		kept = keep()
	}
	for i, be := range bes {
		poolOf(be) // created now: the schedules only read the map of pools
		if dsts != nil {
			ds[i] = dsts[i]
		}
		if trace != nil {
			traces[i] = func(t *Trace, order, leaves []*graph.Node) { trace(i, t, order, leaves) }
		}
		if keep == nil {
			continue
		}
		// A node on another device reads none of be's memory, as a
		// schedule here reads none of another's, and its graph is the
		// other schedule's to change.
		var mine []*graph.Node
		for _, k := range kept {
			if !k.Realized() && k.Dev == be {
				mine = append(mine, k)
			}
		}
		keeps[i] = func() []*graph.Node { return mine }
		if len(mine) > 0 {
			// The schedule may compute some of mine first (see protect),
			// which a replay would not.
			traces[i] = nil
		}
		ts[i], ds[i] = protect(ctx, be, ts[i], ds[i], keeps[i])
	}
	errs := make([]error, len(bes))
	var wg sync.WaitGroup
	for i, be := range bes {
		if len(ts[i]) == 0 {
			continue
		}
		wg.Go(func() { errs[i] = schedule(ctx, be, ts[i], ds[i], keeps[i], true, traces[i]) })
	}
	wg.Wait()
	return errors.Join(errs...)
}

// protect computes, in schedules of their own, the nodes that keep lists
// that read the memory of a destination in dsts, as RealizeInto
// documents, and returns the targets and destinations with those it
// computed added. mu is held.
func protect(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node, keep func() []*graph.Node) ([]*graph.Node, []*graph.Node) {
	if keep != nil {
		if k := endangered(keep(), targets, dsts); len(k) > 0 {
			// The errors are their readers' to see, when they read them.
			// They take no donated memory that the targets read.
			read := func() []*graph.Node { return targets }
			if realize(ctx, be, k, nil, read) != nil {
				for _, n := range k {
					_ = realize(ctx, be, []*graph.Node{n}, nil, read)
				}
			}
			// Those computed join the targets, which costs no kernels: one
			// that is a view of a destination then keeps that memory from
			// being written over or made stale, as a target would.
			targets, dsts = slices.Clip(targets), slices.Clip(dsts)
			for _, n := range k {
				if n.Realized() {
					targets, dsts = append(targets, n), append(dsts, nil)
				}
			}
		}
	}
	return targets, dsts
}

// A node's state says what it is computed from.
type state uint8

const (
	unseen state = iota
	clean        // nothing below
	reads        // memory of interest
	broken       // a stale node, so it cannot be computed
)

// classifier returns a function that tells what unrealized nodes are
// computed from: whether they read memory in mems or a stale node, through
// unrealized nodes other than those in skip. It classifies each node once,
// walking iteratively, like the walker, so deep graphs cannot overflow
// the stack.
func classifier(mems map[*graph.Storage]bool, skip map[*graph.Node]bool) func(*graph.Node) state {
	states := map[*graph.Node]state{}
	return func(root *graph.Node) state {
		if s := states[root]; s != unseen {
			return s
		}
		type frame struct {
			n    *graph.Node
			next int
			s    state
		}
		stack := []frame{{n: root, s: clean}}
		for {
			f := &stack[len(stack)-1]
			if f.next < len(f.n.Inputs) {
				in := f.n.Inputs[f.next]
				f.next++
				switch {
				case in.Stale():
					f.s = broken
				case in.Realized():
					if mems[in.Mem] {
						f.s = max(f.s, reads)
					}
				case skip[in]:
				case states[in] == unseen:
					stack = append(stack, frame{n: in, s: clean})
				default:
					f.s = max(f.s, states[in])
				}
				continue
			}
			s := f.s
			states[f.n] = s
			if stack = stack[:len(stack)-1]; len(stack) == 0 {
				return s
			}
			stack[len(stack)-1].s = max(stack[len(stack)-1].s, s)
		}
	}
}

// endangered returns the nodes of keep that read the memory of a realized
// destination in dsts, which the schedule makes stale, other than through
// a target, and that are not targets, realized or stale already. Targets
// are left out because the schedule computes them, and they stay valid
// after it.
func endangered(keep, targets, dsts []*graph.Node) []*graph.Node {
	mems := map[*graph.Storage]bool{}
	for _, d := range dsts {
		if d != nil && d.Realized() && !d.Stale() {
			mems[d.Mem] = true
		}
	}
	if len(mems) == 0 {
		return nil
	}
	isTarget := make(map[*graph.Node]bool, len(targets))
	for _, t := range targets {
		isTarget[t] = true
	}
	classify := classifier(mems, isTarget)
	var out []*graph.Node
	for _, k := range keep {
		if k.Realized() || isTarget[k] || slices.Contains(out, k) {
			continue
		}
		if classify(k) == reads {
			out = append(out, k)
		}
	}
	return out
}

// generation counts the schedules, which change what nodes are realized
// or stale. A release does not count: it makes stale only memory that no
// kept node reads.
var generation atomic.Uint64

// kept caches keptMemory's answer for a generation and set of nodes.
var kept struct {
	generation uint64
	nodes      map[*graph.Node]bool
	mems       map[*graph.Storage]bool
}

// keptMemory returns the storages that the nodes of keep read, through
// unrealized nodes, leaving out those that can no longer be computed. A
// release of each gradient in turn asks the same question, so the answer
// is kept until a schedule, or a different keep, changes it.
func keptMemory(keep []*graph.Node) map[*graph.Storage]bool {
	nodes := make(map[*graph.Node]bool, len(keep))
	for _, k := range keep {
		nodes[k] = true
	}
	if kept.mems != nil && kept.generation == generation.Load() && maps.Equal(kept.nodes, nodes) {
		return kept.mems
	}
	classify := classifier(nil, nil)
	mems := map[*graph.Storage]bool{}
	seen := map[*graph.Node]bool{}
	for _, k := range keep {
		if k.Realized() || seen[k] || classify(k) == broken {
			continue
		}
		seen[k] = true
		for stack := []*graph.Node{k}; len(stack) > 0; {
			n := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			for _, in := range n.Inputs {
				switch {
				case in.Realized():
					mems[in.Mem] = true
				case !seen[in]:
					seen[in] = true
					stack = append(stack, in)
				}
			}
		}
	}
	kept.generation, kept.nodes, kept.mems = generation.Load(), nodes, mems
	return mems
}

// Release gives the memory of the realized node n to its device's pool for
// later schedules and makes n stale, if n is the only node realized into that
// memory and no node that keep lists, when it is not nil, reads it (see
// RealizeInto). The caller must be done with n. Memory that the pool did
// not obtain, such as an upload's, counts against the device's budget
// from then on.
func Release(n *graph.Node, keep func() []*graph.Node) {
	mu.Lock()
	defer mu.Unlock()
	if !n.Realized() || n.Stale() || n.Mem.Refs != 1 {
		return
	}
	if keep != nil && keptMemory(keep())[n.Mem] {
		return
	}
	n.Mem.Version++
	n.Mem.Refs = 0
	p := poolOf(n.Mem.Dev)
	p.track(n.Buf())
	p.release(n.Buf())
}

// Realized reports whether n holds its result. Unlike n.Realized, it is
// safe while other goroutines realize graphs that n is part of.
func Realized(n *graph.Node) bool {
	mu.Lock()
	defer mu.Unlock()
	return n.Realized()
}

// Device returns the device that holds n's result, if n is realized, or
// its inputs, or nil if n is computed from constants alone. Unlike n.Mem,
// it is safe while other goroutines realize graphs that n is part of.
func Device(n *graph.Node) backend.Backend {
	mu.Lock()
	defer mu.Unlock()
	if n.Realized() {
		return n.Mem.Dev
	}
	return n.Dev
}

// Download realizes n on be, unless it is realized, and copies its
// elements into dst as row-major bytes. len(dst) must equal the tensor's
// size in bytes. keep, if not nil, lists nodes whose memory the schedule
// must not take, as for RealizeInto.
func Download(ctx context.Context, be backend.Backend, n *graph.Node, dst []byte, keep func() []*graph.Node) error {
	mu.Lock()
	defer mu.Unlock()
	if err := realize(ctx, be, []*graph.Node{n}, nil, keep); err != nil {
		return err
	}
	return download(ctx, n, dst)
}

// download copies the elements of the realized node n into dst.
func download(ctx context.Context, n *graph.Node, dst []byte) error {
	be := n.Mem.Dev
	buf, v := n.Buf(), n.View
	if !v.IsContiguous() || v.Offset != 0 || buf.Len() != n.Size() {
		m := poolOf(be)
		s := m.simulate()
		s.take(n.DType, n.Size())
		if err := m.reserve(s); err != nil {
			return err
		}
		tmp, err := runNew(ctx, m, &backend.Kernel{Op: backend.OpContiguous, DType: n.DType, Shape: n.Shape, In: []backend.View{v}}, buf)
		if err != nil {
			return err
		}
		defer m.release(tmp)
		buf = tmp
	}
	return be.Download(ctx, dst, buf)
}

// result is where a computed node's elements live.
type result struct {
	view backend.View
	mem  *graph.Storage
}

// realize computes targets on be, writing them over dsts, as
// RealizeInto documents. mu is held.
func realize(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node, keep func() []*graph.Node) error {
	return schedule(ctx, be, targets, dsts, keep, false, nil)
}

// schedule is realize, which if local is set runs while the schedules of
// other devices do (see RealizeEach), and therefore fails rather than
// copy an input from another device; mu is then held by the caller of
// the goroutines that run them. It calls trace, if not nil, with the
// schedule's trace (see RealizeEachTraced) unless a target is realized.
func schedule(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node, keep func() []*graph.Node, local bool, trace func(*Trace, []*graph.Node, []*graph.Node)) (err error) {
	generation.Add(1)
	m := poolOf(be)
	w := &m.walker
	defer w.forget()
	if err := w.run(be, targets, dsts); err != nil {
		return err
	}
	if local {
		for _, n := range w.leaves {
			if n.Mem.Dev != be {
				return fmt.Errorf("sched: a schedule on %s reads a tensor on %s while that device runs its own: %w", be.Name(), n.Mem.Dev.Name(), backend.ErrDevice)
			}
		}
	}
	w.withhold(keep, targets)
	if trace != nil && slices.ContainsFunc(targets, (*graph.Node).Realized) {
		// The key names a realized target by the leaf it is, which a
		// replay would have to find.
		trace = nil
	}
	if len(w.order) == 0 {
		if trace != nil {
			trace(w.tracing(nil, len(targets)), w.order, w.leaves)
		}
		consume(targets, dsts)
		return nil
	}
	defer m.tick()
	// A backend may run kernels after Run returns and report their errors
	// at Sync, so the schedule waits for them before it changes any node.
	// On error it still waits, so that the errors of kernels already
	// queued are not reported by a later schedule.
	defer func() {
		if err != nil {
			_ = be.Sync(context.WithoutCancel(ctx))
		}
	}()
	prog, err := m.program(w, targets, dsts)
	if err != nil {
		return err
	}
	if trace != nil {
		trace(w.tracing(prog, len(targets)), w.order, w.leaves)
	}
	// The schedule is refused before it allocates anything if it would
	// not fit in the budget.
	if err := prog.reserve(m); err != nil {
		return err
	}
	return prog.run(ctx, m, w, targets, dsts)
}

// program returns the program that computes targets, which w has walked,
// from the cache if it has one for w's key, and otherwise plans and
// compiles one, and caches it.
func (m *pool) program(w *walker, targets, dsts []*graph.Node) (*program, error) {
	cached := caching && keyAttrs != nil
	var hash uint64
	if cached {
		hash = hashKey(w.key)
		if prog := m.programs.lookup(hash, w.key); prog != nil {
			return prog, nil
		}
	}
	isTarget := make(map[*graph.Node]bool, len(targets))
	for _, t := range targets {
		isTarget[t] = true
	}
	p := newPlan(m.be, w.order, isTarget, candidates(m.be, targets, dsts))
	prog, err := compile(p, w, targets, dsts, p.destinations(targets, dsts), isTarget)
	if err != nil {
		return nil, err
	}
	compiled.Add(1)
	if cached {
		m.programs.add(hash, w.key, prog)
	}
	return prog, nil
}

// assign makes t hold the result r, whose storage is set, and drop its
// inputs.
func assign(t *graph.Node, r result) {
	r.mem.Refs++
	t.Mem, t.Ver, t.View = r.mem, r.mem.Version, r.view
	t.Op, t.Attrs, t.Inputs = backend.OpLoad, backend.Attrs{}, nil
}

// consume makes stale every destination whose target was computed into
// memory of its own: the caller is done with it, whether or not its memory
// was reused. A destination already stale is left alone, since its memory
// may hold a newer value. It marks the storages it has seen, as a walk
// does, so that it allocates nothing.
func consume(targets, dsts []*graph.Node) {
	walk := walks.Add(1)
	for _, t := range targets {
		if t.Mem != nil {
			t.Mem.Mark.Walk = walk
		}
	}
	for _, d := range dsts {
		if d != nil && d.Realized() && !d.Stale() && d.Mem.Mark.Walk != walk {
			d.Mem.Version++
			d.Mem.Refs = 0
			d.Mem.Mark.Walk = walk
		}
	}
}

// elements returns the number of elements of k's output.
func elements(k *backend.Kernel) (int, error) {
	n, ok := backend.NumElements(k.Shape)
	if !ok {
		return 0, fmt.Errorf("sched: %s: shape %v: %w", k.Op, k.Shape, backend.ErrShape)
	}
	return n, nil
}

// resultView returns how the node x reads the output of k that holds it:
// contiguously, or, for a constant, the one element broadcast to its
// shape with zero strides.
func resultView(k *backend.Kernel, x *graph.Node) backend.View {
	if k.Op == backend.OpConst {
		return broadcast(x.Shape)
	}
	return backend.Contiguous(x.Shape)
}

// runNew executes k, which computes one value, into a new buffer from m.
// On error the buffer goes back to m and nil is returned.
func runNew(ctx context.Context, m *pool, k *backend.Kernel, ins ...backend.Buffer) (backend.Buffer, error) {
	n, err := elements(k)
	if err != nil {
		return nil, err
	}
	out, err := m.alloc(k.DType, n)
	if err != nil {
		return nil, err
	}
	if err := m.be.Run(ctx, k, []backend.Buffer{out}, ins); err != nil {
		m.release(out)
		return nil, err
	}
	return out, nil
}

// broadcast returns a view that reads element 0 for every index of shape.
func broadcast(shape []int) backend.View {
	return backend.View{Shape: shape, Strides: make([]int, len(shape))}
}

// isBroadcastScalar reports whether every element of v is the same element.
func isBroadcastScalar(v backend.View) bool {
	for i, s := range v.Strides {
		if s != 0 && v.Shape[i] != 1 {
			return false
		}
	}
	return true
}
