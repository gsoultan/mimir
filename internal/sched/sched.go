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
	"fmt"
	"maps"
	"slices"
	"sync"

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
	return realize(ctx, be, targets, nil)
}

// RealizeInto computes every target on be like Realize, but writes
// targets[i] over the memory of dsts[i] where it can instead of into a new
// buffer. A nil dsts[i] is ignored. A target goes into its destination's
// memory when the destination is realized, with the target's dtype and
// shape, into a buffer of its own, and when no kernel reads that memory
// after the one that writes the target, which may itself read it
// elementwise, as an optimizer's update reads the old value it replaces.
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
// once the destinations it reads are written over.
func RealizeInto(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node, keep func() []*graph.Node) error {
	if len(dsts) != len(targets) {
		return fmt.Errorf("sched: %d destinations for %d targets: %w", len(dsts), len(targets), backend.ErrInternal)
	}
	mu.Lock()
	defer mu.Unlock()
	if keep != nil {
		if k := endangered(keep(), targets, dsts); len(k) > 0 {
			// The errors are their readers' to see, when they read them.
			if realize(ctx, be, k, nil) != nil {
				for _, n := range k {
					_ = realize(ctx, be, []*graph.Node{n}, nil)
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
	return realize(ctx, be, targets, dsts)
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
// walking iteratively, like topoSort, so deep graphs cannot overflow the
// stack.
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
var generation uint64

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
	if kept.mems != nil && kept.generation == generation && maps.Equal(kept.nodes, nodes) {
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
	kept.generation, kept.nodes, kept.mems = generation, nodes, mems
	return mems
}

// Release gives the memory of the realized node n to its device's pool for
// later schedules and makes n stale, if n is the only node realized into that
// memory and no node that keep lists, when it is not nil, reads it (see
// RealizeInto). The caller must be done with n.
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
	poolOf(n.Mem.Dev).release(n.Buf())
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
// size in bytes.
func Download(ctx context.Context, be backend.Backend, n *graph.Node, dst []byte) error {
	mu.Lock()
	defer mu.Unlock()
	if err := realize(ctx, be, []*graph.Node{n}, nil); err != nil {
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

// transfer copies the realized node n, which lives on another device, into
// a new buffer from m.
func transfer(ctx context.Context, m *pool, n *graph.Node) (backend.Buffer, error) {
	data := make([]byte, n.Size()*n.DType.Size())
	if err := download(ctx, n, data); err != nil {
		return nil, err
	}
	buf, err := m.alloc(n.DType, n.Size())
	if err != nil {
		return nil, err
	}
	if err := m.be.Upload(ctx, buf, data); err != nil {
		m.release(buf)
		return nil, err
	}
	return buf, nil
}

// result is where a computed node's elements live.
type result struct {
	buf  backend.Buffer
	view backend.View
	mem  *graph.Storage // the storage of buf, if realized nodes read it
}

func realize(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node) (err error) {
	generation++
	order, err := topoSort(targets)
	if err != nil {
		return err
	}
	if len(order) == 0 {
		consume(targets, dsts)
		return nil
	}
	m := poolOf(be)
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
	isTarget := make(map[*graph.Node]bool, len(targets))
	for _, t := range targets {
		isTarget[t] = true
	}
	p := newPlan(order, isTarget, candidates(be, targets, dsts))
	p.be = be
	into := p.destinations(targets, dsts)
	uses := p.uses()
	// The schedule is refused before it allocates anything if it would
	// not fit in the budget.
	s, err := p.forecast(m, uses, into, isTarget)
	if err != nil {
		return err
	}
	if err := m.reserve(s); err != nil {
		return err
	}

	results := make(map[*graph.Node]result, len(order))
	// refs counts the live results that read each buffer allocated here, so
	// a buffer shared by views is freed only after its last reader.
	refs := make(map[backend.Buffer]int)
	// A realized input on another device, such as a constant computed
	// there, is copied to be first. Kernels read the copy until the end of
	// the schedule, directly or through views, so the schedule holds a
	// reference to it as well as the results that view it, and gives it
	// back after the last kernel.
	moved := make(map[*graph.Node]result)
	for _, n := range p.order {
		for _, in := range p.work[n].deps {
			if _, ok := moved[in]; ok || !in.Realized() || in.Mem.Dev == be {
				continue
			}
			buf, err := transfer(ctx, m, in)
			if err != nil {
				for buf := range refs {
					m.release(buf)
				}
				return err
			}
			refs[buf] = 1
			moved[in] = result{buf: buf, view: backend.Contiguous(in.Shape)}
		}
	}
	lookup := func(n *graph.Node) result {
		if r, ok := moved[n]; ok {
			return r
		}
		if n.Realized() {
			return result{n.Buf(), n.View, n.Mem}
		}
		return results[n]
	}
	release := func(n *graph.Node) {
		uses[n]--
		if uses[n] > 0 || isTarget[n] || n.Realized() {
			return
		}
		r := results[n]
		delete(results, n)
		if _, owned := refs[r.buf]; owned {
			if refs[r.buf]--; refs[r.buf] == 0 {
				delete(refs, r.buf)
				m.release(r.buf)
			}
		}
	}

	for _, n := range p.order {
		w := p.work[n]
		ins := make([]result, len(w.deps))
		for i, in := range w.deps {
			ins[i] = lookup(in)
		}
		nodes := w.nodes(n)
		err := ctx.Err()
		if err == nil && slices.ContainsFunc(nodes, func(x *graph.Node) bool { return into[x] != nil }) {
			ctx = context.WithoutCancel(ctx)
		}
		var rs []result
		var fresh []backend.Buffer
		if err == nil {
			rs, fresh, err = step(ctx, m, n, w, ins, into)
		}
		if err != nil {
			for buf := range refs {
				m.release(buf)
			}
			return err
		}
		for _, buf := range fresh {
			refs[buf] = 0
		}
		for i, x := range nodes {
			r := rs[i]
			if _, owned := refs[r.buf]; owned {
				refs[r.buf]++
			}
			results[x] = r
		}
		for _, in := range w.deps {
			release(in)
		}
	}

	for _, t := range targets {
		if _, ok := results[t]; !ok && !t.Realized() {
			for buf := range refs {
				m.release(buf)
			}
			return fmt.Errorf("sched: no kernel computed a %s target: %w", t.Op, backend.ErrInternal)
		}
	}
	if err := be.Sync(context.WithoutCancel(ctx)); err != nil {
		for buf := range refs {
			m.release(buf)
		}
		return err
	}
	for _, r := range moved {
		// A target that views the copy keeps it.
		if refs[r.buf]--; refs[r.buf] == 0 {
			delete(refs, r.buf)
			m.release(r.buf)
		}
	}
	mems := make(map[backend.Buffer]*graph.Storage)
	for _, t := range targets {
		if t.Realized() {
			continue
		}
		r := results[t]
		if r.mem == nil {
			if r.mem = mems[r.buf]; r.mem == nil {
				r.mem = &graph.Storage{Buf: r.buf, Dev: be}
				mems[r.buf] = r.mem
			}
		}
		assign(t, r)
	}
	consume(targets, dsts)
	return nil
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
// may hold a newer value.
func consume(targets, dsts []*graph.Node) {
	held := make(map[*graph.Storage]bool, len(targets))
	for _, t := range targets {
		held[t.Mem] = true
	}
	for _, d := range dsts {
		if d != nil && d.Realized() && !d.Stale() && !held[d.Mem] {
			d.Mem.Version++
			d.Mem.Refs = 0
			held[d.Mem] = true
		}
	}
}

// topoSort returns the unrealized nodes reachable from targets, each after
// its inputs, or ErrOverwritten if a realized node among them is stale. It
// walks iteratively so deep graphs cannot overflow the stack.
func topoSort(targets []*graph.Node) ([]*graph.Node, error) {
	type frame struct {
		n    *graph.Node
		next int
	}
	stale := func(n *graph.Node) error {
		return fmt.Errorf("sched: %s%v: %w", n.DType, n.Shape, graph.ErrOverwritten)
	}
	var order []*graph.Node
	seen := make(map[*graph.Node]bool)
	for _, t := range targets {
		if t.Stale() {
			return nil, stale(t)
		}
		if t.Realized() || seen[t] {
			continue
		}
		seen[t] = true
		stack := []frame{{n: t}}
		for len(stack) > 0 {
			f := &stack[len(stack)-1]
			if f.next < len(f.n.Inputs) {
				in := f.n.Inputs[f.next]
				f.next++
				switch {
				case in.Stale():
					return nil, stale(in)
				case !in.Realized() && !seen[in]:
					seen[in] = true
					stack = append(stack, frame{n: in})
				}
				continue
			}
			order = append(order, f.n)
			stack = stack[:len(stack)-1]
		}
	}
	return order, nil
}

// step computes one node as w plans, from the results of w.deps, and the
// targets its kernel stores (see work.nodes). It returns their results and
// the buffers it allocated, which hold the results of the nodes not in
// into. A node in into is written over the memory of its destination
// instead; the result of a view may also be a view of an input's buffer.
func step(ctx context.Context, m *pool, n *graph.Node, w *work, ins []result, into map[*graph.Node]*graph.Node) ([]result, []backend.Buffer, error) {
	views := make([]backend.View, len(ins))
	bufs := make([]backend.Buffer, len(ins))
	for i, in := range ins {
		views[i], bufs[i] = in.view, in.buf
	}
	k, view, err := w.kernel(n, views)
	if err != nil {
		return nil, nil, err
	}
	if k == nil {
		return []result{{ins[0].buf, view, ins[0].mem}}, nil, nil
	}
	return runInto(ctx, m, k, w.nodes(n), into, bufs...)
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

// runInto executes k, which computes nodes (see resultView): into the
// memory of its destination in into, whose old values it makes stale, or
// into a new buffer from m, as forecast predicts. It returns their results
// and the new buffers. On error the new buffers go back to m.
func runInto(ctx context.Context, m *pool, k *backend.Kernel, nodes []*graph.Node, into map[*graph.Node]*graph.Node, ins ...backend.Buffer) ([]result, []backend.Buffer, error) {
	n, err := elements(k)
	if err != nil {
		return nil, nil, err
	}
	rs := make([]result, len(nodes))
	outs := make([]backend.Buffer, len(nodes))
	var fresh []backend.Buffer
	for i, x := range nodes {
		rs[i].view = resultView(k, x)
		if into[x] != nil {
			continue
		}
		buf, err := m.alloc(k.DType, n)
		if err != nil {
			for _, buf := range fresh {
				m.release(buf)
			}
			return nil, nil, err
		}
		outs[i], rs[i].buf = buf, buf
		fresh = append(fresh, buf)
	}
	for i, x := range nodes {
		if d := into[x]; d != nil {
			outs[i], rs[i].buf, rs[i].mem = d.Buf(), d.Buf(), d.Mem
			d.Mem.Version++
			d.Mem.Refs = 0
		}
	}
	if err := m.be.Run(ctx, k, outs, ins); err != nil {
		for _, buf := range fresh {
			m.release(buf)
		}
		return nil, nil, err
	}
	return rs, fresh, nil
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
