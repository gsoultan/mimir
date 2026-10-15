// Package sched realizes lazy graphs on a backend.
//
// The scheduler runs kernels in dependency order: one per node, except
// that chains of elementwise nodes fuse into one kernel (see newPlan). It
// turns movement ops into views of their input's buffer where the layout
// allows, and frees every intermediate buffer right after its last use.
// A target can also take over the memory of a realized node that the
// caller is done with (see RealizeInto).
package sched

import (
	"context"
	"fmt"
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
func RealizeInto(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node) error {
	if len(dsts) != len(targets) {
		return fmt.Errorf("sched: %d destinations for %d targets: %w", len(dsts), len(targets), backend.ErrInternal)
	}
	mu.Lock()
	defer mu.Unlock()
	return realize(ctx, be, targets, dsts)
}

// Release gives the memory of the realized node n to be's pool for later
// schedules and makes n stale, if n is the only node realized into that
// memory. The caller must be done with n.
func Release(be backend.Backend, n *graph.Node) {
	mu.Lock()
	defer mu.Unlock()
	if !n.Realized() || n.Stale() || n.Mem.Refs != 1 {
		return
	}
	n.Mem.Version++
	n.Mem.Refs = 0
	poolOf(be).release(n.Buf())
}

// Download realizes n and copies its elements into dst as row-major bytes.
// len(dst) must equal the tensor's size in bytes.
func Download(ctx context.Context, be backend.Backend, n *graph.Node, dst []byte) error {
	mu.Lock()
	defer mu.Unlock()
	if err := realize(ctx, be, []*graph.Node{n}, nil); err != nil {
		return err
	}
	buf, v := n.Buf(), n.View
	if !v.IsContiguous() || v.Offset != 0 || buf.Len() != n.Size() {
		m := poolOf(be)
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
	buf  backend.Buffer
	view backend.View
	mem  *graph.Storage // the storage of buf, if realized nodes read it
}

func realize(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node) error {
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
	isTarget := make(map[*graph.Node]bool, len(targets))
	for _, t := range targets {
		isTarget[t] = true
	}
	p := newPlan(order, isTarget)
	into := p.destinations(targets, dsts)
	uses := make(map[*graph.Node]int, len(p.order))
	for _, n := range p.order {
		for _, in := range p.work[n].deps {
			uses[in]++
		}
	}

	results := make(map[*graph.Node]result, len(order))
	// refs counts the live results that read each buffer allocated here, so
	// a buffer shared by views is freed only after its last reader.
	refs := make(map[backend.Buffer]int)
	lookup := func(n *graph.Node) result {
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
		nodes := []*graph.Node{n}
		for _, s := range w.stores {
			nodes = append(nodes, s.n)
		}
		err := ctx.Err()
		if err == nil && slices.ContainsFunc(nodes, func(x *graph.Node) bool { return into[x] != nil }) {
			ctx = context.WithoutCancel(ctx)
		}
		var rs []result
		var fresh []backend.Buffer
		if err == nil {
			rs, fresh, err = step(ctx, m, n, w, ins, nodes, into)
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
	mems := make(map[backend.Buffer]*graph.Storage)
	for _, t := range targets {
		if t.Realized() {
			continue
		}
		r := results[t]
		if r.mem == nil {
			if r.mem = mems[r.buf]; r.mem == nil {
				r.mem = &graph.Storage{Buf: r.buf}
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
// was reused.
func consume(targets, dsts []*graph.Node) {
	held := make(map[*graph.Storage]bool, len(targets))
	for _, t := range targets {
		held[t.Mem] = true
	}
	for _, d := range dsts {
		if d != nil && d.Realized() && !held[d.Mem] {
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
// targets its kernel stores: nodes, n first. It returns their results and
// the buffers it allocated, which hold the results of the nodes not in
// into. A node in into is written over the memory of its destination
// instead; the result of a view may also be a view of an input's buffer.
func step(ctx context.Context, m *pool, n *graph.Node, w *work, ins []result, nodes []*graph.Node, into map[*graph.Node]*graph.Node) ([]result, []backend.Buffer, error) {
	fresh := func(buf backend.Buffer, v backend.View, err error) ([]result, []backend.Buffer, error) {
		if err != nil {
			return nil, nil, err
		}
		return []result{{buf: buf, view: v}}, []backend.Buffer{buf}, nil
	}
	if w.swap {
		// (AB)ᵀ = BᵀAᵀ, from the operands B and A transposed.
		k := &backend.Kernel{Op: backend.OpMatMul, DType: n.DType, Shape: n.Shape, In: []backend.View{
			ins[0].view.Permute(n.Attrs.Axes), ins[1].view.Permute(n.Attrs.Axes),
		}}
		return runInto(ctx, m, k, nodes, into, ins[0].buf, ins[1].buf)
	}
	switch n.Op {
	case backend.OpLoad:
		return nil, nil, fmt.Errorf("sched: load without data: %w", backend.ErrInternal)
	case backend.OpConst:
		// One element, broadcast to the full shape with zero strides.
		buf, err := runNew(ctx, m, &backend.Kernel{Op: backend.OpConst, DType: n.DType, Attrs: n.Attrs})
		return fresh(buf, broadcast(n.Shape), err)
	case backend.OpPermute, backend.OpExpand, backend.OpShrink, backend.OpFlip, backend.OpReshape:
		in := ins[0]
		if v, ok := viewOf(in.view, n); ok {
			return []result{{in.buf, v, in.mem}}, nil, nil
		}
		// Only a reshape can fail, and copying makes it contiguous.
		k := &backend.Kernel{Op: backend.OpContiguous, DType: n.DType, Shape: in.view.Shape, In: []backend.View{in.view}}
		buf, err := runNew(ctx, m, k, in.buf)
		return fresh(buf, backend.Contiguous(n.Shape), err)
	}

	k := &backend.Kernel{Op: n.Op, DType: n.DType, Shape: n.Shape, In: make([]backend.View, len(ins)), Attrs: n.Attrs}
	switch {
	case w.gemm != nil:
		// The product, rewritten in place, holds n in row-major order.
		k.Op, k.Shape, k.Attrs = backend.OpMatMul, w.gemm.Shape, backend.Attrs{Program: w.prog}
	case w.prog != nil:
		k.Op, k.Attrs = backend.OpFused, backend.Attrs{Program: w.prog}
		for _, s := range w.stores {
			k.Attrs.Stores = append(k.Attrs.Stores, s.at)
		}
	}
	bufs := make([]backend.Buffer, len(ins))
	for i, in := range ins {
		k.In[i], bufs[i] = in.view, in.buf
	}
	return runInto(ctx, m, k, nodes, into, bufs...)
}

// runInto executes k, which computes nodes, each contiguous: into the
// memory of its destination in into, whose old values it makes stale, or
// into a new buffer from m. It returns their results and the new buffers.
// On error the new buffers go back to m.
func runInto(ctx context.Context, m *pool, k *backend.Kernel, nodes []*graph.Node, into map[*graph.Node]*graph.Node, ins ...backend.Buffer) ([]result, []backend.Buffer, error) {
	n, ok := backend.NumElements(k.Shape)
	if !ok {
		return nil, nil, fmt.Errorf("sched: %s: shape %v: %w", k.Op, k.Shape, backend.ErrShape)
	}
	rs := make([]result, len(nodes))
	outs := make([]backend.Buffer, len(nodes))
	var fresh []backend.Buffer
	for i, x := range nodes {
		rs[i].view = backend.Contiguous(x.Shape)
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
	n, ok := backend.NumElements(k.Shape)
	if !ok {
		return nil, fmt.Errorf("sched: %s: shape %v: %w", k.Op, k.Shape, backend.ErrShape)
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
