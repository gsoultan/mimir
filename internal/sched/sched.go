// Package sched realizes lazy graphs on a backend.
//
// The scheduler runs kernels in dependency order: one per node, except
// that chains of elementwise nodes fuse into one kernel (see newPlan). It
// turns movement ops into views of their input's buffer where the layout
// allows, and frees every intermediate buffer right after its last use.
package sched

import (
	"context"
	"fmt"
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
	return realize(ctx, be, targets)
}

// Download realizes n and copies its elements into dst as row-major bytes.
// len(dst) must equal the tensor's size in bytes.
func Download(ctx context.Context, be backend.Backend, n *graph.Node, dst []byte) error {
	mu.Lock()
	defer mu.Unlock()
	if err := realize(ctx, be, []*graph.Node{n}); err != nil {
		return err
	}
	buf, v := n.Buf, n.View
	if !v.IsContiguous() || v.Offset != 0 || buf.Len() != n.Size() {
		m := poolOf(be)
		tmp, err := run(ctx, m, &backend.Kernel{Op: backend.OpContiguous, DType: n.DType, Shape: n.Shape, In: []backend.View{v}}, buf)
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
}

func realize(ctx context.Context, be backend.Backend, targets []*graph.Node) error {
	order := topoSort(targets)
	if len(order) == 0 {
		return nil
	}
	m := poolOf(be)
	defer m.tick()
	isTarget := make(map[*graph.Node]bool, len(targets))
	for _, t := range targets {
		isTarget[t] = true
	}
	p := newPlan(order, isTarget)
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
			return result{n.Buf, n.View}
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
		r, fresh, err := step(ctx, m, n, w, ins)
		if err != nil {
			for buf := range refs {
				m.release(buf)
			}
			if fresh != nil {
				m.release(fresh)
			}
			return err
		}
		if fresh != nil {
			refs[fresh] = 0
		}
		if _, owned := refs[r.buf]; owned {
			refs[r.buf]++
		}
		results[n] = r
		// The targets the kernel also wrote follow its result in r.buf.
		for i, s := range w.stores {
			v := backend.Contiguous(s.n.Shape)
			v.Offset = (i + 1) * n.Size()
			results[s.n] = result{r.buf, v}
			refs[r.buf]++
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
	for _, t := range targets {
		if t.Realized() {
			continue
		}
		r := results[t]
		t.Buf, t.View = r.buf, r.view
		t.Op, t.Attrs, t.Inputs = backend.OpLoad, backend.Attrs{}, nil
	}
	return nil
}

// topoSort returns the unrealized nodes reachable from targets, each after
// its inputs. It walks iteratively so deep graphs cannot overflow the stack.
func topoSort(targets []*graph.Node) []*graph.Node {
	type frame struct {
		n    *graph.Node
		next int
	}
	var order []*graph.Node
	seen := make(map[*graph.Node]bool)
	for _, t := range targets {
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
				if !in.Realized() && !seen[in] {
					seen[in] = true
					stack = append(stack, frame{n: in})
				}
				continue
			}
			order = append(order, f.n)
			stack = stack[:len(stack)-1]
		}
	}
	return order
}

// step computes one node as w plans, from the results of w.deps. fresh is
// the buffer it allocated, if any; the result may also be a view of an
// input's buffer.
func step(ctx context.Context, m *pool, n *graph.Node, w *work, ins []result) (r result, fresh backend.Buffer, err error) {
	if err := ctx.Err(); err != nil {
		return result{}, nil, err
	}
	if w.swap {
		// (AB)ᵀ = BᵀAᵀ, from the operands B and A transposed.
		k := &backend.Kernel{Op: backend.OpMatMul, DType: n.DType, Shape: n.Shape, In: []backend.View{
			ins[0].view.Permute(n.Attrs.Axes), ins[1].view.Permute(n.Attrs.Axes),
		}}
		buf, err := run(ctx, m, k, ins[0].buf, ins[1].buf)
		return result{buf, backend.Contiguous(n.Shape)}, buf, err
	}
	switch n.Op {
	case backend.OpLoad:
		return result{}, nil, fmt.Errorf("sched: load without data: %w", backend.ErrInternal)
	case backend.OpConst:
		// One element, broadcast to the full shape with zero strides.
		buf, err := run(ctx, m, &backend.Kernel{Op: backend.OpConst, DType: n.DType, Attrs: n.Attrs})
		return result{buf, broadcast(n.Shape)}, buf, err
	case backend.OpPermute, backend.OpExpand, backend.OpShrink, backend.OpFlip, backend.OpReshape:
		in := ins[0]
		if v, ok := viewOf(in.view, n); ok {
			return result{in.buf, v}, nil, nil
		}
		// Only a reshape can fail, and copying makes it contiguous.
		k := &backend.Kernel{Op: backend.OpContiguous, DType: n.DType, Shape: in.view.Shape, In: []backend.View{in.view}}
		buf, err := run(ctx, m, k, in.buf)
		return result{buf, backend.Contiguous(n.Shape)}, buf, err
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
	buf, err := run(ctx, m, k, bufs...)
	return result{buf, backend.Contiguous(n.Shape)}, buf, err
}

// run takes the output of k from m and executes it. On error the output
// goes back to m and nil is returned.
func run(ctx context.Context, m *pool, k *backend.Kernel, ins ...backend.Buffer) (backend.Buffer, error) {
	n, ok := k.OutputLen()
	if !ok {
		return nil, fmt.Errorf("sched: %s: shape %v: %w", k.Op, k.Shape, backend.ErrShape)
	}
	out, err := m.alloc(k.DType, n)
	if err != nil {
		return nil, err
	}
	if err := m.be.Run(ctx, k, out, ins); err != nil {
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
