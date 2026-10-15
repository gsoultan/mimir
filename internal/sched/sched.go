// Package sched realizes lazy graphs on a backend.
//
// This is the naive milestone-1 scheduler: it runs one kernel per node in
// dependency order, turns movement ops into views of their input's buffer
// where the layout allows, and frees every intermediate buffer right after
// its last use. Kernel fusion comes later; the contract stays the same.
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
		tmp, err := run(ctx, be, &backend.Kernel{Op: backend.OpContiguous, DType: n.DType, Shape: n.Shape, In: []backend.View{v}}, buf)
		if err != nil {
			return err
		}
		defer be.Free(tmp)
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
	isTarget := make(map[*graph.Node]bool, len(targets))
	for _, t := range targets {
		isTarget[t] = true
	}
	uses := make(map[*graph.Node]int, len(order))
	for _, n := range order {
		for _, in := range n.Inputs {
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
				be.Free(r.buf)
			}
		}
	}

	for _, n := range order {
		ins := make([]result, len(n.Inputs))
		for i, in := range n.Inputs {
			ins[i] = lookup(in)
		}
		r, fresh, err := step(ctx, be, n, ins)
		if err != nil {
			for buf := range refs {
				be.Free(buf)
			}
			if fresh != nil {
				be.Free(fresh)
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
		for _, in := range n.Inputs {
			release(in)
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

// step computes one node. fresh is the buffer it allocated, if any; the
// result may also be a view of an input's buffer.
func step(ctx context.Context, be backend.Backend, n *graph.Node, ins []result) (r result, fresh backend.Buffer, err error) {
	if err := ctx.Err(); err != nil {
		return result{}, nil, err
	}
	switch n.Op {
	case backend.OpLoad:
		return result{}, nil, fmt.Errorf("sched: load without data: %w", backend.ErrInternal)
	case backend.OpConst:
		// One element, broadcast to the full shape with zero strides.
		buf, err := run(ctx, be, &backend.Kernel{Op: backend.OpConst, DType: n.DType, Attrs: n.Attrs})
		return result{buf, broadcast(n.Shape)}, buf, err
	case backend.OpPermute:
		return result{ins[0].buf, ins[0].view.Permute(n.Attrs.Axes)}, nil, nil
	case backend.OpExpand:
		return result{ins[0].buf, ins[0].view.Expand(n.Shape)}, nil, nil
	case backend.OpShrink:
		return result{ins[0].buf, ins[0].view.Shrink(n.Attrs.Ranges)}, nil, nil
	case backend.OpFlip:
		return result{ins[0].buf, ins[0].view.Flip(n.Attrs.Axes)}, nil, nil
	case backend.OpReshape:
		in := ins[0]
		if v, ok := in.view.Reshape(n.Shape); ok {
			return result{in.buf, v}, nil, nil
		}
		if isBroadcastScalar(in.view) {
			v := broadcast(n.Shape)
			v.Offset = in.view.Offset
			return result{in.buf, v}, nil, nil
		}
		k := &backend.Kernel{Op: backend.OpContiguous, DType: n.DType, Shape: in.view.Shape, In: []backend.View{in.view}}
		buf, err := run(ctx, be, k, in.buf)
		return result{buf, backend.Contiguous(n.Shape)}, buf, err
	}

	k := &backend.Kernel{Op: n.Op, DType: n.DType, Shape: n.Shape, In: make([]backend.View, len(ins)), Attrs: n.Attrs}
	bufs := make([]backend.Buffer, len(ins))
	for i, in := range ins {
		k.In[i], bufs[i] = in.view, in.buf
	}
	buf, err := run(ctx, be, k, bufs...)
	return result{buf, backend.Contiguous(n.Shape)}, buf, err
}

// run allocates the output of k and executes it. On error the output is
// freed and nil is returned.
func run(ctx context.Context, be backend.Backend, k *backend.Kernel, ins ...backend.Buffer) (backend.Buffer, error) {
	n, ok := backend.NumElements(k.Shape)
	if !ok {
		return nil, fmt.Errorf("sched: %s: shape %v: %w", k.Op, k.Shape, backend.ErrShape)
	}
	out, err := be.Alloc(k.DType, n)
	if err != nil {
		return nil, err
	}
	if err := be.Run(ctx, k, out, ins); err != nil {
		be.Free(out)
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
