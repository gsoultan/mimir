package sched

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// Elementwise nodes are fused: an elementwise node that is not a target,
// and whose consumers are elementwise nodes computed by one kernel, is
// computed inside that kernel instead of into a buffer of its own. Each
// group of fused nodes becomes one OpFused kernel whose program reads the
// group's other inputs through their views, so a chain of elementwise ops
// reads and writes memory once, and a value the group reads several
// times, such as a scaled gradient that both of Adam's moments read, is
// computed once. Constants, through any views, become immediate values of
// the programs that read them.
//
// A target computed elementwise for elementwise consumers of one group
// joins that group too, and the group's kernel writes it as well as the
// group's own result. An optimizer step, which computes new moment
// estimates and from them a new parameter value, then reads the gradient
// and old state once and writes all three in one pass.
//
// A transpose of a product that nothing else reads is computed as the
// product of the transposed operands, (AB)ᵀ = BᵀAᵀ, so it is written in
// the order it is read. Linear layers multiply by transposed weights, and
// the weight gradients come out transposed back.
//
// A sum over batch axes of a product that nothing else reads runs as one
// matmul kernel that adds the products of each output as it computes
// them, so the products never get a buffer of their own. The weight
// gradient of a layer applied to every item of a batch, such as a
// convolution, is such a sum.
//
// A group that reads a product that nothing else reads, of a matmul or
// summed over batch axes, in row-major order (directly or through a
// reshape), runs as that product's epilogue instead: the GEMM applies the
// program to each part of its output while the part is in cache, such as
// adding a bias, a residual or applying an activation, and the product
// never gets a buffer of its own. A group that may be written over a
// value it reads, as a gradient accumulated into is (see RealizeInto),
// keeps the product out of its kernel, which would write the product
// over that value before reading it.

// maxProgram is the most instructions a fused program grows to.
const maxProgram = 256

// plan lists the kernels that compute a schedule's targets.
type plan struct {
	be     backend.Backend             // the device that runs the kernels
	order  []*graph.Node               // nodes computed by kernels, inputs first
	work   map[*graph.Node]*work       // how each is computed
	storer map[*graph.Node]*graph.Node // the node whose kernel writes each stored target
}

// work is how a planned node is computed.
type work struct {
	deps   []*graph.Node   // the nodes its kernel reads
	prog   []backend.Instr // a fused program, or gemm's epilogue
	gemm   *graph.Node     // a product its kernel computes, which prog, if any, reads as input 0
	stores []store         // the targets prog also writes
	swap   bool            // a transposed product, computed from deps transposed
}

// store is a target that a fused kernel computes for its group and writes
// too: the value of instruction at.
type store struct {
	n  *graph.Node
	at int
}

// nodes returns the nodes that the kernel of n, computed as w plans,
// writes: n, then the targets it stores.
func (w *work) nodes(n *graph.Node) []*graph.Node {
	nodes := make([]*graph.Node, 1, 1+len(w.stores))
	nodes[0] = n
	for _, s := range w.stores {
		nodes = append(nodes, s.n)
	}
	return nodes
}

// kernel returns the kernel that computes n as w plans, reading the
// results of w.deps through views, with an output for each of w.nodes(n)
// (see resultView); or nil, and the view of its input's buffer that holds
// n, when n is a view that needs no kernel. It is the one decision that
// step runs and forecast predicts.
func (w *work) kernel(n *graph.Node, views []backend.View) (*backend.Kernel, backend.View, error) {
	if w.swap {
		// (AB)ᵀ = BᵀAᵀ, from the operands B and A transposed.
		k := &backend.Kernel{Op: backend.OpMatMul, DType: n.DType, Shape: n.Shape, In: []backend.View{
			views[0].Permute(n.Attrs.Axes), views[1].Permute(n.Attrs.Axes),
		}}
		return k, backend.View{}, nil
	}
	switch n.Op {
	case backend.OpLoad:
		return nil, backend.View{}, fmt.Errorf("sched: load without data: %w", backend.ErrInternal)
	case backend.OpConst:
		// One element, broadcast to the full shape with zero strides.
		return &backend.Kernel{Op: backend.OpConst, DType: n.DType, Attrs: n.Attrs}, backend.View{}, nil
	case backend.OpPermute, backend.OpExpand, backend.OpShrink, backend.OpFlip, backend.OpReshape:
		in := views[0]
		if v, ok := viewOf(in, n); ok {
			return nil, v, nil
		}
		// Only a reshape can fail, and copying makes it contiguous.
		k := &backend.Kernel{Op: backend.OpContiguous, DType: n.DType, Shape: in.Shape, In: []backend.View{in}}
		return k, backend.View{}, nil
	}

	k := &backend.Kernel{Op: n.Op, DType: n.DType, Shape: n.Shape, In: slices.Clone(views), Attrs: n.Attrs}
	switch {
	case w.gemm != nil:
		// The product, rewritten in place by any epilogue, holds n in
		// row-major order. A summed product's axes are the batch axes it
		// sums.
		k.Op, k.Shape, k.Attrs = backend.OpMatMul, w.gemm.Shape, backend.Attrs{Program: w.prog, Axes: w.gemm.Attrs.Axes}
	case w.prog != nil:
		k.Op, k.Attrs = backend.OpFused, backend.Attrs{Program: w.prog}
		for _, s := range w.stores {
			k.Attrs.Stores = append(k.Attrs.Stores, s.at)
		}
	}
	return k, backend.View{}, nil
}

// uses returns how many times the kernels of p read each node.
func (p *plan) uses() map[*graph.Node]int {
	uses := make(map[*graph.Node]int, len(p.order))
	for _, n := range p.order {
		for _, in := range p.work[n].deps {
			uses[in]++
		}
	}
	return uses
}

// newPlan plans the kernels for the unrealized nodes in order, which lists
// every node before its consumers. over holds the storages that targets
// may be written over; see candidates.
func newPlan(order []*graph.Node, isTarget map[*graph.Node]bool, over map[*graph.Storage]bool) *plan {
	users := make(map[*graph.Node][]*graph.Node, len(order))
	for _, n := range order {
		for i, in := range n.Inputs {
			if !slices.Contains(n.Inputs[:i], in) {
				users[in] = append(users[in], n)
			}
		}
	}
	// private reports whether n is computed for its one consumer only.
	private := func(n *graph.Node) bool {
		return !n.Realized() && !isTarget[n] && len(users[n]) == 1
	}
	// joins reports whether n could be computed inside its consumers'
	// kernel: an unrealized elementwise node read by elementwise nodes
	// only.
	joins := func(n *graph.Node) bool {
		return n.Op.IsElementwise() && !n.Realized() && len(users[n]) > 0 &&
			!slices.ContainsFunc(users[n], func(u *graph.Node) bool { return !u.Op.IsElementwise() })
	}
	// A group stops growing at maxProgram instructions, counting each load
	// and constant as one, and a value read several times once per
	// reader: the inputs that would take it past are computed by kernels
	// of their own. A GPU compiles each program, and a program of a long
	// chain, such as a running sum, would be too large.
	cut := map[*graph.Node]bool{}
	size := make(map[*graph.Node]int, len(order))
	for _, n := range order {
		if !n.Op.IsElementwise() {
			continue
		}
		s := 1
		for i, in := range n.Inputs {
			switch {
			case slices.Contains(n.Inputs[:i], in):
			case !joins(in):
				s++
			case s+size[in] > maxProgram:
				cut[in] = true
				s++
			default:
				s += size[in]
			}
		}
		size[n] = s
	}
	// roots maps each node to the node whose kernel computes it: itself,
	// or the root of the group it joins, which is the root of all its
	// consumers. A target joins only a group whose kernel's output can
	// hold it, which needs the same dtype; elementwise ops keep their
	// inputs' shape. Consumers come after their inputs in order, so each
	// node finds its consumers' roots settled.
	roots := make(map[*graph.Node]*graph.Node, len(order))
	for _, n := range slices.Backward(order) {
		roots[n] = n
		if !joins(n) || cut[n] {
			continue
		}
		r := roots[users[n][0]]
		if slices.ContainsFunc(users[n], func(u *graph.Node) bool { return roots[u] != r }) {
			continue
		}
		if isTarget[n] && r.DType != n.DType {
			continue
		}
		roots[n] = r
	}
	member := func(n *graph.Node) bool {
		r, ok := roots[n]
		return ok && r != n
	}
	// stored reports whether n is a target that the kernel of its
	// consumers' group computes and writes.
	stored := func(n *graph.Node) bool { return member(n) && isTarget[n] }

	p := &plan{work: make(map[*graph.Node]*work, len(order)), storer: map[*graph.Node]*graph.Node{}}
	for _, n := range order {
		switch {
		case member(n):
		case summedProduct(n, private):
			p.work[n] = &work{deps: factors(n), gemm: n}
		case transposes(n) && n.Inputs[0].Op == backend.OpMatMul && private(n.Inputs[0]):
			mm := n.Inputs[0]
			p.work[n] = &work{deps: []*graph.Node{mm.Inputs[1], mm.Inputs[0]}, swap: true}
		case n.Op.IsElementwise():
			prog, nodes, ins := fuse(n, member)
			w := &work{deps: ins, prog: prog}
			for j, x := range nodes {
				if x != n && stored(x) {
					w.stores = append(w.stores, store{x, j})
					p.storer[x] = n
				}
			}
			// The group may be written over a value it loads elementwise;
			// see overwrites.
			overwritable := func(x *graph.Node) bool {
				return x.Realized() && over[x.Mem] && ownBuffer(x) && slices.Equal(x.Shape, n.Shape)
			}
			if w.stores == nil && !slices.ContainsFunc(ins, overwritable) {
				for i, x := range ins {
					if gemm := productOf(x, private); gemm != nil && gemm.DType == n.DType {
						w = epilogue(prog, ins, i, gemm)
						break
					}
				}
			}
			if w.gemm == nil && w.stores == nil && len(prog) == len(ins)+1 {
				// A single op on its own inputs runs as itself.
				w = &work{deps: n.Inputs}
			}
			p.work[n] = w
		default:
			p.work[n] = &work{deps: n.Inputs}
		}
	}
	// Keep the kernels the targets need: a constant read only by programs
	// needs none, nor does a product computed inside its epilogue's kernel
	// or a target its consumer's kernel writes.
	needed := make(map[*graph.Node]bool, len(order))
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		if stored(n) || !isTarget[n] && !needed[n] {
			continue
		}
		needed[n] = true
		p.order = append(p.order, n)
		for _, in := range p.work[n].deps {
			needed[in] = true
		}
	}
	slices.Reverse(p.order)
	return p
}

// fuse returns the program that computes root with the members of its
// group, the node each instruction computes, and the nodes it loads.
func fuse(root *graph.Node, member func(*graph.Node) bool) (prog []backend.Instr, nodes, ins []*graph.Node) {
	vals := map[*graph.Node]int{}
	var value func(n *graph.Node) int
	value = func(n *graph.Node) int {
		if v, ok := vals[n]; ok {
			return v
		}
		var in backend.Instr
		if c, ok := constOf(n); ok {
			in = backend.Instr{Op: backend.OpConst, DType: n.DType, Value: c}
		} else if n == root || member(n) {
			in = backend.Instr{Op: n.Op, DType: n.DType, Args: make([]int, len(n.Inputs))}
			for i, x := range n.Inputs {
				in.Args[i] = value(x)
			}
		} else {
			in = backend.Instr{Op: backend.OpLoad, DType: n.DType, Args: []int{len(ins)}}
			ins = append(ins, n)
		}
		prog, nodes = append(prog, in), append(nodes, n)
		vals[n] = len(prog) - 1
		return len(prog) - 1
	}
	value(root)
	return prog, nodes, ins
}

// epilogue returns the work of a program over ins that runs as the
// epilogue of gemm, whose product is ins[at]. The matmul reads its two
// factors, and the program the product as input 0 and the other loads
// after it.
func epilogue(prog []backend.Instr, ins []*graph.Node, at int, gemm *graph.Node) *work {
	w := &work{deps: factors(gemm), prog: slices.Clone(prog), gemm: gemm}
	renumber := make([]int, len(ins))
	for i, x := range ins {
		if i == at {
			continue
		}
		renumber[i] = len(w.deps) - 1
		w.deps = append(w.deps, x)
	}
	for i, in := range w.prog {
		if in.Op == backend.OpLoad {
			w.prog[i].Args = []int{renumber[in.Args[0]]}
		}
	}
	return w
}

// transposes reports whether n swaps the last two axes of its input and
// keeps the others in place.
func transposes(n *graph.Node) bool {
	r := len(n.Shape)
	if n.Op != backend.OpPermute || r < 2 {
		return false
	}
	for i, a := range n.Attrs.Axes {
		want := i
		switch i {
		case r - 2:
			want = r - 1
		case r - 1:
			want = r - 2
		}
		if a != want {
			return false
		}
	}
	return true
}

// summedProduct reports whether n sums a private matmul over batch axes
// only.
func summedProduct(n *graph.Node, private func(*graph.Node) bool) bool {
	if n.Op != backend.OpReduceSum {
		return false
	}
	mm := n.Inputs[0]
	if mm.Op != backend.OpMatMul || !private(mm) {
		return false
	}
	r := len(mm.Shape)
	return !slices.ContainsFunc(n.Attrs.Axes, func(a int) bool { return a >= r-2 })
}

// factors returns the two matrices whose product gemm is: the inputs of
// a matmul, or of the matmul that a summed product sums.
func factors(gemm *graph.Node) []*graph.Node {
	if gemm.Op == backend.OpReduceSum {
		gemm = gemm.Inputs[0]
	}
	return slices.Clone(gemm.Inputs)
}

// productOf returns the product that x is, of a matmul or summed over
// batch axes, directly or through views that read it in row-major order,
// if x and every node between them is private: computed for its one
// consumer only.
func productOf(x *graph.Node, private func(*graph.Node) bool) *graph.Node {
	var views []*graph.Node
	n := x
	for ; n.Op.IsView(); n = n.Inputs[0] {
		if !private(n) {
			return nil
		}
		views = append(views, n)
	}
	if n.Op != backend.OpMatMul && !summedProduct(n, private) || !private(n) {
		return nil
	}
	v := backend.Contiguous(n.Shape)
	for _, view := range slices.Backward(views) {
		var ok bool
		if v, ok = viewOf(v, view); !ok {
			return nil
		}
	}
	if !v.IsContiguous() || v.Offset != 0 || v.Size() != n.Size() {
		return nil
	}
	return n
}

// viewOf returns how the view node n reads the elements of its input,
// which v reads, or false if a view cannot express it.
func viewOf(v backend.View, n *graph.Node) (backend.View, bool) {
	switch n.Op {
	case backend.OpPermute:
		return v.Permute(n.Attrs.Axes), true
	case backend.OpExpand:
		return v.Expand(n.Shape), true
	case backend.OpShrink:
		return v.Shrink(n.Attrs.Ranges), true
	case backend.OpFlip:
		return v.Flip(n.Attrs.Axes), true
	case backend.OpReshape:
		if r, ok := v.Reshape(n.Shape); ok {
			return r, true
		}
		if isBroadcastScalar(v) {
			r := broadcast(n.Shape)
			r.Offset = v.Offset
			return r, true
		}
	}
	return backend.View{}, false
}

// constOf reports the value of n if every element of it is that constant:
// an unrealized constant, seen through any views.
func constOf(n *graph.Node) (float64, bool) {
	for !n.Realized() && n.Op.IsView() {
		n = n.Inputs[0]
	}
	if n.Realized() || n.Op != backend.OpConst {
		return 0, false
	}
	return n.Attrs.Value, true
}
