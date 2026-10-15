package sched

import (
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// Elementwise nodes are fused: an elementwise node whose only consumer is
// another elementwise node, and which is not a target, is computed inside
// its consumer's kernel instead of into a buffer of its own. Each group of
// fused nodes becomes one OpFused kernel whose program reads the group's
// other inputs through their views, so a chain of elementwise ops reads
// and writes memory once. Constants, through any views, become immediate
// values of the programs that read them.
//
// A group that reads the product of a matmul that nothing else reads, in
// row-major order (directly or through a reshape), runs as that matmul's
// epilogue instead: the GEMM applies the program to each part of its
// output while the part is in cache, such as adding a bias, a residual or
// applying an activation, and the product never gets a buffer of its own.

// plan lists the kernels that compute a schedule's targets.
type plan struct {
	order []*graph.Node         // nodes computed by kernels, inputs first
	work  map[*graph.Node]*work // how each is computed
}

// work is how a planned node is computed.
type work struct {
	deps []*graph.Node   // the nodes its kernel reads
	prog []backend.Instr // a fused program, or gemm's epilogue
	gemm *graph.Node     // the matmul whose product prog reads as input 0
}

// newPlan plans the kernels for the unrealized nodes in order, which lists
// every node before its consumers.
func newPlan(order []*graph.Node, isTarget map[*graph.Node]bool) *plan {
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
	inner := func(n *graph.Node) bool {
		return n.Op.IsElementwise() && private(n) && users[n][0].Op.IsElementwise()
	}

	p := &plan{work: make(map[*graph.Node]*work, len(order))}
	for _, n := range order {
		switch {
		case inner(n):
		case n.Op.IsElementwise():
			prog, ins := fuse(n, inner)
			for i, x := range ins {
				if gemm := productOf(x, private); gemm != nil && gemm.DType == n.DType {
					p.work[n] = epilogue(prog, ins, i, gemm)
					break
				}
			}
			switch {
			case p.work[n] != nil:
			case len(prog) == len(ins)+1:
				// A single op on its own inputs runs as itself.
				p.work[n] = &work{deps: n.Inputs}
			default:
				p.work[n] = &work{deps: ins, prog: prog}
			}
		default:
			p.work[n] = &work{deps: n.Inputs}
		}
	}
	// Keep the kernels the targets need: a constant read only by programs
	// needs none, nor does a product computed inside its epilogue's kernel.
	needed := make(map[*graph.Node]bool, len(order))
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		if !isTarget[n] && !needed[n] {
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

// fuse returns the program that computes root, with its inner nodes, and
// the nodes it loads.
func fuse(root *graph.Node, inner func(*graph.Node) bool) ([]backend.Instr, []*graph.Node) {
	var (
		prog []backend.Instr
		ins  []*graph.Node
		vals = map[*graph.Node]int{}
	)
	var value func(n *graph.Node) int
	value = func(n *graph.Node) int {
		if v, ok := vals[n]; ok {
			return v
		}
		var in backend.Instr
		if c, ok := constOf(n); ok {
			in = backend.Instr{Op: backend.OpConst, DType: n.DType, Value: c}
		} else if n == root || inner(n) {
			in = backend.Instr{Op: n.Op, DType: n.DType, Args: make([]int, len(n.Inputs))}
			for i, x := range n.Inputs {
				in.Args[i] = value(x)
			}
		} else {
			in = backend.Instr{Op: backend.OpLoad, DType: n.DType, Args: []int{len(ins)}}
			ins = append(ins, n)
		}
		prog = append(prog, in)
		vals[n] = len(prog) - 1
		return len(prog) - 1
	}
	value(root)
	return prog, ins
}

// epilogue returns the work of a program over ins that runs as the
// epilogue of gemm, whose product is ins[at]. The matmul reads its two
// operands, and the program the product as input 0 and the other loads
// after it.
func epilogue(prog []backend.Instr, ins []*graph.Node, at int, gemm *graph.Node) *work {
	w := &work{deps: slices.Clone(gemm.Inputs), prog: slices.Clone(prog), gemm: gemm}
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

// productOf returns the matmul whose product x is, directly or through
// views that read it in row-major order, if x and every node between them
// is private: computed for its one consumer only.
func productOf(x *graph.Node, private func(*graph.Node) bool) *graph.Node {
	var views []*graph.Node
	n := x
	for ; n.Op.IsView(); n = n.Inputs[0] {
		if !private(n) {
			return nil
		}
		views = append(views, n)
	}
	if n.Op != backend.OpMatMul || !private(n) {
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
