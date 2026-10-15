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

// plan lists the kernels that compute a schedule's targets.
type plan struct {
	order []*graph.Node                   // nodes computed by kernels, inputs first
	deps  map[*graph.Node][]*graph.Node   // the nodes each kernel reads
	progs map[*graph.Node][]backend.Instr // the programs of fused kernels
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
	inner := func(n *graph.Node) bool {
		u := users[n]
		return n.Op.IsElementwise() && !n.Realized() && !isTarget[n] && len(u) == 1 && u[0].Op.IsElementwise()
	}

	p := &plan{
		deps:  make(map[*graph.Node][]*graph.Node, len(order)),
		progs: make(map[*graph.Node][]backend.Instr),
	}
	for _, n := range order {
		switch {
		case inner(n):
		case n.Op.IsElementwise():
			prog, ins := fuse(n, inner)
			if prog == nil {
				p.deps[n] = n.Inputs
				break
			}
			p.deps[n], p.progs[n] = ins, prog
		default:
			p.deps[n] = n.Inputs
		}
	}
	// Keep the kernels the targets need: a constant read only by programs
	// needs none.
	needed := make(map[*graph.Node]bool, len(order))
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		if !isTarget[n] && !needed[n] {
			continue
		}
		needed[n] = true
		p.order = append(p.order, n)
		for _, in := range p.deps[n] {
			needed[in] = true
		}
	}
	for i, j := 0, len(p.order)-1; i < j; i, j = i+1, j-1 {
		p.order[i], p.order[j] = p.order[j], p.order[i]
	}
	return p
}

// fuse returns the program that computes root and the nodes it loads, or
// nil if root is a single op on unfused inputs, which runs as itself.
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
	if len(prog) == len(ins)+1 {
		return nil, nil
	}
	return prog, ins
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
