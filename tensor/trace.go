package tensor

import (
	"fmt"
	"slices"
	"sync/atomic"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// tracing counts the calls of Trace running their functions, during which
// operations are lazy in eager mode too: eager mode would compute the
// operations on the arguments, which have no values.
var tracing atomic.Int32

// TracedGraph is the graph of primitive operations that a function
// computes, as Trace records it, for exporters that translate it into
// another format.
type TracedGraph struct {
	// Nodes in an order in which each comes after its inputs.
	Nodes []TracedNode
	// Outputs holds the node of each of the function's results.
	Outputs []int
}

// TracedNode is one operation of a TracedGraph.
type TracedNode struct {
	Op    backend.Op
	DType DType
	Shape []int
	Attrs backend.Attrs
	// Inputs holds the nodes of the operation's inputs.
	Inputs []int
	// Arg is the argument of the function that the node stands for, or
	// -1. An argument's node has no inputs and Op OpInvalid.
	Arg int
	// Value is the node's tensor when no argument affects it, and nil
	// otherwise. A computed tensor, such as a parameter's value, is a
	// node of Op OpLoad without inputs.
	Value *Tensor
	// Name is the name that Trace was given for Value, or "".
	Name string
}

// Trace runs f once on tensors that stand for args, with their dtypes
// and shapes, and records the primitive operations that compute its
// results from them and from the tensors f reads, such as parameters.
// The tensors f reads are recorded as they are: a computed tensor is a
// node without inputs whose Value holds it. names labels some of them,
// such as the parameters of a model by their paths.
//
// f must not read the values of its arguments, as Data, Item, Realize and
// Backward do: they have none, and doing so fails. The operations f
// builds are lazy even in eager mode, which would compute them as f
// builds them.
func Trace(f func(x ...*Tensor) []*Tensor, args []*Tensor, names map[string]*Tensor) (*TracedGraph, error) {
	if err := firstErr(args...); err != nil {
		return nil, err
	}
	stand := make([]*Tensor, len(args))
	argOf := map[*graph.Node]int{}
	for i, a := range args {
		n := &graph.Node{Op: backend.OpInvalid, DType: a.node.DType, Shape: slices.Clone(a.node.Shape), Dev: sched.Device(a.node)}
		stand[i] = &Tensor{node: n}
		argOf[n] = i
	}
	ys := func() []*Tensor {
		tracing.Add(1)
		defer tracing.Add(-1)
		return f(stand...)
	}()
	if err := firstErr(ys...); err != nil {
		return nil, err
	}
	nameOf := map[*graph.Node]string{}
	for name, t := range names {
		if t == nil || t.err != nil {
			continue
		}
		if prev, ok := nameOf[t.node]; !ok || name < prev {
			nameOf[t.node] = name
		}
	}

	tg := &TracedGraph{}
	index := map[*graph.Node]int{}
	// Nodes are visited depth first, without recursion, since a long
	// chain of operations would overflow the stack.
	type frame struct {
		n    *graph.Node
		ins  []*graph.Node
		next int
	}
	for _, y := range ys {
		stack := []frame{}
		push := func(n *graph.Node) {
			if _, ok := index[n]; ok {
				return
			}
			var ins []*graph.Node
			if _, isArg := argOf[n]; !isArg && !sched.Realized(n) {
				ins = n.Inputs
			}
			stack = append(stack, frame{n: n, ins: ins})
		}
		push(y.node)
		for len(stack) > 0 {
			top := &stack[len(stack)-1]
			if top.next < len(top.ins) {
				in := top.ins[top.next]
				top.next++
				if _, ok := index[in]; !ok {
					for _, f := range stack {
						if f.n == in {
							return nil, fmt.Errorf("tensor: Trace of a cyclic graph: %w", ErrInternal)
						}
					}
					push(in)
				}
				continue
			}
			stack = stack[:len(stack)-1]
			if _, ok := index[top.n]; ok {
				continue
			}
			index[top.n] = len(tg.Nodes)
			tg.Nodes = append(tg.Nodes, tg.node(top.n, top.ins, index, argOf, nameOf))
		}
		tg.Outputs = append(tg.Outputs, index[y.node])
	}
	return tg, nil
}

// node returns the traced node of n, whose inputs ins are traced.
func (tg *TracedGraph) node(n *graph.Node, ins []*graph.Node, index map[*graph.Node]int, argOf map[*graph.Node]int, nameOf map[*graph.Node]string) TracedNode {
	tn := TracedNode{DType: n.DType, Shape: slices.Clone(n.Shape), Arg: -1, Name: nameOf[n]}
	if a, ok := argOf[n]; ok {
		tn.Op, tn.Arg = backend.OpInvalid, a
		return tn
	}
	if len(ins) == 0 && sched.Realized(n) {
		tn.Op = backend.OpLoad
	} else {
		tn.Op, tn.Attrs = n.Op, n.Attrs
	}
	free := true
	for _, in := range ins {
		i := index[in]
		tn.Inputs = append(tn.Inputs, i)
		free = free && tg.Nodes[i].Value != nil
	}
	if free {
		tn.Value = &Tensor{node: n}
	}
	return tn
}
