package sched

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// Out describes an output of a collective on one device: its dtype and
// shape, and the input whose memory it may take, written over in place.
// An Out of dtype backend.Invalid is no output.
type Out struct {
	DType backend.DType
	Shape []int
	Over  int // the index of the input it may write over, or -1
}

// Collective computes the nodes ins[r] on the device of rank r, bes[r],
// and calls run with buffers that hold their elements contiguously and
// buffers for the outputs outs[r], for run to queue collectives over them
// (see backend.Communicator). A nil input has a nil buffer. It returns a
// realized node per output, in its buffer, and nil for an Out of no dtype.
//
// An output that may write over an input does when the input is computed
// into memory of its own that no other node and no node that keep lists
// reads, as RealizeInto writes a target over a destination: the input is
// then stale afterwards, even if run fails. Otherwise the output gets
// memory of its own and the input is left alone. Inputs that are views of
// a larger buffer are copied first.
func Collective(ctx context.Context, bes []backend.Backend, ins [][]*graph.Node, outs [][]Out, keep func() []*graph.Node, run func(ins, outs [][]backend.Buffer) error) ([][]*graph.Node, error) {
	if len(ins) != len(bes) || len(outs) != len(bes) {
		return nil, fmt.Errorf("sched: collective over %d devices of %d inputs and %d outputs: %w", len(bes), len(ins), len(outs), backend.ErrInternal)
	}
	mu.Lock()
	defer mu.Unlock()
	for r, be := range bes {
		var todo []*graph.Node
		for _, n := range ins[r] {
			switch {
			case n == nil:
			case n.Stale():
				return nil, stale(n)
			case !n.Realized():
				todo = append(todo, n)
			case n.Mem.Dev != be:
				return nil, fmt.Errorf("sched: collective input %s%v on %s, want %s: %w", n.DType, n.Shape, n.Mem.Dev.Name(), be.Name(), backend.ErrDevice)
			}
		}
		if len(todo) > 0 {
			if err := realize(ctx, be, todo, nil, keep); err != nil {
				return nil, err
			}
		}
	}
	generation++
	var kept map[*graph.Storage]bool
	if keep != nil {
		kept = keptMemory(keep())
	}

	// tmps are the copies of inputs that no output takes, which go back
	// to their pools after run.
	type tmp struct {
		m   *pool
		buf backend.Buffer
	}
	var tmps []tmp
	inBufs := make([][]backend.Buffer, len(bes))
	outBufs := make([][]backend.Buffer, len(bes))
	over := make([][]*graph.Node, len(bes)) // the input each output writes over, if any
	owned := make([][]bool, len(bes))       // whether an input's buffer is a copy
	release := func() {
		for _, t := range tmps {
			t.m.release(t.buf)
		}
	}
	for r, be := range bes {
		m := poolOf(be)
		inBufs[r] = make([]backend.Buffer, len(ins[r]))
		owned[r] = make([]bool, len(ins[r]))
		uses := map[*graph.Storage]int{} // inputs of the rank in each storage
		for _, n := range ins[r] {
			if n != nil {
				uses[n.Mem]++
			}
		}
		for i, n := range ins[r] {
			if n == nil {
				continue
			}
			if n.Mem.Dev != be {
				release()
				return nil, fmt.Errorf("sched: collective input %s%v on %s, want %s: %w", n.DType, n.Shape, n.Mem.Dev.Name(), be.Name(), backend.ErrDevice)
			}
			buf, v := n.Buf(), n.View
			if !v.IsContiguous() || v.Offset != 0 || buf.Len() != n.Size() {
				c, err := runNew(ctx, m, &backend.Kernel{Op: backend.OpContiguous, DType: n.DType, Shape: n.Shape, In: []backend.View{v}}, buf)
				if err != nil {
					release()
					return nil, err
				}
				buf, owned[r][i] = c, true
				tmps = append(tmps, tmp{m, c})
			}
			inBufs[r][i] = buf
		}
		outBufs[r] = make([]backend.Buffer, len(outs[r]))
		over[r] = make([]*graph.Node, len(outs[r]))
		for j, o := range outs[r] {
			if o.DType == backend.Invalid {
				continue
			}
			size, ok := backend.NumElements(o.Shape)
			if !ok {
				release()
				return nil, fmt.Errorf("sched: collective output of shape %v: %w", o.Shape, backend.ErrShape)
			}
			if i := o.Over; i >= 0 && i < len(ins[r]) && inBufs[r][i] != nil {
				n, buf := ins[r][i], inBufs[r][i]
				if buf.DType() == o.DType && buf.Len() == size {
					if owned[r][i] {
						// The copy is the output's: it leaves the temporaries.
						for k := range tmps {
							if tmps[k].buf == buf {
								tmps = append(tmps[:k], tmps[k+1:]...)
								break
							}
						}
						owned[r][i] = false
						outBufs[r][j] = buf
						continue
					}
					if uses[n.Mem] == 1 && writable(n, kept) && !takenAlready(over[r], n) {
						outBufs[r][j], over[r][j] = buf, n
						continue
					}
				}
			}
			buf, err := m.alloc(o.DType, size)
			if err != nil {
				release()
				return nil, err
			}
			tmps = append(tmps, tmp{m, buf}) // until run succeeds
			outBufs[r][j] = buf
		}
	}
	err := run(inBufs, outBufs)
	results := make([][]*graph.Node, len(bes))
	for r, be := range bes {
		results[r] = make([]*graph.Node, len(outs[r]))
		for j, o := range outs[r] {
			buf := outBufs[r][j]
			if buf == nil {
				continue
			}
			if n := over[r][j]; n != nil {
				// The input's memory is the output's now, written or not.
				n.Mem.Version++
				n.Mem.Refs = 0
				if err == nil {
					results[r][j] = adopt(n.Mem, o.Shape)
				}
				continue
			}
			if err != nil {
				continue
			}
			for k := range tmps {
				if tmps[k].buf == buf {
					tmps = append(tmps[:k], tmps[k+1:]...)
					break
				}
			}
			node, lerr := graph.Load(be, buf, o.Shape)
			if lerr != nil {
				err = lerr
				continue
			}
			node.Mem.Pooled = true
			results[r][j] = node
		}
	}
	release()
	if err != nil {
		return nil, err
	}
	return results, nil
}

// writable reports whether a collective may write over the memory of the
// realized node n: it holds n alone, it came from the backend, no node
// that keep lists reads it, and it is not donated to a schedule.
func writable(n *graph.Node, kept map[*graph.Storage]bool) bool {
	m := n.Mem
	return m.Pooled && m.Refs == 1 && !m.Donated && !n.Donated && !kept[m]
}

// takenAlready reports whether another output writes over n already.
func takenAlready(over []*graph.Node, n *graph.Node) bool {
	for _, o := range over {
		if o != nil && o.Mem == n.Mem {
			return true
		}
	}
	return false
}

// adopt returns a realized node of shape over all of mem, the only node
// realized into it.
func adopt(mem *graph.Storage, shape []int) *graph.Node {
	n, _ := graph.Load(mem.Dev, mem.Buf, shape)
	mem.Refs = 1
	n.Mem, n.Ver = mem, mem.Version
	return n
}
