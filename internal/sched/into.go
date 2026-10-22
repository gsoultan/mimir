package sched

import (
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// A target written over its destination's memory replaces the values that
// realized nodes, the destination and any views of its buffer, read there.
// The kernel that writes the target bumps the storage's version first, so
// those nodes are stale from then on and a later schedule that reads them
// fails instead of computing with the new values. Within the schedule, the
// kernels that read the old values must run before that kernel, or be it:
// an elementwise kernel reads each element before writing it, so it may
// read the old values contiguously, as an optimizer's update does, and a
// slice scatter or a scatter may read them as the tensor it writes into,
// which it then leaves in place but for the elements it writes.

// destinations returns the targets that can be computed into the memory
// of their destination, each mapped to it. dsts is nil or holds the
// destination of each target, or nil.
func (p *plan) destinations(targets, dsts []*graph.Node) map[*graph.Node]*graph.Node {
	if dsts == nil {
		return nil
	}
	pos := make(map[*graph.Node]int, len(p.order))
	for i, n := range p.order {
		pos[n] = i
	}
	into := map[*graph.Node]*graph.Node{}
	taken := map[*graph.Storage]bool{}
	for i, t := range targets {
		d := dsts[i]
		if !candidate(t, d, p.be) || into[t] != nil || taken[d.Mem] {
			continue
		}
		writer := t
		if s, ok := p.storer[t]; ok {
			writer = s
		}
		if _, ok := pos[writer]; !ok || writer.Op.IsView() && !p.work[writer].swap || writer.Op == backend.OpConst {
			continue // no kernel writes t, or one of a single element
		}
		if p.overwrites(writer, d, targets, pos) {
			into[t] = d
			taken[d.Mem] = true
		}
	}
	return into
}

// candidate reports whether the target t could be computed on be into the
// memory of its destination d, as far as the two alone tell: d is
// realized on be and not stale, with t's dtype and shape, in a buffer of
// its own, and t is not computed yet.
func candidate(t, d *graph.Node, be backend.Backend) bool {
	return d != nil && d.Realized() && !d.Stale() && d.Mem.Dev == be && !t.Realized() &&
		t.DType == d.DType && slices.Equal(t.Shape, d.Shape) && ownBuffer(d)
}

// candidates returns the storages of the destinations that targets could
// be computed into, which the plan avoids reading after writing. dsts is
// nil or holds the destination of each target, or nil.
func candidates(be backend.Backend, targets, dsts []*graph.Node) map[*graph.Storage]bool {
	mems := map[*graph.Storage]bool{}
	for i, t := range targets {
		if dsts != nil && candidate(t, dsts[i], be) {
			mems[dsts[i].Mem] = true
		}
	}
	return mems
}

// ownBuffer reports whether the realized node n reads all of its buffer in
// row-major order.
func ownBuffer(n *graph.Node) bool {
	return n.View.IsContiguous() && n.View.Offset == 0 && n.Buf().Len() == n.Size()
}

// overwrites reports whether the kernel of writer can write over the
// memory of d: no target reads d's values, and no kernel reads them after
// writer, which reads them only elementwise and contiguously, or as the
// tensor a slice scatter or a scatter writes into, if at all.
func (p *plan) overwrites(writer, d *graph.Node, targets []*graph.Node, pos map[*graph.Node]int) bool {
	// reads reports whether n's values are d's values, or a view of them.
	known := map[*graph.Node]bool{}
	var reads func(n *graph.Node) bool
	reads = func(n *graph.Node) bool {
		if n.Realized() {
			return n.Mem == d.Mem
		}
		r, ok := known[n]
		if !ok {
			r = n.Op.IsView() && reads(n.Inputs[0])
			known[n] = r
		}
		return r
	}
	if slices.ContainsFunc(targets, reads) {
		return false
	}
	w := p.work[writer]
	elementwise := w.prog != nil && w.gemm == nil || w.prog == nil && !w.swap && writer.Op.IsElementwise()
	scatter := (writer.Op == backend.OpSliceScatter || writer.Op == backend.OpScatter) && w.prog == nil
	for _, n := range p.order {
		for i, in := range p.work[n].deps {
			if !reads(in) {
				continue
			}
			switch {
			case pos[n] < pos[writer]:
			case n != writer || !elementwise && !(scatter && i == 0):
				return false
			case !in.Realized() || !ownBuffer(in) || !slices.Equal(in.Shape, writer.Shape):
				return false
			}
		}
	}
	return true
}
