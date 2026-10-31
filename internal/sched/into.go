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
	r := p.readers(targets)
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
		if p.overwrites(writer, d, r, pos) {
			into[t] = d
			taken[d.Mem] = true
		}
	}
	return into
}

// reader is a kernel input whose values are a storage's values, or a view
// of them: input i of the kernel of n.
type reader struct {
	n  *graph.Node
	i  int
	in *graph.Node
}

// readers holds, for each storage, the kernel inputs that read it, and
// whether a target reads it.
type readers struct {
	kernels map[*graph.Storage][]reader
	targets map[*graph.Storage]bool
}

// readers maps each storage to what reads it in the plan, in one pass, so
// that checking each destination does not walk the whole plan again.
func (p *plan) readers(targets []*graph.Node) readers {
	mem := map[*graph.Node]*graph.Storage{}
	var memOf func(n *graph.Node) *graph.Storage
	memOf = func(n *graph.Node) *graph.Storage {
		if n.Realized() {
			return n.Mem
		}
		m, ok := mem[n]
		if !ok {
			if n.Op.IsView() {
				m = memOf(n.Inputs[0])
			}
			mem[n] = m
		}
		return m
	}
	r := readers{kernels: map[*graph.Storage][]reader{}, targets: map[*graph.Storage]bool{}}
	for _, t := range targets {
		if m := memOf(t); m != nil {
			r.targets[m] = true
		}
	}
	for _, n := range p.order {
		for i, in := range p.work[n].deps {
			if m := memOf(in); m != nil {
				r.kernels[m] = append(r.kernels[m], reader{n, i, in})
			}
		}
	}
	return r
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
func (p *plan) overwrites(writer, d *graph.Node, r readers, pos map[*graph.Node]int) bool {
	if r.targets[d.Mem] {
		return false
	}
	w := p.work[writer]
	elementwise := w.prog != nil && w.gemm == nil || w.prog == nil && !w.swap && writer.Op.IsElementwise()
	scatter := (writer.Op == backend.OpSliceScatter || writer.Op == backend.OpScatter) && w.prog == nil
	for _, k := range r.kernels[d.Mem] {
		n, i, in := k.n, k.i, k.in
		switch {
		case pos[n] < pos[writer]:
		case n != writer || !elementwise && !(scatter && i == 0):
			return false
		case !in.Realized() || !ownBuffer(in) || !slices.Equal(in.Shape, writer.Shape):
			return false
		}
	}
	return true
}
