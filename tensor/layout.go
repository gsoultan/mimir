package tensor

import (
	"fmt"
	"runtime"
	"slices"
	"strings"
	"sync"
	"weak"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// The layout of a sharded tensor says which block of it each device of a
// mesh holds. Each of its axes is whole on every device or split over
// one mesh axis, possibly by sections (see Split). Between operations a
// tensor may also be partial over some mesh axes: what the devices along
// them hold must still be summed, or for a maximum maxed, to give the
// value, as the partial products of a matrix product whose contracted
// axis is split. Attention's packed output is one-dimensional and holds
// parts each laid out by its own layout, flattened.

// axisSet is a set of mesh axes: bit a for axis a.
type axisSet uint8

func (s axisSet) has(a int) bool { return s&(1<<a) != 0 }

// dimLayout is how one axis of a tensor is split.
type dimLayout struct {
	axis int8  // the mesh axis that splits it, or -1
	sec  []int // the sections each split separately, or nil for one
}

var unsplit = dimLayout{axis: -1}

func (d dimLayout) split() bool { return d.axis >= 0 }

func (d dimLayout) equal(e dimLayout) bool { return d.axis == e.axis && slices.Equal(d.sec, e.sec) }

// sections returns the lengths of the sections of an axis of length n.
func (d dimLayout) sections(n int) []int {
	if d.sec == nil {
		return []int{n}
	}
	return d.sec
}

// layout is how a tensor's elements are spread over a mesh.
type layout struct {
	dims    []dimLayout
	partial axisSet // the mesh axes over which values are still to be reduced
	max     bool    // partial values reduce by their maximum, not their sum
	// packed, if not nil, describes a one-dimensional tensor holding its
	// parts one after another, and dims is nil.
	packed []packedPart
}

// packedPart is one part of a packed tensor: the flattened elements of
// a tensor of shape, laid out as lay, from offset off of the whole.
type packedPart struct {
	off   int
	shape []int
	lay   layout
}

// replicated returns the layout of a tensor of rank r held whole on
// every device.
func replicated(r int) layout {
	l := layout{dims: make([]dimLayout, r)}
	for d := range l.dims {
		l.dims[d] = unsplit
	}
	return l
}

// used returns the mesh axes that split l's axes.
func (l layout) used() axisSet {
	var s axisSet
	for _, d := range l.dims {
		if d.split() {
			s |= 1 << d.axis
		}
	}
	return s
}

func (l layout) clone() layout {
	c := layout{dims: make([]dimLayout, len(l.dims)), partial: l.partial, max: l.max, packed: l.packed}
	copy(c.dims, l.dims)
	return c
}

// withPartial returns l with partial values over the axes of p, reduced
// by a maximum if max is set.
func (l layout) withPartial(p axisSet, max bool) layout {
	c := l.clone()
	c.partial, c.max = p, max && p != 0
	return c
}

func (l layout) equal(o layout) bool {
	if l.partial != o.partial || l.max != o.max || len(l.dims) != len(o.dims) || len(l.packed) != len(o.packed) {
		return false
	}
	for d := range l.dims {
		if !l.dims[d].equal(o.dims[d]) {
			return false
		}
	}
	for i, p := range l.packed {
		q := o.packed[i]
		if p.off != q.off || !slices.Equal(p.shape, q.shape) || !p.lay.equal(q.lay) {
			return false
		}
	}
	return true
}

// key returns a string that equal layouts share.
func (l layout) key() string {
	var b strings.Builder
	l.write(&b)
	return b.String()
}

func (l layout) write(b *strings.Builder) {
	for _, d := range l.dims {
		fmt.Fprintf(b, "%d%v,", d.axis, d.sec)
	}
	fmt.Fprintf(b, "p%d%v", l.partial, l.max)
	for _, p := range l.packed {
		fmt.Fprintf(b, "[%d%v:", p.off, p.shape)
		p.lay.write(b)
		b.WriteByte(']')
	}
}

// local returns the length of a rank's block of an axis of length n.
func (m *Mesh) local(n int, d dimLayout) int {
	if !d.split() {
		return n
	}
	return n / m.shape[d.axis]
}

// localShape returns the shape of a rank's block of a tensor of shape
// laid out as l.
func (m *Mesh) localShape(shape []int, l layout) []int {
	if l.packed != nil {
		n := 0
		for _, p := range l.packed {
			n += prod(m.localShape(p.shape, p.lay))
		}
		return []int{n}
	}
	out := make([]int, len(shape))
	for d, n := range shape {
		out[d] = m.local(n, l.dims[d])
	}
	return out
}

// constraints holds the layouts that WithSharding demands of its nodes,
// which they keep as long as they live.
var constraints constraintSet

type constraintSet struct {
	mu sync.Mutex
	m  map[weak.Pointer[graph.Node]]layout
}

func (c *constraintSet) add(n *graph.Node, l layout) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil {
		c.m = map[weak.Pointer[graph.Node]]layout{}
	}
	k := weak.Make(n)
	c.m[k] = l
	runtime.AddCleanup(n, c.remove, k)
}

func (c *constraintSet) remove(k weak.Pointer[graph.Node]) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.m, k)
}

func (c *constraintSet) get(n *graph.Node) (layout, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	l, ok := c.m[weak.Make(n)]
	return l, ok
}

// isZero reports whether n is computed from a constant zero by movement
// and casts alone: a tensor of zeros, which any device may hold as its
// share of a partial sum.
func isZero(n *graph.Node) bool {
	for range 16 {
		switch n.Op {
		case backend.OpConst:
			return n.Attrs.Value == 0
		case backend.OpReshape, backend.OpPermute, backend.OpExpand, backend.OpShrink, backend.OpFlip, backend.OpCast:
		case backend.OpPad:
			if n.Attrs.Value != 0 {
				return false
			}
		default:
			return false
		}
		if n.Realized() || len(n.Inputs) != 1 {
			return false
		}
		n = n.Inputs[0]
	}
	return false
}

// rule returns the layout of n, an operation on a mesh, computed from
// inputs laid out as in, and the layout each input must take first. The
// inputs are converted where they differ.
func (m *Mesh) rule(n *graph.Node, in []layout) (layout, []layout) {
	if l, ok := constraints.get(n); ok {
		return l, []layout{l}
	}
	ins := n.Inputs
	rank := len(n.Shape)
	for _, l := range in {
		if l.packed != nil && !(n.Op == backend.OpShrink || n.Op == backend.OpReshape) {
			return m.wholeRule(n, in)
		}
	}
	op := n.Op
	switch {
	case op == backend.OpConst || op == backend.OpRandUniform || op == backend.OpRandNormal:
		return replicated(rank), nil
	case op == backend.OpNeg || op == backend.OpCast && n.DType == ins[0].DType:
		l := in[0].clone()
		if in[0].max {
			l = l.withPartial(0, false)
		}
		return l, []layout{l}
	case op.IsUnary() || op == backend.OpGammaGrad:
		return m.elementwise(n, in, nil)
	case op == backend.OpAdd:
		return m.elementwise(n, in, func(ps []axisSet) axisSet { return sameOrZero(ins, in, ps, 0, 1) })
	case op == backend.OpMul:
		return m.elementwise(n, in, func(ps []axisSet) axisSet {
			switch {
			case ps[0] != 0 && ps[1] == 0:
				return ps[0]
			case ps[1] != 0 && ps[0] == 0:
				return ps[1]
			}
			return 0
		})
	case op == backend.OpWhere:
		return m.elementwise(n, in, func(ps []axisSet) axisSet {
			if ps[0] != 0 {
				return 0
			}
			return sameOrZero(ins, in, ps, 1, 2)
		})
	case op.IsBinary():
		return m.elementwise(n, in, nil)
	case op.IsReduce():
		return m.reduceRule(n, in[0])
	case op == backend.OpReshape:
		return m.reshapeRule(n, in[0])
	case op == backend.OpPermute:
		l := layout{dims: make([]dimLayout, rank), partial: in[0].partial, max: in[0].max}
		for i, p := range n.Attrs.Axes {
			l.dims[i] = in[0].dims[p]
		}
		return l, []layout{in[0]}
	case op == backend.OpExpand:
		l := in[0].clone()
		for d := range l.dims {
			if ins[0].Shape[d] != n.Shape[d] {
				l.dims[d] = unsplit
			}
		}
		return l, []layout{in[0]}
	case op == backend.OpPad:
		req := in[0].clone()
		for d, p := range n.Attrs.Pads {
			if p != [2]int{} {
				req.dims[d] = unsplit
			}
		}
		if n.Attrs.Value != 0 || req.max {
			req = req.withPartial(0, false)
		}
		return req, []layout{req}
	case op == backend.OpShrink:
		return m.shrinkRule(n, in[0])
	case op == backend.OpFlip:
		req := in[0].clone()
		for _, a := range n.Attrs.Axes {
			req.dims[a] = unsplit
		}
		return req, []layout{req}
	case op == backend.OpMatMul:
		return m.matmulRule(n, in[0], in[1])
	case op == backend.OpGather:
		return m.gatherRule(n, in)
	case op == backend.OpScatterAdd:
		return m.scatterRule(n, in)
	case op == backend.OpQMatMul:
		x := in[0].clone()
		x.dims[len(x.dims)-1] = unsplit
		if x.max {
			x = x.withPartial(0, false)
		}
		out := layout{dims: append(slices.Clone(x.dims[:len(x.dims)-1]), unsplit), partial: x.partial}
		return out, []layout{x, replicated(len(ins[1].Shape))}
	case op == backend.OpUnfold || op == backend.OpFold:
		s := len(n.Attrs.Kernel)
		lead := len(ins[0].Shape) - s
		if op == backend.OpFold {
			lead = rank - s
		}
		req := replicated(len(ins[0].Shape))
		copy(req.dims, in[0].dims[:lead])
		if !in[0].max && (op == backend.OpFold || n.Attrs.Value == 0) {
			req.partial = in[0].partial
		}
		out := replicated(rank)
		copy(out.dims, req.dims[:lead])
		out.partial = req.partial
		return out, []layout{req}
	case op == backend.OpScan:
		req := in[0].clone()
		req.dims[n.Attrs.Axis] = unsplit
		if req.max || n.Attrs.Combine != backend.CombineSum {
			req = req.withPartial(0, false)
		}
		return req, []layout{req}
	case op == backend.OpCat:
		return m.catRule(n, in)
	case op == backend.OpAttention || op == backend.OpAttentionGrad:
		return m.attentionRule(n, in)
	case op == backend.OpSliceScatter:
		return m.sliceScatterRule(n, in)
	case op == backend.OpArgsort:
		req := in[0].withPartial(0, false)
		req.dims[n.Attrs.Axis] = unsplit
		return req, []layout{req}
	case op == backend.OpFFT:
		req := in[0].clone()
		req.dims[n.Attrs.Axis] = unsplit
		req.dims[rank-1] = unsplit
		if req.max {
			req = req.withPartial(0, false)
		}
		return req, []layout{req}
	case op == backend.OpSegment:
		req := in[0].clone()
		req.dims[0] = unsplit
		if req.max || n.Attrs.Combine != backend.CombineSum {
			req = req.withPartial(0, false)
		}
		return req, []layout{req, replicated(1), replicated(1)}
	}
	return m.wholeRule(n, in)
}

// wholeRule computes n whole on every device from whole inputs: the rule
// of operations whose blocks need all of their inputs, such as random
// samplers, whose values depend on every element's position.
func (m *Mesh) wholeRule(n *graph.Node, in []layout) (layout, []layout) {
	req := make([]layout, len(in))
	for i := range in {
		req[i] = replicated(len(n.Inputs[i].Shape))
	}
	return replicated(len(n.Shape)), req
}

// sameOrZero returns the axes over which inputs i and j of an elementwise
// operation linear in both are partial, partial over the same axes, or
// one partial and the other zeros, or none when they are not so.
func sameOrZero(ins []*graph.Node, in []layout, ps []axisSet, i, j int) axisSet {
	switch {
	case in[i].max || in[j].max:
		return 0
	case ps[i] == ps[j]:
		return ps[i]
	case ps[j] == 0 && isZero(ins[j]):
		return ps[i]
	case ps[i] == 0 && isZero(ins[i]):
		return ps[j]
	}
	return 0
}

// elementwise is the rule of elementwise operations over inputs of one
// shape: the result is split as the inputs that are split most are, and
// the others are cut to their blocks. linear, if not nil, returns, from
// the axes over which each input is partial, those over which the
// result may be partial too: the operation is linear in its partial
// inputs, and their partial sums can be combined with the other inputs
// as they are.
func (m *Mesh) elementwise(n *graph.Node, in []layout, linear func([]axisSet) axisSet) (layout, []layout) {
	out := replicated(len(n.Shape))
	order := make([]int, len(in))
	for i := range order {
		order[i] = i
	}
	splits := func(l layout) int {
		k := 0
		for _, d := range l.dims {
			if d.split() {
				k++
			}
		}
		return k
	}
	slices.SortStableFunc(order, func(a, b int) int { return splits(in[b]) - splits(in[a]) })
	var used axisSet
	for _, i := range order {
		for d, dl := range in[i].dims {
			if dl.split() && !out.dims[d].split() && !used.has(int(dl.axis)) {
				out.dims[d] = dl
				used |= 1 << dl.axis
			}
		}
	}
	var keep axisSet
	if linear != nil {
		ps := make([]axisSet, len(in))
		for i, l := range in {
			if !l.max {
				ps[i] = l.partial
			} else if l.partial != 0 {
				linear = nil
			}
		}
		if linear != nil {
			keep = linear(ps) &^ used
		}
	}
	req := make([]layout, len(in))
	for i, l := range in {
		r := layout{dims: out.dims}
		if keep != 0 && l.partial != 0 {
			r.partial = keep
		}
		req[i] = r
	}
	out.partial = keep
	return out, req
}

func (m *Mesh) reduceRule(n *graph.Node, in layout) (layout, []layout) {
	isMax := n.Op == backend.OpReduceMax
	req := in.clone()
	if in.max != isMax {
		req = req.withPartial(0, false)
	}
	out := req.clone()
	for _, a := range n.Attrs.Axes {
		if d := in.dims[a]; d.split() {
			out.partial |= 1 << d.axis
		}
		out.dims[a] = unsplit
	}
	out.max = isMax && out.partial != 0
	return out, []layout{req}
}

// reshapeGroups returns the groups of axes of shapes a and b with equal
// numbers of elements: group i holds axes [g[0], g[1]) of a and [g[2],
// g[3]) of b.
func reshapeGroups(a, b []int) [][4]int {
	var gs [][4]int
	i, j := 0, 0
	for i < len(a) || j < len(b) {
		si, sj := i, j
		pa, pb := 1, 1
		if i < len(a) {
			pa *= a[i]
			i++
		}
		if j < len(b) {
			pb *= b[j]
			j++
		}
		for pa != pb {
			switch {
			case pa < pb && i < len(a):
				pa *= a[i]
				i++
			case pb < pa && j < len(b):
				pb *= b[j]
				j++
			default:
				return nil
			}
		}
		gs = append(gs, [4]int{si, i, sj, j})
	}
	return gs
}

// reshapeRule keeps an axis split when it leads its group of axes, so
// that each device's elements stay one contiguous run of the group's:
// the group's first axis in the result is split then, if it divides
// evenly.
func (m *Mesh) reshapeRule(n *graph.Node, in layout) (layout, []layout) {
	src, dst := n.Inputs[0].Shape, n.Shape
	if in.packed != nil {
		if len(in.packed) == 1 && slices.Equal(in.packed[0].shape, dst) {
			return in.packed[0].lay, []layout{in}
		}
		return m.wholeRule(n, []layout{in})
	}
	out := replicated(len(dst))
	out.partial, out.max = in.partial, in.max
	req := in.clone()
	groups := reshapeGroups(src, dst)
	if groups == nil || prod(src) == 0 {
		return out, []layout{req.dims0()}
	}
	for _, g := range groups {
		// The axes of the group longer than 1: an axis split stays split
		// if it is the only one on both sides, or leads the group.
		var ins, outs []int
		for d := g[0]; d < g[1]; d++ {
			if src[d] != 1 {
				ins = append(ins, d)
			}
		}
		for d := g[2]; d < g[3]; d++ {
			if dst[d] != 1 {
				outs = append(outs, d)
			}
		}
		for _, d := range ins {
			dl := in.dims[d]
			if !dl.split() {
				continue
			}
			k := m.shape[dl.axis]
			switch {
			case len(ins) == 1 && len(outs) == 1:
				out.dims[outs[0]] = dl // the axis is unchanged
			case d == ins[0] && dl.sec == nil && dst[outs[0]]%k == 0:
				out.dims[outs[0]] = dl
			default:
				req.dims[d] = unsplit
			}
		}
	}
	return out, []layout{req}
}

// dims0 returns l with every axis whole.
func (l layout) dims0() layout {
	c := l.clone()
	for d := range c.dims {
		c.dims[d] = unsplit
	}
	return c
}

func (m *Mesh) shrinkRule(n *graph.Node, in layout) (layout, []layout) {
	ranges := n.Attrs.Ranges
	src := n.Inputs[0].Shape
	if in.packed != nil {
		for _, p := range in.packed {
			if ranges[0] == [2]int{p.off, p.off + prod(p.shape)} {
				return layout{packed: []packedPart{{shape: p.shape, lay: p.lay}}}, []layout{in}
			}
		}
		return m.wholeRule(n, []layout{in})
	}
	req, out := in.clone(), in.clone()
	for d, r := range ranges {
		dl := in.dims[d]
		if r == [2]int{0, src[d]} || !dl.split() {
			continue
		}
		at := 0
		found := false
		for _, s := range dl.sec {
			if r == [2]int{at, at + s} {
				found = true
				break
			}
			at += s
		}
		if found {
			out.dims[d] = dimLayout{axis: dl.axis}
		} else {
			req.dims[d], out.dims[d] = unsplit, unsplit
		}
	}
	return out, []layout{req}
}

// matmulRule gives each mesh axis to at most one role among the batch
// axes, the rows, the columns and the contracted axis of a product. Where
// both operands split the same mesh axis in different roles, the batch
// axes win over the rows, the rows over the contracted axis and that over
// the columns, then the larger operand: activations, which carry their
// rows, keep their split and a weight split another way is gathered, as
// fully sharded data parallelism does. A split contracted axis makes the
// result partial.
func (m *Mesh) matmulRule(n *graph.Node, x, y layout) (layout, []layout) {
	xs, ys := n.Inputs[0].Shape, n.Inputs[1].Shape
	r := len(xs)
	sx, sy := prod(xs), prod(ys)
	type cand struct {
		role   int // a batch axis, r-2 for the rows, r-1 for the columns, r for the contracted axis
		dl     dimLayout
		weight int
	}
	var cands []cand
	add := func(role int, dl dimLayout, w int) {
		if dl.split() {
			cands = append(cands, cand{role, dl, w})
		}
	}
	for b := range r - 2 {
		w := sx + sy
		add(b, x.dims[b], w)
		if !y.dims[b].equal(x.dims[b]) {
			add(b, y.dims[b], w)
		}
	}
	add(r-2, x.dims[r-2], sx)
	add(r-1, y.dims[r-1], sy)
	// Both operands hold the same positions of a split contracted axis,
	// in the same order, sections or not, so each device's products sum
	// the terms of its positions.
	kx, ky := x.dims[r-1], y.dims[r-2]
	add(r, kx, sx+sy)
	if !ky.equal(kx) {
		add(r, ky, sx+sy)
	}
	rank := func(role int) int {
		switch role {
		case r - 2:
			return 1
		case r:
			return 2
		case r - 1:
			return 3
		}
		return 0
	}
	slices.SortStableFunc(cands, func(a, b cand) int {
		if ra, rb := rank(a.role), rank(b.role); ra != rb {
			return ra - rb
		}
		return b.weight - a.weight
	})
	roles := make([]dimLayout, r+1)
	for i := range roles {
		roles[i] = unsplit
	}
	var used axisSet
	for _, c := range cands {
		if roles[c.role].split() || used.has(int(c.dl.axis)) {
			continue
		}
		roles[c.role] = c.dl
		used |= 1 << c.dl.axis
	}
	rx, ry := replicated(r), replicated(r)
	out := replicated(r)
	for b := range r - 2 {
		rx.dims[b], ry.dims[b], out.dims[b] = roles[b], roles[b], roles[b]
	}
	rx.dims[r-2], out.dims[r-2] = roles[r-2], roles[r-2]
	ry.dims[r-1], out.dims[r-1] = roles[r-1], roles[r-1]
	rx.dims[r-1], ry.dims[r-2] = roles[r], roles[r]
	if k := roles[r]; k.split() {
		out.partial = 1 << k.axis
	}
	// A product is linear in each operand, so one partial operand gives
	// partial results.
	switch {
	case x.partial != 0 && y.partial == 0 && !x.max:
		keep := x.partial &^ used
		rx.partial = keep
		out.partial |= keep
	case y.partial != 0 && x.partial == 0 && !y.max:
		keep := y.partial &^ used
		ry.partial = keep
		out.partial |= keep
	}
	return out, []layout{rx, ry}
}

func (m *Mesh) gatherRule(n *graph.Node, in []layout) (layout, []layout) {
	xs, is := n.Inputs[0].Shape, n.Inputs[1].Shape
	axis := n.Attrs.Axis
	out := replicated(len(is))
	var used axisSet
	take := func(d int, dl dimLayout) {
		if dl.split() && !out.dims[d].split() && !used.has(int(dl.axis)) {
			out.dims[d] = dl
			used |= 1 << dl.axis
		}
	}
	for d := range is {
		if d == axis || is[d] == xs[d] {
			take(d, in[1].dims[d])
		}
	}
	for d := range is {
		if d != axis && is[d] == xs[d] {
			take(d, in[0].dims[d])
		}
	}
	rx, ri := replicated(len(xs)), layout{dims: out.dims}
	for d := range xs {
		if d != axis && is[d] == xs[d] {
			rx.dims[d] = out.dims[d]
		}
	}
	return out, []layout{rx, ri}
}

func (m *Mesh) scatterRule(n *graph.Node, in []layout) (layout, []layout) {
	x := n.Inputs[0]
	xs, is := x.Shape, n.Inputs[1].Shape
	axis := n.Attrs.Axis
	out := replicated(len(xs))
	src := replicated(len(is))
	var used axisSet
	for _, l := range in {
		for d, dl := range l.dims {
			if d == axis || is[d] != xs[d] || !dl.split() || out.dims[d].split() || used.has(int(dl.axis)) {
				continue
			}
			out.dims[d], src.dims[d] = dl, dl
			used |= 1 << dl.axis
		}
	}
	// Scattering into zeros from a split run of positions along the axis
	// gives each device a partial sum.
	zero := isZero(x)
	for _, l := range in[1:] {
		if dl := l.dims[axis]; zero && dl.split() && dl.sec == nil && !used.has(int(dl.axis)) {
			src.dims[axis] = dl
			out.partial = 1 << dl.axis
			used |= 1 << dl.axis
			break
		}
	}
	rx := layout{dims: out.dims}
	return out, []layout{rx, src, src}
}

func (m *Mesh) catRule(n *graph.Node, in []layout) (layout, []layout) {
	axis := n.Attrs.Axis
	out := replicated(len(n.Shape))
	var used axisSet
	// Inputs split along the axis by one mesh axis, each evenly, join
	// into a result split by sections.
	a := in[0].dims[axis]
	var sec []int
	for i, l := range in {
		dl := l.dims[axis]
		if !dl.split() || dl.axis != a.axis || dl.sec != nil {
			a = unsplit
			break
		}
		sec = append(sec, n.Inputs[i].Shape[axis])
	}
	if a.split() {
		if len(sec) == 1 {
			sec = nil
		}
		out.dims[axis] = dimLayout{axis: a.axis, sec: sec}
		used |= 1 << a.axis
	}
	for _, l := range in {
		for d, dl := range l.dims {
			if d != axis && dl.split() && !out.dims[d].split() && !used.has(int(dl.axis)) {
				out.dims[d] = dl
				used |= 1 << dl.axis
			}
		}
	}
	keep := in[0].partial &^ used
	for i, l := range in {
		if l.max || l.partial != in[0].partial && !(l.partial == 0 && isZero(n.Inputs[i])) {
			keep = 0
		}
	}
	req := make([]layout, len(in))
	for i := range in {
		r := layout{dims: slices.Clone(out.dims)}
		r.dims[axis] = dimLayout{axis: out.dims[axis].axis}
		if !out.dims[axis].split() {
			r.dims[axis] = unsplit
		}
		if keep != 0 && in[i].partial != 0 {
			r.partial = keep
		}
		req[i] = r
	}
	out.partial = keep
	return out, req
}

// attentionRule splits attention over its batch axes and heads: the
// queries' heads and the key heads they read split together.
func (m *Mesh) attentionRule(n *graph.Node, in []layout) (layout, []layout) {
	ins := n.Inputs
	q, k := ins[0].Shape, ins[1].Shape
	r := len(q)
	lead := r - 3
	dims := make([]dimLayout, r)
	for d := range dims {
		dims[d] = unsplit
	}
	var used axisSet
	for _, l := range in {
		if len(l.dims) != r {
			continue
		}
		for d := range lead {
			if dl := l.dims[d]; dl.split() && dl.sec == nil && !dims[d].split() && !used.has(int(dl.axis)) {
				dims[d] = dl
				used |= 1 << dl.axis
			}
		}
	}
	heads := unsplit
	for _, i := range []int{0, 1, 2} {
		dl := in[i].dims[r-3]
		if dl.split() && dl.sec == nil && !used.has(int(dl.axis)) && q[r-3]%m.shape[dl.axis] == 0 && k[r-3]%m.shape[dl.axis] == 0 {
			heads = dl
			break
		}
	}
	at := func(shape []int) layout {
		l := replicated(len(shape))
		copy(l.dims, dims[:lead])
		l.dims[lead] = heads
		return l
	}
	req := make([]layout, len(ins))
	for i, x := range ins {
		req[i] = at(x.Shape)
	}
	s, err := backend.NewAttentionShape(q, k, ins[2].Shape, nil)
	if err != nil {
		return m.wholeRule(n, in)
	}
	var shapes [][]int
	if n.Op == backend.OpAttention {
		shapes = [][]int{s.OutShape(), s.LSEShape()}
	} else {
		shapes = [][]int{s.QShape(), s.KShape(), s.VShape()}
	}
	out := layout{}
	off := 0
	for _, sh := range shapes {
		out.packed = append(out.packed, packedPart{off: off, shape: sh, lay: at(sh)})
		off += prod(sh)
	}
	return out, req
}

func (m *Mesh) sliceScatterRule(n *graph.Node, in []layout) (layout, []layout) {
	xs := n.Inputs[0].Shape
	out := replicated(len(xs))
	var used axisSet
	for _, l := range in {
		for d, dl := range l.dims {
			if n.Attrs.Ranges[d] == [2]int{0, xs[d]} && dl.split() && !out.dims[d].split() && !used.has(int(dl.axis)) {
				out.dims[d] = dl
				used |= 1 << dl.axis
			}
		}
	}
	var keep axisSet
	if !in[0].max && !in[1].max && in[0].partial == in[1].partial {
		keep = in[0].partial &^ used
	}
	r := layout{dims: out.dims, partial: keep}
	out.partial = keep
	return out, []layout{r, r}
}
