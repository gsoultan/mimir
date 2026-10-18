package tensor

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// A sharded tensor's graph is computed in two passes over it. The first
// gives each node a layout by the rules of layout.go, from its inputs'
// layouts, and notes the layout each input must take, and so where
// collectives are needed. Each node's depth is the number of collectives
// on the longest path to it. The second pass builds, depth by depth, one
// graph per device of the mesh from the blocks of the inputs, and runs
// the collectives that the next depth needs all at once: first the
// devices compute what later depths still read, in one schedule each,
// then the collectives run as one group. The last depth's graphs are
// computed into the targets' shards.

// plan is one computation of sharded tensors on a mesh.
type plan struct {
	m     *Mesh
	ctx   context.Context
	walk  uint64        // the plan's number, which marks the nodes it knows (graph.Node.Plan)
	pns   []*pnode      // what it knows of each node it marked, by Plan.Num
	free  []pnode       // pnodes allocated and not yet used
	order []*graph.Node // the nodes to compute, each after its inputs
	convs map[convKey]*value
	temps []*graph.Node // shards realized on the way, which go back to their pools at the end
	cuts  []*value      // the values convert left to cut, until lowerAll cuts them

	an     *analysis     // the analysis of the graph, as the mesh's cache keeps it, or nil
	leaves []*graph.Node // the graph's leaves, as its key numbers them
	rec    *recorder     // the recording of the run, if it is recorded (see replay.go)
}

// pnode is what a plan knows of a node of the graph.
type pnode struct {
	ins   []*graph.Node
	lay   layout
	req   []layout // the layout each input takes
	depth int
	last  int // the greatest depth that reads it
	leaf  bool
	val   *value // its shards, once built
	num   int    // its place in the plan's order, or for a leaf in the key (see structure)
	keyed bool   // whether a leaf has its number
}

// value is a sharded tensor that a plan built: a node per device, laid
// out as lay.
type value struct {
	shards []*graph.Node
	lay    layout
	shape  []int
	dt     DType
	// cutOf, if not nil, is the value whose blocks this one's shards cut
	// out, which a depth's lowering cuts on each device as it reaches
	// them (see lowerAll); a nil shard is yet to be cut.
	cutOf *value
}

type convKey struct {
	n   *graph.Node
	lay string
}

// plans counts the plans, which number their marks on nodes. It is
// guarded by meshMu.
var plans uint64

func newPlan(ctx context.Context, m *Mesh) *plan {
	plans++
	return &plan{m: m, ctx: ctx, walk: plans, convs: map[convKey]*value{}}
}

// node returns what p knows of n, or nil.
func (p *plan) node(n *graph.Node) *pnode {
	if n.Plan.Walk != p.walk {
		return nil
	}
	return p.pns[n.Plan.Num]
}

// seen reports whether p has marked n.
func (p *plan) seen(n *graph.Node) bool { return n.Plan.Walk == p.walk }

// set makes pn what p knows of n, marking n.
func (p *plan) set(n *graph.Node, pn *pnode) {
	if n.Plan.Walk != p.walk {
		n.Plan = graph.Mark{Walk: p.walk, Num: int32(len(p.pns))}
		p.pns = append(p.pns, pn)
		return
	}
	p.pns[n.Plan.Num] = pn
}

// pnode returns a pointer to a copy of v, allocated with others.
func (p *plan) pnode(v pnode) *pnode {
	if len(p.free) == 0 {
		p.free = make([]pnode, 256)
	}
	pn := &p.free[0]
	p.free = p.free[1:]
	*pn = v
	return pn
}

// analyze walks the graphs of targets and gives each node its layout
// and depth.
func (p *plan) analyze(targets []*graph.Node) error {
	type frame struct {
		n    *graph.Node
		next int
	}
	for _, t := range targets {
		if p.seen(t) {
			continue
		}
		stack := []frame{{n: t}}
		p.set(t, nil) // on the stack
		for len(stack) > 0 {
			f := &stack[len(stack)-1]
			if f.next == 0 {
				pn, err := p.visit(f.n)
				if err != nil {
					return err
				}
				p.set(f.n, pn)
				if pn.leaf {
					stack = stack[:len(stack)-1]
					continue
				}
			}
			pn := p.node(f.n)
			if f.next < len(pn.ins) {
				in := pn.ins[f.next]
				f.next++
				if !p.seen(in) {
					p.set(in, nil)
					stack = append(stack, frame{n: in})
				}
				continue
			}
			p.order = append(p.order, f.n)
			stack = stack[:len(stack)-1]
		}
	}
	key, leaves, keyed := p.structure()
	p.leaves = leaves
	var hash uint64
	if keyed {
		hash = sched.HashKey(key)
		if p.an = p.m.plans.lookup(hash, key); p.an != nil && !checkPlans {
			p.apply(p.an, leaves)
			return nil
		}
	}
	if err := p.rules(); err != nil {
		return err
	}
	switch {
	case p.an != nil:
		return p.same(p.an, leaves)
	case keyed:
		p.an = p.record(key, leaves)
		p.m.plans.add(hash, p.an)
	}
	return nil
}

// rules gives the nodes of p's order their layouts, the layouts their
// inputs take, and their depths, by the rules of layout.go.
func (p *plan) rules() error {
	for _, n := range p.order {
		pn := p.node(n)
		in := make([]layout, len(pn.ins))
		for i, x := range pn.ins {
			in[i] = p.node(x).lay
		}
		pn.lay, pn.req = p.m.rule(n, in)
		if len(pn.req) != len(pn.ins) {
			return fmt.Errorf("tensor: sharded %s: %d layouts for %d inputs: %w", n.Op, len(pn.req), len(pn.ins), ErrInternal)
		}
		for i, x := range pn.ins {
			d := p.node(x).depth
			if p.needsCollective(x, pn.req[i]) {
				d++
			}
			pn.depth = max(pn.depth, d)
		}
		for i, x := range pn.ins {
			if p.needsCollective(x, pn.req[i]) {
				if src, _, _, ok := p.underlying(x, pn.req[i]); ok {
					x = src
				}
			}
			px := p.node(x)
			px.last = max(px.last, pn.depth)
		}
	}
	return nil
}

// visit returns what the plan knows of n before its inputs are walked.
func (p *plan) visit(n *graph.Node) (*pnode, error) {
	if !sched.Realized(n) {
		return p.pnode(pnode{ins: sched.Inputs(n)}), nil
	}
	if n.Stale() {
		return nil, fmt.Errorf("tensor: sharded tensor of shape %v: %w", n.Shape, ErrOverwritten)
	}
	buf, ok := n.Mem.Buf.(*meshBuffer)
	if !ok {
		// Computed on a device, as a constant can be: it is copied
		// whole to every device of the mesh when it is read.
		return p.pnode(pnode{leaf: true, lay: replicated(len(n.Shape))}), nil
	}
	if buf.lay.used() != 0 && (len(buf.shards) != len(p.m.devs) || meshOf(n) != p.m) {
		return nil, fmt.Errorf("tensor: sharded tensor of another mesh: %w", ErrDevice)
	}
	for _, s := range buf.shards {
		if s.Stale() {
			return nil, fmt.Errorf("tensor: sharded tensor of shape %v: %w", n.Shape, ErrOverwritten)
		}
	}
	return p.pnode(pnode{leaf: true, lay: buf.lay, val: &value{shards: buf.shards, lay: buf.lay, shape: n.Shape, dt: n.DType}}), nil
}

// underlying returns the tensor that n permutes or broadcasts, through
// any chain of permutations and expansions, with the layout to convert it
// to in place of converting n to l, and the nodes from n down to it. The
// collectives then move the tensor before it is broadcast, such as a
// weight before a batched product repeats it for each example, and a
// weight read transposed and as it is converts once; each device then
// expands and permutes its block.
func (p *plan) underlying(n *graph.Node, l layout) (*graph.Node, layout, []*graph.Node, bool) {
	var chain []*graph.Node
	for l.packed == nil {
		pn := p.node(n)
		if pn == nil || pn.leaf || pn.lay.packed != nil || len(pn.ins) != 1 {
			break
		}
		x := pn.ins[0]
		next := l.clone()
		switch n.Op {
		case backend.OpPermute:
			for i, a := range n.Attrs.Axes {
				next.dims[a] = l.dims[i]
			}
		case backend.OpExpand:
			for d := range next.dims {
				if x.Shape[d] != n.Shape[d] {
					next.dims[d] = unsplit
				}
			}
		default:
			if len(chain) == 0 {
				return nil, layout{}, nil, false
			}
			return n, l, chain, true
		}
		chain = append(chain, n)
		n, l = x, next
	}
	if px := p.node(n); len(chain) == 0 || px == nil || px.lay.packed != nil {
		return nil, layout{}, nil, false
	}
	return n, l, chain, true
}

// needsCollective reports whether converting x to the layout to needs a
// collective: making a split axis whole, or adding up partial values.
func (p *plan) needsCollective(x *graph.Node, to layout) bool {
	return needsCollective(p.node(x).lay, to)
}

func needsCollective(from, to layout) bool {
	if from.packed != nil {
		return !from.equal(to)
	}
	if from.partial&^to.partial != 0 {
		return true
	}
	for d, dl := range from.dims {
		if dl.split() && !dl.equal(to.dims[d]) {
			return true
		}
	}
	return false
}

// final returns the layout a target computed as l is stored in: whole
// where it is packed, and with its partial values added up.
func final(n *graph.Node, l layout) layout {
	if l.packed != nil {
		return replicated(len(n.Shape))
	}
	return l.withPartial(0, false)
}

// predict returns the layout that computing n on m would store it in.
// It runs with meshMu held.
func (m *Mesh) predict(n *graph.Node) (layout, error) {
	if sched.Realized(n) {
		if buf, ok := n.Mem.Buf.(*meshBuffer); ok {
			return buf.lay, nil
		}
		return replicated(len(n.Shape)), nil
	}
	p := newPlan(context.Background(), m)
	if err := p.analyze([]*graph.Node{n}); err != nil {
		return layout{}, err
	}
	return final(n, p.node(n).lay), nil
}

// realize computes the sharded tensors targets on m, writing targets[i]
// over the memory of dsts[i] where dsts is not nil and dsts[i] is not
// nil, as RealizeInto does on a device. It runs with meshMu held.
func (m *Mesh) realize(ctx context.Context, targets, dsts []*graph.Node) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	var todo []int
	for i, t := range targets {
		if !sched.Realized(t) || dsts != nil && dsts[i] != nil {
			todo = append(todo, i)
		}
	}
	if len(todo) == 0 {
		return nil
	}
	p := newPlan(ctx, m)
	roots := make([]*graph.Node, len(todo))
	for j, i := range todo {
		roots[j] = targets[i]
	}
	if err := p.analyze(roots); err != nil {
		return err
	}
	// The layout each target is stored in: its destination's, or its
	// own with partial values added up.
	want := make([]layout, len(todo))
	maxDepth := 0
	for j, i := range todo {
		t := targets[i]
		pt := p.node(t)
		want[j] = final(t, pt.lay)
		if dsts != nil && dsts[i] != nil {
			d := dsts[i]
			buf, ok := d.Mem.Buf.(*meshBuffer)
			if !sched.Realized(d) || !ok || meshOf(d) != m {
				return fmt.Errorf("tensor: write a sharded tensor over a tensor on %v: %w", Device{sched.Device(d)}, ErrDevice)
			}
			want[j] = buf.lay
		}
		maxDepth = max(maxDepth, pt.depth)
	}
	if t := p.scripted(roots, want, todo, dsts); t != nil {
		return p.replay(t, roots, want, todo, dsts)
	}
	err := p.lower(roots, want, todo, dsts, maxDepth)
	p.stopRecording(err)
	return err
}

// lower lowers p's graph to each device's and computes roots, the
// targets todo, laid out as want, over dsts.
func (p *plan) lower(roots []*graph.Node, want []layout, todo []int, dsts []*graph.Node, maxDepth int) error {
	last := maxDepth + 1
	for _, t := range roots {
		p.node(t).last = last
	}
	byDepth := make([][]*graph.Node, maxDepth+1)
	for _, n := range p.order {
		pn := p.node(n)
		byDepth[pn.depth] = append(byDepth[pn.depth], n)
	}
	err := p.build(byDepth, func(k int) []convReq {
		if k != last {
			return nil
		}
		reqs := make([]convReq, len(roots))
		for j, t := range roots {
			reqs[j] = convReq{t, want[j]}
		}
		return reqs
	}, last)
	if err != nil {
		p.release(nil)
		return err
	}
	return p.store(roots, want, todo, dsts)
}

// convReq asks for node n in layout lay.
type convReq struct {
	n   *graph.Node
	lay layout
}

// build builds the nodes of each depth up to last, after running the
// collectives that the depth's nodes, and extra(k), need.
func (p *plan) build(byDepth [][]*graph.Node, extra func(k int) []convReq, last int) error {
	for k := 0; k <= last; k++ {
		var reqs []convReq
		if k < len(byDepth) {
			for _, n := range byDepth[k] {
				pn := p.node(n)
				if _, ok := constraint(n); ok && len(pn.ins) == 1 {
					reqs = append(reqs, convReq{pn.ins[0], pn.req[0]})
					continue
				}
				for i, x := range pn.ins {
					reqs = append(reqs, convReq{x, pn.req[i]})
				}
			}
		}
		reqs = append(reqs, extra(k)...)
		var lanes []*lane
		var expands []convReq
		seen := map[convKey]bool{}
		for _, r := range reqs {
			if !p.needsCollective(r.n, r.lay) {
				continue
			}
			key := convKey{r.n, r.lay.key()}
			if seen[key] {
				continue
			}
			if _, ok := p.convs[key]; ok {
				continue
			}
			seen[key] = true
			if src, sl, _, ok := p.underlying(r.n, r.lay); ok {
				expands = append(expands, r)
				r = convReq{src, sl}
				key = convKey{src, sl.key()}
				if _, ok := p.convs[key]; ok || seen[key] {
					continue
				}
				seen[key] = true
			}
			v, err := p.valueOf(r.n)
			if err != nil {
				return err
			}
			ln, err := p.newLane(key, v, r.lay)
			if err != nil {
				return err
			}
			lanes = append(lanes, ln)
		}
		if len(lanes) > 0 {
			if err := p.flush(k, lanes); err != nil {
				return err
			}
		}
		for _, r := range expands {
			if err := p.expand(r); err != nil {
				return err
			}
		}
		if k < len(byDepth) {
			if err := p.lowerAll(byDepth[k]); err != nil {
				return err
			}
		}
	}
	return nil
}

// expand converts r.n to r.lay by converting what it permutes or
// broadcasts, as underlying says, then expanding and permuting each
// device's block.
func (p *plan) expand(r convReq) error {
	src, sl, chain, _ := p.underlying(r.n, r.lay)
	v, err := p.input(src, sl)
	if err != nil {
		return err
	}
	for i := len(chain) - 1; i >= 0; i-- {
		n := chain[i]
		next := &value{shards: make([]*graph.Node, len(v.shards)), lay: v.lay.clone(), shape: n.Shape, dt: n.DType}
		if n.Op == backend.OpPermute {
			for d, a := range n.Attrs.Axes {
				next.lay.dims[d] = v.lay.dims[a]
			}
		}
		local := p.m.localShape(n.Shape, next.lay)
		for j, s := range v.shards {
			if n.Op == backend.OpPermute {
				next.shards[j], err = graph.Permute(s, n.Attrs.Axes)
			} else {
				next.shards[j], err = graph.Expand(s, local)
			}
			if err != nil {
				return err
			}
		}
		v = next
	}
	out, err := p.cut(v, r.lay)
	if err != nil {
		return err
	}
	p.convs[convKey{r.n, r.lay.key()}] = out
	return nil
}

// valueOf returns the shards of n, which the plan has built or which n
// holds.
func (p *plan) valueOf(n *graph.Node) (*value, error) {
	pn := p.node(n)
	if pn.val != nil {
		return pn.val, nil
	}
	if !pn.leaf {
		return nil, fmt.Errorf("tensor: sharded %s read before it is built: %w", n.Op, ErrInternal)
	}
	// A tensor on a device: copied to every device of the mesh.
	data, err := hostBytes(p.ctx, n)
	if err != nil {
		return nil, err
	}
	v := &value{shards: make([]*graph.Node, len(p.m.devs)), lay: pn.lay, shape: n.Shape, dt: n.DType}
	for r, be := range p.m.bes {
		s, err := sched.Upload(p.ctx, be, n.DType, data, n.Shape)
		if err != nil {
			return nil, err
		}
		v.shards[r] = s
	}
	if p.rec != nil {
		p.rec.upload(pn, v.shards)
	}
	pn.val = v
	return v, nil
}

// input returns the shards of n laid out as l: built by a collective
// already, or converted on each device now.
func (p *plan) input(n *graph.Node, l layout) (*value, error) {
	return p.convert(n, l, false)
}

// convert is input, which leaves the cutting of blocks to each device's
// lowering if later is set, returning a value whose shards are cut from
// another's as shard asks for them.
func (p *plan) convert(n *graph.Node, l layout, later bool) (*value, error) {
	if v := p.node(n).val; v != nil && v.lay.equal(l) {
		return v, nil
	}
	key := convKey{n, l.key()}
	if v, ok := p.convs[key]; ok {
		return v, nil
	}
	v, err := p.valueOf(n)
	if err != nil {
		return nil, err
	}
	if v.lay.equal(l) {
		return v, nil
	}
	if needsCollective(v.lay, l) {
		return nil, fmt.Errorf("tensor: sharded %s needs a collective that did not run: %w", n.Op, ErrInternal)
	}
	var out *value
	if n.Op == backend.OpConst && !sched.Realized(n) && l.packed == nil && l.partial == 0 {
		out = &value{shards: make([]*graph.Node, len(p.m.devs)), lay: l, shape: n.Shape, dt: n.DType}
		local := p.m.localShape(n.Shape, l)
		for r := range out.shards {
			if out.shards[r], err = graph.Const(n.DType, local, n.Attrs.Value); err != nil {
				return nil, err
			}
			if p.rec != nil {
				p.rec.source(r, out.shards[r], n, p.node(n).num)
			}
		}
	} else if later {
		if err := p.cuttable(v, l); err != nil {
			return nil, err
		}
		out = &value{shards: make([]*graph.Node, len(p.m.devs)), lay: l, shape: v.shape, dt: v.dt, cutOf: v}
		p.cuts = append(p.cuts, out)
	} else if out, err = p.cut(v, l); err != nil {
		return nil, err
	}
	p.convs[key] = out
	return out, nil
}

// lowerAll builds the shards of the nodes ns, each after its inputs,
// from their inputs' shards. It first gives each node, and each input it
// reads converted, a value whose shards are to be built; then each
// device builds its shards of them all, on a goroutine of its own, so
// that the devices' graphs are built at once.
func (p *plan) lowerAll(ns []*graph.Node) error {
	type step struct {
		n   *graph.Node
		ins []*value
		val *value
	}
	steps := make([]step, 0, len(ns))
	for _, n := range ns {
		pn := p.node(n)
		ins := make([]*value, len(pn.ins))
		for i, x := range pn.ins {
			v, err := p.convert(x, pn.req[i], true)
			if err != nil {
				return err
			}
			ins[i] = v
		}
		if _, ok := constraint(n); ok && len(ins) == 1 {
			pn.val = ins[0]
			continue
		}
		pn.val = &value{shards: make([]*graph.Node, len(p.m.devs)), lay: pn.lay, shape: n.Shape, dt: n.DType}
		steps = append(steps, step{n, ins, pn.val})
	}
	// Every cut is made, including those that no node of ns reads but a
	// later depth or a target does.
	cuts := p.cuts
	p.cuts = nil
	defer func() {
		for _, v := range cuts {
			v.cutOf = nil
		}
	}()
	build := func(r int) error {
		var locals []*graph.Node
		for _, s := range steps {
			locals = locals[:0]
			for _, x := range s.ins {
				b, err := p.shard(x, r)
				if err != nil {
					return err
				}
				locals = append(locals, b)
			}
			b, err := p.local(s.n, locals, s.ins, s.val.lay)
			if err != nil {
				return fmt.Errorf("tensor: sharded %s: %w", s.n.Op, err)
			}
			if p.rec != nil {
				p.rec.source(r, b, s.n, p.node(s.n).num)
			}
			s.val.shards[r] = b
		}
		for _, v := range cuts {
			if _, err := p.shard(v, r); err != nil {
				return err
			}
		}
		return nil
	}
	if len(p.m.devs) == 1 {
		return build(0)
	}
	errs := make([]error, len(p.m.devs))
	var wg sync.WaitGroup
	for r := range p.m.devs {
		wg.Go(func() { errs[r] = build(r) })
	}
	wg.Wait()
	return errors.Join(errs...)
}

// shard returns device r's shard of v, cutting it out of the shard it is
// cut from first if v is a cut whose shard is yet to be built. Only the
// goroutine that builds device r's shards calls it with r.
func (p *plan) shard(v *value, r int) (*graph.Node, error) {
	if s := v.shards[r]; s != nil || v.cutOf == nil {
		return s, nil
	}
	from, err := p.shard(v.cutOf, r)
	if err != nil {
		return nil, err
	}
	s, err := p.cutShard(from, v.cutOf.lay, v.lay, v.shape, r)
	if err != nil {
		return nil, err
	}
	v.shards[r] = s
	return s, nil
}

// local builds one device's block of n, laid out as l, from its blocks
// of the inputs, laid out as ins.
func (p *plan) local(n *graph.Node, x []*graph.Node, ins []*value, l layout) (*graph.Node, error) {
	m := p.m
	shape := m.localShape(n.Shape, l)
	a := n.Attrs
	switch op := n.Op; {
	case op == backend.OpConst:
		return graph.Const(n.DType, shape, a.Value)
	case op == backend.OpRandUniform || op == backend.OpRandNormal:
		return graph.Rand(n.Shape, a.Seed, op == backend.OpRandNormal)
	case op == backend.OpCast:
		return graph.Cast(x[0], n.DType)
	case op.IsUnary():
		return graph.Unary(op, x[0])
	case op.IsBinary():
		return graph.Binary(op, x[0], x[1])
	case op == backend.OpWhere:
		return graph.Where(x[0], x[1], x[2])
	case op.IsReduce():
		return graph.Reduce(op, x[0], a.Axes)
	case op == backend.OpReshape:
		return graph.Reshape(x[0], shape)
	case op == backend.OpPermute:
		return graph.Permute(x[0], a.Axes)
	case op == backend.OpExpand:
		return graph.Expand(x[0], shape)
	case op == backend.OpPad:
		return graph.Pad(x[0], a.Pads, a.Value)
	case op == backend.OpShrink:
		return graph.Shrink(x[0], m.localRanges(a.Ranges, ins[0], x[0].Shape))
	case op == backend.OpFlip:
		return graph.Flip(x[0], a.Axes)
	case op == backend.OpMatMul:
		return graph.MatMul(x[0], x[1])
	case op == backend.OpGather:
		return graph.Gather(x[0], x[1], a.Axis)
	case op == backend.OpScatterAdd:
		return graph.ScatterAdd(x[0], x[1], x[2], a.Axis)
	case op == backend.OpQMatMul:
		return graph.QMatMul(x[0], x[1])
	case op == backend.OpUnfold:
		return graph.Unfold(x[0], window(a), a.Value)
	case op == backend.OpFold:
		return graph.Fold(x[0], shape[len(shape)-len(a.Kernel):], window(a))
	case op == backend.OpScan:
		return graph.Scan(x[0], a.Axis, a.Combine)
	case op == backend.OpCat:
		return graph.Cat(x, a.Axis)
	case op == backend.OpAttention:
		var mask *graph.Node
		if len(x) == 4 {
			mask = x[3]
		}
		return graph.Attention(x[0], x[1], x[2], mask, attentionAttrs(a))
	case op == backend.OpAttentionGrad:
		var mask *graph.Node
		if len(x) == 7 {
			mask = x[6]
		}
		return graph.AttentionGrad(x[0], x[1], x[2], mask, x[3], x[4], x[5], attentionAttrs(a))
	case op == backend.OpSliceScatter:
		ranges := slices.Clone(a.Ranges)
		for d := range ranges {
			if l.dims[d].split() {
				ranges[d] = [2]int{0, shape[d]}
			}
		}
		return graph.SliceScatter(x[0], x[1], ranges)
	case op == backend.OpArgsort:
		return graph.Argsort(x[0], a.Axis, n.Shape[a.Axis], a.Descending)
	case op == backend.OpFFT:
		return graph.FFT(x[0], a.Axis, a.Inverse)
	case op == backend.OpCholesky || op == backend.OpLU || op == backend.OpQR || op == backend.OpSVD || op == backend.OpEigh:
		return graph.Factor(op, x[0])
	case op == backend.OpTriangularSolve:
		return graph.TriangularSolve(x[0], x[1], a.Upper, a.Unit)
	case op == backend.OpRandGamma || op == backend.OpRandPoisson:
		return graph.Sample(op, x[0], a.Seed)
	case op == backend.OpGammaGrad:
		return graph.GammaGrad(x[0], x[1])
	case op == backend.OpSegment:
		return graph.Segment(x[0], x[1], x[2], a.Combine)
	}
	return nil, fmt.Errorf("no sharded form of %s: %w", n.Op, ErrInternal)
}

func window(a backend.Attrs) graph.Window {
	return graph.Window{Kernel: a.Kernel, Stride: a.Stride, Dilation: a.Dilation, Pads: a.Pads}
}

func attentionAttrs(a backend.Attrs) graph.AttentionAttrs {
	return graph.AttentionAttrs{Scale: a.Scale, Causal: a.Causal, Offset: a.CausalOffset}
}

// localRanges returns the ranges of a shrink of a block, of shape local,
// of a tensor laid out as in.lay, by the ranges of the whole tensor.
func (m *Mesh) localRanges(ranges [][2]int, in *value, local []int) [][2]int {
	if in.lay.packed != nil {
		off := 0
		for _, part := range in.lay.packed {
			n := prod(m.localShape(part.shape, part.lay))
			if ranges[0] == [2]int{part.off, part.off + prod(part.shape)} {
				return [][2]int{{off, off + n}}
			}
			off += n
		}
		return ranges
	}
	out := slices.Clone(ranges)
	for d, dl := range in.lay.dims {
		if !dl.split() {
			continue
		}
		if ranges[d] == [2]int{0, in.shape[d]} {
			out[d] = [2]int{0, local[d]}
			continue
		}
		// A section of an axis split by sections.
		k := m.shape[dl.axis]
		at, lat := 0, 0
		for _, s := range dl.sec {
			if ranges[d] == [2]int{at, at + s} {
				out[d] = [2]int{lat, lat + s/k}
				break
			}
			at, lat = at+s, lat+s/k
		}
	}
	return out
}

// store computes the shards of the targets roots, into their
// destinations' where there are some, and makes them the targets'
// results.
func (p *plan) store(roots []*graph.Node, want []layout, todo []int, dsts []*graph.Node) error {
	m := p.m
	vals := make([]*value, len(roots))
	for j, t := range roots {
		v, err := p.input(t, want[j])
		if err != nil {
			p.release(nil)
			return err
		}
		vals[j] = v
	}
	final := make([][]*graph.Node, len(m.devs))
	for r := range m.bes {
		ts := make([]*graph.Node, len(roots))
		seen := map[*graph.Node]bool{}
		for j, v := range vals {
			s := v.shards[r]
			var d *graph.Node
			if dsts != nil {
				d = dsts[todo[j]]
			}
			// A shard that another target is, or that is computed
			// already and is to be written over a destination, is
			// copied.
			if seen[s] || d != nil && sched.Realized(s) {
				c, err := graph.Cast(s, s.DType)
				if err != nil {
					p.release(nil)
					return err
				}
				s = c
			}
			seen[s] = true
			ts[j] = s
		}
		final[r] = ts
	}
	return p.finish(roots, want, todo, dsts, func(over [][]*graph.Node) ([][]*graph.Node, error) {
		return final, p.schedule(final, over, true)
	})
}

// schedule computes the nodes of each device on it, writing them over
// the destinations over, as sched.RealizeEach does, recording the
// schedules if p's run is recorded. final says whether the nodes are the
// shards of the run's targets.
func (p *plan) schedule(nodes, over [][]*graph.Node, final bool) error {
	if p.rec == nil {
		return sched.RealizeEach(p.ctx, p.m.bes, nodes, over, lazy.values)
	}
	ev := p.rec.realize(nodes, final)
	return sched.RealizeEachTraced(p.ctx, p.m.bes, nodes, over, lazy.values, func(r int, t *sched.Trace, order, leaves []*graph.Node) {
		p.rec.traced(ev, r, t, order, leaves)
	})
}

// finish computes each device's shards of the targets roots, by compute,
// into their destinations', over, where there are some, and makes them
// the targets' results, laid out as want.
func (p *plan) finish(roots []*graph.Node, want []layout, todo []int, dsts []*graph.Node, compute func(over [][]*graph.Node) ([][]*graph.Node, error)) error {
	m := p.m
	var over [][]*graph.Node
	if dsts != nil {
		for j := range roots {
			d := dsts[todo[j]]
			if d == nil {
				continue
			}
			if over == nil {
				over = make([][]*graph.Node, len(m.bes))
				for r := range over {
					over[r] = make([]*graph.Node, len(roots))
				}
			}
			for r, s := range d.Mem.Buf.(*meshBuffer).shards {
				over[r][j] = s
			}
		}
	}
	final, err := compute(over)
	if err != nil {
		p.release(nil)
		if over != nil {
			// The destinations are stale, as on a device.
			for j := range roots {
				if d := dsts[todo[j]]; d != nil {
					sched.Invalidate(d)
				}
			}
		}
		return err
	}
	keep := map[*graph.Node]bool{}
	for j, t := range roots {
		shards := make([]*graph.Node, len(m.devs))
		for r := range shards {
			shards[r] = final[r][j]
			keep[shards[r]] = true
		}
		var over *graph.Node
		if dsts != nil {
			over = dsts[todo[j]]
		}
		sched.Bind(t, m.dev, &meshBuffer{dt: t.DType, n: t.Size(), shards: shards, lay: want[j]}, over)
	}
	p.release(keep)
	return nil
}

// release gives the shards the plan realized on the way, other than
// those of keep, back to their devices' pools.
func (p *plan) release(keep map[*graph.Node]bool) {
	for _, n := range p.temps {
		if !keep[n] {
			sched.Release(n, lazy.values)
		}
	}
	p.temps = nil
}
