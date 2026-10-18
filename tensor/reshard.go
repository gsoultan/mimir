package tensor

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// Converting a sharded tensor from one layout to another takes up to three
// collectives, in rounds that every conversion of a depth shares: partial
// values are added up, by an all-reduce over the axes they are partial
// over or a reduce-scatter when an axis is to be split over the mesh axis
// the values are partial over; then one all-gather makes whole every axis
// that is split otherwise than wanted. Last, each device cuts its block
// of the axes that are to be split and are whole, which needs no
// collective. A packed tensor is cut into its parts, each made whole, and
// joined again.

// lane is one conversion in progress.
type lane struct {
	key   convKey
	cur   *value
	to    layout
	parts []*lane // the parts of a packed value, each made whole
}

func (p *plan) newLane(key convKey, v *value, to layout) (*lane, error) {
	ln := &lane{key: key, cur: v, to: to}
	if v.lay.packed == nil {
		return ln, nil
	}
	off := 0
	for _, part := range v.lay.packed {
		local := p.m.localShape(part.shape, part.lay)
		size := prod(local)
		pv := &value{shards: make([]*graph.Node, len(v.shards)), lay: part.lay, shape: part.shape, dt: v.dt}
		for r, s := range v.shards {
			x, err := graph.Shrink(s, [][2]int{{off, off + size}})
			if err == nil {
				x, err = graph.Reshape(x, local)
			}
			if err != nil {
				return nil, err
			}
			pv.shards[r] = x
		}
		off += size
		ln.parts = append(ln.parts, &lane{cur: pv, to: replicated(len(part.shape))})
	}
	return ln, nil
}

// job is one collective of a round: a kind of collective over the groups
// of the mesh axes of axes, on each device's input in, giving outputs of
// local shape out, from which done makes the converted value.
type job struct {
	kind backend.CollectiveKind
	op   backend.Reduction
	axes axisSet
	in   []*graph.Node
	out  []int
	dt   DType
	done func(out []*graph.Node) (*value, error)
}

// flush runs the collectives of lanes, which the nodes of depth k read,
// after computing the shards that depth k and later read.
func (p *plan) flush(k int, lanes []*lane) error {
	if err := p.realizeLive(k); err != nil {
		return err
	}
	for {
		var jobs []*job
		var owners []*lane
		for _, ln := range lanes {
			for _, l := range append([]*lane{ln}, ln.parts...) {
				if l.parts != nil {
					continue
				}
				j, err := p.next(l.cur, l.to)
				if err != nil {
					return err
				}
				if j != nil {
					jobs, owners = append(jobs, j), append(owners, l)
				}
			}
		}
		if len(jobs) == 0 {
			break
		}
		vals, err := p.run(jobs)
		if err != nil {
			return err
		}
		for i, l := range owners {
			l.cur = vals[i]
		}
	}
	for _, ln := range lanes {
		v := ln.cur
		if ln.parts != nil {
			var err error
			if v, err = p.join(ln); err != nil {
				return err
			}
		}
		out, err := p.cut(v, ln.to)
		if err != nil {
			return err
		}
		p.convs[ln.key] = out
	}
	return nil
}

// realizeLive computes, on each device in one schedule, the shards built
// before depth k that depth k or later reads, so that no device computes
// them twice.
func (p *plan) realizeLive(k int) error {
	live := make([][]*graph.Node, len(p.m.devs))
	seen := map[*graph.Node]bool{}
	for _, n := range p.order {
		pn := p.node(n)
		if pn.val == nil || pn.depth >= k || pn.last < k {
			continue
		}
		for r, s := range pn.val.shards {
			if seen[s] || sched.Device(s) == nil || sched.Realized(s) {
				continue
			}
			seen[s] = true
			live[r] = append(live[r], s)
		}
	}
	if p.rec != nil {
		p.rec.realize(live)
	}
	err := sched.RealizeEach(p.ctx, p.m.bes, live, nil, lazy.values)
	for _, ns := range live {
		for _, n := range ns {
			if sched.Realized(n) {
				p.temps = append(p.temps, n)
			}
		}
	}
	return err
}

// next returns the next collective that converting v to the layout to
// takes, or nil if cutting blocks is all that is left.
func (p *plan) next(v *value, to layout) (*job, error) {
	l := v.lay
	if l.packed != nil {
		if !l.equal(to) {
			return nil, fmt.Errorf("tensor: convert a packed sharded tensor: %w", ErrInternal)
		}
		return nil, nil
	}
	if r := l.partial &^ to.partial; r != 0 {
		for d, t := range to.dims {
			if t.split() && r.has(int(t.axis)) && !l.dims[d].split() {
				if rest := r &^ (1 << t.axis); rest != 0 {
					return p.allReduce(v, rest), nil
				}
				return p.reduceScatter(v, d, t)
			}
		}
		return p.allReduce(v, r), nil
	}
	var gather axisSet
	var dims []int
	for d, dl := range l.dims {
		if dl.split() && !dl.equal(to.dims[d]) {
			gather |= 1 << dl.axis
			dims = append(dims, d)
		}
	}
	if gather != 0 {
		return p.allGather(v, gather, dims), nil
	}
	return nil, nil
}

func (v *value) with(shards []*graph.Node, l layout) *value {
	return &value{shards: shards, lay: l, shape: v.shape, dt: v.dt}
}

func (p *plan) reduction(l layout) backend.Reduction {
	if l.max {
		return backend.ReduceMax
	}
	return backend.ReduceSum
}

func (p *plan) allReduce(v *value, axes axisSet) *job {
	l := v.lay.withPartial(v.lay.partial&^axes, v.lay.max)
	return &job{
		kind: backend.AllReduce, op: p.reduction(v.lay), axes: axes,
		in: v.shards, out: p.m.localShape(v.shape, v.lay), dt: v.dt,
		done: func(out []*graph.Node) (*value, error) { return v.with(out, l), nil },
	}
}

// front returns the permutation that moves axis d of r to the front.
func front(d, r int) []int {
	perm := []int{d}
	for i := range r {
		if i != d {
			perm = append(perm, i)
		}
	}
	return perm
}

// inverse returns the inverse of the permutation perm.
func inverse(perm []int) []int {
	inv := make([]int, len(perm))
	for i, p := range perm {
		inv[p] = i
	}
	return inv
}

// reduceScatter adds up v's partial values over the mesh axis of t and
// leaves each device the block of axis d that t gives it.
func (p *plan) reduceScatter(v *value, d int, t dimLayout) (*job, error) {
	m := p.m
	k := m.shape[t.axis]
	local := m.localShape(v.shape, v.lay)
	perm := front(d, len(local))
	in := make([]*graph.Node, len(v.shards))
	for r, s := range v.shards {
		x := s
		if t.sec != nil {
			// Each device's blocks of the sections, one after another.
			var parts []*graph.Node
			for q := range k {
				at := 0
				for _, n := range t.sec {
					b := n / k
					y, err := graph.Shrink(x, rangeAxis(local, d, at+q*b, at+(q+1)*b))
					if err != nil {
						return nil, err
					}
					parts = append(parts, y)
					at += n
				}
			}
			var err error
			if x, err = catAll(parts, d); err != nil {
				return nil, err
			}
		}
		if d != 0 {
			var err error
			if x, err = graph.Permute(x, perm); err != nil {
				return nil, err
			}
		}
		in[r] = x
	}
	out := make([]int, len(local))
	for i, a := range perm {
		out[i] = local[a]
	}
	out[0] /= k
	l := v.lay.withPartial(v.lay.partial&^(1<<t.axis), v.lay.max)
	l.dims[d] = t
	return &job{
		kind: backend.ReduceScatter, op: p.reduction(v.lay), axes: 1 << t.axis,
		in: in, out: out, dt: v.dt,
		done: func(out []*graph.Node) (*value, error) {
			if d == 0 {
				return v.with(out, l), nil
			}
			shards := make([]*graph.Node, len(out))
			for r, o := range out {
				var e error
				if shards[r], e = graph.Permute(o, inverse(perm)); e != nil {
					return nil, e
				}
			}
			return v.with(shards, l), nil
		},
	}, nil
}

// allGather makes the axes dims of v, split over the mesh axes of axes,
// whole: each device receives every device's block in the order of their
// coordinates along those axes, and puts them in place.
func (p *plan) allGather(v *value, axes axisSet, dims []int) *job {
	m := p.m
	local := m.localShape(v.shape, v.lay)
	var ga []int
	var sizes []int
	for a := range m.shape {
		if axes.has(a) {
			ga, sizes = append(ga, a), append(sizes, m.shape[a])
		}
	}
	l := v.lay.clone()
	for _, d := range dims {
		l.dims[d] = unsplit
	}
	return &job{
		kind: backend.AllGather, axes: axes,
		in: v.shards, out: slices.Concat(sizes, local), dt: v.dt,
		done: func(out []*graph.Node) (*value, error) {
			shards := make([]*graph.Node, len(out))
			for r, o := range out {
				x, err := p.regroup(o, ga, v.lay, dims, v.shape)
				if err != nil {
					return nil, err
				}
				shards[r] = x
			}
			return v.with(shards, l), nil
		},
	}
}

// regroup arranges y, the blocks of an all-gather over the mesh axes ga,
// [blocks along each of ga..., block...], as the tensor of shape whose
// axes dims, laid out as l, it makes whole.
func (p *plan) regroup(y *graph.Node, ga []int, l layout, dims []int, shape []int) (*graph.Node, error) {
	nb := len(ga)
	var perm []int
	for d := range shape {
		if slices.Contains(dims, d) {
			perm = append(perm, slices.Index(ga, int(l.dims[d].axis)))
		}
		perm = append(perm, nb+d)
	}
	y, err := graph.Permute(y, perm)
	if err != nil {
		return nil, err
	}
	for i := len(dims) - 1; i >= 0; i-- {
		d := dims[i]
		pos := d + i // the blocks' axis, after the i gathered axes before d have theirs
		dl := l.dims[d]
		k := p.m.shape[dl.axis]
		cur := y.Shape
		merged := func(n int) []int { return slices.Concat(cur[:pos], []int{n}, cur[pos+2:]) }
		if dl.sec == nil {
			if y, err = graph.Reshape(y, merged(shape[d])); err != nil {
				return nil, err
			}
			continue
		}
		var parts []*graph.Node
		at := 0
		for _, n := range dl.sec {
			b := n / k
			x, err := graph.Shrink(y, rangeAxis(cur, pos+1, at, at+b))
			if err == nil {
				x, err = graph.Reshape(x, merged(n))
			}
			if err != nil {
				return nil, err
			}
			parts = append(parts, x)
			at += b
		}
		if y, err = catAll(parts, pos); err != nil {
			return nil, err
		}
	}
	return y, nil
}

// run runs the collectives of jobs as one group and returns each job's
// converted value. A group of one device needs no collective.
func (p *plan) run(jobs []*job) ([]*value, error) {
	m := p.m
	vals := make([]*value, len(jobs))
	var coll []int
	groups := make([][]meshGroup, len(jobs))
	for i, j := range jobs {
		if m.span(j.axes) > 1 {
			gs, err := m.meshGroups(j.axes)
			if err != nil {
				return nil, err
			}
			groups[i] = gs
			coll = append(coll, i)
			continue
		}
		out := make([]*graph.Node, len(j.in))
		for r, x := range j.in {
			o, err := graph.Reshape(x, j.out)
			if err != nil {
				return nil, err
			}
			out[r] = o
		}
		v, err := j.done(out)
		if err != nil {
			return nil, err
		}
		vals[i] = v
	}
	if len(coll) == 0 {
		return vals, nil
	}
	ins := make([][]*graph.Node, len(m.devs))
	outs := make([][]sched.Out, len(m.devs))
	descs := make([]collDesc, len(coll))
	for at, i := range coll {
		j := jobs[i]
		for r := range ins {
			ins[r] = append(ins[r], j.in[r])
			outs[r] = append(outs[r], sched.Out{DType: j.dt, Shape: j.out, Over: -1})
		}
		descs[at] = collDesc{kind: j.kind, op: j.op, groups: groups[i], dt: j.dt, elems: prod(j.in[0].Shape)}
	}
	if p.rec != nil {
		p.rec.collective(ins, outs, descs)
	}
	nodes, err := m.collect(p.ctx, ins, outs, descs)
	if err != nil {
		return nil, err
	}
	if p.rec != nil {
		p.rec.produced(nodes)
	}
	for at, i := range coll {
		out := make([]*graph.Node, len(m.devs))
		for r := range out {
			out[r] = nodes[r][at]
		}
		p.temps = append(p.temps, out...)
		v, err := jobs[i].done(out)
		if err != nil {
			return nil, err
		}
		vals[i] = v
	}
	return vals, nil
}

// collDesc is one collective of a group that collect runs: its kind and
// reduction, the groups of ranks it connects, and, for the mesh's
// traffic, the dtype and number of elements of each rank's input.
type collDesc struct {
	kind   backend.CollectiveKind
	op     backend.Reduction
	groups []meshGroup
	dt     DType
	elems  int
}

// collect computes the inputs ins[r] on each device r and runs the
// collectives descs over them as one group, the at-th of each device's
// inputs and outputs, outs[r][at], being those of descs[at]. It returns
// each device's outputs.
func (m *Mesh) collect(ctx context.Context, ins [][]*graph.Node, outs [][]sched.Out, descs []collDesc) ([][]*graph.Node, error) {
	nodes, err := sched.Collective(ctx, m.bes, ins, outs, lazy.values, func(in, out [][]backend.Buffer) error {
		var comms []backend.Communicator
		byComm := map[backend.Communicator][]backend.Collective{}
		for at, d := range descs {
			for _, g := range d.groups {
				c := backend.Collective{Kind: d.kind, Reduce: d.op, In: make([]backend.Buffer, len(g.ranks)), Out: make([]backend.Buffer, len(g.ranks))}
				for q, r := range g.ranks {
					c.In[q], c.Out[q] = in[r][at], out[r][at]
				}
				if _, ok := byComm[g.comm]; !ok {
					comms = append(comms, g.comm)
				}
				byComm[g.comm] = append(byComm[g.comm], c)
			}
		}
		for _, c := range comms {
			if err := c.Run(ctx, byComm[c]); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	for _, d := range descs {
		m.count(d.kind, d.dt, d.elems, len(m.devs))
	}
	return nodes, nil
}

// join joins the whole parts of a packed value, each flattened, into the
// packed tensor held whole on every device.
func (p *plan) join(ln *lane) (*value, error) {
	v := ln.cur
	shards := make([]*graph.Node, len(v.shards))
	for r := range shards {
		parts := make([]*graph.Node, len(ln.parts))
		for i, part := range ln.parts {
			if err := part.check(); err != nil {
				return nil, err
			}
			x, err := graph.Reshape(part.cur.shards[r], []int{prod(part.cur.shape)})
			if err != nil {
				return nil, err
			}
			parts[i] = x
		}
		x, err := catAll(parts, 0)
		if err != nil {
			return nil, err
		}
		shards[r] = x
	}
	return v.with(shards, replicated(1)), nil
}

// check reports a part of a packed value that its collectives did not
// make whole.
func (ln *lane) check() error {
	if !ln.cur.lay.withPartial(0, false).equal(ln.to) || ln.cur.lay.partial != 0 {
		return fmt.Errorf("tensor: part of a packed sharded tensor laid out as %s: %w", ln.cur.lay.key(), ErrInternal)
	}
	return nil
}

// cut converts v to the layout l on each device alone: l splits axes
// that v holds whole, or keeps v's partial values partial.
func (p *plan) cut(v *value, l layout) (*value, error) {
	if v.lay.equal(l) {
		return v, nil
	}
	if err := p.cuttable(v, l); err != nil {
		return nil, err
	}
	shards := make([]*graph.Node, len(v.shards))
	for r, x := range v.shards {
		var err error
		if shards[r], err = p.cutShard(x, v.lay, l, v.shape, r); err != nil {
			return nil, err
		}
	}
	return v.with(shards, l), nil
}

// cuttable returns an error unless cut can convert v to l.
func (p *plan) cuttable(v *value, l layout) error {
	if v.lay.packed != nil || l.packed != nil || needsCollective(v.lay, l) || l.partial&^v.lay.partial != 0 {
		return fmt.Errorf("tensor: convert a sharded tensor from %s to %s on each device: %w", v.lay.key(), l.key(), ErrInternal)
	}
	return nil
}

// cutShard cuts device r's block of a tensor of shape laid out as to out
// of x, its block laid out as from, which cuttable allows.
func (p *plan) cutShard(x *graph.Node, from, to layout, shape []int, r int) (*graph.Node, error) {
	m := p.m
	for d, t := range to.dims {
		if from.dims[d].equal(t) {
			continue
		}
		k := m.shape[t.axis]
		c := m.coord(r, int(t.axis))
		var parts []*graph.Node
		at := 0
		for _, n := range t.sections(shape[d]) {
			b := n / k
			y, err := graph.Shrink(x, rangeAxis(x.Shape, d, at+c*b, at+(c+1)*b))
			if err != nil {
				return nil, err
			}
			parts = append(parts, y)
			at += n
		}
		y, err := catAll(parts, d)
		if err != nil {
			return nil, err
		}
		x = y
	}
	return x, nil
}

// catAll joins xs along axis, in Cats of at most backend.MaxCatInputs
// inputs; one input is returned as it is.
func catAll(xs []*graph.Node, axis int) (*graph.Node, error) {
	for len(xs) > 1 {
		var next []*graph.Node
		for i := 0; i < len(xs); i += backend.MaxCatInputs {
			chunk := xs[i:min(i+backend.MaxCatInputs, len(xs))]
			if len(chunk) == 1 {
				next = append(next, chunk[0])
				continue
			}
			c, err := graph.Cat(chunk, axis)
			if err != nil {
				return nil, err
			}
			next = append(next, c)
		}
		xs = next
	}
	return xs[0], nil
}
