package tensor

import (
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// A training step builds the same graphs on every device step after step:
// lowering a graph whose analysis the mesh has cached (see plancache.go)
// gives each device a graph of the same structure, read from the new
// shards of the same leaves, and runs the same collectives. So the first
// run of a graph records a script of what it did: each device's nodes, in
// the order it built them, and the schedules, collectives and uploads it
// ran between them. Later runs of a graph of the same key, with the same
// targets and destinations, replay the script: they build each device's
// nodes straight from it, without the layouts, conversions and checks of
// lowering, and run what it ran. The scheduler still walks each device's
// graph and finds its program by its key, as it does on a single device.
//
// A lowered node takes its constant or seed, which the key leaves out,
// from the node of the graph that it lowers. A run that builds a node
// with a constant or seed of no node of the graph, or reads a computed
// tensor that it cannot name by the graph's leaves or what the run
// itself computed, is not recorded, and runs of its graph lower anew.

// replays turns the replay of recorded runs on and off, for tests.
var replays = true

// script is the record of a plan's run.
type script struct {
	roots []int    // the place of each target in the plan's order
	todo  []int    // the targets the run computed, by their index
	want  []layout // the layout of each target's result
	over  []bool   // whether each target is written over a destination

	slots  [][]slot  // each device's nodes, in the order the run built or got them
	built  []int     // how many of each device's slots are nodes the run built
	owned  []int     // how many of those are allocated alone (see slot.own)
	inputs []int     // how many inputs each device's nodes read in all
	events []event   // what the run ran, in order
	final  [][]int32 // each device's slots of the targets' shards
}

type slotKind uint8

const (
	slotNode   slotKind = iota // a node the run built
	slotLeaf                   // a shard of a leaf of the graph
	slotOutput                 // a node that an event made
)

// slot is one node of a device's graph in the script.
type slot struct {
	kind slotKind
	// A built node: its fields, its inputs' slots, and the place in the
	// plan's order of the node whose constant and seed it takes, or -1.
	op    backend.Op
	dt    DType
	shape []int
	attrs backend.Attrs
	dev   backend.Backend
	ins   []int32
	src   int32
	// own marks a node that is the shard of a target, which outlives
	// the run, and so is not allocated with the others.
	own bool
	// The leaf's number, or the event and the place among its outputs.
	from, at int32
}

type eventKind uint8

const (
	eventRealize    eventKind = iota // a schedule on each device (realizeLive)
	eventCollective                  // a group of collectives (collect)
	eventUpload                      // a leaf copied to every device (valueOf)
)

// event is a schedule, collective or upload of the run, which reads the
// slots of each device that come before mark[r].
type event struct {
	kind  eventKind
	mark  []int
	nodes [][]int32 // the targets of the schedules, or the collectives' inputs, by slot; -1 for none
	outs  [][]sched.Out
	descs []collDesc
	leaf  int // the leaf uploaded
}

// recorder records a plan's run in a script.
type recorder struct {
	t       *script
	known   []map[*graph.Node]int32 // each device's slot of each node the script names
	sources []map[*graph.Node]int32 // each device's built nodes, by the place in the order of what they lower
	failed  bool
}

// ambiguous is the source of a node that lowers two nodes of the graph.
const ambiguous = -2

// scripted returns the script of p's graph if the mesh has one that fits the
// run's targets and their destinations. Otherwise it starts recording the
// run if the graph has an analysis and no script yet.
func (p *plan) scripted(roots []*graph.Node, want []layout, todo []int, dsts []*graph.Node) *script {
	a := p.an
	if a == nil || !replays {
		return nil
	}
	if t := a.script; t != nil && t.fits(p, roots, want, todo, dsts) {
		return t
	}
	if a.script == nil && !a.unscripted {
		p.startRecording(roots, want, todo, dsts)
	}
	return nil
}

// fits reports whether t replays a run of p's graph that computes roots,
// the targets todo, laid out as want, over dsts.
func (t *script) fits(p *plan, roots []*graph.Node, want []layout, todo []int, dsts []*graph.Node) bool {
	if !slices.Equal(t.todo, todo) || len(t.roots) != len(roots) {
		return false
	}
	for j, r := range roots {
		pn := p.node(r)
		if pn.leaf || pn.num != t.roots[j] || !t.want[j].equal(want[j]) || t.over[j] != (dsts != nil && dsts[todo[j]] != nil) {
			return false
		}
	}
	return true
}

// startRecording records p's run, unless the targets or the leaves keep
// a script from naming what it reads.
func (p *plan) startRecording(roots []*graph.Node, want []layout, todo []int, dsts []*graph.Node) {
	nd := len(p.m.devs)
	t := &script{
		todo:   slices.Clone(todo),
		want:   slices.Clone(want),
		roots:  make([]int, len(roots)),
		over:   make([]bool, len(roots)),
		slots:  make([][]slot, nd),
		built:  make([]int, nd),
		owned:  make([]int, nd),
		inputs: make([]int, nd),
	}
	rc := &recorder{t: t, known: make([]map[*graph.Node]int32, nd), sources: make([]map[*graph.Node]int32, nd)}
	for r := range nd {
		rc.known[r], rc.sources[r] = map[*graph.Node]int32{}, map[*graph.Node]int32{}
	}
	for j, x := range roots {
		pn := p.node(x)
		if pn.leaf {
			return
		}
		t.roots[j] = pn.num
		t.over[j] = dsts != nil && dsts[todo[j]] != nil
	}
	for j, x := range p.leaves {
		v := p.node(x).val
		if v == nil {
			continue
		}
		for r, s := range v.shards {
			if s == nil {
				continue
			}
			if _, ok := rc.known[r][s]; ok {
				// Two leaves share a shard, which a replay could not tell.
				return
			}
			rc.define(r, s, slot{kind: slotLeaf, from: int32(j)})
		}
	}
	p.rec = rc
}

// stopRecording keeps the script of p's run on its analysis if the run
// was recorded whole and succeeded.
func (p *plan) stopRecording(err error) {
	rc := p.rec
	p.rec = nil
	if rc == nil || err != nil {
		return
	}
	if rc.failed {
		p.an.unscripted = true
		return
	}
	p.an.script = rc.t
}

func (rc *recorder) define(r int, n *graph.Node, s slot) int32 {
	at := int32(len(rc.t.slots[r]))
	rc.t.slots[r] = append(rc.t.slots[r], s)
	rc.known[r][n] = at
	return at
}

// source notes that b, built on device r, lowers the node n of the
// graph, at the place at in its order: b takes its constant and seed
// from n, unless they differ, or b lowers another node too.
func (rc *recorder) source(r int, b, n *graph.Node, at int) {
	src := int32(at)
	if was, ok := rc.sources[r][b]; ok && was != src || b.Attrs.Value != n.Attrs.Value || b.Attrs.Seed != n.Attrs.Seed {
		src = ambiguous
	}
	rc.sources[r][b] = src
}

// slot returns the slot of x on device r, recording x and the nodes it
// reads that the script does not name yet, or -1 for a nil x.
func (rc *recorder) slot(r int, x *graph.Node) int32 {
	if x == nil || rc.failed {
		return -1
	}
	if s, ok := rc.known[r][x]; ok {
		return s
	}
	type frame struct {
		n    *graph.Node
		next int
	}
	stack := []frame{{n: x}}
	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		if f.next < len(f.n.Inputs) {
			in := f.n.Inputs[f.next]
			f.next++
			if _, ok := rc.known[r][in]; !ok {
				stack = append(stack, frame{n: in})
			}
			continue
		}
		n := f.n
		stack = stack[:len(stack)-1]
		if _, ok := rc.known[r][n]; ok {
			continue
		}
		if n.Realized() || n.Placeholder {
			// A computed tensor that the script cannot name.
			rc.failed = true
			return -1
		}
		src, ok := rc.sources[r][n]
		if !ok {
			src = -1
		}
		if src == ambiguous || src < 0 && (n.Attrs.Value != 0 || n.Attrs.Seed != 0) {
			// A constant or seed of no single node of the graph.
			rc.failed = true
			return -1
		}
		s := slot{kind: slotNode, op: n.Op, dt: n.DType, shape: n.Shape, attrs: n.Attrs, dev: n.Dev, src: src}
		if len(n.Inputs) > 0 {
			s.ins = make([]int32, len(n.Inputs))
			for i, in := range n.Inputs {
				s.ins[i] = rc.known[r][in]
			}
			rc.t.inputs[r] += len(n.Inputs)
		}
		rc.t.built[r]++
		rc.define(r, n, s)
	}
	return rc.known[r][x]
}

// slots records the nodes of each device, and returns their slots.
func (rc *recorder) slots(nodes [][]*graph.Node) [][]int32 {
	out := make([][]int32, len(nodes))
	for r, ns := range nodes {
		out[r] = make([]int32, len(ns))
		for i, n := range ns {
			out[r][i] = rc.slot(r, n)
		}
	}
	return out
}

func (rc *recorder) event(e event) {
	e.mark = make([]int, len(rc.t.slots))
	for r := range rc.t.slots {
		e.mark[r] = len(rc.t.slots[r])
	}
	rc.t.events = append(rc.t.events, e)
}

// realize records schedules that compute the nodes live of each device.
func (rc *recorder) realize(live [][]*graph.Node) {
	rc.event(event{kind: eventRealize, nodes: rc.slots(live)})
}

// collective records collectives over ins, as collect runs them; produced
// records their outputs.
func (rc *recorder) collective(ins [][]*graph.Node, outs [][]sched.Out, descs []collDesc) {
	rc.event(event{kind: eventCollective, nodes: rc.slots(ins), outs: outs, descs: descs})
}

func (rc *recorder) produced(nodes [][]*graph.Node) {
	e := int32(len(rc.t.events) - 1)
	for r, ns := range nodes {
		for i, n := range ns {
			if n != nil {
				rc.define(r, n, slot{kind: slotOutput, from: e, at: int32(i)})
			}
		}
	}
}

// upload records the copies of a leaf, which is a tensor on a device, to
// every device.
func (rc *recorder) upload(leaf *pnode, shards []*graph.Node) {
	if !leaf.keyed {
		rc.failed = true
		return
	}
	rc.event(event{kind: eventUpload, leaf: leaf.num})
	e := int32(len(rc.t.events) - 1)
	for r, n := range shards {
		rc.define(r, n, slot{kind: slotOutput, from: e})
	}
}

// store records each device's shards of the targets.
func (rc *recorder) store(final [][]*graph.Node) {
	rc.t.final = rc.slots(final)
	for r, ss := range rc.t.final {
		for _, s := range ss {
			if sl := &rc.t.slots[r][s]; s >= 0 && sl.kind == slotNode && !sl.own {
				sl.own = true
				rc.t.owned[r]++
			}
		}
	}
}

// replay runs the script t of p's graph, computing roots, the targets todo,
// laid out as want, over dsts.
func (p *plan) replay(t *script, roots []*graph.Node, want []layout, todo []int, dsts []*graph.Node) error {
	m := p.m
	m.plans.replays++
	nd := len(m.devs)
	devs := make([]replayer, nd)
	for r := range devs {
		d := &devs[r]
		d.slots = make([]*graph.Node, len(t.slots[r]))
		d.nodes = make([]graph.Node, t.built[r]-t.owned[r])
		d.ins = make([]*graph.Node, t.inputs[r])
	}
	outs := make([][][]*graph.Node, len(t.events)) // each event's outputs, by device
	// Each device builds its nodes on a goroutine of its own, as lowerAll
	// does.
	build := func(upto func(r int) int) {
		if nd == 1 {
			devs[0].build(p, t, 0, upto(0), outs)
			return
		}
		var wg sync.WaitGroup
		for r := range devs {
			wg.Go(func() { devs[r].build(p, t, r, upto(r), outs) })
		}
		wg.Wait()
	}
	nodes := func(slots [][]int32) [][]*graph.Node {
		ns := make([][]*graph.Node, nd)
		for r, ss := range slots {
			ns[r] = make([]*graph.Node, len(ss))
			for i, s := range ss {
				if s >= 0 {
					ns[r][i] = devs[r].slots[s]
				}
			}
		}
		return ns
	}
	fail := func(err error) error {
		p.release(nil)
		return err
	}
	for e := range t.events {
		ev := &t.events[e]
		build(func(r int) int { return ev.mark[r] })
		switch ev.kind {
		case eventRealize:
			live := nodes(ev.nodes)
			err := sched.RealizeEach(p.ctx, m.bes, live, nil, lazy.values)
			for _, ns := range live {
				for _, n := range ns {
					if sched.Realized(n) {
						p.temps = append(p.temps, n)
					}
				}
			}
			if err != nil {
				return fail(err)
			}
		case eventCollective:
			out, err := m.collect(p.ctx, nodes(ev.nodes), ev.outs, ev.descs)
			if err != nil {
				return fail(err)
			}
			for _, ns := range out {
				for _, n := range ns {
					if n != nil {
						p.temps = append(p.temps, n)
					}
				}
			}
			outs[e] = out
		case eventUpload:
			x := p.leaves[ev.leaf]
			data, err := hostBytes(p.ctx, x)
			if err != nil {
				return fail(err)
			}
			outs[e] = make([][]*graph.Node, nd)
			for r, be := range m.bes {
				s, err := sched.Upload(p.ctx, be, x.DType, data, x.Shape)
				if err != nil {
					return fail(err)
				}
				outs[e][r] = []*graph.Node{s}
			}
		}
	}
	build(func(r int) int { return len(t.slots[r]) })
	return p.finish(roots, want, todo, dsts, nodes(t.final))
}

// replayer builds one device's nodes from a script.
type replayer struct {
	slots []*graph.Node // the device's nodes, by slot
	built int           // how many slots are built
	nodes []graph.Node  // the memory of the nodes to build, but those allocated alone
	ins   []*graph.Node // the memory of their inputs
}

// build builds device r's slots up to upto, from the script t of p's
// graph and the outputs of the events run so far.
func (d *replayer) build(p *plan, t *script, r, upto int, outs [][][]*graph.Node) {
	for i := d.built; i < upto; i++ {
		s := &t.slots[r][i]
		switch s.kind {
		case slotLeaf:
			d.slots[i] = p.node(p.leaves[s.from]).val.shards[r]
		case slotOutput:
			d.slots[i] = outs[s.from][r][s.at]
		case slotNode:
			var n *graph.Node
			if s.own {
				n = new(graph.Node)
			} else {
				n, d.nodes = &d.nodes[0], d.nodes[1:]
			}
			*n = graph.Node{Op: s.op, DType: s.dt, Shape: s.shape, Attrs: s.attrs, Dev: s.dev}
			if s.src >= 0 {
				a := &p.order[s.src].Attrs
				n.Attrs.Value, n.Attrs.Seed = a.Value, a.Seed
			}
			if k := len(s.ins); k > 0 {
				in := d.ins[:k:k]
				d.ins = d.ins[k:]
				for j, x := range s.ins {
					in[j] = d.slots[x]
				}
				n.Inputs = in
			}
			d.slots[i] = n
		}
	}
	d.built = max(d.built, upto)
}
