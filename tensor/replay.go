package tensor

import (
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// A training step builds the same graphs on every device step after step:
// lowering a graph whose analysis the mesh has cached (see plancache.go)
// gives each device a graph of the same structure, read from the new
// shards of the same leaves, and runs the same collectives. So the first
// run of a graph records a script of what it did: each device's nodes,
// and the schedules, collectives and uploads it ran. Later runs of a
// graph of the same key, with the same targets and destinations, replay
// the script. A schedule replays the program the scheduler compiled for
// it (see sched.ReplayEach), which reads the leaves and writes the
// targets alone, so only the nodes that hold its targets are built; the
// scheduler checks that the leaves and destinations would have found the
// same program, and a device whose do not computes the graph the script
// builds for it, as the run did.
//
// A lowered node takes its constant or seed, which the key leaves out,
// from the node of the graph that it lowers. A run that builds a node
// with a constant or seed of no node of the graph, or reads a computed
// tensor that it cannot name by the graph's leaves or what the run
// itself computed, is not recorded, and runs of its graph lower anew.

// replays turns the replay of recorded runs on and off, and
// replayPrograms that of the schedules' programs, without which a replay
// computes the graphs the script builds, for tests.
var replays, replayPrograms = true, true

// script is the record of a plan's run.
type script struct {
	roots []int    // the place of each target in the plan's order
	todo  []int    // the targets the run computed, by their index
	want  []layout // the layout of each target's result
	over  []bool   // whether each target is written over a destination

	slots  [][]slot // each device's nodes, in the order the run built or got them
	events []*event // what the run ran, in order
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
	// The leaf's number, or the event and the place among its outputs.
	from, at int32
}

type eventKind uint8

const (
	eventSchedule   eventKind = iota // a schedule on each device (plan.schedule)
	eventCollective                  // a group of collectives (collect)
	eventUpload                      // a leaf copied to every device (valueOf)
)

// event is a schedule, collective or upload of the run.
type event struct {
	kind  eventKind
	nodes [][]int32 // the targets of the schedules, or the collectives' inputs, by slot; -1 for none
	final bool      // whether the schedules compute the shards of the run's targets
	// The schedules' traces, the slots of their leaves, and the sources
	// of the constants and seeds of their programs (see slot.src), by
	// device; a device without a trace computes its graph.
	traces []*sched.Trace
	leaves [][]int32
	args   [][]int32
	argBuf [][]*graph.Node // scratch for the arguments of a replay
	outs   [][]sched.Out
	descs  []collDesc
	leaf   int // the leaf uploaded
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

// zeroArg is the node whose constant and seed a program reads for a node
// of no source, whose are zero.
var zeroArg = &graph.Node{}

// scripted returns the script of p's graph if the mesh has one that fits
// the run's targets and their destinations. Otherwise it starts
// recording the run if the graph has an analysis and no script yet.
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
		todo:  slices.Clone(todo),
		want:  slices.Clone(want),
		roots: make([]int, len(roots)),
		over:  make([]bool, len(roots)),
		slots: make([][]slot, nd),
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
		}
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

func (rc *recorder) event(e *event) *event {
	rc.t.events = append(rc.t.events, e)
	return e
}

// realize records schedules that compute the nodes of each device, the
// shards of the run's targets if final is set, and returns their event,
// for traced to fill in.
func (rc *recorder) realize(nodes [][]*graph.Node, final bool) *event {
	nd := len(nodes)
	return rc.event(&event{
		kind: eventSchedule, nodes: rc.slots(nodes), final: final,
		traces: make([]*sched.Trace, nd), leaves: make([][]int32, nd), args: make([][]int32, nd), argBuf: make([][]*graph.Node, nd),
	})
}

// traced records the trace of the schedule of ev on device r, which
// computes order, from leaves. It runs on the device's goroutine.
func (rc *recorder) traced(ev *event, r int, t *sched.Trace, order, leaves []*graph.Node) {
	known, slots := rc.known[r], rc.t.slots[r]
	ls := make([]int32, t.Leaves())
	for i := range ls {
		s, ok := known[leaves[i]]
		if !ok {
			return
		}
		ls[i] = s
	}
	var args []int32
	if as := t.Args(); len(as) > 0 {
		args = make([]int32, as[len(as)-1]+1)
		for i := range args {
			args[i] = -1
		}
		for _, a := range as {
			s, ok := known[order[a]]
			if !ok || slots[s].kind != slotNode {
				return
			}
			args[a] = slots[s].src
		}
		ev.argBuf[r] = make([]*graph.Node, len(args))
	}
	ev.traces[r], ev.leaves[r], ev.args[r] = t, ls, args
}

// collective records collectives over ins, as collect runs them; produced
// records their outputs.
func (rc *recorder) collective(ins [][]*graph.Node, outs [][]sched.Out, descs []collDesc) {
	rc.event(&event{kind: eventCollective, nodes: rc.slots(ins), outs: outs, descs: descs})
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
	rc.event(&event{kind: eventUpload, leaf: leaf.num})
	e := int32(len(rc.t.events) - 1)
	for r, n := range shards {
		rc.define(r, n, slot{kind: slotOutput, from: e})
	}
}

// replay runs the script t of p's graph, computing roots, the targets
// todo, laid out as want, over dsts.
func (p *plan) replay(t *script, roots []*graph.Node, want []layout, todo []int, dsts []*graph.Node) error {
	m := p.m
	m.plans.replays++
	nd := len(m.devs)
	devs := make([]replayer, nd)
	for r := range devs {
		devs[r] = replayer{p: p, t: t, r: r, slots: make([]*graph.Node, len(t.slots[r])), outs: make([][]*graph.Node, len(t.events))}
	}
	var final *event
	for e, ev := range t.events {
		switch ev.kind {
		case eventSchedule:
			if ev.final {
				final = ev
				continue
			}
			live, err := p.replaySchedule(ev, devs, nil)
			for _, ns := range live {
				for _, n := range ns {
					if n != nil && sched.Realized(n) {
						p.temps = append(p.temps, n)
					}
				}
			}
			if err != nil {
				p.release(nil)
				return err
			}
		case eventCollective:
			ins := make([][]*graph.Node, nd)
			for r, ss := range ev.nodes {
				ins[r] = make([]*graph.Node, len(ss))
				for i, s := range ss {
					if s >= 0 {
						ins[r][i] = devs[r].node(s)
					}
				}
			}
			out, err := m.collect(p.ctx, ins, ev.outs, ev.descs)
			if err != nil {
				p.release(nil)
				return err
			}
			for r, ns := range out {
				devs[r].outs[e] = ns
				for _, n := range ns {
					if n != nil {
						p.temps = append(p.temps, n)
					}
				}
			}
		case eventUpload:
			x := p.leaves[ev.leaf]
			data, err := hostBytes(p.ctx, x)
			if err != nil {
				p.release(nil)
				return err
			}
			for r, be := range m.bes {
				s, err := sched.Upload(p.ctx, be, x.DType, data, x.Shape)
				if err != nil {
					p.release(nil)
					return err
				}
				devs[r].outs[e] = []*graph.Node{s}
			}
		}
	}
	return p.finish(roots, want, todo, dsts, func(over [][]*graph.Node) ([][]*graph.Node, error) {
		return p.replaySchedule(final, devs, over)
	})
}

// replaySchedule replays the schedules of ev, over the destinations
// over, and returns each device's targets: computed by the traced
// program where the scheduler finds that it fits, and otherwise by
// computing the graph the script builds.
func (p *plan) replaySchedule(ev *event, devs []replayer, over [][]*graph.Node) ([][]*graph.Node, error) {
	nd := len(devs)
	targets := make([][]*graph.Node, nd)
	leaves := make([][]*graph.Node, nd)
	args := make([][]*graph.Node, nd)
	traces := ev.traces
	if !replayPrograms {
		traces = make([]*sched.Trace, nd)
	}
	for r := range devs {
		if traces[r] == nil {
			continue
		}
		d := &devs[r]
		leaves[r] = make([]*graph.Node, len(ev.leaves[r]))
		for i, s := range ev.leaves[r] {
			leaves[r][i] = d.get(s)
		}
		buf := ev.argBuf[r]
		for a, src := range ev.args[r] {
			if src >= 0 {
				buf[a] = p.order[src]
			} else {
				buf[a] = zeroArg
			}
		}
		args[r] = buf
	}
	missed, err := sched.ReplayEach(p.ctx, traces, leaves, args, func(r int) []*graph.Node {
		d := &devs[r]
		ts := make([]*graph.Node, len(ev.nodes[r]))
		for i, s := range ev.nodes[r] {
			ts[i] = d.bare(s)
		}
		targets[r] = ts
		return ts
	}, over, lazy.values)
	for _, buf := range args {
		clear(buf)
	}
	if err != nil {
		return targets, err
	}
	var bes []backend.Backend
	var rest, restOver [][]*graph.Node
	for r, miss := range missed {
		if !miss {
			continue
		}
		if traces[r] != nil {
			p.m.plans.missed++
		}
		d := &devs[r]
		ts := make([]*graph.Node, len(ev.nodes[r]))
		for i, s := range ev.nodes[r] {
			ts[i] = d.node(s)
		}
		targets[r] = ts
		bes, rest = append(bes, p.m.bes[r]), append(rest, ts)
		if over != nil {
			restOver = append(restOver, over[r])
		}
	}
	if len(bes) == 0 {
		return targets, nil
	}
	return targets, sched.RealizeEach(p.ctx, bes, rest, restOver, lazy.values)
}

// replayer gets one device's nodes from a script.
type replayer struct {
	p     *plan
	t     *script
	r     int
	slots []*graph.Node   // the device's nodes, by slot, once got
	outs  [][]*graph.Node // the outputs of each event run so far
}

// get returns the node of slot s if it is a shard of a leaf, the output
// of an event or a node built already, and otherwise nil.
func (d *replayer) get(s int32) *graph.Node {
	if n := d.slots[s]; n != nil {
		return n
	}
	sl := &d.t.slots[d.r][s]
	switch sl.kind {
	case slotLeaf:
		d.slots[s] = d.p.node(d.p.leaves[sl.from]).val.shards[d.r]
	case slotOutput:
		d.slots[s] = d.outs[sl.from][sl.at]
	}
	return d.slots[s]
}

// bare returns the node of slot s, building it without inputs if it is a
// node the run built that is not built yet: a target that a replayed
// program computes.
func (d *replayer) bare(s int32) *graph.Node {
	if n := d.get(s); n != nil {
		return n
	}
	n := d.build(&d.t.slots[d.r][s])
	d.slots[s] = n
	return n
}

// node returns the node of slot s, building it, and the nodes it reads
// that are not built yet, if it is not built yet.
func (d *replayer) node(s int32) *graph.Node {
	if n := d.get(s); n != nil {
		return n
	}
	stack := []int32{s}
	for len(stack) > 0 {
		x := stack[len(stack)-1]
		sl := &d.t.slots[d.r][x]
		ready := true
		for _, in := range sl.ins {
			if d.get(in) == nil {
				stack, ready = append(stack, in), false
			}
		}
		if !ready {
			continue
		}
		stack = stack[:len(stack)-1]
		if d.slots[x] != nil {
			continue
		}
		n := d.build(sl)
		if len(sl.ins) > 0 {
			n.Inputs = make([]*graph.Node, len(sl.ins))
			for i, in := range sl.ins {
				n.Inputs[i] = d.slots[in]
			}
		}
		d.slots[x] = n
	}
	return d.slots[s]
}

// build returns a node of the fields of sl, without inputs.
func (d *replayer) build(sl *slot) *graph.Node {
	n := &graph.Node{Op: sl.op, DType: sl.dt, Shape: sl.shape, Attrs: sl.attrs, Dev: sl.dev}
	if sl.src >= 0 {
		a := &d.p.order[sl.src].Attrs
		n.Attrs.Value, n.Attrs.Seed = a.Value, a.Seed
	}
	return n
}
