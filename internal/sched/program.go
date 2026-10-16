package sched

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// A schedule runs as a program, compiled from its plan: the kernels in
// order, the pooled buffers each takes and gives back, and the buffer and
// view that hold each target. A program names buffers by slot rather
// than holding them: the buffers of the leaves the walk numbered, of the
// targets' destinations and of the pool, which each run binds anew. It
// names the nodes whose values and seeds its kernels take by their number
// in the walk's order. So a program compiled for one graph runs any graph
// with the same key, which is what the cache does (see cache.go), and a
// run allocates nothing but the storage of each new result.
//
// A run first does the pool's bookkeeping for the whole schedule, taking
// and giving back buffers in the order the kernels would, then runs the
// kernels. A buffer given back before a later kernel takes it is written
// by that kernel after the earlier ones read it, as when each kernel took
// its buffers as it ran, since a backend runs kernels in order. Taking
// every buffer first lets a backend that runs a whole batch at once, such
// as CUDA through a CUDA graph, see the schedule's kernels and buffers
// together (see backend.Batch).
//
// A donated leaf whose memory the schedule takes (see donate.go) is
// given to the pool as a pooled buffer is, after its last reader, so
// that the kernels after it can take it; it has a pooled buffer's slot
// whose taking is the binding of the leaf's buffer at the start of the
// run.

// program is a compiled schedule. Its scratch slices are bound for one
// run at a time, under mu.
type program struct {
	leaves, targets int       // the slots of the walk's leaves, then of the destinations
	tmps            []poolKey // the pooled buffers, in the order taken; their slots follow
	events          []int32   // the pool's bookkeeping in order: k takes tmps[k], ^k gives it back
	peaks           []peak    // the most buffers of each key held at once
	moves           []move    // inputs copied from other devices before the kernels
	steps           []kstep
	results         []int32 // the slot of each target's result, or -1 if it was realized
	views           []backend.View
	into            bool // some kernel writes over a destination

	batch backend.Batch    // the kernels, and the buffers of each run
	bufs  []backend.Buffer // by slot, for one run
	held  []bool           // by pooled buffer: taken and not given back
	last  []uintptr        // by pooled buffer: the address of the one the last run took
	keyOf []int32          // by pooled buffer: the index of its key in peaks
	short []int            // by peak: the buffers of its key the run has yet to obtain
	mems  []*graph.Storage // by pooled buffer: the storage of the results it holds

	donated []donation // the donated leaves the program gives to the pool
}

// donation is a donated leaf, in slot leaf, whose buffer the program
// holds as the pooled buffer tmp until it gives it to the pool.
type donation struct{ tmp, leaf int32 }

// kstep is one kernel of a program.
type kstep struct {
	k         *backend.Kernel
	outs, ins []int32 // slots
	into      []int32 // the targets whose destinations it writes over
	// argsOf is the number of the node whose Value and Seed k takes, or
	// -1, and consts the number of the constant whose value each
	// constant of k's program takes.
	argsOf int32
	consts []constArg
}

type constArg struct{ instr, node int32 }

// move copies the leaf in slot leaf, which lives on another device, into
// the pooled buffer in slot to.
type move struct{ leaf, to int32 }

// peak is the most buffers of one key that a program holds at once.
type peak struct {
	key poolKey
	n   int
}

// compile returns the program that computes the plan p, whose nodes w
// numbered, with into mapping targets to the destinations they are
// written over. It mirrors, at compile time, how each kernel reads the
// results of earlier ones and when the last reader of a buffer has run.
func compile(p *plan, w *walker, targets, dsts []*graph.Node, into map[*graph.Node]*graph.Node, isTarget map[*graph.Node]bool) (*program, error) {
	prog := &program{leaves: len(w.leaves), targets: len(targets)}
	base := int32(prog.leaves + prog.targets) // the slot of the first pooled buffer
	type result struct {
		slot int32
		view backend.View
	}
	take := func(dt backend.DType, n int) int32 {
		k := int32(len(prog.tmps))
		prog.tmps = append(prog.tmps, poolKey{dt, n})
		prog.events = append(prog.events, k)
		return base + k
	}
	give := func(slot int32) { prog.events = append(prog.events, ^(slot - base)) }
	results := make(map[*graph.Node]result, len(p.order))
	// refs counts the live results that read each pooled buffer, so a
	// buffer shared by views is given back only after its last reader.
	refs := map[int32]int{}
	moved := map[*graph.Node]result{}
	for _, n := range p.order {
		for _, in := range p.work[n].deps {
			if _, ok := moved[in]; ok || !in.Realized() || in.Mem.Dev == p.be {
				continue
			}
			// The schedule holds the copy until its end.
			slot := take(in.DType, in.Size())
			refs[slot] = 1
			moved[in] = result{slot, backend.Contiguous(in.Shape)}
			prog.moves = append(prog.moves, move{in.Mark.Num, slot})
		}
	}
	lookup := func(n *graph.Node) result {
		if r, ok := moved[n]; ok {
			return r
		}
		if n.Realized() {
			return result{n.Mark.Num, n.View}
		}
		return results[n]
	}
	uses := p.uses()
	// gifts holds the storages of the donated leaves that the program
	// gives to the pool, each with the number of reads of it by kernels
	// and live views still to come. A leaf that a target or destination
	// shares memory with keeps it.
	type donor struct {
		leaf int32
		refs int
	}
	gifts := map[*graph.Storage]*donor{}
	for _, g := range w.gifts {
		if l := w.leaves[g.leaf]; uses[l] > 0 {
			gifts[l.Mem] = &donor{g.leaf, uses[l]}
		}
	}
	if len(gifts) > 0 {
		for i, t := range targets {
			if d := destination(dsts, i); d != nil && d.Realized() {
				delete(gifts, d.Mem)
			}
			if t.Realized() {
				delete(gifts, t.Mem)
			}
		}
	}
	// unref counts a read of the memory of the leaf in slot, which the
	// program gives to the pool after the last.
	unref := func(slot int32) {
		g := gifts[w.leaves[slot].Mem]
		if g == nil {
			return
		}
		if g.refs--; g.refs > 0 {
			return
		}
		buf := w.leaves[g.leaf].Buf()
		k := int32(len(prog.tmps))
		prog.tmps = append(prog.tmps, poolKey{buf.DType(), buf.Len()})
		prog.events = append(prog.events, ^k)
		prog.donated = append(prog.donated, donation{k, g.leaf})
	}
	release := func(n *graph.Node) {
		uses[n]--
		if n.Realized() {
			if len(gifts) > 0 {
				unref(n.Mark.Num)
			}
			return
		}
		if uses[n] > 0 || isTarget[n] {
			return
		}
		r := results[n]
		delete(results, n)
		switch {
		case r.slot >= base:
			if refs[r.slot]--; refs[r.slot] == 0 {
				delete(refs, r.slot)
				give(r.slot)
			}
		case int(r.slot) < prog.leaves && len(gifts) > 0:
			unref(r.slot)
		}
	}
	// dest returns the slot of the destination that target x is written
	// over, d, and its target's number.
	dest := func(x, d *graph.Node) int32 {
		for i, t := range targets {
			if t == x && dsts[i] == d {
				return int32(i)
			}
		}
		return -1
	}

	for _, n := range p.order {
		wk := p.work[n]
		views := make([]backend.View, len(wk.deps))
		ins := make([]int32, len(wk.deps))
		for i, in := range wk.deps {
			r := lookup(in)
			views[i], ins[i] = r.view, r.slot
		}
		// A weight gradient in parts sums them into a buffer that the
		// kernel of n then adds up.
		parts := int32(-1)
		if pk := wk.parts(n, views); pk != nil {
			size, err := elements(pk)
			if err != nil {
				return nil, err
			}
			parts = take(pk.DType, size)
			prog.steps = append(prog.steps, kstep{k: pk, ins: ins, outs: []int32{parts}, argsOf: -1})
			views, ins = []backend.View{backend.Contiguous(pk.Shape)}, []int32{parts}
		}
		k, view, err := wk.kernel(n, views)
		if err != nil {
			return nil, err
		}
		if k == nil {
			r := result{ins[0], view}
			switch {
			case r.slot >= base:
				refs[r.slot]++
			case int(r.slot) < prog.leaves:
				if g := gifts[w.leaves[r.slot].Mem]; g != nil {
					g.refs++
				}
			}
			results[n] = r
		} else {
			size, err := elements(k)
			if err != nil {
				return nil, err
			}
			s := kstep{k: k, ins: ins, argsOf: -1}
			if wk.prog == nil && wk.gemm == nil && !wk.swap && k.Op == n.Op {
				s.argsOf = n.Mark.Num // k.Attrs is n.Attrs
			}
			if k.Attrs.Program != nil {
				k.Attrs.Program = slices.Clone(k.Attrs.Program)
				for i, in := range k.Attrs.Program {
					if in.Op == backend.OpConst {
						s.consts = append(s.consts, constArg{int32(i), constNode(wk.src[i]).Mark.Num})
					}
				}
			}
			for _, x := range wk.nodes(n) {
				r := result{view: resultView(k, x)}
				if d := into[x]; d != nil {
					i := dest(x, d)
					if i < 0 {
						return nil, fmt.Errorf("sched: a target written over no destination of its own: %w", backend.ErrInternal)
					}
					r.slot = int32(prog.leaves) + i
					s.into = append(s.into, i)
					prog.into = true
				} else {
					r.slot = take(k.DType, size)
				}
				s.outs = append(s.outs, r.slot)
				results[x] = r
			}
			for _, x := range wk.nodes(n) {
				if slot := results[x].slot; slot >= base {
					refs[slot]++
				}
			}
			prog.steps = append(prog.steps, s)
			if parts >= 0 {
				give(parts)
			}
		}
		for _, in := range wk.deps {
			release(in)
		}
	}

	prog.results = make([]int32, len(targets))
	prog.views = make([]backend.View, len(targets))
	for i, t := range targets {
		prog.results[i] = -1
		if t.Realized() {
			continue
		}
		r, ok := results[t]
		if !ok {
			return nil, fmt.Errorf("sched: no kernel computed a %s target: %w", t.Op, backend.ErrInternal)
		}
		prog.results[i], prog.views[i] = r.slot, r.view
	}
	// The copies go back after the last kernel, unless a target views one.
	for _, mv := range prog.moves {
		if refs[mv.to]--; refs[mv.to] == 0 {
			give(mv.to)
		}
	}
	prog.peaks = peaks(prog.tmps, prog.events)

	prog.bufs = make([]backend.Buffer, int(base)+len(prog.tmps))
	prog.held = make([]bool, len(prog.tmps))
	prog.last = make([]uintptr, len(prog.tmps))
	prog.keyOf = make([]int32, len(prog.tmps))
	prog.short = make([]int, len(prog.peaks))
	for e, key := range prog.tmps {
		i, _ := slices.BinarySearchFunc(prog.peaks, key, func(pk peak, key poolKey) int {
			return cmp.Or(cmp.Compare(pk.key.dt, key.dt), cmp.Compare(pk.key.n, key.n))
		})
		prog.keyOf[e] = int32(i)
	}
	prog.mems = make([]*graph.Storage, len(prog.tmps))
	prog.batch.Kernels = make([]*backend.Kernel, len(prog.steps))
	prog.batch.Outs = make([][]backend.Buffer, len(prog.steps))
	prog.batch.Ins = make([][]backend.Buffer, len(prog.steps))
	for i, s := range prog.steps {
		prog.batch.Kernels[i] = s.k
		prog.batch.Outs[i] = make([]backend.Buffer, len(s.outs))
		prog.batch.Ins[i] = make([]backend.Buffer, len(s.ins))
	}
	return prog, nil
}

// peaks returns, for each key of the pooled buffers tmps, the most of
// them that events hold at once.
func peaks(tmps []poolKey, events []int32) []peak {
	held, top := map[poolKey]int{}, map[poolKey]int{}
	for _, e := range events {
		if e >= 0 {
			k := tmps[e]
			held[k]++
			top[k] = max(top[k], held[k])
		} else {
			held[tmps[^e]]--
		}
	}
	out := make([]peak, 0, len(top))
	for _, k := range slices.SortedFunc(maps.Keys(top), func(a, b poolKey) int {
		return cmp.Or(cmp.Compare(a.dt, b.dt), cmp.Compare(a.n, b.n))
	}) {
		out = append(out, peak{k, top[k]})
	}
	return out
}

// simulate returns a sim of m's free lists after the program takes and
// gives back its buffers. The pool serves the most recently freed buffer
// of a key first, so a program that holds at most n buffers of a key at
// once takes n of them from the pool, as far as the pool has them, and
// the rest from the backend.
func (p *program) simulate(m *pool) *sim {
	s := m.simulate()
	for _, pk := range p.peaks {
		have := s.pooled[pk.key]
		taken := min(have, pk.n)
		s.pooled[pk.key] = have - taken
		s.extra += int64(pk.n-taken) * pk.key.bytes()
	}
	return s
}

// reserve admits a run of the program on m, or refuses it with
// ErrMemoryBudget, as pool.reserve does, without the allocations of a sim
// when the run fits the budget.
func (p *program) reserve(m *pool) error {
	var extra int64
	for _, pk := range p.peaks {
		if have := len(m.free[pk.key]); pk.n > have {
			extra += int64(pk.n-have) * pk.key.bytes()
		}
	}
	if m.budget > 0 && m.held()+extra > m.budget {
		return m.reserve(p.simulate(m))
	}
	m.predicted += extra
	return nil
}

// run computes targets, writing them over dsts as compiled, on m's
// backend, reading the leaves and constants of the walk w. It assigns
// each target its result and makes stale the destinations it consumes,
// as realize documents.
func (p *program) run(ctx context.Context, m *pool, w *walker, targets, dsts []*graph.Node) (err error) {
	defer p.unbind()
	for i, n := range w.leaves {
		p.bufs[i] = n.Buf()
	}
	for i := range p.targets {
		if d := destination(dsts, i); d != nil && d.Realized() && !d.Stale() {
			p.bufs[p.leaves+i] = d.Buf()
		}
	}
	base := p.leaves + p.targets
	// The donated leaves are the pool's from now on, to be given to it
	// after their last readers, and their nodes are stale.
	for _, d := range p.donated {
		n := w.leaves[d.leaf]
		m.track(n.Buf())
		p.bufs[base+int(d.tmp)], p.held[d.tmp] = n.Buf(), true
		n.Mem.Version++
		n.Mem.Refs = 0
		n.Mem.Donated = true
	}
	// The run obtains from the backend the buffers that the pool lacks at
	// its peaks, as reserve predicted: those that results of earlier runs
	// took away.
	for i, pk := range p.peaks {
		p.short[i] = max(pk.n-len(m.free[pk.key]), 0)
	}
	for _, e := range p.events {
		if e < 0 {
			m.release(p.bufs[base+int(^e)])
			p.held[^e] = false
			continue
		}
		k := p.keyOf[e]
		buf, obtained, err := m.take(p.tmps[e], p.last[e], p.last[e] != 0 && p.short[k] > 0)
		if err != nil {
			p.giveBack(m)
			return err
		}
		if obtained {
			p.short[k]--
		}
		p.bufs[base+int(e)], p.held[e] = buf, true
		p.last[e] = uintptr(addressOf(buf))
	}
	for _, mv := range p.moves {
		if err := copyFrom(ctx, w.leaves[mv.leaf], m.be, p.bufs[mv.to]); err != nil {
			p.giveBack(m)
			return err
		}
	}
	if err := p.launch(ctx, m.be, w, dsts); err != nil {
		p.giveBack(m)
		return err
	}
	if err := m.be.Sync(uncancelled(ctx)); err != nil {
		p.giveBack(m)
		return err
	}
	for i, t := range targets {
		slot := p.results[i]
		if t.Realized() || slot < 0 {
			continue
		}
		var mem *graph.Storage
		switch k := int(slot) - base; {
		case int(slot) < p.leaves:
			mem = w.leaves[slot].Mem
		case k < 0:
			mem = dsts[int(slot)-p.leaves].Mem
		default:
			if p.mems[k] == nil {
				p.mems[k] = &graph.Storage{Buf: p.bufs[slot], Dev: m.be, Pooled: true}
			}
			mem = p.mems[k]
		}
		assign(t, result{view: p.views[i], mem: mem})
	}
	clear(p.held)
	consume(targets, dsts)
	return nil
}

// launch runs the program's kernels on be, as one batch if be runs
// batches, taking their arguments from the nodes of w. Once a kernel has
// begun to write over a destination, it ignores cancellation, so that the
// schedule does not stop with some targets written and others not.
func (p *program) launch(ctx context.Context, be backend.Backend, w *walker, dsts []*graph.Node) error {
	for i := range p.steps {
		s := &p.steps[i]
		if s.argsOf >= 0 {
			a := &w.order[s.argsOf].Attrs
			s.k.Attrs.Value, s.k.Attrs.Seed = a.Value, a.Seed
		}
		for _, c := range s.consts {
			s.k.Attrs.Program[c.instr].Value = w.order[c.node].Attrs.Value
		}
		for j, slot := range s.outs {
			p.batch.Outs[i][j] = p.bufs[slot]
		}
		for j, slot := range s.ins {
			p.batch.Ins[i][j] = p.bufs[slot]
		}
	}
	overwrite := func(s *kstep) {
		for _, t := range s.into {
			d := dsts[t]
			d.Mem.Version++
			d.Mem.Refs = 0
		}
	}
	if b, ok := be.(backend.Batcher); ok && len(p.steps) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		if p.into {
			ctx = uncancelled(ctx)
			for i := range p.steps {
				overwrite(&p.steps[i])
			}
		}
		return b.RunBatch(ctx, &p.batch)
	}
	writing := false
	for i := range p.steps {
		s := &p.steps[i]
		if !writing {
			if err := ctx.Err(); err != nil {
				return err
			}
			if len(s.into) > 0 {
				writing, ctx = true, uncancelled(ctx)
			}
		}
		overwrite(s)
		if err := be.Run(ctx, s.k, p.batch.Outs[i], p.batch.Ins[i]); err != nil {
			return err
		}
	}
	return nil
}

// giveBack returns to m the pooled buffers the run still holds.
func (p *program) giveBack(m *pool) {
	base := p.leaves + p.targets
	for k, h := range p.held {
		if h {
			m.release(p.bufs[base+k])
			p.held[k] = false
		}
	}
}

// unbind drops the buffers and storages of the last run, so that the
// program keeps no memory alive.
func (p *program) unbind() {
	clear(p.bufs)
	clear(p.mems)
	for i := range p.batch.Outs {
		clear(p.batch.Outs[i])
		clear(p.batch.Ins[i])
	}
	detached.parent = nil
}

// copyFrom copies the realized node n, which lives on another device,
// into dst, a buffer of be.
func copyFrom(ctx context.Context, n *graph.Node, be backend.Backend, dst backend.Buffer) error {
	size, _ := n.DType.Bytes(n.Size())
	data := make([]byte, size)
	if err := download(ctx, n, data); err != nil {
		return err
	}
	return be.Upload(ctx, dst, data)
}

// detached is the context that uncancelled returns, reused by every
// schedule under mu so that a schedule allocates none.
var detached detachedCtx

// uncancelled returns a context with the values of ctx that is never
// cancelled, like context.WithoutCancel, valid until the schedule ends.
func uncancelled(ctx context.Context) context.Context {
	if ctx == context.Context(&detached) {
		return ctx
	}
	detached.parent = ctx
	return &detached
}

type detachedCtx struct{ parent context.Context }

func (*detachedCtx) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*detachedCtx) Done() <-chan struct{}       { return nil }
func (*detachedCtx) Err() error                  { return nil }
func (d *detachedCtx) Value(key any) any         { return d.parent.Value(key) }
