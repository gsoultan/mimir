package tensor

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"weak"

	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// ErrNoGrad reports a Backward with nothing to differentiate: the tensor
// was not computed from any tensor that requires gradients, or an earlier
// Backward already released the graph it was computed from.
var ErrNoGrad = errors.New("tensor does not require gradients")

// gradMu guards the mutable state of tapes: the gradients accumulated in
// leaves and the release of graphs after Backward. It is held only while
// code of this package reads or writes them, never while user code runs,
// so a checkpointed function or a Func may read gradients. Building
// tensors reads only the immutable parts of tapes and never takes it.
//
// Lock order: backwardMu, then gradMu, then Param.mu.
var gradMu sync.Mutex

// backwardMu serialises Backward, so that two of them never add to the
// same gradient at once. Backward holds it from start to end, including
// while its gradient rules run user code: the functions Checkpoint
// recomputes and the rules of Func. A release of a graph happens under
// both backwardMu and gradMu, so a walk of tapes under either one sees it
// whole.
var backwardMu sync.Mutex

// tape records how a tracked tensor was computed, so that the chain rule
// can run through it in either direction. A leaf's tape has no rule and
// instead holds the gradient accumulated for it.
type tape struct {
	seq  uint64      // the count of tapes when this one was made
	node *graph.Node // the tensor's: gradients have its shape, and rules read it as their result

	// ins are the inputs of the operation as it received them. An input
	// without a tape has no gradient.
	ins  []*Tensor
	rule rule
	// effects are computed in the same schedule as the gradients.
	effects []*Tensor
	// ckpt, if not nil, is the call of Checkpoint that computed the
	// tensor, whose rule recomputes the function (see Checkpoint).
	ckpt *checkpointCall
	// part, if not nil, makes the tensor a part of its one input (see
	// split): it has no rule, and the reverse pass joins the gradients of
	// the parts into the input's.
	part *part

	leaf     bool
	param    *Param  // the parameter a leaf stands for, or nil
	grad     *Tensor // accumulated gradient of a leaf
	own      bool    // grad's memory is the tape's to reuse
	released bool    // Backward has run through this node
	// stand marks a tensor that stands for an argument of a function a
	// transformation calls (see standIn): to Checkpoint, a leaf.
	stand bool
}

// deps returns the number of tapes t depends on: those of its inputs
// and, for a checkpoint, those the function reached besides its inputs,
// which its recompute reaches again (see Checkpoint).
func (t *tape) deps() int {
	if t.ckpt != nil {
		return len(t.ins) + len(t.ckpt.reaches)
	}
	return len(t.ins)
}

// dep returns the i-th tape t depends on, nil for an input without one,
// and the shape of its derivatives.
func (t *tape) dep(i int) (*tape, []int) {
	if i < len(t.ins) {
		if x := t.ins[i]; x != nil {
			return x.tape, x.node.Shape
		}
		return nil, nil
	}
	d := t.ckpt.reaches[i-len(t.ins)]
	return d, d.node.Shape
}

// A rule differentiates the operation that computed a tensor, backward
// and forward. Its functions receive the operation's inputs and result:
// the tracked tensors when what they compute is to be differentiated in
// turn, and detached ones otherwise. Written with tensor operations on
// what they receive, a rule gives derivatives of every order.
type rule struct {
	// vjp maps the gradient g of the result to one gradient per input,
	// shaped like the input, or nil for an input without one.
	vjp func(g *Tensor, in []*Tensor, out *Tensor) []*Tensor
	// jvp maps one tangent per input, nil for zero, to the result's
	// tangent, or nil for zero. It runs only when a tangent is not nil. A
	// rule without one has it derived from vjp, which is linear in g, by
	// transposing vjp (see transposed).
	jvp func(t, in []*Tensor, out *Tensor) *Tensor
	// firstOrder marks a vjp that reads tensors it captured besides g, in
	// and out: differentiating what it computes would take them for
	// constants, so it gives first derivatives only.
	firstOrder bool
}

// identity is the rule of an operation that passes its one input on.
var identity = rule{
	vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(g) },
	jvp: func(t, _ []*Tensor, _ *Tensor) *Tensor { return t[0] },
}

// tapes counts the tapes made. A tape's seq tells whether it was made
// before or after a point in time: Checkpoint tells the tapes of a
// recomputed function from those of tensors it captured by it, and JVP
// the tapes of the function it pushes tangents through from others.
var tapes atomic.Uint64

// newLeaf returns the tape of a leaf of n.
func newLeaf(n *graph.Node) *tape {
	return &tape{seq: tapes.Add(1), node: n, leaf: true}
}

// tracked reports whether any of ts requires gradients.
func tracked(ts ...*Tensor) bool {
	for _, t := range ts {
		if t != nil && t.tape != nil {
			return true
		}
	}
	return false
}

// anyTracked reports whether any of ts requires gradients at some level:
// itself, or, for a tensor inside VMap, the batch it stands for. A
// derivative computed from such tensors must be differentiable.
func anyTracked(ts ...*Tensor) bool {
	for _, t := range ts {
		for t != nil {
			if t.tape != nil {
				return true
			}
			b := t.batch()
			if b == nil {
				break
			}
			t = b.phys
		}
	}
	return false
}

// record returns out, which an operation computed from ins, with a tape
// whose rule is r when any input requires gradients, and out unchanged
// otherwise.
func record(out *Tensor, r rule, ins ...*Tensor) *Tensor {
	if out.err != nil || !tracked(ins...) {
		return out
	}
	tp := &tape{seq: tapes.Add(1), node: out.node, ins: slices.Clone(ins), rule: r}
	return &Tensor{node: out.node, tape: tp}
}

// track is record with a first-order rule: vjp maps the gradient of the
// result to one gradient per input, from g and detached tensors it
// captured. The gradients it computes cannot be differentiated in turn:
// that fails with ErrNotDifferentiable. JVP derives the forward
// derivative from it by transposition. record with a rule that reads its
// inputs and result gives derivatives of every order. vjp computes
// with captured tensors outside VMap's batching, so a batched g fails
// with ErrNotBatchable.
func track(out *Tensor, vjp func(g *Tensor) []*Tensor, ins ...*Tensor) *Tensor {
	if out.err != nil || !tracked(ins...) {
		return out
	}
	return record(out, rule{
		vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor {
			if batchLevel(g) != 0 {
				err := fail(fmt.Errorf("tensor: batched gradient of a first-order rule: %w", ErrNotBatchable))
				return slices.Repeat([]*Tensor{err}, len(ins))
			}
			return vjp(g)
		},
		firstOrder: true,
	}, ins...)
}

// firstOrderOnly returns d, a derivative computed from deps, which may
// hold nils, but valid to first order only, with a tape that fails with
// ErrNotDifferentiable when it is differentiated with respect to a
// tracked one of deps. It is nil for a nil d.
func firstOrderOnly(d *Tensor, deps []*Tensor) *Tensor {
	if d == nil {
		return nil
	}
	deps = slices.DeleteFunc(slices.Clone(deps), func(t *Tensor) bool { return t == nil })
	return record(d, failing(fmt.Errorf("tensor: derivative of a derivative that a first-order rule computed: %w", ErrNotDifferentiable)), deps...)
}

// failing returns a rule whose derivatives all fail with err.
func failing(err error) rule {
	return rule{
		vjp: func(_ *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return failAll(len(in), err) },
		jvp: func(_, _ []*Tensor, _ *Tensor) *Tensor { return fail(err) },
	}
}

// grads is shorthand for a gradient rule's result.
func grads(gs ...*Tensor) []*Tensor { return gs }

// standIn returns a tensor with x's values whose tape stands for x's in a
// function that a transformation calls: its walks stop there, and the
// derivatives that reach it flow on to x, if x requires gradients.
func standIn(x *Tensor) *Tensor {
	return &Tensor{node: x.node, tape: &tape{seq: tapes.Add(1), node: x.node, ins: []*Tensor{x}, rule: identity, stand: true}}
}

// detached returns t without its tape. Gradient rules that do not need
// to be differentiated receive detached inputs, so that computing
// gradients records nothing. A tensor inside VMap keeps the batch it
// stands for, which may require gradients of its own: what detached
// leaves out is the tape of this level only (see Detach).
func (t *Tensor) detached() *Tensor {
	if t == nil || t.tape == nil {
		return t
	}
	return &Tensor{node: t.node, err: t.err}
}

// Detach returns t's values without gradient tracking, like
// torch.Tensor.detach and jax.lax.stop_gradient. Gradients do not flow
// through the result, inside VMap or around it.
func (t *Tensor) Detach() *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return t.stopGradient()
}

// RequiresGrad returns a tensor with t's values whose gradient Backward
// computes and accumulates in Grad. Only float tensors have gradients; a
// half tensor's gradient has the dtype the chain rule gives it, its own
// or Float32.
func (t *Tensor) RequiresGrad() *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if !t.node.DType.IsFloat() {
		return fail(fmt.Errorf("tensor: gradient of %s tensor: %w", t.node.DType, ErrDType))
	}
	return &Tensor{node: t.node, tape: newLeaf(t.node)}
}

// Tracked reports whether t requires gradients: it comes from
// RequiresGrad or a Param, or was computed from such a tensor.
func (t *Tensor) Tracked() bool { return firstErr(t) == nil && t.tape != nil }

// Grad returns the gradient Backward accumulated for t, which must come
// from RequiresGrad or Param.Value. It returns nil if no gradient has been
// accumulated since t was created or last zeroed.
func (t *Tensor) Grad() *Tensor {
	if firstErr(t) != nil || t.tape == nil || !t.tape.leaf {
		return nil
	}
	gradMu.Lock()
	defer gradMu.Unlock()
	return t.tape.grad
}

// ZeroGrad discards t's accumulated gradient. The next Backward computes
// the new gradient into the old one's memory, so the old gradient, if
// kept, is stale: reading it returns ErrOverwritten. A parameter's value
// not computed yet that reads the gradient keeps it instead.
func (t *Tensor) ZeroGrad() {
	if firstErr(t) != nil || t.tape == nil || !t.tape.leaf {
		return
	}
	gradMu.Lock()
	defer gradMu.Unlock()
	if g := t.tape.grad; g != nil && t.tape.own {
		sched.Release(g.node, lazy.values)
	}
	t.tape.grad, t.tape.own = nil, false
}

// Backward computes the gradient of t, which has one element, with respect
// to every tensor that requires gradients it was computed from, and adds
// each gradient to that tensor's Grad, like torch.Tensor.backward.
//
// Backward computes t itself in the same schedule, so work shared by the
// forward and backward passes runs once: call Backward before reading t.
// Functions computed with Checkpoint are recomputed in that schedule
// instead of keeping their activations from the forward pass. On success
// the graph between t and its leaves is released, and a later Backward
// through it returns ErrNoGrad. A gradient that Backward adds to is
// updated in place, so a gradient read from Grad before it is stale after
// it (see RealizeInto). If Backward fails before it starts updating
// gradients no gradient changes; cancelling ctx after that does not stop
// it.
//
// The gradients Backward computes are values, not differentiable
// tensors. Gradients and the transformations built on it, such as Grad
// and Hessian, return gradients that can be differentiated again.
//
// Backwards on different goroutines run one at a time. A function that
// Backward recomputes, or a gradient rule of Func, may read gradients but
// must not call Backward, which would wait for the Backward running it.
func (t *Tensor) Backward(ctx context.Context) error {
	if err := firstErr(t); err != nil {
		return err
	}
	if err := computable(t); err != nil {
		return err
	}
	if t.tape == nil {
		return fmt.Errorf("tensor: backward: %w", ErrNoGrad)
	}
	if n := t.node.Size(); n != 1 {
		return fmt.Errorf("tensor: backward of %d elements, want 1: %w", n, ErrShape)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	backwardMu.Lock()
	defer backwardMu.Unlock()

	// The gradient rules run without gradMu: those of checkpoints run
	// user code, which may read gradients.
	r := reverse{backward: true}
	_, walked, effects, err := r.propagate([]*tape{t.tape}, []*Tensor{Ones(t.node.DType, t.node.Shape...)}, nil)
	if err != nil {
		return err
	}
	leaves := r.order

	// From reading the leaves' gradients to writing their sums back,
	// gradMu keeps ZeroGrad and Param.To from changing or freeing them:
	// they see the gradients before this Backward or after it.
	gradMu.Lock()
	defer gradMu.Unlock()

	// The sums go over the memory of the gradients they add to, where
	// that memory is the tapes' own. A sum that Backward computes, rather
	// than one that is already realized, is in memory of its own; leaves
	// whose gradients are the same value, as those of a and b in a+b, each
	// get a copy, so ZeroGrad on one leaves the others alone.
	sums := make([]*Tensor, len(leaves))
	own := make([]bool, len(leaves))
	targets := append(make([]*Tensor, 0, 1+len(leaves)+len(effects)), t)
	dsts := make([]*Tensor, 1, cap(targets))
	computed := map[*graph.Node]bool{t.node: true}
	for i, l := range leaves {
		sums[i] = r.got[l]
		var dst *Tensor
		if l.grad != nil {
			sums[i] = l.grad.Add(sums[i])
			if l.own && sums[i].node.DType == l.grad.node.DType {
				dst = l.grad
			}
		}
		if n := sums[i].node; computed[n] && !n.Realized() {
			sums[i] = sums[i].Clone()
		}
		computed[sums[i].node] = true
		own[i] = !sums[i].node.Realized()
		targets, dsts = append(targets, sums[i]), append(dsts, dst)
	}
	targets = append(targets, effects...)
	dsts = append(dsts, make([]*Tensor, len(effects))...)
	if err := RealizeInto(ctx, dsts, targets); err != nil {
		return err
	}
	for i, l := range leaves {
		// A gradient that shares its memory, as those of a and b in
		// a.Reshape(...).Add(b) do, is not the tape's to write over.
		l.grad, l.own = sums[i], own[i] && sums[i].node.Mem.Refs == 1
	}
	for _, n := range walked.tapes {
		if !n.leaf {
			n.ins, n.rule, n.effects, n.ckpt, n.part, n.released = nil, rule{}, nil, nil, nil, true
		}
	}
	return nil
}

// A walk is the tapes reachable from a result, each after the tapes of
// its inputs, as a pass of the chain rule sees them.
type walk struct {
	tapes []*tape
	pos   map[*tape]int // the index of each tape in tapes
	// copies holds a copy of each tape, taken under gradMu, for a pass
	// that may run while a Backward releases the tapes; nil otherwise.
	copies []tape
}

// at returns the i-th tape as the walk saw it.
func (w *walk) at(i int) *tape {
	if w.copies != nil {
		return &w.copies[i]
	}
	return w.tapes[i]
}

// walkTapes returns the walk of the tapes reachable from roots. It does
// not go past a tape that stop, if not nil, reports, nor into a tape made
// when no more than since tapes existed, if since is not 0, unless stop
// reports it. It walks
// iteratively, so deep graphs cannot overflow the stack, and fails with
// ErrNoGrad on a tape it would go past that Backward released. With
// copies it reads copies of the tapes, for a pass outside Backward.
func walkTapes(roots []*tape, stop func(*tape) bool, since uint64, copies bool) (*walk, error) {
	gradMu.Lock()
	defer gradMu.Unlock()
	type frame struct {
		n    *tape
		next int
	}
	// pos holds -1 for a tape on the stack, until its index is known.
	w := &walk{pos: map[*tape]int{}}
	var stack []frame
	for _, root := range roots {
		if _, seen := w.pos[root]; root == nil || seen || since > 0 && root.seq <= since && (stop == nil || !stop(root)) {
			continue
		}
		w.pos[root] = -1
		stack = append(stack, frame{n: root})
		for len(stack) > 0 {
			f := &stack[len(stack)-1]
			last := stop != nil && stop(f.n)
			if !last && f.n.released {
				return nil, fmt.Errorf("tensor: derivative through a graph an earlier Backward released: %w", ErrNoGrad)
			}
			if !last && f.next < f.n.deps() {
				p, _ := f.n.dep(f.next)
				f.next++
				if _, seen := w.pos[p]; p != nil && !seen && (since == 0 || p.seq > since || stop != nil && stop(p)) {
					w.pos[p] = -1
					stack = append(stack, frame{n: p})
				}
				continue
			}
			w.pos[f.n] = len(w.tapes)
			w.tapes = append(w.tapes, f.n)
			stack = stack[:len(stack)-1]
		}
	}
	if copies {
		w.copies = make([]tape, len(w.tapes))
		for i, n := range w.tapes {
			w.copies[i] = *n
		}
	}
	return w, nil
}

// A reverse pass applies the chain rule from a result back to the tapes
// it was computed from, and collects the gradients of the tapes it
// wants.
type reverse struct {
	// create hands the rules tracked tensors, so that the gradients they
	// compute can be differentiated in turn. Otherwise the rules get
	// detached tensors.
	create bool
	// backward marks the pass of Backward, which holds backwardMu, so
	// that no other Backward releases the tapes it walks. Other passes
	// read copies of them.
	backward bool
	// want reports the tapes whose gradients the pass collects: every
	// leaf if nil, as Backward collects.
	want map[*tape]bool
	// fenced makes the stops of the pass bound its walk alone: their
	// gradients are not wanted.
	fenced bool

	got   map[*tape]*Tensor // the gradients collected
	order []*tape           // the tapes collected, in the order reached
	// firstOrder holds the inputs of the first-order rules walked in a
	// pass that creates differentiable gradients (see result).
	firstOrder []*Tensor
}

// result returns g, a gradient that r collected, as it is to be
// differentiated in turn. If the pass walked a first-order rule, g's
// derivatives fail with ErrNotDifferentiable, whether or not g came
// through the rule: such a rule may stand for derivatives that the rules
// around it leave out at every order but the first, as nn's recurrent
// layers give W_hh the gradient that the steps, which read it detached,
// do not.
func (r *reverse) result(g *Tensor) *Tensor {
	if g == nil || r.firstOrder == nil {
		return g
	}
	return firstOrderOnly(g, append([]*Tensor{g}, r.firstOrder...))
}

// wants reports whether r collects the gradient of t.
func (r *reverse) wants(t *tape) bool {
	if r.want == nil {
		return t.leaf
	}
	return r.want[t]
}

// add adds g to the gradient collected for t.
func (r *reverse) add(t *tape, g *Tensor) {
	if r.got == nil {
		r.got = map[*tape]*Tensor{}
	}
	if prev, ok := r.got[t]; ok {
		g = prev.Add(g)
	} else {
		r.order = append(r.order, t)
	}
	r.got[t] = g
}

// propagate applies the chain rule from roots, whose gradients are gs,
// through the tapes reachable from them, not past those of stops: each rule
// whose result leads to a wanted tape runs once, on the sum of the
// gradients its result received, and the gradient of each wanted tape
// reached is added to r. It returns the gradients that reached the tapes
// of stops, nil for none, the walk and the effects of the tapes walked.
// Backward uses it for the loss, Checkpoint for the function it
// recomputes, and Gradients and the transformations for theirs.
func (r *reverse) propagate(roots []*tape, gs []*Tensor, stops []*tape) (atStops []*Tensor, w *walk, effects []*Tensor, err error) {
	var stop func(*tape) bool
	if len(stops) > 0 {
		stop = func(t *tape) bool { return slices.Contains(stops, t) }
	}
	w, err = walkTapes(roots, stop, 0, !r.backward)
	if err != nil {
		return nil, nil, nil, err
	}
	n := len(w.tapes)
	// need tells the tapes from which a wanted tape or a stop is reached:
	// the rules of the others need not run.
	need := make([]bool, n)
	for i, t := range w.tapes {
		v := w.at(i)
		if stop != nil && stop(t) && !r.fenced || r.wants(t) {
			need[i] = true
			continue
		}
		for k := range v.deps() {
			if d, _ := v.dep(k); d != nil {
				if j, ok := w.pos[d]; ok && need[j] {
					need[i] = true
					break
				}
			}
		}
	}
	var args *arena
	if !r.create {
		args = newArena(w)
	}
	acc := make([]*Tensor, n)
	for k, root := range roots {
		i := w.pos[root]
		acc[i] = addGrad(acc[i], gs[k])
	}
	atStops = make([]*Tensor, len(stops))
	// The gradients of the parts of each split tensor, by the tensor's
	// tape, are joined once every part has had its turn: the parts come
	// after the tensor they were split from.
	var joins map[*tape][]*joining
	for i := n - 1; i >= 0; i-- {
		t, v := w.tapes[i], w.at(i)
		effects = append(effects, v.effects...)
		for _, j := range joins[t] {
			acc[i] = addGrad(acc[i], j.join())
		}
		delete(joins, t)
		g := acc[i]
		if r.create && v.rule.firstOrder {
			r.firstOrder = append(r.firstOrder, v.ins...)
		}
		if g == nil || !need[i] {
			continue
		}
		acc[i] = nil // its inputs' gradients are all that is needed now
		if stop != nil && stop(t) {
			atStops[slices.Index(stops, t)] = g
			continue
		}
		if r.wants(t) {
			r.add(t, g)
		}
		if v.leaf {
			continue
		}
		var gi []*Tensor
		switch {
		case v.part != nil:
			whole := v.ins[0].tape
			if joins == nil {
				joins = map[*tape][]*joining{}
			}
			k := slices.IndexFunc(joins[whole], func(j *joining) bool { return j.of == v.part.of })
			if k < 0 {
				joins[whole] = append(joins[whole], &joining{of: v.part.of, grads: make([]*Tensor, len(v.part.of.shapes))})
				k = len(joins[whole]) - 1
			}
			joins[whole][k].grads[v.part.i] = g
			continue
		case v.ckpt != nil:
			if gi, err = v.ckpt.recompute(r, v.ins, g); err != nil {
				return nil, nil, nil, err
			}
		case v.rule.firstOrder && r.create:
			gi = v.rule.vjp(g.detached(), detachAll(v.ins), &Tensor{node: v.node})
			deps := append([]*Tensor{g}, v.ins...)
			for j := range gi {
				gi[j] = firstOrderOnly(gi[j], deps)
			}
		default:
			in, out := v.ins, &Tensor{node: v.node, tape: t}
			if args != nil {
				in, out = args.detach(v)
			}
			gi = v.rule.vjp(g, in, out)
		}
		for j := range min(v.deps(), len(gi)) {
			d, shape := v.dep(j)
			k, ok := w.pos[d]
			if d == nil || !ok || !need[k] || gi[j] == nil {
				continue
			}
			gj := gi[j]
			if !r.create {
				gj = gj.detached()
			}
			if err := gj.Err(); err != nil {
				return nil, nil, nil, fmt.Errorf("tensor: backward: %w", err)
			}
			if !slices.Equal(gj.node.Shape, shape) {
				return nil, nil, nil, fmt.Errorf("tensor: backward: gradient of shape %v for input of shape %v: %w", gj.node.Shape, shape, ErrShape)
			}
			acc[k] = addGrad(acc[k], gj)
		}
	}
	return atStops, w, effects, nil
}

// addGrad returns the sum of two gradients of one tensor, either of which
// may be nil for none.
func addGrad(a, b *Tensor) *Tensor {
	switch {
	case a == nil:
		return b
	case b == nil:
		return a
	}
	return a.Add(b)
}

// An arena hands the rules of a pass whose results are not differentiated
// detached inputs and results, taken from memory allocated once for the
// pass rather than once per rule.
type arena struct {
	vals []Tensor
	ptrs []*Tensor
}

func newArena(w *walk) *arena {
	n, k := 0, 0
	for i := range w.tapes {
		if v := w.at(i); !v.leaf && v.part == nil && v.ckpt == nil {
			n += len(v.ins) + 1
			k += len(v.ins)
		}
	}
	return &arena{vals: make([]Tensor, n), ptrs: make([]*Tensor, k)}
}

// detach returns the inputs and result of v's rule, detached.
func (a *arena) detach(v *tape) ([]*Tensor, *Tensor) {
	k := len(v.ins)
	vals, ptrs := a.vals[:k+1:k+1], a.ptrs[:k:k]
	a.vals, a.ptrs = a.vals[k+1:], a.ptrs[k:]
	for i, x := range v.ins {
		if x != nil {
			vals[i] = Tensor{node: x.node}
			ptrs[i] = &vals[i]
		}
	}
	vals[k] = Tensor{node: v.node}
	return ptrs, &vals[k]
}

// WithEffects returns t, with the same values and gradients, carrying
// effects: tensors that Backward computes in the same schedule as the
// gradients whenever it runs on a result computed from t. Layers use it
// for state that a training step updates on the side, such as the running
// statistics of batch normalization, so the update costs no extra pass.
// Gradients and the transformations ignore effects.
func (t *Tensor) WithEffects(effects ...*Tensor) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if err := firstErr(effects...); err != nil {
		return fail(err)
	}
	if len(effects) == 0 {
		return t
	}
	det := make([]*Tensor, len(effects))
	for i, e := range effects {
		det[i] = e.detached()
	}
	tp := &tape{seq: tapes.Add(1), node: t.node, ins: []*Tensor{t}, rule: identity, effects: det}
	return &Tensor{node: t.node, tape: tp}
}

// Func is an operation with a hand-written gradient, like
// torch.autograd.Function. Forward computes the result from inputs that
// do not require gradients. Backward receives the gradient of the result
// along with the inputs and result, and returns one gradient per input,
// each with that input's shape, or nil for an input without one.
//
// JVP pushes tangents through a Func by transposing Backward, which must
// be linear in the gradient it receives, as every gradient is.
type Func struct {
	Forward  func(in []*Tensor) *Tensor
	Backward func(g *Tensor, in []*Tensor, out *Tensor) []*Tensor
	// Differentiable declares that Backward computes the gradients from
	// g, in and out alone, with tensor operations, so that they can be
	// differentiated in turn (see Gradients): Backward then receives the
	// tracked inputs and result when they are. Otherwise differentiating
	// the gradients of a Func fails with ErrNotDifferentiable, since the
	// tensors Backward captured from elsewhere would count as constants.
	Differentiable bool
}

// Apply computes f on in, recording f.Backward as the gradient rule.
//
// Inside VMap, Backward gives the derivatives that the function mapped
// computes, such as per-example gradients. A derivative through the
// result taken outside VMap fails with ErrNotBatchable.
func (f Func) Apply(in ...*Tensor) *Tensor {
	if err := firstErr(in...); err != nil {
		return fail(err)
	}
	det := make([]*Tensor, len(in))
	for i, t := range in {
		det[i] = t.stopGradient()
	}
	out := f.Forward(det)
	if err := firstErr(out); err != nil {
		return fail(err)
	}
	out = severed(out.stopGradient(), in)
	if !tracked(in...) {
		return out
	}
	return record(out, rule{
		vjp:        func(g *Tensor, in []*Tensor, out *Tensor) []*Tensor { return f.Backward(g, in, out) },
		firstOrder: !f.Differentiable,
	}, in...)
}

// Param is a trainable tensor: a value that an optimizer replaces between
// steps, and the gradient that Backward accumulates for it. Computations
// use Value. Set replaces the value, and the tensors built before keep the
// value they were built with; an optimizer step instead writes the new
// value over the old one's memory, after which the old value is stale (see
// RealizeInto). A Param is safe for concurrent use.
type Param struct {
	mu        sync.Mutex
	value     *Tensor
	trainable bool
}

// NewParam returns a trainable parameter holding a copy of init's values.
// It computes them now, so the parameter holds memory of its own rather
// than a recipe. If that fails, Value carries the error, as it does for a
// block-quantized init, which cannot be trained.
func NewParam(init *Tensor) *Param { return newParam(init, true) }

// NewBuffer returns a parameter that is saved and restored with a model
// but not trained: Backward computes no gradient for it. Batch
// normalization keeps its running statistics in buffers. A
// block-quantized init, such as weights read from a GGUF file, is held
// as it is rather than copied, since nothing writes over it.
func NewBuffer(init *Tensor) *Param { return newParam(init, false) }

func newParam(init *Tensor, trainable bool) *Param {
	v := init.Detach()
	switch {
	case v.err != nil:
	case !v.node.DType.IsQuantized():
		v = v.Clone()
	case trainable:
		v = fail(fmt.Errorf("tensor: trainable parameter of %s: %w", v.node.DType, ErrDType))
	}
	if v.err == nil {
		if err := v.Realize(context.Background()); err != nil {
			v = fail(err)
		}
	}
	p := &Param{trainable: trainable}
	if trainable {
		v = v.RequiresGrad()
		if v.tape != nil {
			v.tape.param = p
		}
	}
	p.value = v
	return p
}

// Value returns the parameter's current value.
func (p *Param) Value() *Tensor {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.value
}

// Trainable reports whether Backward computes gradients for p.
func (p *Param) Trainable() bool { return p.trainable }

// Grad returns the gradient accumulated since the last ZeroGrad, or nil.
func (p *Param) Grad() *Tensor { return p.Value().Grad() }

// ZeroGrad discards the accumulated gradient.
func (p *Param) ZeroGrad() { p.Value().ZeroGrad() }

// Set replaces the value with v, which must have the same shape. A
// trainable parameter keeps a float dtype, which v may change, as a cast
// of a model to BFloat16 does; a buffer takes any dtype. The accumulated
// gradient is kept. Optimizers compute v first, so the parameter never
// holds a growing recipe; a lazy v is allowed for buffers updated with
// WithEffects. The parameter takes over v's memory, which optimizer steps
// write over, so pass v.Clone() to keep using v.
//
// A lazy v stays computable: a RealizeInto that would write over memory
// it reads, such as an optimizer step over the weights that running
// statistics were computed from, computes it first, and a ZeroGrad of a
// gradient it reads keeps the gradient's memory.
func (p *Param) Set(v *Tensor) error {
	if err := firstErr(v); err != nil {
		return err
	}
	if err := computable(v); err != nil {
		return err
	}
	if err := p.set(v); err != nil {
		return err
	}
	if !sched.Realized(v.node) {
		lazy.add(p)
	}
	return nil
}

func (p *Param) set(v *Tensor) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.value.err; err != nil {
		return err
	}
	if !slices.Equal(v.node.Shape, p.value.node.Shape) {
		return fmt.Errorf("tensor: set %s%v parameter to %s%v: %w", p.value.node.DType, p.value.node.Shape, v.node.DType, v.node.Shape, ErrShape)
	}
	if p.trainable && !v.node.DType.IsFloat() {
		return fmt.Errorf("tensor: set trainable %s parameter to %s: %w", p.value.node.DType, v.node.DType, ErrDType)
	}
	p.value = &Tensor{node: v.node, tape: p.value.tape}
	return nil
}

// To moves the parameter's value, and the gradient accumulated since the
// last ZeroGrad, to d. Tensors built from the old value keep it.
func (p *Param) To(d Device) error {
	v := p.Value()
	if err := firstErr(v); err != nil {
		return err
	}
	moved := v.Detach().To(d)
	if err := moved.Realize(context.Background()); err != nil {
		return err
	}
	// The gradient is read, copied and replaced under gradMu, so that a
	// Backward adding to it waits for the copy instead of adding to the
	// gradient the copy then replaces.
	gradMu.Lock()
	defer gradMu.Unlock()
	var grad *Tensor
	if v.tape != nil && v.tape.grad != nil {
		g := v.tape.grad
		if grad = g.To(d); grad.err == nil {
			grad.err = grad.Realize(context.Background())
		}
		if grad.err != nil {
			return grad.err
		}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.value != v {
		return fmt.Errorf("tensor: parameter set while moving to %v: %w", d, ErrInternal)
	}
	p.value = &Tensor{node: moved.node, tape: v.tape}
	// On the device it is on already the gradient is not copied, and
	// keeps whether its memory is the tape's.
	if grad != nil && grad != v.tape.grad {
		v.tape.grad, v.tape.own = grad, true
	}
	return nil
}

// lazy holds the parameters that were set to values not yet computed.
// Weak pointers let a parameter that nothing else holds go.
var lazy lazyParams

type lazyParams struct {
	mu sync.Mutex
	ps map[weak.Pointer[Param]]struct{}
}

func (l *lazyParams) add(p *Param) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.ps == nil {
		l.ps = map[weak.Pointer[Param]]struct{}{}
	}
	l.ps[weak.Make(p)] = struct{}{}
}

// values returns the values of the parameters that are still not
// computed, and forgets the others. The scheduler calls it with its lock
// held, which makes reading whether a value is computed safe.
func (l *lazyParams) values() []*graph.Node {
	l.mu.Lock()
	defer l.mu.Unlock()
	var nodes []*graph.Node
	for wp := range l.ps {
		p := wp.Value()
		if p == nil {
			delete(l.ps, wp)
			continue
		}
		v := p.Value()
		if v.err != nil || v.node.Realized() {
			delete(l.ps, wp)
			continue
		}
		nodes = append(nodes, v.node)
	}
	return nodes
}
