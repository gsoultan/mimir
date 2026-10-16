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
// leaves and the release of graphs after Backward. Building tensors reads
// only the immutable parts of tapes and never takes it.
var gradMu sync.Mutex

// tape records how a tracked tensor was computed so Backward can apply the
// chain rule. A leaf's tape has no rule and instead holds the gradient
// accumulated for it.
type tape struct {
	seq   uint64 // the count of tapes when this one was made
	shape []int  // of the tensor; gradients must match it

	// parents has one entry per input: the input's tape, or nil when the
	// input does not require gradients.
	parents []*tape
	// vjp maps the gradient of the result to one gradient per input. It
	// runs only during Backward, on detached snapshots it captured.
	vjp func(g *Tensor) []*Tensor
	// effects are computed in the same schedule as the gradients.
	effects []*Tensor

	leaf     bool
	grad     *Tensor // accumulated gradient of a leaf
	own      bool    // grad's memory is the tape's to reuse
	released bool    // Backward has run through this node
}

// tapes counts the tapes made. A tape's seq tells whether it was made
// before or after a point in time: Checkpoint tells the tapes of a
// recomputed function from those of tensors it captured by it.
var tapes atomic.Uint64

// newTape returns the tape of a tensor of shape computed from the tensors
// of parents by the gradient rule vjp.
func newTape(shape []int, parents []*tape, vjp func(g *Tensor) []*Tensor) *tape {
	return &tape{seq: tapes.Add(1), shape: shape, parents: parents, vjp: vjp}
}

// newLeaf returns the tape of a leaf of shape.
func newLeaf(shape []int) *tape {
	return &tape{seq: tapes.Add(1), shape: shape, leaf: true}
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

// track returns out with a tape whose gradient rule is vjp when any input
// requires gradients, and out unchanged otherwise.
func track(out *Tensor, vjp func(g *Tensor) []*Tensor, ins ...*Tensor) *Tensor {
	if out.err != nil || !tracked(ins...) {
		return out
	}
	parents := make([]*tape, len(ins))
	for i, in := range ins {
		parents[i] = in.tape
	}
	return &Tensor{node: out.node, tape: newTape(out.node.Shape, parents, vjp)}
}

// grads is shorthand for a gradient rule's result.
func grads(gs ...*Tensor) []*Tensor { return gs }

// detached returns t without its tape. Gradient rules capture detached
// snapshots so that computing gradients records nothing.
func (t *Tensor) detached() *Tensor {
	if t == nil || t.tape == nil {
		return t
	}
	return &Tensor{node: t.node, err: t.err}
}

// Detach returns t's values without gradient tracking, like
// torch.Tensor.detach. Gradients do not flow through the result.
func (t *Tensor) Detach() *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return t.detached()
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
	return &Tensor{node: t.node, tape: newLeaf(t.node.Shape)}
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
func (t *Tensor) Backward(ctx context.Context) error {
	if err := firstErr(t); err != nil {
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
	gradMu.Lock()
	defer gradMu.Unlock()
	bwd = backwardState{acc: map[*tape]*Tensor{}}
	defer func() { bwd = backwardState{} }()

	order, effects, err := propagate(t.tape, Ones(t.node.DType, t.node.Shape...))
	if err != nil {
		return err
	}
	effects = append(effects, bwd.effects...)
	// A leaf read only by a checkpointed function is not among order: its
	// gradient came from the function's recompute.
	leaves := bwd.leaves()

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
		sums[i] = bwd.acc[l]
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
	for _, n := range order {
		if !n.leaf {
			n.parents, n.vjp, n.effects, n.released = nil, nil, nil, true
		}
	}
	return nil
}

// bwd is the state of the Backward that is running: the gradient
// accumulated for each leaf reached, the order the leaves were reached in,
// the effects of checkpointed functions recomputed on the way and the
// first error one of them met. Gradient rules run while Backward holds
// gradMu, which guards it; the rule of a checkpoint (see Checkpoint)
// propagates gradients through the function it recomputes into it, since
// the parameters the function reads are leaves of this Backward, not
// inputs of the checkpoint.
var bwd backwardState

type backwardState struct {
	acc     map[*tape]*Tensor
	order   []*tape
	effects []*Tensor
	err     error
}

// add adds g to the gradient accumulated for the leaf l.
func (s *backwardState) add(l *tape, g *Tensor) {
	if prev, ok := s.acc[l]; ok {
		g = prev.Add(g)
	} else {
		s.order = append(s.order, l)
	}
	s.acc[l] = g
}

// take removes and returns the gradient accumulated for the leaf l, or nil
// if it received none.
func (s *backwardState) take(l *tape) *Tensor {
	g := s.acc[l]
	delete(s.acc, l)
	return g
}

// leaves returns the leaves holding a gradient, each once, in the order
// they were reached.
func (s *backwardState) leaves() []*tape {
	var ls []*tape
	seen := make(map[*tape]bool, len(s.acc))
	for _, l := range s.order {
		if _, ok := s.acc[l]; ok && !seen[l] {
			seen[l] = true
			ls = append(ls, l)
		}
	}
	return ls
}

// fail records err as the error of the running Backward, unless one is
// recorded already.
func (s *backwardState) fail(err error) {
	if s.err == nil {
		s.err = err
	}
}

// propagate applies the chain rule from root, whose gradient is g, through
// the tapes reachable from it: each gradient rule runs once, on the sum of
// the gradients its result received, and the gradient of each leaf
// reached is added to bwd. It returns the tapes it walked, each after its
// parents, and the effects they carry. Backward uses it for the loss, and
// Checkpoint for the function it recomputes.
func propagate(root *tape, g *Tensor) (order []*tape, effects []*Tensor, err error) {
	order, err = tapeOrder(root)
	if err != nil {
		return nil, nil, err
	}
	acc := map[*tape]*Tensor{root: g}
	for _, n := range slices.Backward(order) {
		effects = append(effects, n.effects...)
		g, ok := acc[n]
		if !ok {
			continue
		}
		delete(acc, n) // its inputs' gradients are all that is needed now
		if n.leaf {
			bwd.add(n, g)
			continue
		}
		gs := n.vjp(g)
		if bwd.err != nil {
			return nil, nil, bwd.err
		}
		for i, p := range n.parents {
			if p == nil || i >= len(gs) || gs[i] == nil {
				continue
			}
			gi := gs[i].detached()
			if err := gi.Err(); err != nil {
				return nil, nil, fmt.Errorf("tensor: backward: %w", err)
			}
			if !slices.Equal(gi.node.Shape, p.shape) {
				return nil, nil, fmt.Errorf("tensor: backward: gradient of shape %v for input of shape %v: %w", gi.node.Shape, p.shape, ErrShape)
			}
			if prev, ok := acc[p]; ok {
				gi = prev.Add(gi)
			}
			acc[p] = gi
		}
	}
	return order, effects, nil
}

// tapeOrder returns the tapes reachable from root, each after its parents.
// It walks iteratively so deep graphs cannot overflow the stack.
func tapeOrder(root *tape) ([]*tape, error) {
	type frame struct {
		n    *tape
		next int
	}
	var order []*tape
	seen := map[*tape]bool{root: true}
	stack := []frame{{n: root}}
	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		if f.n.released {
			return nil, fmt.Errorf("tensor: backward through a graph an earlier Backward released: %w", ErrNoGrad)
		}
		if f.next < len(f.n.parents) {
			p := f.n.parents[f.next]
			f.next++
			if p != nil && !seen[p] {
				seen[p] = true
				stack = append(stack, frame{n: p})
			}
			continue
		}
		order = append(order, f.n)
		stack = stack[:len(stack)-1]
	}
	return order, nil
}

// WithEffects returns t, with the same values and gradients, carrying
// effects: tensors that Backward computes in the same schedule as the
// gradients whenever it runs on a result computed from t. Layers use it
// for state that a training step updates on the side, such as the running
// statistics of batch normalization, so the update costs no extra pass.
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
	identity := func(g *Tensor) []*Tensor { return grads(g) }
	tp := newTape(t.node.Shape, []*tape{t.tape}, identity)
	tp.effects = det
	return &Tensor{node: t.node, tape: tp}
}

// Func is an operation with a hand-written gradient, like
// torch.autograd.Function. Forward computes the result from inputs that
// do not require gradients. Backward receives the gradient of the result
// along with the inputs and result, and returns one gradient per input,
// each with that input's shape, or nil for an input without one.
type Func struct {
	Forward  func(in []*Tensor) *Tensor
	Backward func(g *Tensor, in []*Tensor, out *Tensor) []*Tensor
}

// Apply computes f on in, recording f.Backward as the gradient rule.
func (f Func) Apply(in ...*Tensor) *Tensor {
	if err := firstErr(in...); err != nil {
		return fail(err)
	}
	det := make([]*Tensor, len(in))
	for i, t := range in {
		det[i] = t.detached()
	}
	out := f.Forward(det)
	if err := firstErr(out); err != nil {
		return fail(err)
	}
	out = out.detached()
	return track(out, func(g *Tensor) []*Tensor { return f.Backward(g, det, out) }, in...)
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
// than a recipe. If that fails, Value carries the error.
func NewParam(init *Tensor) *Param { return newParam(init, true) }

// NewBuffer returns a parameter that is saved and restored with a model
// but not trained: Backward computes no gradient for it. Batch
// normalization keeps its running statistics in buffers.
func NewBuffer(init *Tensor) *Param { return newParam(init, false) }

func newParam(init *Tensor, trainable bool) *Param {
	v := init.Detach().Clone()
	if v.err == nil {
		if err := v.Realize(context.Background()); err != nil {
			v = fail(err)
		}
	}
	if trainable {
		v = v.RequiresGrad()
	}
	return &Param{value: v, trainable: trainable}
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
	var grad *Tensor
	if g := v.Grad(); g != nil {
		if grad = g.To(d); grad.err == nil {
			grad.err = grad.Realize(context.Background())
		}
		if grad.err != nil {
			return grad.err
		}
	}
	if grad != nil {
		// Backward holds gradMu while the functions it recomputes read
		// parameters, which takes p.mu: the same order here.
		gradMu.Lock()
		defer gradMu.Unlock()
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.value != v {
		return fmt.Errorf("tensor: parameter set while moving to %v: %w", d, ErrInternal)
	}
	p.value = &Tensor{node: moved.node, tape: v.tape}
	if grad != nil {
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
