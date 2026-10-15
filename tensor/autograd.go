package tensor

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
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
	shape []int // of the tensor; gradients must match it

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
	released bool    // Backward has run through this node
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
	return &Tensor{node: out.node, tape: &tape{shape: out.node.Shape, parents: parents, vjp: vjp}}
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
// computes and accumulates in Grad. Only Float32 tensors have gradients.
func (t *Tensor) RequiresGrad() *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if !t.node.DType.IsFloat() {
		return fail(fmt.Errorf("tensor: gradient of %s tensor: %w", t.node.DType, ErrDType))
	}
	return &Tensor{node: t.node, tape: &tape{shape: t.node.Shape, leaf: true}}
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

// ZeroGrad discards t's accumulated gradient.
func (t *Tensor) ZeroGrad() {
	if firstErr(t) != nil || t.tape == nil || !t.tape.leaf {
		return
	}
	gradMu.Lock()
	defer gradMu.Unlock()
	t.tape.grad = nil
}

// Backward computes the gradient of t, which has one element, with respect
// to every tensor that requires gradients it was computed from, and adds
// each gradient to that tensor's Grad, like torch.Tensor.backward.
//
// Backward computes t itself in the same schedule, so work shared by the
// forward and backward passes runs once: call Backward before reading t.
// On success the graph between t and its leaves is released, and a later
// Backward through it returns ErrNoGrad. On error no gradient changes.
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

	order, err := tapeOrder(t.tape)
	if err != nil {
		return err
	}
	acc := map[*tape]*Tensor{t.tape: Ones(t.node.DType, t.node.Shape...)}
	var leaves []*tape
	var effects []*Tensor
	for _, n := range slices.Backward(order) {
		effects = append(effects, n.effects...)
		g, ok := acc[n]
		if !ok {
			continue
		}
		if n.leaf {
			leaves = append(leaves, n)
			continue
		}
		delete(acc, n) // its inputs' gradients are all that is needed now
		gs := n.vjp(g)
		for i, p := range n.parents {
			if p == nil || i >= len(gs) || gs[i] == nil {
				continue
			}
			gi := gs[i].detached()
			if err := gi.Err(); err != nil {
				return fmt.Errorf("tensor: backward: %w", err)
			}
			if !slices.Equal(gi.node.Shape, p.shape) {
				return fmt.Errorf("tensor: backward: gradient of shape %v for input of shape %v: %w", gi.node.Shape, p.shape, ErrShape)
			}
			if prev, ok := acc[p]; ok {
				gi = prev.Add(gi)
			}
			acc[p] = gi
		}
	}

	sums := make([]*Tensor, len(leaves))
	targets := append(make([]*Tensor, 0, 1+len(leaves)+len(effects)), t)
	for i, l := range leaves {
		sums[i] = acc[l]
		if l.grad != nil {
			sums[i] = l.grad.Add(sums[i])
		}
		targets = append(targets, sums[i])
	}
	if err := Realize(ctx, append(targets, effects...)...); err != nil {
		return err
	}
	for i, l := range leaves {
		l.grad = sums[i]
	}
	for _, n := range order {
		if !n.leaf {
			n.parents, n.vjp, n.effects, n.released = nil, nil, nil, true
		}
	}
	return nil
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
	return &Tensor{node: t.node, tape: &tape{shape: t.node.Shape, parents: []*tape{t.tape}, vjp: identity, effects: det}}
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
// use Value; the tensors they build keep the value they were built with,
// even after Set replaces it. A Param is safe for concurrent use.
type Param struct {
	mu        sync.Mutex
	value     *Tensor
	trainable bool
}

// NewParam returns a trainable parameter holding init's values. It
// computes init now, so the parameter holds memory rather than a recipe.
// If that fails, Value carries the error.
func NewParam(init *Tensor) *Param { return newParam(init, true) }

// NewBuffer returns a parameter that is saved and restored with a model
// but not trained: Backward computes no gradient for it. Batch
// normalization keeps its running statistics in buffers.
func NewBuffer(init *Tensor) *Param { return newParam(init, false) }

func newParam(init *Tensor, trainable bool) *Param {
	v := init.Detach()
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

// Set replaces the value with v, which must have the same shape and dtype.
// The accumulated gradient is kept. Optimizers compute v first, so the
// parameter never holds a growing recipe; a lazy v is allowed for buffers
// updated with WithEffects.
func (p *Param) Set(v *Tensor) error {
	if err := firstErr(v); err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.value.err; err != nil {
		return err
	}
	if !slices.Equal(v.node.Shape, p.value.node.Shape) || v.node.DType != p.value.node.DType {
		return fmt.Errorf("tensor: set %s%v parameter to %s%v: %w", p.value.node.DType, p.value.node.Shape, v.node.DType, v.node.Shape, ErrShape)
	}
	p.value = &Tensor{node: v.node, tape: p.value.tape}
	return nil
}
