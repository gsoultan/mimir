package tensor

import (
	"errors"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/internal/graph"
)

// ErrCheckpoint reports a checkpointed function that reads a tracked
// tensor it captured from its surroundings instead of receiving it as an
// input.
var ErrCheckpoint = errors.New("tensor: checkpointed function breaks a rule of Checkpoint")

// Checkpoint computes f(in...) like calling f, but trades compute for
// memory: Backward recomputes f to get the gradients instead of keeping
// the activations inside f alive from the forward pass to the backward
// pass, as torch.utils.checkpoint does. f must be deterministic and must
// read tracked tensors only through in and through parameters (tensors
// from RequiresGrad or Param.Value); a tracked tensor it captures any
// other way fails with ErrCheckpoint, at the call or in Backward. Random
// tensors inside f are deterministic in mimir (counter-based), so a
// dropout mask drawn from a seed is recomputed the same. f may read
// gradients, but must not call Backward (see Backward).
//
// The recompute reads the parameters as they are when Backward runs. If
// one of them has changed since the call, as an optimizer step changes it,
// Backward fails with ErrOverwritten, as it does without Checkpoint when
// the step overwrote the values the gradient rules need.
//
// In the lazy graph the result is a recipe that reads f's inputs, like
// any other, and Backward computes it in the schedule of the gradients.
// What Checkpoint changes is the gradient graph: the gradient rules of
// the operations inside f capture the activations they need, which keeps
// those alive from the forward pass to their use in the backward pass;
// here those rules are dropped, and the one rule recorded for the result
// runs f again on the inputs and takes the gradients from the copies.
// The forward copies are then read by the forward pass alone and freed
// as soon as it is done with them, and the recomputed ones live only
// while the backward pass reads them.
//
// The effects f's result carries (see WithEffects) are those of this
// call: the result carries them, and Backward computes them once, as
// without Checkpoint. The recompute's are dropped. The recompute is still
// a second call of f, so other state that f changes it changes twice: a
// parameter it sets with Param.Set, as a layer's running statistics, is
// set again from the value it holds then, and a layer that counts its
// calls to seed its random tensors, as a dropout layer does, draws a new
// mask. nn.Checkpoint runs the recompute from the state its module had
// before this call and then puts back the state it found, so that every
// call updates the module's state once, in the order of the calls, and
// the recompute draws the mask this call drew.
func Checkpoint(f func(in ...*Tensor) *Tensor, in ...*Tensor) *Tensor {
	if f == nil {
		return fail(fmt.Errorf("tensor: Checkpoint(nil): %w", ErrInternal))
	}
	if err := firstErr(in...); err != nil {
		return fail(err)
	}
	det := make([]*Tensor, len(in))
	for i, t := range in {
		det[i] = t.detached()
	}
	since := tapes.Load()
	out := f(det...)
	if err := firstErr(out); err != nil {
		return fail(err)
	}
	c := &checkpointCall{f: f}
	captured, effects := c.scan(out.tape, since)
	if captured {
		return fail(fmt.Errorf("tensor: checkpointed function reads a tracked tensor that is not an input or a leaf: %w", ErrCheckpoint))
	}
	if !tracked(in...) && out.tape == nil {
		return out
	}
	// The result's own tape, if f built one through parameters, goes: the
	// recompute builds it again. Unlike record, the node is recorded even
	// when no input is tracked, for the parameters f reads. The checkpoint
	// keeps what the dropped tape held that the recompute does not give
	// back: the effects and the values of the parameters read.
	tp := &tape{seq: tapes.Add(1), node: out.node, ins: slices.Clone(in), effects: effects, ckpt: c}
	return &Tensor{node: out.node, tape: tp}
}

// checkpointCall is a call of Checkpoint: the function, which its rule
// recomputes, the parameters it read and the tapes it reached besides
// its inputs': leaves, which parameters are, and tensors standing for
// the arguments of a transformation. A derivative of the result flows to
// those as well as to the inputs.
type checkpointCall struct {
	f       func(in ...*Tensor) *Tensor
	reads   []paramRead
	reaches []*tape
}

// paramRead is a parameter that a checkpointed function read, and the
// node of the value it read.
type paramRead struct {
	p    *Param
	node *graph.Node
}

// recompute is the gradient rule of the checkpoint c over the inputs ins,
// in the reverse pass r: it runs the function again on tensors standing
// for the tracked inputs, propagates g through the result and returns the
// gradients that reach them, followed by those that reach c.reaches. The
// recompute's effects are dropped: those of the call that the checkpoint
// carries are the ones Backward computes.
//
// What stands for an input is a fresh leaf when the gradients are not to
// be differentiated, and otherwise a tensor whose tape leads to the input
// (see standIn), so that the gradients are functions of the inputs.
func (c *checkpointCall) recompute(r *reverse, ins []*Tensor, g *Tensor) ([]*Tensor, error) {
	for _, rd := range c.reads {
		if rd.p.Value().node != rd.node {
			return nil, fmt.Errorf("tensor: backward: recompute of a checkpointed function: a parameter it read changed after the call: %w", ErrOverwritten)
		}
	}
	since := tapes.Load()
	args := make([]*Tensor, len(ins))
	var stops []*tape
	for i, x := range ins {
		switch {
		case x.tape == nil:
			args[i] = x
		case r.create:
			args[i] = standIn(x)
		default:
			args[i] = x.detached().RequiresGrad()
		}
		if x.tape != nil {
			stops = append(stops, args[i].tape)
		}
	}
	out := c.f(args...)
	if err := firstErr(out); err != nil {
		return nil, fmt.Errorf("tensor: backward: recompute: %w", err)
	}
	if !slices.Equal(out.node.Shape, g.node.Shape) {
		return nil, fmt.Errorf("tensor: backward: recompute of shape %v, want %v: %w", out.node.Shape, g.node.Shape, ErrShape)
	}
	if captured, _ := new(checkpointCall).scan(out.tape, since); captured {
		return nil, fmt.Errorf("tensor: backward: recompute reads a tracked tensor that is not an input or a leaf: %w", ErrCheckpoint)
	}
	gs := make([]*Tensor, len(ins)+len(c.reaches))
	if out.tape == nil {
		return gs, nil
	}
	at, _, _, err := r.propagate([]*tape{out.tape}, []*Tensor{g}, append(stops, c.reaches...))
	if err != nil {
		return nil, err
	}
	k := 0
	for i, x := range ins {
		if x.tape != nil {
			gs[i] = at[k]
			k++
		}
	}
	copy(gs[len(ins):], at[k:])
	return gs, nil
}

// scan walks the tapes reachable from root, the tape of what a function
// returned that ran when since tapes existed. It reports whether one that
// is not a leaf or a stand-in was made by then: a tracked tensor that the
// function captured, since those it computed itself came later and its
// inputs are leaves or stand-ins. Otherwise it adds to c the parameters
// the function read, with their values now, and the leaves and stand-ins
// it reached, those of checkpoints inside it included, and returns the
// effects of the tapes, those of checkpoints inside it included.
func (c *checkpointCall) scan(root *tape, since uint64) (captured bool, effects []*Tensor) {
	if root == nil {
		return false, nil
	}
	gradMu.Lock()
	defer gradMu.Unlock()
	seen := map[*tape]bool{root: true}
	reach := func(t *tape) {
		if !slices.Contains(c.reaches, t) {
			c.reaches = append(c.reaches, t)
		}
	}
	for stack := []*tape{root}; len(stack) > 0; {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		switch {
		case n.leaf || n.stand:
			if n.param != nil {
				c.reads = append(c.reads, paramRead{p: n.param, node: n.param.Value().node})
			}
			reach(n)
			continue
		case n.seq <= since:
			return true, nil
		}
		if n.ckpt != nil {
			c.reads = append(c.reads, n.ckpt.reads...)
			for _, t := range n.ckpt.reaches {
				reach(t)
			}
		}
		effects = append(effects, n.effects...)
		for i := range n.ins {
			if p, _ := n.dep(i); p != nil && !seen[p] {
				seen[p] = true
				stack = append(stack, p)
			}
		}
	}
	return false, effects
}
