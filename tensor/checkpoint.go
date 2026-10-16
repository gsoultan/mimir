package tensor

import (
	"errors"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/internal/graph"
)

// ErrCheckpoint reports a checkpointed function that breaks a rule of
// Checkpoint: it reads a tracked tensor it captured from its surroundings
// instead of receiving it as an input, or it calls Backward while Backward
// recomputes it.
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
// gradients, but calling Backward from the recompute fails with
// ErrCheckpoint.
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
	parents := make([]*tape, len(in))
	for i, t := range in {
		det[i], parents[i] = t.detached(), t.tape
	}
	since := tapes.Load()
	out := f(det...)
	if err := firstErr(out); err != nil {
		return fail(err)
	}
	captured, reads, effects := scan(out.tape, since)
	if captured {
		return fail(fmt.Errorf("tensor: checkpointed function reads a tracked tensor that is not an input or a leaf: %w", ErrCheckpoint))
	}
	if !tracked(in...) && out.tape == nil {
		return out
	}
	// The result's own tape, if f built one through parameters, goes: the
	// recompute builds it again. Unlike track, the node is recorded even
	// when no input is tracked, for the parameters f reads. The checkpoint
	// keeps what the dropped tape held that the recompute does not give
	// back: the effects and the values of the parameters read.
	vjp := func(g *Tensor) []*Tensor { return recompute(f, det, parents, reads, g) }
	tp := newTape(out.node.Shape, parents, vjp)
	tp.effects, tp.reads = effects, reads
	return &Tensor{node: out.node, tape: tp}
}

// paramRead is a parameter that a checkpointed function read, and the
// node of the value it read.
type paramRead struct {
	p    *Param
	node *graph.Node
}

// recompute is the gradient rule of a checkpoint of f over the detached
// inputs det, whose tapes are parents and which read the parameters
// reads: it runs f again on fresh leaves standing for the tracked inputs,
// propagates g through the result into bwd and returns the gradients the
// leaves received. The gradients of the other leaves f reads, its
// parameters, stay in bwd for Backward to collect. The recompute's effects
// are dropped: those of the call that the checkpoint carries are the ones
// Backward computes. An error is left in bwd.
func recompute(f func(in ...*Tensor) *Tensor, det []*Tensor, parents []*tape, reads []paramRead, g *Tensor) []*Tensor {
	for _, r := range reads {
		if r.p.Value().node != r.node {
			bwd.fail(fmt.Errorf("tensor: backward: recompute of a checkpointed function: a parameter it read changed after the call: %w", ErrOverwritten))
			return nil
		}
	}
	leaves := make([]*Tensor, len(det))
	for i, d := range det {
		leaves[i] = d
		if parents[i] != nil {
			leaves[i] = d.RequiresGrad()
		}
	}
	since := tapes.Load()
	out := f(leaves...)
	if err := firstErr(out); err != nil {
		bwd.fail(fmt.Errorf("tensor: backward: recompute: %w", err))
		return nil
	}
	if !slices.Equal(out.node.Shape, g.node.Shape) {
		bwd.fail(fmt.Errorf("tensor: backward: recompute of shape %v, want %v: %w", out.node.Shape, g.node.Shape, ErrShape))
		return nil
	}
	if captured, _, _ := scan(out.tape, since); captured {
		bwd.fail(fmt.Errorf("tensor: backward: recompute reads a tracked tensor that is not an input or a leaf: %w", ErrCheckpoint))
		return nil
	}
	if out.tape != nil {
		if _, _, err := propagate(out.tape, g); err != nil {
			bwd.fail(err)
			return nil
		}
	}
	gs := make([]*Tensor, len(leaves))
	for i, l := range leaves {
		if parents[i] != nil {
			gs[i] = bwd.take(l.tape)
		}
	}
	return gs
}

// scan walks the tapes reachable from root, the tape of what a function
// returned that ran when since tapes existed. It reports whether one that
// is not a leaf was made by then: a tracked tensor that the function
// captured, since those it computed itself came later and its inputs and
// parameters are leaves. Otherwise it also returns the parameters the
// function read, with their values now, those that checkpoints inside it
// read, and the effects of the tapes, those of checkpoints inside it
// included.
func scan(root *tape, since uint64) (captured bool, reads []paramRead, effects []*Tensor) {
	if root == nil {
		return false, nil, nil
	}
	seen := map[*tape]bool{root: true}
	for stack := []*tape{root}; len(stack) > 0; {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		switch {
		case n.leaf:
			if n.param != nil {
				reads = append(reads, paramRead{p: n.param, node: n.param.Value().node})
			}
			continue
		case n.seq <= since:
			return true, nil, nil
		}
		reads = append(reads, n.reads...)
		effects = append(effects, n.effects...)
		for _, p := range n.parents {
			if p != nil && !seen[p] {
				seen[p] = true
				stack = append(stack, p)
			}
		}
	}
	return false, reads, effects
}
