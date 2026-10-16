package tensor

import (
	"errors"
	"fmt"
	"slices"
)

// ErrCheckpoint reports a checkpointed function that reads a tracked
// tensor it captured from its surroundings instead of receiving it as an
// input (see Checkpoint).
var ErrCheckpoint = errors.New("tensor: checkpointed function reads a tracked tensor that is not an input or a leaf")

// Checkpoint computes f(in...) like calling f, but trades compute for
// memory: Backward recomputes f to get the gradients instead of keeping
// the activations inside f alive from the forward pass to the backward
// pass, as torch.utils.checkpoint does. f must be deterministic and must
// read tracked tensors only through in and through parameters (tensors
// from RequiresGrad or Param.Value); a tracked tensor it captures any
// other way fails with ErrCheckpoint, at the call or in Backward. Random
// tensors inside f are deterministic in mimir (counter-based), so a
// dropout mask drawn from a seed is recomputed the same.
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
// The recompute is a second call of f, so state that f changes it
// changes twice: a parameter it sets with Param.Set, as a layer's running
// statistics, is set again from the value the first call gave it, and a
// layer that counts its calls to seed its random tensors, as a dropout
// layer does, draws a new mask. nn.Checkpoint puts the state of its
// module back before the recompute, so that a checkpointed layer updates
// its running statistics once and recomputes the mask it drew. The
// effects f carries (see WithEffects) are computed once, from the
// recompute.
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
	if readsOutside(out.tape, since) {
		return fail(ErrCheckpoint)
	}
	if !tracked(in...) && out.tape == nil {
		return out
	}
	// The result's own tape, if f built one through parameters, goes: the
	// recompute builds it again. Unlike track, the node is recorded even
	// when no input is tracked, for the parameters f reads.
	vjp := func(g *Tensor) []*Tensor { return recompute(f, det, parents, g) }
	return &Tensor{node: out.node, tape: newTape(out.node.Shape, parents, vjp)}
}

// recompute is the gradient rule of a checkpoint of f over the detached
// inputs det, whose tapes are parents: it runs f again on fresh leaves
// standing for the tracked inputs, propagates g through the result into
// bwd and returns the gradients the leaves received. The gradients of the
// other leaves f reads, its parameters, stay in bwd for Backward to
// collect, and so do the effects. An error is left in bwd.
func recompute(f func(in ...*Tensor) *Tensor, det []*Tensor, parents []*tape, g *Tensor) []*Tensor {
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
	if readsOutside(out.tape, since) {
		bwd.fail(fmt.Errorf("tensor: backward: %w", ErrCheckpoint))
		return nil
	}
	if out.tape != nil {
		_, effects, err := propagate(out.tape, g)
		if err != nil {
			bwd.fail(err)
			return nil
		}
		bwd.effects = append(bwd.effects, effects...)
	}
	gs := make([]*Tensor, len(leaves))
	for i, l := range leaves {
		if parents[i] != nil {
			gs[i] = bwd.take(l.tape)
		}
	}
	return gs
}

// readsOutside reports whether a tape reachable from root, other than a
// leaf, was made when at most since tapes existed: a tracked tensor that
// the function computing root captured, since those it computed itself
// came later and its inputs and parameters are leaves.
func readsOutside(root *tape, since uint64) bool {
	if root == nil {
		return false
	}
	seen := map[*tape]bool{root: true}
	for stack := []*tape{root}; len(stack) > 0; {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if !n.leaf && n.seq <= since {
			return true
		}
		for _, p := range n.parents {
			if p != nil && !seen[p] {
				seen[p] = true
				stack = append(stack, p)
			}
		}
	}
	return false
}
