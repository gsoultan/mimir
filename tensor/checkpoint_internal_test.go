package tensor

import (
	"testing"

	"github.com/gsoultan/mimir/internal/sched"
)

func TestCheckpointComputesTheRecomputedEffectsOnce(t *testing.T) {
	w := FromSlice([]float32{1, 2}).RequiresGrad()
	// f carries an effect, as a layer updating running statistics does.
	// The forward call's effect goes with the tape Checkpoint drops; the
	// recompute's is computed in the schedule of the gradients.
	var effects []*Tensor
	f := func(in ...*Tensor) *Tensor {
		h := in[0].Mul(in[0])
		e := h.Detach().Sum()
		effects = append(effects, e)
		return h.WithEffects(e)
	}
	if err := Checkpoint(f, w).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if len(effects) != 2 {
		t.Fatalf("f ran %d times, want 2", len(effects))
	}
	if !eager && sched.Realized(effects[0].node) {
		t.Error("the forward call's effect was computed")
	}
	if !sched.Realized(effects[1].node) {
		t.Error("the recompute's effect was not computed by Backward")
	}
	if bwd.acc != nil || bwd.err != nil {
		t.Error("Backward left its state behind")
	}
}

func TestCheckpointNodeHasNoTapeOfItsOwn(t *testing.T) {
	w := FromSlice([]float32{1, 2}).RequiresGrad()
	y := Checkpoint(func(in ...*Tensor) *Tensor { return in[0].Exp().Mul(w) }, FromSlice([]float32{3, 4}))
	if y.tape == nil || len(y.tape.parents) != 1 || y.tape.parents[0] != nil {
		t.Fatalf("checkpoint over an untracked input has tape %+v, want one with a nil parent", y.tape)
	}
	// Nothing but the checkpoint's own rule stands between the result
	// and w: the exponential inside is not kept for the backward pass.
	if readsOutside(y.tape, y.tape.seq-1) {
		t.Error("the checkpoint's tape reaches tapes made before it")
	}
}
