package tensor

import (
	"errors"
	"testing"
)

func TestEagerModeComputesAtTheCall(t *testing.T) {
	defer func(old bool) { eager = old }(eager)
	eager = true

	if y := FromSlice([]float32{1, 2}).Exp(); !y.node.Realized() {
		t.Error("eager operation was not computed")
	}
	g := FromSlice([]float32{1, 2, 3}).Gather(0, FromSlice([]int32{7}))
	if !errors.Is(g.Err(), ErrIndex) {
		t.Fatalf("Err() = %v, want ErrIndex at the gather", g.Err())
	}
	if !errors.Is(g.Add(g).Err(), ErrIndex) {
		t.Error("error did not propagate")
	}
}
