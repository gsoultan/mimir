package tensor

import (
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

func TestSplitJoinsGradientsOnce(t *testing.T) {
	ctx := t.Context()
	x := Arange[float32](0, 12, 1).Reshape(4, 3).RequiresGrad()
	parts := x.split(0, []int{1, 2, 1})
	if !slices.Equal(parts[1].Shape(), []int{2, 3}) {
		t.Fatalf("part shape %v", parts[1].Shape())
	}
	// Part 0 is read twice, part 1 not at all and part 2 in bfloat16.
	loss := parts[0].Mul(Scalar[float32](2)).Sum().Add(parts[0].Sum()).Add(parts[2].Cast(BFloat16).Sum())
	if err := loss.Backward(ctx); err != nil {
		t.Fatal(err)
	}
	g, err := x.Grad().Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1}
	if !slices.Equal(g, want) {
		t.Errorf("gradient %v, want %v", g, want)
	}
}

func TestCatJoinsInATreeOfKernels(t *testing.T) {
	n := 3*backend.MaxCatInputs + 1
	ts := make([]*Tensor, n)
	want := make([]float32, n)
	for i := range ts {
		ts[i] = Full(Float32, float64(i), 1)
		want[i] = float32(i)
	}
	out := Cat(0, ts...)
	// Four groups of at most MaxCatInputs, then one join of the groups.
	if out.node.Op != backend.OpCat || len(out.node.Inputs) != 4 {
		t.Fatalf("top join of %d inputs, want 4", len(out.node.Inputs))
	}
	got, err := out.Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Errorf("cat %v, want %v", got, want)
	}
}
