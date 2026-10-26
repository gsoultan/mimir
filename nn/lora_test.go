package nn_test

import (
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

func TestLoRA(t *testing.T) {
	ctx := t.Context()
	base := nn.NewLinear(6, 4, nn.Seed(1))
	x := tensor.Rand(2, 3, 6)
	want, err := base.Forward(x).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	l := nn.NewLoRA(base, 2, 3, nn.Seed(3))
	if ps := nn.Parameters(l); len(ps) != 2 || ps[0] != l.A || ps[1] != l.B {
		t.Fatalf("trainable parameters %v, want A and B alone", ps)
	}
	got, err := l.Forward(x).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	near(t, "fresh", got, want, 0)

	// With B set, it is a Linear layer of weight W + 1.5·B·A.
	if err := l.B.Set(tensor.Rand(4, 4, 2)); err != nil {
		t.Fatal(err)
	}
	w := base.Weight.Value().Add(l.B.Value().MatMul(l.A.Value()).Mul(tensor.Full(tensor.Float32, 1.5)))
	merged := &nn.Linear{Weight: tensor.NewParam(w), Bias: base.Bias}
	if want, err = merged.Forward(x).Data[float32](ctx); err != nil {
		t.Fatal(err)
	}
	y := l.Forward(x)
	if got, err = y.Data[float32](ctx); err != nil {
		t.Fatal(err)
	}
	near(t, "trained", got, want, 1e-5)

	if err := y.Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if l.A.Grad() == nil || l.B.Grad() == nil {
		t.Error("no gradient for A or B")
	}
	if l.Base.Weight.Grad() != nil {
		t.Error("a gradient for the base weight")
	}

	if bad := nn.NewLoRA(base, 0, 3); bad.A.Value().Err() == nil {
		t.Error("LoRA of rank 0")
	}
}
