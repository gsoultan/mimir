package tensor_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

func TestRealizeIntoMakesTheDestinationStale(t *testing.T) {
	x := tensor.FromSlice([]float32{1, 2, 3})
	if err := x.Realize(t.Context()); err != nil {
		t.Fatal(err)
	}
	pending := x.Add(tensor.Full(tensor.Float32, 1))
	y := x.Mul(tensor.Full(tensor.Float32, 2))
	if err := tensor.RealizeInto(t.Context(), []*tensor.Tensor{x}, []*tensor.Tensor{y}); err != nil {
		t.Fatal(err)
	}
	if got, err := y.Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{2, 4, 6}) {
		t.Errorf("result = %v, %v; want [2 4 6]", got, err)
	}
	if _, err := x.Data[float32](t.Context()); !errors.Is(err, tensor.ErrOverwritten) {
		t.Errorf("reading the destination: got %v, want ErrOverwritten", err)
	}
	// In eager mode the pending sum was computed when it was created.
	if _, err := pending.Data[float32](t.Context()); !tensor.Eager() && !errors.Is(err, tensor.ErrOverwritten) {
		t.Errorf("computing from the destination: got %v, want ErrOverwritten", err)
	}
	bad := tensor.Zeros(tensor.Float32, 2)
	if err := tensor.RealizeInto(t.Context(), []*tensor.Tensor{bad}, []*tensor.Tensor{y.Clone()}); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("mismatched destination: got %v, want ErrShape", err)
	}
}

func TestCloneKeepsItsValuesAndGradients(t *testing.T) {
	x := tensor.FromSlice([]float32{1, -2}).RequiresGrad()
	c := x.Clone()
	if err := c.Mul(c).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got, err := x.Grad().Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{2, -4}) {
		t.Errorf("gradient through a clone = %v, %v; want [2 -4]", got, err)
	}
	src := tensor.FromSlice([]float32{5, 6})
	kept := src.Clone()
	if err := kept.Realize(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := tensor.RealizeInto(t.Context(), []*tensor.Tensor{src}, []*tensor.Tensor{src.Neg()}); err != nil {
		t.Fatal(err)
	}
	if got, err := kept.Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{5, 6}) {
		t.Errorf("clone after its source was overwritten = %v, %v; want [5 6]", got, err)
	}
}

func TestGradientsReuseTheirMemory(t *testing.T) {
	p := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	backward := func() {
		t.Helper()
		if err := p.Value().Mul(p.Value()).Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	backward()
	first := p.Grad()
	backward() // adds to the gradient in place
	if got, err := p.Grad().Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{4, 8}) {
		t.Errorf("accumulated gradient = %v, %v; want [4 8]", got, err)
	}
	// Eager mode computes the sums before Backward sees them, into memory
	// that is not the gradients' own.
	if _, err := first.Data[float32](t.Context()); !tensor.Eager() && !errors.Is(err, tensor.ErrOverwritten) {
		t.Errorf("gradient from before accumulating: got %v, want ErrOverwritten", err)
	}
	kept := p.Grad()
	p.ZeroGrad()
	if _, err := kept.Data[float32](t.Context()); !tensor.Eager() && !errors.Is(err, tensor.ErrOverwritten) {
		t.Errorf("gradient kept across ZeroGrad: got %v, want ErrOverwritten", err)
	}
	backward()
	if got, err := p.Grad().Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{2, 4}) {
		t.Errorf("gradient after ZeroGrad = %v, %v; want [2 4]", got, err)
	}
}

func TestNewParamCopiesItsInitialValue(t *testing.T) {
	init := tensor.FromSlice([]float32{3, 4})
	p := tensor.NewParam(init)
	if err := p.Value().Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	next := p.Value().Sub(p.Grad())
	if err := tensor.RealizeInto(t.Context(), []*tensor.Tensor{p.Value()}, []*tensor.Tensor{next}); err != nil {
		t.Fatal(err)
	}
	if got, err := init.Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{3, 4}) {
		t.Errorf("initial value after the parameter was overwritten = %v, %v; want [3 4]", got, err)
	}
}
