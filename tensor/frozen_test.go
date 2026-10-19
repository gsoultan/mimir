package tensor_test

import (
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

func TestFrozenParamRecordsNoTape(t *testing.T) {
	ctx := t.Context()
	p := tensor.NewParam(tensor.FromSlice([]float32{1, 2, 3}))
	v := p.Frozen()
	if v.Frozen() != v {
		t.Error("the view of a view is another view")
	}
	if v.Trainable() || v.Value().Tracked() {
		t.Error("frozen view requires gradients")
	}
	y := v.Value().Mul(v.Value()).Sum()
	if y.Tracked() {
		t.Error("a computation through a frozen view is tracked")
	}
	if err := y.Backward(ctx); !errors.Is(err, tensor.ErrNoGrad) {
		t.Errorf("Backward through a frozen view: %v, want ErrNoGrad", err)
	}
	if got := data[float32](t, y); got[0] != 14 {
		t.Errorf("sum of squares %v, want 14", got)
	}
	if v.Grad() != nil {
		t.Error("frozen view has a gradient")
	}
	v.ZeroGrad() // a no-op

	// The view follows the parameter.
	if err := p.Set(tensor.FromSlice([]float32{4, 5, 6})); err != nil {
		t.Fatal(err)
	}
	if got := data[float32](t, v.Value()); !slices.Equal(got, []float32{4, 5, 6}) {
		t.Errorf("view reads %v after Set, want the new value", got)
	}
	// Training through the parameter is unaffected by the view.
	if err := p.Value().Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if got := data[float32](t, p.Grad()); !slices.Equal(got, []float32{1, 1, 1}) {
		t.Errorf("gradient %v", got)
	}

	if err := v.Set(tensor.FromSlice([]float32{0, 0, 0})); !errors.Is(err, tensor.ErrFrozen) {
		t.Errorf("Set through a view: %v, want ErrFrozen", err)
	}
	if err := v.To(tensor.CPU()); !errors.Is(err, tensor.ErrFrozen) {
		t.Errorf("To through a view: %v, want ErrFrozen", err)
	}
	if got := data[float32](t, p.Value()); !slices.Equal(got, []float32{4, 5, 6}) {
		t.Errorf("parameter %v after a refused Set through its view", got)
	}
	b := tensor.NewBuffer(tensor.FromSlice([]float32{7}))
	if got := data[float32](t, b.Frozen().Value()); got[0] != 7 {
		t.Errorf("view of a buffer reads %v", got)
	}
}

// Inference through views and training through the parameter run at once
// on separate goroutines.
func TestFrozenParamConcurrentWithTraining(t *testing.T) {
	ctx := t.Context()
	p := tensor.NewParam(tensor.Full(tensor.Float32, 1, 64))
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			v := p.Frozen()
			for range 50 {
				y := v.Value().Mul(tensor.Scalar[float32](2)).Sum()
				if y.Tracked() {
					t.Error("inference is tracked")
					return
				}
				if err := y.Realize(ctx); err != nil && !errors.Is(err, tensor.ErrOverwritten) {
					t.Error(err)
					return
				}
			}
		})
	}
	wg.Go(func() {
		for range 50 {
			if err := p.Value().Sum().Backward(ctx); err != nil {
				t.Error(err)
				return
			}
			if err := p.Set(p.Value().Detach().Sub(p.Grad().Mul(tensor.Scalar[float32](0.01))).Clone()); err != nil {
				t.Error(err)
				return
			}
			p.ZeroGrad()
		}
	})
	wg.Wait()
}
