package tensor_test

import (
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/optim"
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

func TestLeavesWhoseGradientsShareMemoryKeepTheirs(t *testing.T) {
	a := tensor.NewParam(tensor.Zeros(tensor.Float32, 2, 3))
	b := tensor.NewParam(tensor.Zeros(tensor.Float32, 6))
	c := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6})
	twice := []float32{2, 4, 6, 8, 10, 12}
	grad := func(p *tensor.Param, want []float32) {
		t.Helper()
		if got, err := p.Grad().Data[float32](t.Context()); err != nil || !slices.Equal(got, want) {
			t.Fatalf("gradient = %v, %v; want %v", got, err, want)
		}
	}
	// a's gradient is a view of b's.
	if err := a.Value().Reshape(6).Add(b.Value()).Mul(c).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	// Adding to a's gradient alone leaves b's.
	if err := a.Value().Reshape(6).Mul(c).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	grad(a, twice)
	grad(b, []float32{1, 2, 3, 4, 5, 6})
	// So does zeroing it.
	a.ZeroGrad()
	grad(b, []float32{1, 2, 3, 4, 5, 6})
	if err := b.Value().Mul(c).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	grad(b, twice)
}

func TestLazyParameterValuesSurviveWritesOverWhatTheyRead(t *testing.T) {
	a := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	b := tensor.NewParam(tensor.Zeros(tensor.Float32, 2))
	if err := b.Set(a.Value().Clone()); err != nil {
		t.Fatal(err)
	}
	old := a.Value()
	if err := tensor.RealizeInto(t.Context(), []*tensor.Tensor{old}, []*tensor.Tensor{old.Neg()}); err != nil {
		t.Fatal(err)
	}
	if got, err := b.Value().Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{1, 2}) {
		t.Errorf("copy of a value written over = %v, %v; want [1 2]", got, err)
	}
}

func TestLeavesWithTheSameGradientEachOwnIt(t *testing.T) {
	a := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	b := tensor.NewParam(tensor.FromSlice([]float32{3, 4}))
	for range 2 {
		if err := a.Value().Add(b.Value()).Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		a.ZeroGrad()
		if got, err := b.Grad().Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{1, 1}) {
			t.Fatalf("b's gradient after a.ZeroGrad = %v, %v; want [1 1]", got, err)
		}
		b.ZeroGrad()
	}
}

func TestStepsSurviveLazyValuesThatFail(t *testing.T) {
	if tensor.Eager() {
		t.Skip("eager mode computes the value when it is set")
	}
	w := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	u := tensor.NewParam(tensor.FromSlice([]float32{3, 4}))
	buf := tensor.NewBuffer(tensor.Zeros(tensor.Float32, 2))
	// Index 9 is out of range, which only computing the value finds.
	if err := buf.Set(w.Value().Detach().Gather(0, tensor.FromSlice([]int32{0, 9}))); err != nil {
		t.Fatal(err)
	}
	opt := optim.SGD([]*tensor.Param{w, u}, optim.LR(1))
	for i := range 2 {
		if err := w.Value().Mul(u.Value()).Sum().Backward(t.Context()); err != nil {
			t.Fatalf("backward %d: %v", i, err)
		}
		if err := opt.Step(t.Context()); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		opt.ZeroGrad()
	}
	// u -= w and w -= u, twice.
	if got, err := u.Value().Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{4, 4}) {
		t.Errorf("u = %v, %v; want [4 4]", got, err)
	}
	if _, err := buf.Value().Data[float32](t.Context()); !errors.Is(err, tensor.ErrOverwritten) {
		t.Errorf("value that failed: got %v, want ErrOverwritten", err)
	}
}

func TestLazyValuesKeepTheGradientsTheyRead(t *testing.T) {
	w := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	last := tensor.NewBuffer(tensor.Zeros(tensor.Float32, 2))
	if err := w.Value().Mul(w.Value()).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := last.Set(w.Grad().Clone()); err != nil {
		t.Fatal(err)
	}
	w.ZeroGrad()
	if got, err := last.Value().Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{2, 4}) {
		t.Errorf("copy of a gradient zeroed after = %v, %v; want [2 4]", got, err)
	}
}

func TestParamsTakeValuesThatOtherGoroutinesRealize(t *testing.T) {
	// The race detector checks that Set looks at whether its value is
	// computed only under the scheduler's lock.
	p := tensor.NewBuffer(tensor.Zeros(tensor.Float32, 64))
	x := tensor.Ones(tensor.Float32, 64)
	for range 100 {
		v := x.Neg()
		var wg sync.WaitGroup
		wg.Go(func() {
			if err := v.Realize(t.Context()); err != nil {
				t.Error(err)
			}
		})
		wg.Go(func() {
			if err := p.Set(v); err != nil {
				t.Error(err)
			}
		})
		wg.Wait()
	}
	if got, err := p.Value().Data[float32](t.Context()); err != nil || got[0] != -1 {
		t.Errorf("value = %v, %v; want -1s", got, err)
	}
}

func TestParamToKeepsWhatAConcurrentBackwardAdds(t *testing.T) {
	// To copies the gradient and replaces it with the copy. A Backward
	// adding to the gradient in between would add to the gradient the
	// copy then replaces, and its contribution would be lost.
	ctx := t.Context()
	w := tensor.NewParam(tensor.Zeros(tensor.Float32, 64))
	const rounds = 300
	for range rounds {
		l := w.Value().Sum()
		var wg sync.WaitGroup
		wg.Go(func() {
			if err := l.Backward(ctx); err != nil {
				t.Error(err)
			}
		})
		wg.Go(func() {
			if err := w.To(tensor.CPU()); err != nil {
				t.Error(err)
			}
		})
		wg.Wait()
	}
	got, err := w.Grad().Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != rounds {
			t.Fatalf("gradient[%d] = %v, want %d: a Backward's contribution was lost", i, v, rounds)
		}
	}
}
