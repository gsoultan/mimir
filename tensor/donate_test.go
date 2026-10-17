package tensor_test

import (
	"errors"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func TestReadingADonatedTensorFails(t *testing.T) {
	if tensor.Eager() {
		t.Skip("Donate does nothing in eager mode")
	}
	ctx := t.Context()
	x := tensor.FromSlice([]float32{1, 2, 3, 4}, 2, 2)
	view := x.Detach() // the same memory
	y := x.Donate().Mul(tensor.Scalar[float32](2)).Add(x)
	got, err := y.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if want := []float32{3, 6, 9, 12}; !slices.Equal(got, want) {
		t.Errorf("result %v, want %v", got, want)
	}
	for name, read := range map[string]func() error{
		"Data":    func() error { _, err := x.Data[float32](ctx); return err },
		"view":    func() error { _, err := view.Data[float32](ctx); return err },
		"compute": func() error { return x.Neg().Realize(ctx) },
	} {
		if err := read(); !errors.Is(err, tensor.ErrDonated) || !errors.Is(err, tensor.ErrOverwritten) {
			t.Errorf("%s of a donated tensor: %v, want ErrDonated and ErrOverwritten", name, err)
		}
	}
	// The result computed from it stays valid.
	if _, err := y.Data[float32](ctx); err != nil {
		t.Errorf("reading a result computed from a donated tensor: %v", err)
	}
}

func TestDonatingWhatOthersShareTakesNothing(t *testing.T) {
	ctx := t.Context()
	base := tensor.FromSlice([]float32{1, 2, 3, 4}, 2, 2)
	view := base.Reshape(4)
	if err := view.Realize(ctx); err != nil {
		t.Fatal(err)
	}
	lazy := tensor.FromSlice([]float32{5, 6}).Exp() // not computed yet
	for name, x := range map[string]*tensor.Tensor{"view": view, "lazy": lazy} {
		if err := x.Donate().Neg().Realize(ctx); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	for name, x := range map[string]*tensor.Tensor{"base": base, "lazy": lazy} {
		if _, err := x.Data[float32](ctx); err != nil {
			t.Errorf("%s after its view was donated: %v", name, err)
		}
	}
	// A batch-normalized donated batch: the running statistics, which
	// the forward pass sets to values computed from the batch, keep it.
	bn := nn.NewBatchNorm1d(4)
	x := tensor.Randn(1, 8, 4)
	if err := x.Realize(ctx); err != nil {
		t.Fatal(err)
	}
	if err := bn.Forward(x.Donate()).Realize(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := bn.RunningMean.Value().Data[float32](ctx); err != nil {
		t.Errorf("running mean after a forward pass over a donated batch: %v", err)
	}
}

// allocRecorder counts the buffers a batchRecorder allocates.
type allocRecorder struct {
	*batchRecorder
	allocs int
}

func (a *allocRecorder) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	a.allocs++
	return a.batchRecorder.Alloc(dt, n)
}

// TestDonatedBatchesRunOverTheSameBuffers trains on a new batch every
// step, as a loop over a dataset does, uploaded and donated: the steps
// after the first upload each batch into the buffer that the step before
// gave back, allocate nothing, and run their kernels over the buffers of
// the step before, but for the loss; and they compute what steps over
// batches not donated compute, bit for bit.
func TestDonatedBatchesRunOverTheSameBuffers(t *testing.T) {
	if tensor.Eager() {
		t.Skip("Donate does nothing in eager mode")
	}
	const batch, steps = 16, 8
	rng := rand.New(rand.NewPCG(1, 2))
	images := make([][]float32, steps)
	for i := range images {
		images[i] = make([]float32, batch*28*28)
		for j := range images[i] {
			images[i][j] = rng.Float32()
		}
	}
	labels := make([]int32, batch)
	for i := range labels {
		labels[i] = int32(i % 10)
	}
	train := func(donate bool) (params [][]float32, allocs []int, moved [][]backend.Buffer) {
		be := &allocRecorder{batchRecorder: &batchRecorder{Backend: cpu.New(), runs: map[*backend.Batch][][]backend.Buffer{}}}
		defer tensor.SetDevice(be)()
		nn.ManualSeed(1)
		m := nn.NewSequential(nn.Flatten(), nn.NewLinear(28*28, 32), nn.ReLU(), nn.NewLinear(32, 10))
		opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
		for step := range steps {
			if step == 3 {
				be.moved, be.ops = nil, nil
			}
			before := be.allocs
			x, y := tensor.FromSlice(images[step], batch, 1, 28, 28), tensor.FromSlice(labels, batch)
			if donate {
				x, y = x.Donate(), y.Donate()
			}
			if err := loss.CrossEntropy(m.Forward(x), y).Backward(t.Context()); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(t.Context()); err != nil {
				t.Fatal(err)
			}
			opt.ZeroGrad()
			allocs = append(allocs, be.allocs-before)
			if donate && x.Err() == nil {
				if _, err := x.Data[float32](t.Context()); !errors.Is(err, tensor.ErrDonated) {
					t.Fatalf("step %d: reading the donated batch: %v, want ErrDonated", step, err)
				}
			}
		}
		for _, p := range nn.Parameters(m) {
			v, err := p.Value().Data[float32](t.Context())
			if err != nil {
				t.Fatal(err)
			}
			params = append(params, v)
		}
		return params, allocs, be.moved
	}
	want, kept, _ := train(false)
	got, allocs, moved := train(true)
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Fatalf("parameter %d after training on donated batches differs", i)
		}
	}
	t.Logf("allocations per step: %v kept, %v donated", kept, allocs)
	for step, n := range allocs[3:] {
		// The loss, which each step returns.
		if n > 1 {
			t.Errorf("step %d allocated %d buffers, want only the loss's", step+3, n)
		}
	}
	if len(moved) == 0 {
		t.Fatal("no batch was replayed")
	}
	for i, m := range moved {
		if len(m) > 1 {
			t.Errorf("replayed batch %d ran over %d other buffers than the run before, want the loss's alone", i, len(m))
		}
	}
}
