package tensor_test

import (
	"errors"
	"os"
	"testing"

	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/sched"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func TestTrainingStepsPredictTheirMemoryExactly(t *testing.T) {
	be := cpu.New()
	defer tensor.SetDevice(be)()
	nn.ManualSeed(3)
	model := nn.NewSequential(nn.NewLinear(8, 32), nn.ReLU(), nn.NewLinear(32, 4))
	opt := optim.Adam(nn.Parameters(model), optim.LR(0.01))
	x := tensor.Randn(1, 16, 8)
	y := tensor.FromSlice([]int32{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3})
	before := sched.MemoryOf(be)
	for step := range 3 {
		// The forward and backward passes, in one schedule, then the
		// update in place, then the gradients' memory back to the pool.
		l := loss.CrossEntropy(model.Forward(x), y)
		if err := l.Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
		after := sched.MemoryOf(be)
		got, want := after.Obtained-before.Obtained, after.Predicted-before.Predicted
		if got != want {
			t.Fatalf("step %d obtained %d bytes from the backend, predicted %d", step, got, want)
		}
		if step == 0 && got == 0 {
			t.Fatal("the first step obtained nothing")
		}
		before = after
	}
}

func TestMemoryBudgetRefusesTrainingStepsThatDoNotFit(t *testing.T) {
	be := cpu.New()
	defer tensor.SetDevice(be)()
	dev := tensor.DefaultDevice()
	if got := tensor.MemoryBudget(dev); got != 0 {
		t.Fatalf("budget of a new device = %d, want 0", got)
	}
	w := tensor.NewParam(tensor.Randn(1, 64, 64))
	x := tensor.Randn(2, 64, 64)
	if err := x.Realize(t.Context()); err != nil {
		t.Fatal(err)
	}
	// The loss is built afresh each time: in eager mode a tensor built
	// past the budget carries the error.
	lossOf := func() *tensor.Tensor { return x.Mul(w.Value()).ReLU().Sum() }
	// Results dropped while building the inputs, as eager mode computes
	// each operation, count until they are collected, which a refusal does
	// first: after one, what is held is what the inputs hold.
	tensor.SetMemoryBudget(dev, 1)
	defer tensor.SetMemoryBudget(dev, 0)
	if _, err := x.Neg().Data[float32](t.Context()); !errors.Is(err, tensor.ErrMemoryBudget) {
		t.Fatalf("Data in a budget of one byte: got %v, want ErrMemoryBudget", err)
	}
	held := sched.MemoryOf(be).Allocated
	// The loss needs more than one byte beyond what the inputs hold.
	tensor.SetMemoryBudget(dev, held+1)
	if got := tensor.MemoryBudget(dev); got != held+1 {
		t.Fatalf("MemoryBudget = %d, want %d", got, held+1)
	}
	err := lossOf().Backward(t.Context())
	if !errors.Is(err, tensor.ErrMemoryBudget) {
		t.Fatalf("Backward past the budget: got %v, want ErrMemoryBudget", err)
	}
	if w.Grad() != nil {
		t.Error("a refused Backward left a gradient")
	}
	if _, err := x.Neg().Data[float32](t.Context()); !errors.Is(err, tensor.ErrMemoryBudget) {
		t.Errorf("Data past the budget: got %v, want ErrMemoryBudget", err)
	}
	tensor.SetMemoryBudget(dev, 0)
	if err := lossOf().Backward(t.Context()); err != nil {
		t.Fatalf("Backward without a budget: %v", err)
	}
	if w.Grad() == nil {
		t.Error("no gradient after Backward")
	}
}

func TestGradientScalingAllocatesNothingExtra(t *testing.T) {
	if os.Getenv("MIMIR_EAGER") == "1" {
		t.Skip("eager mode computes the scaled gradient into memory of its own")
	}
	// The scaled gradient is read by both of Adam's moments, and is
	// computed inside their kernel rather than into a buffer of its own:
	// a step allocates the same with and without a gradient scale.
	step := func(scale float64) (int64, []float32) {
		be := cpu.New()
		defer tensor.SetDevice(be)()
		nn.ManualSeed(3)
		model := nn.NewSequential(nn.NewLinear(8, 32), nn.ReLU(), nn.NewLinear(32, 4))
		opt := optim.Adam(nn.Parameters(model), optim.LR(0.01))
		opt.SetGradScale(scale)
		x := tensor.Randn(1, 16, 8)
		y := tensor.FromSlice([]int32{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3})
		l := loss.CrossEntropy(model.Forward(x), y).Mul(tensor.Scalar(float32(scale)))
		if err := l.Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		before := sched.MemoryOf(be).Obtained
		if err := opt.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		var w []float32
		for _, p := range nn.Parameters(model) {
			v, err := p.Value().Data[float32](t.Context())
			if err != nil {
				t.Fatal(err)
			}
			w = append(w, v...)
		}
		return sched.MemoryOf(be).Obtained - before, w
	}
	plain, w := step(1)
	scaled, ws := step(1024)
	if scaled != plain {
		t.Errorf("a step at gradient scale 1024 obtained %d bytes, one without %d", scaled, plain)
	}
	for i := range w {
		if w[i] != ws[i] {
			t.Fatalf("parameter value %d = %v at scale 1024, %v without", i, ws[i], w[i])
		}
	}
}
