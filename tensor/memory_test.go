package tensor_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// allocCounter counts the elements a backend allocates.
type allocCounter struct {
	*cpu.Backend
	mu       sync.Mutex
	elements int
}

func (c *allocCounter) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	c.mu.Lock()
	c.elements += n
	c.mu.Unlock()
	return c.Backend.Alloc(dt, n)
}

func TestTrainingStepsReuseTheirMemory(t *testing.T) {
	if tensor.Eager() {
		t.Skip("eager mode computes every operation into memory of its own")
	}
	for name, newOpt := range map[string]func([]*tensor.Param) *optim.Optimizer{
		"AdamW":        func(ps []*tensor.Param) *optim.Optimizer { return optim.AdamW(ps) },
		"SGD momentum": func(ps []*tensor.Param) *optim.Optimizer { return optim.SGD(ps, optim.Momentum(0.9)) },
	} {
		// Several backward passes per step accumulate the gradients.
		for _, backwards := range []int{1, 3} {
			t.Run(fmt.Sprintf("%s/%d backward", name, backwards), func(t *testing.T) {
				trainingStepsReuseTheirMemory(t, newOpt, backwards)
			})
		}
	}
}

func trainingStepsReuseTheirMemory(t *testing.T, newOpt func([]*tensor.Param) *optim.Optimizer, backwards int) {
	be := &allocCounter{Backend: cpu.New()}
	defer tensor.SetDevice(be)()
	model := nn.NewSequential(
		nn.NewConv2d(1, 4, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
		nn.Flatten(), nn.NewLinear(4*4*4, 16), nn.ReLU(), nn.NewLinear(16, 3),
	)
	opt := newOpt(nn.Parameters(model))
	x := tensor.Randn(1, 8, 1, 8, 8)
	y := tensor.FromSlice([]int32{0, 1, 2, 0, 1, 2, 0, 1})
	for i := range 6 {
		before := be.elements
		for range backwards {
			l := loss.CrossEntropy(model.Forward(x), y)
			if err := l.Backward(t.Context()); err != nil {
				t.Fatal(err)
			}
		}
		if err := opt.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
		// After the first steps fill the pool, a step allocates only
		// the losses it returns.
		if n := be.elements - before; i >= 2 && n > backwards {
			t.Errorf("step %d allocated %d elements, want at most %d", i, n, backwards)
		}
	}
}
