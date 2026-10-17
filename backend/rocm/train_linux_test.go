//go:build linux && (amd64 || arm64) && !norocm

package rocm_test

import (
	"math"
	"testing"

	"github.com/gsoultan/mimir/backend/rocm"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// TestTrainingStepsRunAsHIPGraphs trains a small network on the GPU with
// HIP graphs on and checks that its steps launch as graphs, captured a
// few times rather than at every step, and compute what the steps
// launched one by one compute.
func TestTrainingStepsRunAsHIPGraphs(t *testing.T) {
	dev, err := tensor.GPU()
	if err != nil || dev.Name() != "rocm" {
		t.Skip("no ROCm device")
	}
	const steps = 12
	train := func(graphs bool) []float32 {
		t.Helper()
		defer rocm.SetGraphs(graphs)()
		nn.ManualSeed(1)
		m := nn.NewSequential(nn.Flatten(), nn.NewLinear(28*28, 64), nn.ReLU(), nn.NewLinear(64, 10))
		if err := nn.To(m, dev); err != nil {
			t.Fatal(err)
		}
		opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
		labels := make([]int32, 32)
		for i := range labels {
			labels[i] = int32(i % 10)
		}
		x, y := tensor.Rand(1, 32, 1, 28, 28).To(dev), tensor.FromSlice(labels, 32).To(dev)
		if err := tensor.Realize(t.Context(), x, y); err != nil {
			t.Fatal(err)
		}
		var losses []float32
		for range steps {
			l := loss.CrossEntropy(m.Forward(x), y)
			if err := l.Backward(t.Context()); err != nil {
				t.Fatal(err)
			}
			v, err := l.Item[float32](t.Context())
			if err != nil {
				t.Fatal(err)
			}
			losses = append(losses, v)
			if err := opt.Step(t.Context()); err != nil {
				t.Fatal(err)
			}
			opt.ZeroGrad()
		}
		return losses
	}
	direct := train(false)
	graphs0, _, captures0 := rocm.GraphLaunches()
	withGraphs := train(true)
	graphs, _, captures := rocm.GraphLaunches()
	graphs, captures = graphs-graphs0, captures-captures0
	t.Logf("%d steps: %d batches launched as graphs, %d graphs captured", steps, graphs, captures)
	// Each step runs a batch for the forward and backward passes and one
	// for the update, which differ in the first steps, while the
	// optimizer's state is new.
	if graphs < steps {
		t.Errorf("%d batches launched as graphs in %d steps, want at least %d", graphs, steps, steps)
	}
	if captures > 8 {
		t.Errorf("%d graphs captured in %d steps: the steps capture their graphs again", captures, steps)
	}
	for i := range steps {
		if d := math.Abs(float64(withGraphs[i] - direct[i])); d > 1e-5*math.Abs(float64(direct[i])) {
			t.Errorf("step %d: loss %v with graphs, %v without", i, withGraphs[i], direct[i])
		}
	}
}
