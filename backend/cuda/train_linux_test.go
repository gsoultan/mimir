//go:build linux && (amd64 || arm64) && !nocuda

package cuda_test

import (
	"math"
	"testing"

	"github.com/gsoultan/mimir/backend/cuda"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// TestTrainingStepsRunAsCUDAGraphs trains a small network on the GPU with
// CUDA graphs on and checks that its steps launch as graphs, captured a
// few times rather than at every step, and compute what the steps
// launched one by one compute.
func TestTrainingStepsRunAsCUDAGraphs(t *testing.T) {
	dev, err := tensor.GPU()
	if err != nil || dev.Name() != "cuda" {
		t.Skip("no CUDA device")
	}
	const steps = 12
	train := func(graphs bool) []float32 {
		t.Helper()
		defer cuda.SetGraphs(graphs)()
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
	graphs0, _, captures0 := cuda.GraphLaunches()
	withGraphs := train(true)
	graphs, _, captures := cuda.GraphLaunches()
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

// BenchmarkTrainingStepGraphs times the training step of
// BenchmarkStepOverhead's MLP on the GPU, with CUDA graphs off and on,
// and reports per step what its batches did with graphs: those launched
// as graphs and one call at a time, the graphs captured and the kernel
// nodes updated in them. A step that launches its batches as graphs but
// captures them again, or updates most of their nodes, pays more on the
// host than launching the calls one by one.
func BenchmarkTrainingStepGraphs(b *testing.B) {
	dev, err := tensor.GPU()
	if err != nil || dev.Name() != "cuda" {
		b.Skip("no CUDA device")
	}
	for _, graphs := range []bool{false, true} {
		b.Run(map[bool]string{false: "direct", true: "graphs"}[graphs], func(b *testing.B) {
			defer cuda.SetGraphs(graphs)()
			nn.ManualSeed(1)
			m := nn.NewSequential(nn.Flatten(), nn.NewLinear(28*28, 256), nn.ReLU(), nn.NewLinear(256, 10))
			if err := nn.To(m, dev); err != nil {
				b.Fatal(err)
			}
			opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
			labels := make([]int32, 128)
			for i := range labels {
				labels[i] = int32(i % 10)
			}
			x, y := tensor.Rand(1, 128, 1, 28, 28).To(dev), tensor.FromSlice(labels, 128).To(dev)
			if err := tensor.Realize(b.Context(), x, y); err != nil {
				b.Fatal(err)
			}
			step := func() {
				l := loss.CrossEntropy(m.Forward(x), y)
				if err := l.Backward(b.Context()); err != nil {
					b.Fatal(err)
				}
				if err := opt.Step(b.Context()); err != nil {
					b.Fatal(err)
				}
				opt.ZeroGrad()
			}
			for range 5 {
				step()
			}
			launched, direct, captures := cuda.GraphLaunches()
			updates := cuda.GraphUpdates()
			for b.Loop() {
				step()
			}
			l2, d2, c2 := cuda.GraphLaunches()
			n := float64(b.N)
			b.ReportMetric(float64(l2-launched)/n, "graphs/op")
			b.ReportMetric(float64(d2-direct)/n, "direct/op")
			b.ReportMetric(float64(c2-captures)/n, "captures/op")
			b.ReportMetric(float64(cuda.GraphUpdates()-updates)/n, "node-updates/op")
		})
	}
}
