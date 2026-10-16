package tensor_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/sched"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// kernelTimer wraps the CPU backend and adds up the time its kernels
// take. The CPU backend runs each kernel before Run returns, so the time
// in Run is the kernel's.
type kernelTimer struct {
	*cpu.Backend
	mu      sync.Mutex
	kernels int
	busy    time.Duration
}

func (k *kernelTimer) Run(ctx context.Context, kern *backend.Kernel, outs, ins []backend.Buffer) error {
	start := time.Now()
	err := k.Backend.Run(ctx, kern, outs, ins)
	d := time.Since(start)
	k.mu.Lock()
	k.kernels++
	k.busy += d
	k.mu.Unlock()
	return err
}

// BenchmarkStepOverhead times the host side of a training step of the
// MNIST MLP and CNN on a batch of 128: everything but the kernels, which
// is building the graph, differentiating it, scheduling it and launching
// its kernels. It reports host-ns/op, the step's time outside kernels,
// and kernel-ns/op, the time in them, with the schedule cache, which
// replays the steps after the first, and without it, which plans each.
func BenchmarkStepOverhead(b *testing.B) {
	for _, bm := range []struct {
		name  string
		model func() nn.Module
	}{
		{"mlp", func() nn.Module {
			return nn.NewSequential(nn.Flatten(), nn.NewLinear(28*28, 256), nn.ReLU(), nn.NewLinear(256, 10))
		}},
		{"cnn", func() nn.Module {
			return nn.NewSequential(
				nn.NewConv2d(1, 32, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
				nn.NewConv2d(32, 64, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
				nn.Flatten(), nn.NewLinear(64*7*7, 128), nn.ReLU(), nn.NewLinear(128, 10),
			)
		}},
	} {
		for _, cache := range []string{"cached", "planned"} {
			b.Run(bm.name+"/"+cache, func(b *testing.B) {
				defer sched.SetCaching(sched.SetCaching(cache == "cached"))
				be := &kernelTimer{Backend: cpu.New()}
				defer tensor.SetDevice(be)()
				nn.ManualSeed(1)
				m := bm.model()
				opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
				x := tensor.Rand(1, 128, 1, 28, 28)
				labels := make([]int32, 128)
				for i := range labels {
					labels[i] = int32(i % 10)
				}
				y := tensor.FromSlice(labels, 128)
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
				for range 3 {
					step()
				}
				be.busy, be.kernels = 0, 0
				var total time.Duration
				for b.Loop() {
					start := time.Now()
					step()
					total += time.Since(start)
				}
				b.ReportMetric(float64(total-be.busy)/float64(b.N), "host-ns/op")
				b.ReportMetric(float64(be.busy)/float64(b.N), "kernel-ns/op")
				b.ReportMetric(float64(be.kernels)/float64(b.N), "kernels/op")
			})
		}
	}
}
