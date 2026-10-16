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

// waitTimer wraps a GPU backend that runs batches and convolution
// kernels, as both GPU backends do, and adds up the time it waits for
// the device. A GPU runs kernels after Run or RunBatch returns, so time
// in them is the launches', and the device's time shows only where the
// host waits for it: in Sync, at the end of every schedule, and in
// Download.
type waitTimer struct {
	backend.Backend
	batcher   backend.Batcher
	convolver backend.Convolver
	mu        sync.Mutex
	kernels   int
	wait      time.Duration
}

func (w *waitTimer) Run(ctx context.Context, k *backend.Kernel, outs, ins []backend.Buffer) error {
	w.count(1)
	return w.Backend.Run(ctx, k, outs, ins)
}

func (w *waitTimer) RunBatch(ctx context.Context, b *backend.Batch) error {
	w.count(len(b.Kernels))
	return w.batcher.RunBatch(ctx, b)
}

func (w *waitTimer) ConvParts(k *backend.Kernel) int { return w.convolver.ConvParts(k) }

func (w *waitTimer) Sync(ctx context.Context) error {
	defer w.waited(time.Now())
	return w.Backend.Sync(ctx)
}

func (w *waitTimer) Download(ctx context.Context, dst []byte, src backend.Buffer) error {
	defer w.waited(time.Now())
	return w.Backend.Download(ctx, dst, src)
}

func (w *waitTimer) count(kernels int) {
	w.mu.Lock()
	w.kernels += kernels
	w.mu.Unlock()
}

func (w *waitTimer) waited(start time.Time) {
	d := time.Since(start)
	w.mu.Lock()
	w.wait += d
	w.mu.Unlock()
}

// BenchmarkStepOverhead times the host side of a training step of the
// MNIST MLP and CNN on a batch of 128: everything but the kernels, which
// is building the graph, differentiating it, scheduling it and launching
// its kernels, with the schedule cache, which replays the steps after the
// first, and without it, which plans each.
//
// It runs on the CPU backend whatever the default device is: the CPU runs
// each kernel before Run returns, so the time outside Run is exactly the
// host's. It reports host-ns/op, the step's time outside kernels, and
// kernel-ns/op, the time in the CPU's kernels, which is TrainStep's time
// on the CPU, not on a GPU, and varies with what else the host's cores
// run.
//
// When the default device is a GPU, the gpu sub-benchmarks time the same
// steps there. A GPU's kernels run while the host goes on, so they report
// host-ns/op, the step's time outside waits for the device, which
// includes the launches, and wait-ns/op, the time the host waits for the
// device to finish what it launched: the kernels' time that the host's
// work did not hide.
func BenchmarkStepOverhead(b *testing.B) {
	type device struct {
		name  string
		be    func() backend.Backend
		times func(backend.Backend) (kernels int, kernel time.Duration)
	}
	devices := []device{{
		name: "",
		be:   func() backend.Backend { return &kernelTimer{Backend: cpu.New()} },
		times: func(be backend.Backend) (int, time.Duration) {
			k := be.(*kernelTimer)
			defer func() { k.busy, k.kernels = 0, 0 }()
			return k.kernels, k.busy
		},
	}}
	gpu := tensor.BackendOf(tensor.DefaultDevice())
	batcher, isBatcher := gpu.(backend.Batcher)
	convolver, isConvolver := gpu.(backend.Convolver)
	if gpu.Name() != "cpu" && isBatcher && isConvolver {
		devices = append(devices, device{
			name: "/gpu",
			be:   func() backend.Backend { return &waitTimer{Backend: gpu, batcher: batcher, convolver: convolver} },
			times: func(be backend.Backend) (int, time.Duration) {
				w := be.(*waitTimer)
				defer func() { w.wait, w.kernels = 0, 0 }()
				return w.kernels, w.wait
			},
		})
	}
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
			for _, dev := range devices {
				b.Run(bm.name+"/"+cache+dev.name, func(b *testing.B) {
					defer sched.SetCaching(sched.SetCaching(cache == "cached"))
					be := dev.be()
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
					for range warmup {
						step()
					}
					dev.times(be)
					var total time.Duration
					for b.Loop() {
						start := time.Now()
						step()
						total += time.Since(start)
					}
					kernels, busy := dev.times(be)
					metric := "kernel-ns/op"
					if dev.name != "" {
						metric = "wait-ns/op"
					}
					b.ReportMetric(float64(total-busy)/float64(b.N), "host-ns/op")
					b.ReportMetric(float64(busy)/float64(b.N), metric)
					b.ReportMetric(float64(kernels)/float64(b.N), "kernels/op")
				})
			}
		}
	}
}
