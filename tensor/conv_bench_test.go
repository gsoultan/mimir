package tensor_test

import (
	"context"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/tensor"
)

// windowed is a backend that runs convolutions as the products of their
// windows, as the CPU does, by hiding that the backend it wraps runs
// convolution kernels.
type windowed struct{ backend.Backend }

// windowedBatcher is windowed for a backend that runs kernels in batches.
type windowedBatcher struct {
	windowed
	b backend.Batcher
}

func (w windowedBatcher) RunBatch(ctx context.Context, b *backend.Batch) error {
	return w.b.RunBatch(ctx, b)
}

func withoutConvolutions(be backend.Backend) backend.Backend {
	if b, ok := be.(backend.Batcher); ok {
		return windowedBatcher{windowed{be}, b}
	}
	return windowed{be}
}

// BenchmarkConv2d times a convolution layer's forward and backward pass,
// with the gradients of its weights and, but for a network's first layer,
// of its input, at the shapes of examples/mnist's CNN on a batch of 128
// and of ResNet-18 on a batch of 16. On a device that runs convolution
// kernels (see backend.Convolver) "kernel" runs them and "windows" runs
// the products of the windows that the other devices run.
func BenchmarkConv2d(b *testing.B) {
	for _, c := range []struct {
		name             string
		n, c, o, size, k int
		opts             tensor.ConvOptions
		first            bool // the input needs no gradient
	}{
		{"cnn/conv1", 128, 1, 32, 28, 3, tensor.ConvOptions{Padding: 1}, true},
		{"cnn/conv2", 128, 32, 64, 14, 3, tensor.ConvOptions{Padding: 1}, false},
		{"resnet18/stem", 16, 3, 64, 224, 7, tensor.ConvOptions{Stride: 2, Padding: 3}, true},
		{"resnet18/64", 16, 64, 64, 56, 3, tensor.ConvOptions{Padding: 1}, false},
		{"resnet18/128s2", 16, 64, 128, 56, 3, tensor.ConvOptions{Stride: 2, Padding: 1}, false},
		{"resnet18/128", 16, 128, 128, 28, 3, tensor.ConvOptions{Padding: 1}, false},
		{"resnet18/128down", 16, 64, 128, 56, 1, tensor.ConvOptions{Stride: 2}, false},
		{"resnet18/256", 16, 256, 256, 14, 3, tensor.ConvOptions{Padding: 1}, false},
		{"resnet18/512", 16, 512, 512, 7, 3, tensor.ConvOptions{Padding: 1}, false},
	} {
		dev := tensor.BackendOf(tensor.DefaultDevice())
		paths := map[string]backend.Backend{"windows": withoutConvolutions(dev)}
		if _, ok := dev.(backend.Convolver); ok {
			paths["kernel"] = dev
		}
		for _, path := range []string{"windows", "kernel"} {
			be, ok := paths[path]
			if !ok {
				continue
			}
			b.Run(c.name+"/"+path, func(b *testing.B) {
				defer tensor.SetDevice(be)()
				x := realized(b, 1, c.n, c.c, c.size, c.size)
				w := realized(b, 2, c.o, c.c, c.k, c.k)
				if !c.first {
					x = x.RequiresGrad()
				}
				w = w.RequiresGrad()
				stride := max(c.opts.Stride, 1)
				out := (c.size+2*c.opts.Padding-c.k)/stride + 1
				flops := 2 * float64(c.n*c.o*c.c*c.k*c.k*out*out)
				passes := 2.0 // forward and weight gradient
				if !c.first {
					passes++
				}
				step := func() {
					y := x.Conv2d(w, nil, c.opts)
					if err := y.Mul(y).Sum().Backward(b.Context()); err != nil {
						b.Fatal(err)
					}
					x.ZeroGrad()
					w.ZeroGrad()
				}
				for range warmup {
					step()
				}
				for b.Loop() {
					step()
				}
				b.ReportMetric(passes*flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}
