package train_test

import (
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkTrainStep times one training step on a batch of 128 MNIST-sized
// images: the forward pass, the loss, the backward pass and an AdamW
// update, for the two models of examples/mnist.
func BenchmarkTrainStep(b *testing.B) {
	models := map[string]func() nn.Module{
		"mlp": func() nn.Module {
			return nn.NewSequential(
				nn.Flatten(),
				nn.NewLinear(28*28, 256), nn.ReLU(),
				nn.NewLinear(256, 10),
			)
		},
		"cnn": func() nn.Module {
			return nn.NewSequential(
				nn.NewConv2d(1, 16, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
				nn.NewConv2d(16, 32, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
				nn.Flatten(),
				nn.NewLinear(32*7*7, 10),
			)
		},
	}
	for _, name := range []string{"mlp", "cnn"} {
		b.Run(name, func(b *testing.B) {
			const batch = 128
			nn.ManualSeed(1)
			m := models[name]()
			opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
			x := tensor.Rand(1, batch, 1, 28, 28)
			labels := make([]int32, batch)
			for i := range labels {
				labels[i] = int32(i % 10)
			}
			y := tensor.FromSlice(labels, batch)
			if err := tensor.Realize(b.Context(), x, y); err != nil {
				b.Fatal(err)
			}
			for b.Loop() {
				l := loss.CrossEntropy(m.Forward(x), y)
				if err := l.Backward(b.Context()); err != nil {
					b.Fatal(err)
				}
				if err := opt.Step(b.Context()); err != nil {
					b.Fatal(err)
				}
				opt.ZeroGrad()
			}
		})
	}
}
