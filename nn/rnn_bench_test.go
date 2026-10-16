package nn_test

import (
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkLSTM times a 2-layer LSTM of 256 hidden units over sequences
// of 128 steps of 64 features, in batches of 32: the forward pass alone,
// the forward and backward passes, and a whole training step with an
// AdamW update.
func BenchmarkLSTM(b *testing.B) {
	const steps, batch, in, hidden = 128, 32, 64, 256
	setup := func(b *testing.B, opts ...nn.Option) (*nn.LSTM, *tensor.Tensor, *tensor.Tensor) {
		nn.ManualSeed(1)
		m := nn.NewLSTM(in, hidden, append([]nn.Option{nn.NumLayers(2)}, opts...)...)
		x, y := tensor.Randn(1, steps, batch, in), tensor.Randn(2, steps, batch, hidden)
		if err := tensor.Realize(b.Context(), x, y); err != nil {
			b.Fatal(err)
		}
		return m, x, y
	}
	b.Run("forward", func(b *testing.B) {
		m, x, _ := setup(b)
		step := func() {
			if err := m.Forward(x).Realize(b.Context()); err != nil {
				b.Fatal(err)
			}
		}
		for range warmup {
			step()
		}
		for b.Loop() {
			step()
		}
	})
	backward := func(b *testing.B, opts ...nn.Option) {
		m, x, y := setup(b, opts...)
		step := func() {
			if err := loss.MSE(m.Forward(x), y).Backward(b.Context()); err != nil {
				b.Fatal(err)
			}
			for _, p := range nn.Parameters(m) {
				p.ZeroGrad()
			}
		}
		for range warmup {
			step()
		}
		for b.Loop() {
			step()
		}
	}
	b.Run("backward", func(b *testing.B) { backward(b) })
	b.Run("backward/checkpoint16", func(b *testing.B) { backward(b, nn.StepCheckpoint(16)) })
	b.Run("train", func(b *testing.B) {
		m, x, y := setup(b)
		opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
		step := func() {
			if err := loss.MSE(m.Forward(x), y).Backward(b.Context()); err != nil {
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
		for b.Loop() {
			step()
		}
	})
}
