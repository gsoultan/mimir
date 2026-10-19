package nn_test

import (
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkForward times a serving-sized forward pass of a GPT-2 small
// block (768 features in 12 heads) over one sequence of 32 tokens, with
// the outputs computed: tracked runs the trainable block in evaluation
// mode, which records a tape through every operation that reads a
// parameter, and inference runs it through nn.Inference, which records
// none.
func BenchmarkForward(b *testing.B) {
	const batch, seq, embed, heads = 1, 32, 768, 12
	b.Run("tracked", func(b *testing.B) {
		block := nn.NewGPT2Block(embed, heads, nn.Seed(1))
		nn.Eval(block)
		forward(b, block, tensor.Randn(2, batch, seq, embed))
	})
	b.Run("inference", func(b *testing.B) {
		block := nn.Inference(nn.NewGPT2Block(embed, heads, nn.Seed(1)))
		forward(b, block, tensor.Randn(2, batch, seq, embed))
	})
}

// forward times m.Forward(x) and the computation of its result.
func forward(b *testing.B, m nn.Module, x *tensor.Tensor) {
	ctx := b.Context()
	if err := x.Realize(ctx); err != nil {
		b.Fatal(err)
	}
	step := func() {
		if err := m.Forward(x).Realize(ctx); err != nil {
			b.Fatal(err)
		}
	}
	for range warmup {
		step()
	}
	for b.Loop() {
		step()
	}
}
