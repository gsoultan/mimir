package nn_test

import (
	"fmt"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// warmup is the number of steps a benchmark runs before it starts timing:
// the first steps compile the GPU kernels, plan and cache the schedules,
// create the optimizer's state and, on CUDA, capture the step's graph,
// work a program does once, which timing would report as seconds per
// step when b.N is small.
const warmup = 3

// BenchmarkDecoderBlockStep times one training step of a GPT-2 small
// block (768 features in 12 heads) over 8 sequences: the forward pass, a
// loss, the backward pass and an AdamW update. In the upload and donated
// sub-benchmarks each step reads a new input, uploaded from host memory
// as a batch is, and donated reads it once only (see
// tensor.Tensor.Donate). Each reports the page faults of a step, where
// the platform counts them.
func BenchmarkDecoderBlockStep(b *testing.B) {
	const batch, embed, heads = 8, 768, 12
	for _, seq := range []int{256, 1024} {
		for _, feed := range []string{"", "/upload", "/donated"} {
			if seq > 256 && feed != "" {
				continue
			}
			b.Run(fmt.Sprintf("seq%d%s", seq, feed), func(b *testing.B) {
				decoderBlockStep(b, batch, seq, embed, heads, feed)
			})
		}
	}
}

// decoderBlockStep times a step of BenchmarkDecoderBlockStep, with its
// input fed as feed says.
func decoderBlockStep(b *testing.B, batch, seq, embed, heads int, feed string) {
	ctx := b.Context()
	block := nn.NewGPT2Block(embed, heads, nn.Seed(1))
	opt := optim.AdamW(nn.Parameters(block))
	x := tensor.Randn(2, batch, seq, embed)
	w := tensor.Randn(3, batch, seq, embed)
	if err := tensor.Realize(ctx, x, w); err != nil {
		b.Fatal(err)
	}
	var inputs [2][]float32
	for i := range inputs {
		var err error
		if inputs[i], err = tensor.Randn(uint64(i+4), batch, seq, embed).Data[float32](ctx); err != nil {
			b.Fatal(err)
		}
	}
	n := 0
	step := func() {
		x := x
		if feed != "" {
			n++
			x = tensor.FromSlice(inputs[n%len(inputs)], batch, seq, embed)
			if feed == "/donated" {
				x = x.Donate()
			}
		}
		if err := block.Forward(x).Mul(w).Mean().Backward(ctx); err != nil {
			b.Fatal(err)
		}
		if err := opt.Step(ctx); err != nil {
			b.Fatal(err)
		}
		opt.ZeroGrad()
	}
	for range warmup {
		step()
	}
	faults, counted := pageFaults()
	for b.Loop() {
		step()
	}
	if after, _ := pageFaults(); counted {
		b.ReportMetric(float64(after-faults)/float64(b.N), "faults/op")
	}
}

// llama1B returns a block at the shapes of Llama 3.2 1B: 2048 features,
// 32 query and 8 key heads of 64 features, and an MLP of 8192.
func llama1B() *nn.DecoderBlock {
	rope := nn.NewRotaryEmbedding(64, 500000)
	rope.Scaling = nn.Llama3RoPEScaling(32, 1, 4, 8192)
	return nn.NewLlamaBlock(2048, 32, 8, 8192, rope, nn.Seed(1))
}

// BenchmarkDecode times a decoding step of one token with 512 positions
// in the cache at the shapes of Llama 3.2 1B: the attention alone, with
// its cache append, and a whole block. The model has 16 blocks.
func BenchmarkDecode(b *testing.B) {
	const past = 512
	b.Run("attention", func(b *testing.B) {
		ctx := b.Context()
		cache := nn.NewKVCache(1, 8, past+1, 64, tensor.Float32)
		if _, _, err := cache.Append(ctx, tensor.Randn(1, 1, 8, past, 64), tensor.Randn(2, 1, 8, past, 64)); err != nil {
			b.Fatal(err)
		}
		q, k, v := tensor.Randn(3, 1, 32, 1, 64), tensor.Randn(4, 1, 8, 1, 64), tensor.Randn(5, 1, 8, 1, 64)
		if err := tensor.Realize(ctx, q, k, v); err != nil {
			b.Fatal(err)
		}
		step := func() {
			cache.Truncate(past)
			keys, values, err := cache.Append(ctx, k, v)
			if err != nil {
				b.Fatal(err)
			}
			if err := tensor.ScaledDotProductAttention(q, keys, values, tensor.CausalOffset(past)).Realize(ctx); err != nil {
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
	b.Run("block", func(b *testing.B) {
		ctx := b.Context()
		block := llama1B()
		nn.Eval(block)
		cache := nn.NewKVCache(1, 8, past+1, 64, tensor.Float32)
		prompt := tensor.Randn(2, 1, past, 2048)
		if _, err := block.Decode(ctx, prompt, cache); err != nil {
			b.Fatal(err)
		}
		x := tensor.Randn(3, 1, 1, 2048)
		if err := x.Realize(ctx); err != nil {
			b.Fatal(err)
		}
		step := func() {
			cache.Truncate(past)
			y, err := block.Decode(ctx, x, cache)
			if err != nil {
				b.Fatal(err)
			}
			if err := y.Realize(ctx); err != nil {
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
}
