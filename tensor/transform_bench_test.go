package tensor_test

import (
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkPerExampleGradients computes the gradients of each example's
// loss with respect to the parameters of an MLP, for a batch of 64: all at
// once with VMap, and one example at a time with Grad, each example's
// gradients realized together in one schedule either way.
func BenchmarkPerExampleGradients(b *testing.B) {
	nn.ManualSeed(1)
	m := nn.NewSequential(nn.NewLinear(32, 64), nn.ReLU(), nn.NewLinear(64, 10))
	ps := nn.Parameters(m)
	values := make([]*tensor.Tensor, len(ps))
	for i, p := range ps {
		values[i] = p.Value()
	}
	const n = 64
	x := tensor.Rand(1, n, 32)
	labels := make([]int32, n)
	for i := range labels {
		labels[i] = int32(i % 10)
	}
	y := tensor.FromSlice(labels, n)
	if err := tensor.Realize(b.Context(), x, y); err != nil {
		b.Fatal(err)
	}
	grads := func(x, y *tensor.Tensor) []*tensor.Tensor {
		return tensor.Gradients(loss.CrossEntropy(m.Forward(x.Unsqueeze(0)), y.Unsqueeze(0)), values...)
	}
	b.Run("vmap", func(b *testing.B) {
		for b.Loop() {
			gs := tensor.VMap(func(in ...*tensor.Tensor) []*tensor.Tensor { return grads(in[0], in[1]) })(x, y)
			if err := tensor.Realize(b.Context(), gs...); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("loop", func(b *testing.B) {
		for b.Loop() {
			var gs []*tensor.Tensor
			for k := range n {
				gs = append(gs, grads(x.Shrink([][2]int{{k, k + 1}, {0, 32}}).Squeeze(0), y.Shrink([][2]int{{k, k + 1}}).Squeeze(0))...)
			}
			if err := tensor.Realize(b.Context(), gs...); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkHessianVectorProduct times a Hessian-vector product of the
// loss of an MLP with respect to its parameters, forward over reverse,
// against a gradient alone.
func BenchmarkHessianVectorProduct(b *testing.B) {
	nn.ManualSeed(1)
	m := nn.NewSequential(nn.NewLinear(32, 64), nn.ReLU(), nn.NewLinear(64, 10))
	ps := nn.Parameters(m)
	x := tensor.Rand(1, 64, 32)
	labels := make([]int32, 64)
	for i := range labels {
		labels[i] = int32(i % 10)
	}
	y := tensor.FromSlice(labels, 64)
	values := make([]*tensor.Tensor, len(ps))
	vs := make([]*tensor.Tensor, len(ps))
	for i, p := range ps {
		values[i] = p.Value().Detach()
		vs[i] = tensor.Randn(uint64(i+2), p.Value().Shape()...)
	}
	if err := tensor.Realize(b.Context(), append(append([]*tensor.Tensor{x, y}, values...), vs...)...); err != nil {
		b.Fatal(err)
	}
	// f is the loss as a function of the parameters' values.
	f := func(w ...*tensor.Tensor) *tensor.Tensor {
		h := x.MatMulT(w[0]).Add(w[1]).ReLU().MatMulT(w[2]).Add(w[3])
		return loss.CrossEntropy(h, y)
	}
	b.Run("gradient", func(b *testing.B) {
		for b.Loop() {
			if err := tensor.Realize(b.Context(), tensor.Grad(f)(values...)...); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("hvp", func(b *testing.B) {
		for b.Loop() {
			_, hv := tensor.JVP(tensor.Grad(f), values, vs)
			if err := tensor.Realize(b.Context(), hv...); err != nil {
				b.Fatal(err)
			}
		}
	})
}
