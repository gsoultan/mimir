package nn_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/internal/quanttest"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// TestQuantizedLinearAndEmbedding checks layers whose weights are
// quantized buffers, as a model read from a GGUF file holds them: they
// compute what the same layers with the dequantized weights compute, up to
// the CPU's rounding of activations to int8, gradients reach the inputs and
// the bias, and Save refuses the quantized weights.
func TestQuantizedLinearAndEmbedding(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const in, out = 512, 6
	for _, dt := range []tensor.DType{tensor.Q8_0, tensor.Q4_K, tensor.Q6_K} {
		raw := quanttest.Random(rng, dt, out*in)
		q := tensor.FromBytes(dt, raw, out, in).To(tensor.DefaultDevice())
		deq := tensor.FromSlice(quanttest.Dequantize(dt, raw), out, in)

		lin := &nn.Linear{Weight: tensor.NewBuffer(q), Bias: tensor.NewParam(tensor.Randn(1, out))}
		ref := &nn.Linear{Weight: tensor.NewBuffer(deq), Bias: lin.Bias}
		xv := make([]float32, 3*in)
		for i := range xv {
			xv[i] = rng.Float32()*2 - 1
		}
		x := tensor.FromSlice(xv, 3, in).RequiresGrad()
		y := lin.Forward(x)
		got, want := data(t, y), data(t, ref.Forward(x))
		// Rounding x, whose values are at most 1, to int8 moves each by at
		// most 1/254, and the result by that times the weights' magnitudes.
		w := quanttest.Dequantize(dt, raw)
		for i := range want {
			j := i % out
			var bound float64
			for _, v := range w[j*in : (j+1)*in] {
				bound += math.Abs(float64(v)) / 254
			}
			if math.Abs(float64(got[i]-want[i])) > bound+1e-4*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("%s linear: element %d = %v, want about %v", dt, i, got[i], want[i])
			}
		}
		if err := y.Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		if x.Grad() == nil || lin.Bias.Grad() == nil {
			t.Fatalf("%s linear: no gradient for the input or the bias", dt)
		}
		if lin.Weight.Grad() != nil {
			t.Errorf("%s linear: a gradient for the quantized weight", dt)
		}

		emb := &nn.Embedding{Weight: tensor.NewBuffer(q)}
		idx := tensor.FromSlice([]int32{5, 0, 5, 2}, 2, 2)
		rows := emb.Forward(idx)
		if rows.DType() != tensor.Float32 || !slices.Equal(rows.Shape(), []int{2, 2, in}) {
			t.Fatalf("%s embedding: %v", dt, rows)
		}
		gotRows, table := data(t, rows), data(t, deq)
		for r, j := range []int{5, 0, 5, 2} {
			for c := range in {
				if gotRows[r*in+c] != table[j*in+c] {
					t.Fatalf("%s embedding: row %d differs from row %d of the table", dt, r, j)
				}
			}
		}
	}
	if _, err := tensor.NewParam(tensor.FromBytes(tensor.Q8_0, make([]byte, 34), 32)).Value().Data[float32](t.Context()); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("trainable quantized parameter: %v", err)
	}
}
