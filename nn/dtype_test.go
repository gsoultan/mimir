package nn_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// floats returns x's elements, through float32 for a half tensor.
func floats(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	return data(t, x.Cast(tensor.Float32))
}

func TestCastConvertsTrainableParameters(t *testing.T) {
	m := nn.NewSequential(nn.NewConv2d(1, 2, 3), nn.NewBatchNorm2d(2), nn.Flatten(), nn.NewLinear(8, 3))
	x := tensor.Randn(5, 3, 1, 4, 4)
	if err := m.Forward(x).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	before := map[string][]float32{}
	for name, p := range nn.NamedParams(m) {
		before[name] = data(t, p.Value())
	}
	if err := nn.Cast(m, tensor.BFloat16); err != nil {
		t.Fatal(err)
	}
	for name, p := range nn.NamedParams(m) {
		want := tensor.Float32 // a buffer keeps its dtype
		if p.Trainable() {
			want = tensor.BFloat16
		}
		if got := p.Value().DType(); got != want {
			t.Errorf("%s is %s after the cast, want %s", name, got, want)
		}
		if got, want := floats(t, p.Value()), floats(t, tensor.FromSlice(before[name]).Cast(want)); !slices.Equal(got, want) {
			t.Errorf("%s = %v after the cast, want %v", name, got, want)
		}
		if p.Trainable() && p.Grad() == nil {
			t.Errorf("%s lost its gradient", name)
		}
	}
	// The cast model trains: its float32 activations and gradients reach
	// the half parameters.
	nn.Train(m)
	if err := m.Forward(x).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := nn.Cast(m, tensor.Int32); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("cast to int32: %v, want ErrDType", err)
	}
	if err := nn.Cast(m, tensor.Float32); err != nil {
		t.Fatal(err)
	}
	for name, p := range nn.NamedParams(m) {
		if got := p.Value().DType(); got != tensor.Float32 {
			t.Errorf("%s is %s after a cast back, want float32", name, got)
		}
	}
}

// TestHalfLinearTrains fits a bfloat16 Linear layer with master weights
// as closely as its float32 twin, and without them still fits.
func TestHalfLinearTrains(t *testing.T) {
	ctx := t.Context()
	x := tensor.Randn(7, 64, 4)
	y := x.MatMul(tensor.FromSlice([]float32{1.1, -2.3, 0.7, 3.3}, 4, 1)).Add(tensor.Scalar(float32(0.45)))
	train := func(dt tensor.DType, opts ...optim.Option) []float32 {
		nn.ManualSeed(1)
		m := nn.NewLinear(4, 1)
		if err := nn.Cast(m, dt); err != nil {
			t.Fatal(err)
		}
		opt := optim.SGD(nn.Parameters(m), append([]optim.Option{optim.LR(0.1)}, opts...)...)
		var losses []float32
		for range 60 {
			l := loss.MSE(m.Forward(x), y)
			losses = append(losses, data(t, l)[0])
			if err := l.Backward(ctx); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(ctx); err != nil {
				t.Fatal(err)
			}
			opt.ZeroGrad()
		}
		if got := m.Weight.Value().DType(); got != dt {
			t.Fatalf("weights are %s after training, want %s", got, dt)
		}
		return losses
	}
	full := train(tensor.Float32)
	master := train(tensor.BFloat16, optim.MasterWeights())
	plain := train(tensor.BFloat16)
	last := len(full) - 1
	t.Logf("losses after %d steps: float32 %v, bfloat16 with master weights %v, bfloat16 %v", last+1, full[last], master[last], plain[last])
	if full[last] > full[0]/100 {
		t.Fatalf("float32 loss went from %v to %v", full[0], full[last])
	}
	if math.Abs(float64(master[0]-full[0])) > 1e-3*float64(full[0]) {
		t.Errorf("first losses differ: bfloat16 %v, float32 %v", master[0], full[0])
	}
	if math.Abs(float64(master[last]-full[last])) > 1e-2 {
		t.Errorf("bfloat16 loss %v after training with master weights, float32 %v", master[last], full[last])
	}
	if !(plain[last] < plain[0]/2) {
		t.Errorf("bfloat16 loss went from %v to %v without master weights", plain[0], plain[last])
	}
}
