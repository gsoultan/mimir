package vision_test

import (
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/train"
	"github.com/gsoultan/mimir/vision"
)

// colours returns n photos of one colour, red or blue, with a little
// noise, labelled by colour.
func colours(t *testing.T, n, size int, seed uint64) (*tensor.Tensor, []int32) {
	t.Helper()
	noise, err := tensor.Randn(seed, n*3*size*size).Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, n*3*size*size)
	y := make([]int32, n)
	for i := range n {
		y[i] = int32(i % 2)
		for c := range 3 {
			v := float32(-1.4)
			if (y[i] == 0 && c == 0) || (y[i] == 1 && c == 2) {
				v = 1.2
			}
			for p := range size * size {
				at := (i*3+c)*size*size + p
				x[at] = v + 0.1*noise[at]
			}
		}
	}
	return tensor.FromSlice(x, n, 3, size, size), y
}

// A MobileNet taught in a few steps answers in inference mode as it does
// in training once its batch norms' statistics are recomputed; with the
// running averages kept while training, it answers every photo alike.
func TestPreciseBatchNormMobileNet(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a MobileNet")
	}
	ctx := t.Context()
	x, y := colours(t, 16, 16, 7)
	ds, err := data.FromTensors(x, tensor.FromSlice(y), data.BatchSize(8))
	if err != nil {
		t.Fatal(err)
	}
	right := func(m nn.Module) int {
		nn.Eval(m)
		out, err := m.Forward(x).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		n := 0
		for i, want := range y {
			if (out[2*i+1] > out[2*i]) == (want == 1) {
				n++
			}
		}
		return n
	}
	for _, precise := range []bool{false, true} {
		nn.ManualSeed(3)
		m := vision.NewMobileNetV2(2, vision.Small(), vision.Width(0.35))
		opts := []train.Option{train.Loss(loss.CrossEntropy), train.Epochs(4)}
		if precise {
			opts = append(opts, train.PreciseBatchNorm(2))
		}
		if err := train.Fit(ctx, m, optim.AdamW(nn.Parameters(m), optim.LR(3e-3)), ds, opts...); err != nil {
			t.Fatal(err)
		}
		n := right(m)
		t.Logf("precise %v: %d of %d right", precise, n, len(y))
		if precise && n != len(y) {
			t.Errorf("with precise batch norm, %d of %d right", n, len(y))
		}
	}
}

func TestBatchNorms(t *testing.T) {
	if n := len(nn.BatchNorms(vision.ResNet18(2))); n != 20 {
		t.Errorf("ResNet-18 has %d batch norms, want 20", n)
	}
	if n := len(nn.BatchNorms(nn.NewSequential(nn.NewLinear(2, 2)))); n != 0 {
		t.Errorf("a linear layer has %d batch norms", n)
	}
}
