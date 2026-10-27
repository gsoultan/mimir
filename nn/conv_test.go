package nn_test

import (
	"context"
	"math"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

func values(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, err := x.Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestGroupNorm(t *testing.T) {
	x := tensor.Randn(1, 2, 6, 4, 5).Mul(tensor.Full(tensor.Float32, 3)).Add(tensor.Full(tensor.Float32, 2))
	y := values(t, nn.NewGroupNorm(3, 6).Forward(x))
	// Each example's group of two channels has mean 0 and variance 1.
	const per = 2 * 4 * 5
	for g := range 2 * 3 {
		var s, sq float64
		for _, v := range y[g*per : (g+1)*per] {
			s += float64(v)
			sq += float64(v) * float64(v)
		}
		if m, v := s/per, sq/per-(s/per)*(s/per); math.Abs(m) > 1e-4 || math.Abs(v-1) > 1e-3 {
			t.Errorf("group %d: mean %v, variance %v", g, m, v)
		}
	}
	if err := nn.NewGroupNorm(4, 6).Forward(x).Err(); err == nil {
		t.Error("6 channels in 4 groups are accepted")
	}
}

func TestPoolingAndUpsampling(t *testing.T) {
	x := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 2, 2, 2)
	if got := values(t, nn.GlobalAvgPool()(x)); got[0] != 2.5 || got[1] != 6.5 {
		t.Errorf("global average %v", got)
	}
	up := nn.Upsample(2)(x)
	if s := up.Shape(); s[2] != 4 || s[3] != 4 {
		t.Fatalf("upsampled %v", s)
	}
	if got := values(t, up)[:8]; got[0] != 1 || got[1] != 1 || got[2] != 2 || got[4] != 1 || got[6] != 2 {
		t.Errorf("upsampled rows %v", got)
	}
}

// Gradients reach every weight of the new convolutions.
func TestConvLayersTrain(t *testing.T) {
	ctx := context.Background()
	for name, tc := range map[string]struct {
		m nn.Module
		x *tensor.Tensor
	}{
		"conv1d":           {nn.NewConv1d(3, 4, 3, nn.Padding(1)), tensor.Randn(1, 2, 3, 10)},
		"conv3d":           {nn.NewConv3d(2, 3, 3, nn.Padding(1)), tensor.Randn(2, 1, 2, 4, 4, 4)},
		"depthwise":        {nn.NewDepthwiseConv2d(4, 2, 3, nn.Padding(1)), tensor.Randn(3, 1, 4, 6, 6)},
		"conv_transpose2d": {nn.NewConvTranspose2d(3, 2, 4, nn.Stride(2), nn.Padding(1)), tensor.Randn(4, 1, 3, 5, 5)},
		"conv_transpose1d": {nn.NewConvTranspose1d(3, 2, 3, nn.Stride(2)), tensor.Randn(5, 1, 3, 7)},
	} {
		y := tc.m.Forward(tc.x)
		if err := y.Mul(y).Sum().Backward(ctx); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, p := range nn.Parameters(tc.m) {
			if p.Grad() == nil {
				t.Errorf("%s: a parameter %v has no gradient", name, p.Value().Shape())
			}
		}
	}
	if s := nn.NewConvTranspose2d(3, 2, 4, nn.Stride(2), nn.Padding(1)).Forward(tensor.Randn(4, 1, 3, 5, 5)).Shape(); s[2] != 10 || s[3] != 10 {
		t.Errorf("2× transposed conv of 5×5 is %v", s)
	}
}
