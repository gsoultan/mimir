package tensor_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// firstOf and secondOf pick one result of CumMax.
func firstOf(v, _ *tensor.Tensor) *tensor.Tensor  { return v }
func secondOf(_, i *tensor.Tensor) *tensor.Tensor { return i }

func TestCumSum(t *testing.T) {
	x := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	for _, tt := range []struct {
		axis int
		want []float32
	}{
		{1, []float32{1, 3, 6, 4, 9, 15}},
		{-1, []float32{1, 3, 6, 4, 9, 15}},
		{0, []float32{1, 2, 3, 5, 7, 9}},
	} {
		if got := data[float32](t, x.CumSum(tt.axis)); !slices.Equal(got, tt.want) {
			t.Errorf("axis %d: %v, want %v", tt.axis, got, tt.want)
		}
	}
	// Int32 sums wrap.
	i := tensor.FromSlice([]int32{math.MaxInt32, 1, -5})
	if got := data[int32](t, i.CumSum(0)); !slices.Equal(got, []int32{math.MaxInt32, math.MinInt32, math.MinInt32 - 5 + 1<<32}) {
		t.Errorf("int32: %v", got)
	}
	// A scalar is its own running sum.
	if got := data[float32](t, tensor.Scalar[float32](2).CumSum(0)); !slices.Equal(got, []float32{2}) {
		t.Errorf("scalar: %v", got)
	}
	// A half tensor sums in float32 into its own dtype.
	h := x.Cast(tensor.BFloat16).CumSum(1)
	if h.DType() != tensor.BFloat16 {
		t.Errorf("bfloat16 sum of dtype %s", h.DType())
	}
	if got := data[float32](t, h.Cast(tensor.Float32)); !slices.Equal(got, []float32{1, 3, 6, 4, 9, 15}) {
		t.Errorf("bfloat16: %v", got)
	}
}

func TestCumMax(t *testing.T) {
	nan := float32(math.NaN())
	x := tensor.FromSlice([]float32{1, 3, 2, 3, nan, 5, 0, 2, 1, 2, 7, -1}, 2, 6)
	v, idx := x.CumMax(1)
	got, pos := data[float32](t, v), data[int32](t, idx)
	want := []float32{1, 3, 3, 3, nan, nan, 0, 2, 2, 2, 7, 7}
	for i := range want {
		if got[i] != want[i] && !(math.IsNaN(float64(got[i])) && math.IsNaN(float64(want[i]))) {
			t.Fatalf("values %v, want %v", got, want)
		}
	}
	// Ties and NaN take the latest position, like torch.cummax.
	if want := []int32{0, 1, 1, 3, 4, 4, 0, 1, 1, 3, 4, 4}; !slices.Equal(pos, want) {
		t.Errorf("indices %v, want %v", pos, want)
	}
	// Along the first axis of a transposed view.
	v, idx = tensor.FromSlice([]int32{5, 1, 7, 6, 6, 2}, 2, 3).Transpose(0, 1).CumMax(0)
	if got := data[int32](t, v); !slices.Equal(got, []int32{5, 6, 5, 6, 7, 6}) {
		t.Errorf("int32 values %v", got)
	}
	if got := data[int32](t, idx); !slices.Equal(got, []int32{0, 0, 0, 1, 2, 1}) {
		t.Errorf("int32 indices %v", got)
	}
}

func TestLogCumSumExp(t *testing.T) {
	inf := float32(math.Inf(1))
	in := []float32{-1000, 0, 1000, 1000, -inf, 3, -inf, -inf, 0.5, -2}
	got := data[float32](t, tensor.FromSlice(in, 2, 5).LogCumSumExp(-1))
	for row := range 2 {
		acc := math.Inf(-1)
		for j := range 5 {
			x := float64(in[row*5+j])
			hi, lo := math.Max(acc, x), math.Min(acc, x)
			if !math.IsInf(lo, -1) {
				hi += math.Log1p(math.Exp(lo - hi))
			}
			acc = hi
			if g := float64(got[row*5+j]); g != float64(float32(acc)) && math.Abs(g-acc) > 1e-6*math.Abs(acc) {
				t.Fatalf("element %d: got %v, want %v", row*5+j, g, acc)
			}
		}
	}
}

func TestCumulativeErrors(t *testing.T) {
	x := tensor.Ones(tensor.Float32, 2, 3)
	if err := x.CumSum(2).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("axis 2 of rank 2: %v, want ErrShape", err)
	}
	if v, i := x.CumMax(-3); !errors.Is(v.Err(), tensor.ErrShape) || !errors.Is(i.Err(), tensor.ErrShape) {
		t.Errorf("axis -3 of rank 2: %v, %v, want ErrShape", v.Err(), i.Err())
	}
	if err := tensor.Ones(tensor.Int32, 3).LogCumSumExp(0).Err(); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("int32: %v, want ErrDType", err)
	}
	if err := tensor.Scalar[float32](1).CumSum(1).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("axis 1 of a scalar: %v, want ErrShape", err)
	}
}

func TestHalfScansHaveGradients(t *testing.T) {
	ctx := t.Context()
	for _, dt := range []tensor.DType{tensor.BFloat16, tensor.Float16} {
		x := tensor.FromSlice([]float32{0.5, -1, 2, 0.25, 1, -0.5}, 2, 3).Cast(dt).RequiresGrad()
		v, _ := x.CumMax(1)
		loss := x.CumSum(0).Add(v).Add(x.LogCumSumExp(1)).Sum()
		if err := loss.Backward(ctx); err != nil {
			t.Fatalf("%s: %v", dt, err)
		}
		if g := x.Grad(); g == nil || !slices.Equal(g.Shape(), []int{2, 3}) || g.Err() != nil {
			t.Fatalf("%s: gradient %v", dt, g)
		}
	}
}

func TestCumMaxGradientFlowsToThePositions(t *testing.T) {
	ctx := t.Context()
	x := tensor.FromSlice([]float32{2, 1, 3, 3, 0}).RequiresGrad()
	v, _ := x.CumMax(0)
	if err := v.Mul(tensor.FromSlice([]float32{1, 2, 4, 8, 16})).Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	// Positions 0, 0, 2, 3, 3 hold the maxima.
	if got := data[float32](t, x.Grad()); !slices.Equal(got, []float32{3, 0, 4, 24, 0}) {
		t.Errorf("gradient %v", got)
	}
}
