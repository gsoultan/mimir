package tensor

import (
	"math"
	"testing"
)

// TestBluesteinMatchesTheTransform checks the transform GPUs build for
// lengths that are not powers of two, run here on the CPU, against the
// CPU's kernel, values and gradients.
func TestBluesteinMatchesTheTransform(t *testing.T) {
	ctx := t.Context()
	for _, inverse := range []bool{false, true} {
		x := Randn(3, 2, 100, 3, 2)
		w := Randn(4, 2, 100, 3, 2)
		var vals, gs [2][]float32
		for i, f := range []func(*Tensor) *Tensor{
			func(x *Tensor) *Tensor { return x.bluestein(1, inverse) },
			func(x *Tensor) *Tensor { return x.dft(1, inverse) },
		} {
			p := x.RequiresGrad()
			y := f(p)
			if err := y.Mul(w).Sum().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			var err error
			if vals[i], err = y.Detach().Data[float32](ctx); err != nil {
				t.Fatal(err)
			}
			if gs[i], err = p.Grad().Data[float32](ctx); err != nil {
				t.Fatal(err)
			}
		}
		for i := range vals[0] {
			if d := math.Abs(float64(vals[0][i] - vals[1][i])); d > 1e-4*math.Max(1, math.Abs(float64(vals[1][i]))) {
				t.Fatalf("inverse %v: element %d is %v, want %v", inverse, i, vals[0][i], vals[1][i])
			}
			if d := math.Abs(float64(gs[0][i] - gs[1][i])); d > 1e-4*math.Max(1, math.Abs(float64(gs[1][i]))) {
				t.Fatalf("inverse %v: gradient %d is %v, want %v", inverse, i, gs[0][i], gs[1][i])
			}
		}
	}
}
