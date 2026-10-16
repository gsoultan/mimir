package tensor_test

import (
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/tensor"
)

// linalgTol bounds the differences of float32 factorizations and their
// gradients from float64 ones.
var linalgTol = goldentest.Tol{Atol: 2e-5, Rtol: 2e-4}

// shiftBy returns the matrices x plus by times the identity.
func shiftBy(x *tensor.Tensor, by float32) *tensor.Tensor {
	n := x.Shape()[x.Rank()-1]
	return x.Add(eyeOf(n).Mul(tensor.Scalar(by)))
}

// flatOf returns ts flattened and concatenated.
func flatOf(ts ...*tensor.Tensor) *tensor.Tensor {
	parts := make([]*tensor.Tensor, len(ts))
	for i, t := range ts {
		parts[i] = t.Reshape(-1)
	}
	return tensor.Cat(0, parts...)
}

func svdParts(x *tensor.Tensor) *tensor.Tensor {
	u, s, vh := x.SVD()
	return flatOf(s, u.Mul(u), vh.Mul(vh))
}

// linalgGoldenCases mirrors linalg_cases in tools/golden/gen.py.
var linalgGoldenCases = map[string]goldenCase{
	"sort_values": un(func(x *tensor.Tensor) *tensor.Tensor { v, _ := x.Sort(1, false); return v }, exact),
	"sort_descending_indices": un(func(x *tensor.Tensor) *tensor.Tensor {
		_, i := x.Sort(1, true)
		return i
	}, exact),
	"argsort_axis0": un(func(x *tensor.Tensor) *tensor.Tensor { return x.Argsort(0, false) }, exact),
	"topk_values":   un(func(x *tensor.Tensor) *tensor.Tensor { v, _ := x.TopK(3, 1, true); return v }, exact),
	"topk_smallest_indices": un(func(x *tensor.Tensor) *tensor.Tensor {
		_, i := x.TopK(2, 0, false)
		return i
	}, exact),

	"fft":              un(func(x *tensor.Tensor) *tensor.Tensor { return x.FFT(1) }, loose),
	"fft_bluestein":    un(func(x *tensor.Tensor) *tensor.Tensor { return x.FFT(0) }, loose),
	"ifft":             un(func(x *tensor.Tensor) *tensor.Tensor { return x.IFFT(-1) }, loose),
	"rfft":             un(func(x *tensor.Tensor) *tensor.Tensor { return x.RFFT(-1) }, loose),
	"irfft":            un(func(x *tensor.Tensor) *tensor.Tensor { return x.IRFFT(-1, 8) }, loose),
	"irfft_odd_padded": un(func(x *tensor.Tensor) *tensor.Tensor { return x.IRFFT(0, 9) }, loose),

	"cholesky": un(func(x *tensor.Tensor) *tensor.Tensor {
		return shiftBy(x.MatMul(x.Transpose(-2, -1)), 2).Cholesky()
	}, linalgTol),
	"solve":        bin(func(a, b *tensor.Tensor) *tensor.Tensor { return tensor.Solve(shiftBy(a, 3), b) }, linalgTol),
	"solve_vector": bin(func(a, b *tensor.Tensor) *tensor.Tensor { return tensor.Solve(shiftBy(a, 3), b) }, linalgTol),
	"triangular_solve_lower": bin(func(a, b *tensor.Tensor) *tensor.Tensor {
		return tensor.TriangularSolve(shiftBy(a, 3), b, false)
	}, linalgTol),
	"triangular_solve_upper": bin(func(a, b *tensor.Tensor) *tensor.Tensor {
		return tensor.TriangularSolve(shiftBy(a, 3), b, true)
	}, linalgTol),
	"inv":               un(func(x *tensor.Tensor) *tensor.Tensor { return shiftBy(x, 3).Inv() }, linalgTol),
	"det":               un(func(x *tensor.Tensor) *tensor.Tensor { return shiftBy(x, 1).Det() }, linalgTol),
	"slogdet_sign":      un(func(x *tensor.Tensor) *tensor.Tensor { s, _ := x.SlogDet(); return s }, exact),
	"slogdet_logabsdet": un(func(x *tensor.Tensor) *tensor.Tensor { _, l := x.SlogDet(); return l }, linalgTol),
	"qr_tall":           un(qrParts, linalgTol),
	"qr_wide":           un(qrParts, linalgTol),
	"svd_tall":          un(svdParts, linalgTol),
	"svd_wide":          un(svdParts, linalgTol),
	"eigh": un(func(x *tensor.Tensor) *tensor.Tensor {
		w, v := x.Add(x.Transpose(-2, -1)).Eigh()
		return flatOf(w, v.Mul(v))
	}, linalgTol),
}

func qrParts(x *tensor.Tensor) *tensor.Tensor {
	q, r := x.QR()
	return flatOf(q, r)
}

// TestGoldenLinalg checks the sorts, Fourier transforms and linear
// algebra, and their gradients, against PyTorch or NumPy (see
// tools/golden/gen.py).
func TestGoldenLinalg(t *testing.T) {
	cases, meta := goldentest.Load(t, "linalg")
	t.Logf("golden oracle: %s", meta["oracle"])
	for name := range linalgGoldenCases {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			gc, ok := linalgGoldenCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := goldentest.Inputs(t, c)
			if len(c.Grads) == 0 {
				goldentest.CompareTensor(t, c.Want, gc.fn(in...), gc.tol)
				return
			}
			grads := map[int]goldentest.Grader{}
			for i := range c.Grads {
				grads[i] = in[i]
			}
			tol := gc.tol
			if tol == exact {
				tol = loose
			}
			goldentest.CheckGradients(t, c, gc.fn(in...), grads, tol)
		})
	}
}
