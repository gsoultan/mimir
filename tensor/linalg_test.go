package tensor_test

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// eyeOf returns the n×n identity.
func eyeOf(n int) *tensor.Tensor {
	v := make([]float32, n*n)
	for i := range n {
		v[i*n+i] = 1
	}
	return tensor.FromSlice(v, n, n)
}

// matmul64 returns the product of the m×k a and k×n b in float64.
func matmul64(a, b []float32, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := range m {
		for p := range k {
			for j := range n {
				out[i*n+j] += float64(a[i*k+p]) * float64(b[p*n+j])
			}
		}
	}
	return out
}

// closeTo fails unless got and want agree to within tol.
func closeTo(t *testing.T, what string, got []float32, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i]) - want[i]); !(d <= tol*max(1, math.Abs(want[i]))) {
			t.Fatalf("%s: element %d is %v, want %v", what, i, got[i], want[i])
		}
	}
}

func TestSolveAndInverse(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	n, k := 5, 3
	a := vals(r, 2*n*n, -1, 1)
	for b := range 2 {
		for i := range n {
			a[b*n*n+i*n+i] += 4
		}
	}
	bv := vals(r, n*k, -1, 1)
	at := tensor.FromSlice(a, 2, n, n)
	// b broadcasts over a's batch.
	x := data[float32](t, tensor.Solve(at, tensor.FromSlice(bv, n, k)))
	for b := range 2 {
		closeTo(t, "A·X", f32(matmul64(a[b*n*n:], x[b*n*k:], n, n, k)), widen(bv), 1e-5)
	}
	// A vector b.
	v := tensor.Solve(at, tensor.FromSlice(bv[:2*n], 2, n))
	if got := v.Shape(); !slices.Equal(got, []int{2, n}) {
		t.Fatalf("vector solve shape %v", got)
	}
	inv := data[float32](t, at.Inv())
	for b := range 2 {
		got := f32(matmul64(a[b*n*n:], inv[b*n*n:], n, n, n))
		closeTo(t, "A·A⁻¹", got, widen(data[float32](t, eyeOf(n))), 1e-5)
	}
	// A singular matrix fails when computed.
	sing := tensor.FromSlice([]float32{1, 2, 2, 4}, 2, 2)
	if _, err := sing.Inv().Data[float32](t.Context()); !errors.Is(err, tensor.ErrSingular) {
		t.Errorf("inverse of a singular matrix: %v, want ErrSingular", err)
	}
}

func TestDeterminants(t *testing.T) {
	for _, c := range []struct {
		a        []float32
		n        int
		det      float64
		sign     float32
		logAbsOK bool
	}{
		{[]float32{2, 0, 0, 3}, 2, 6, 1, true},
		{[]float32{0, 1, 1, 0}, 2, -1, -1, true},
		{[]float32{1, 2, 3, 4, 5, 6, 7, 8, 10}, 3, -3, -1, true},
		{[]float32{1, 2, 2, 4}, 2, 0, 0, false},
		{[]float32{-2}, 1, -2, -1, true},
	} {
		at := tensor.FromSlice(c.a, c.n, c.n)
		closeTo(t, "det", data[float32](t, at.Det()), []float64{c.det}, 1e-5)
		sign, logAbs := at.SlogDet()
		if s := data[float32](t, sign)[0]; s != c.sign {
			t.Errorf("sign of det of %v = %v, want %v", c.a, s, c.sign)
		}
		l := data[float32](t, logAbs)[0]
		if c.logAbsOK {
			closeTo(t, "logabsdet", []float32{l}, []float64{math.Log(math.Abs(c.det))}, 1e-5)
		} else if !math.IsInf(float64(l), -1) {
			t.Errorf("logabsdet of a singular matrix = %v, want -Inf", l)
		}
	}
}

func TestFactorizationsReconstruct(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	m, n := 6, 4
	a := vals(r, 2*m*n, -1, 1)
	at := tensor.FromSlice(a, 2, m, n)
	q, rr := at.QR()
	if !slices.Equal(q.Shape(), []int{2, m, n}) || !slices.Equal(rr.Shape(), []int{2, n, n}) {
		t.Fatalf("QR shapes %v %v", q.Shape(), rr.Shape())
	}
	closeTo(t, "Q·R", data[float32](t, q.MatMul(rr)), widen(a), 1e-5)
	u, s, vh := at.Transpose(-2, -1).SVD()
	if !slices.Equal(u.Shape(), []int{2, n, n}) || !slices.Equal(s.Shape(), []int{2, n}) || !slices.Equal(vh.Shape(), []int{2, n, m}) {
		t.Fatalf("SVD shapes %v %v %v", u.Shape(), s.Shape(), vh.Shape())
	}
	usv := u.Mul(s.Unsqueeze(-2)).MatMul(vh).Transpose(-2, -1)
	closeTo(t, "U·S·Vh", data[float32](t, usv), widen(a), 1e-5)

	sym := at.Transpose(-2, -1).MatMul(at).Add(eyeOf(n))
	spd := data[float32](t, sym)
	l := sym.Cholesky()
	closeTo(t, "L·Lᵀ", data[float32](t, l.MatMul(l.Transpose(-2, -1))), widen(spd), 1e-5)
	w, v := sym.Eigh()
	vwv := v.Mul(w.Unsqueeze(-2)).MatMul(v.Transpose(-2, -1))
	closeTo(t, "V·W·Vᵀ", data[float32](t, vwv), widen(spd), 1e-5)
	ws := data[float32](t, w)
	if !slices.IsSorted(ws[:n]) || !slices.IsSorted(ws[n:]) {
		t.Errorf("eigenvalues %v not ascending", ws)
	}

	x := data[float32](t, tensor.TriangularSolve(l, tensor.FromSlice(a[:n*2], n, 2), false))
	lv := data[float32](t, l)
	closeTo(t, "L·X", f32(matmul64(lv[:n*n], x[:n*2], n, n, 2)), widen(a[:n*2]), 1e-5)

	if _, err := tensor.FromSlice([]float32{1, 2, 2, 1}, 2, 2).Cholesky().Data[float32](t.Context()); !errors.Is(err, tensor.ErrSingular) {
		t.Errorf("Cholesky of an indefinite matrix: %v, want ErrSingular", err)
	}
}

func TestLinalgErrors(t *testing.T) {
	sq := tensor.Ones(tensor.Float32, 3, 3)
	for name, err := range map[string]error{
		"cholesky of a vector":    tensor.Ones(tensor.Float32, 3).Cholesky().Err(),
		"inverse of non-square":   tensor.Ones(tensor.Float32, 2, 3).Inv().Err(),
		"det of int":              tensor.Ones(tensor.Int32, 2, 2).Det().Err(),
		"solve of mismatched":     tensor.Solve(sq, tensor.Ones(tensor.Float32, 4, 2)).Err(),
		"solve of unbroadcast":    tensor.Solve(tensor.Ones(tensor.Float32, 2, 3, 3), tensor.Ones(tensor.Float32, 3, 3, 1)).Err(),
		"triangular of non-sq":    tensor.TriangularSolve(tensor.Ones(tensor.Float32, 2, 3), sq, false).Err(),
		"eigh of non-square":      first(tensor.Ones(tensor.Float32, 2, 3).Eigh()).Err(),
		"svd of a vector":         first3(tensor.Ones(tensor.Float32, 3).SVD()).Err(),
		"qr of int":               first(tensor.Ones(tensor.Int32, 3, 3).QR()).Err(),
		"fft of non-complex":      tensor.Ones(tensor.Float32, 3, 3).FFT(0).Err(),
		"fft of the parts axis":   tensor.Ones(tensor.Float32, 3, 2).FFT(1).Err(),
		"irfft of one frequency":  tensor.Ones(tensor.Float32, 1, 2).IRFFT(0, 0).Err(),
		"topk of too many":        first(sq.TopK(4, 0, true)).Err(),
		"sort along a bad axis":   first(sq.Sort(2, false)).Err(),
		"argsort of a bad tensor": tensor.Failed(tensor.ErrShape).Argsort(0, false).Err(),
	} {
		if err == nil {
			t.Errorf("%s: no error", name)
		} else if !errors.Is(err, tensor.ErrShape) && !errors.Is(err, tensor.ErrDType) {
			t.Errorf("%s: %v, want ErrShape or ErrDType", name, err)
		}
	}
}

func first(a, _ *tensor.Tensor) *tensor.Tensor     { return a }
func first3(a, _, _ *tensor.Tensor) *tensor.Tensor { return a }

func TestSortAndTopK(t *testing.T) {
	nan, negZero := float32(math.NaN()), float32(math.Copysign(0, -1))
	x := tensor.FromSlice([]float32{3, nan, 1, 2, 1, negZero, 0, 5}, 2, 4)
	v, i := x.Sort(1, false)
	gotV, gotI := data[float32](t, v), data[int32](t, i)
	wantI := []int32{2, 3, 0, 1, 1, 2, 0, 3}
	if !slices.Equal(gotI, wantI) {
		t.Errorf("ascending positions %v, want %v", gotI, wantI)
	}
	if gotV[0] != 1 || gotV[2] != 3 || !math.IsNaN(float64(gotV[3])) || gotV[7] != 5 {
		t.Errorf("ascending values %v", gotV)
	}
	_, i = x.Sort(1, true)
	if got, want := data[int32](t, i), []int32{1, 0, 3, 2, 3, 0, 1, 2}; !slices.Equal(got, want) {
		t.Errorf("descending positions %v, want %v", got, want)
	}
	if got, want := data[int32](t, x.Argsort(0, false)), []int32{1, 1, 1, 0, 0, 0, 0, 1}; !slices.Equal(got, want) {
		t.Errorf("argsort along axis 0 %v, want %v", got, want)
	}
	tv, ti := x.TopK(2, -1, true)
	if got, want := data[int32](t, ti), []int32{1, 0, 3, 0}; !slices.Equal(got, want) {
		t.Errorf("top 2 positions %v, want %v", got, want)
	}
	if got := data[float32](t, tv); !math.IsNaN(float64(got[0])) || got[1] != 3 || got[2] != 5 || got[3] != 1 {
		t.Errorf("top 2 values %v", got)
	}
	_, ti = x.TopK(1, 1, false)
	if got, want := data[int32](t, ti), []int32{2, 1}; !slices.Equal(got, want) {
		t.Errorf("bottom 1 positions %v, want %v", got, want)
	}
	// More than the kernels select of a longer axis.
	r := rand.New(rand.NewPCG(5, 6))
	long := vals(r, 3000, -1, 1)
	lv, _ := tensor.FromSlice(long, 3000).TopK(1500, 0, false)
	sorted := slices.Clone(long)
	slices.Sort(sorted)
	if got := data[float32](t, lv); !slices.Equal(got, sorted[:1500]) {
		t.Errorf("bottom 1500 of 3000 differ from the sort")
	}
	// A scalar sorts as one element.
	sv, si := tensor.Scalar[float32](2).Sort(0, false)
	if !slices.Equal(sv.Shape(), []int{}) || data[int32](t, si)[0] != 0 {
		t.Errorf("sort of a scalar: %v %v", sv.Shape(), si.Shape())
	}
}

// dft64 returns the discrete Fourier transform of x in float64.
func dft64(x []complex128, inverse bool) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	sign := -1.0
	if inverse {
		sign = 1
	}
	for k := range n {
		for j, v := range x {
			out[k] += v * cmplx.Rect(1, sign*2*math.Pi*float64(j*k%n)/float64(n))
		}
	}
	return out
}

func TestFourierTransforms(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, n := range []int{1, 4, 6, 17, 64, 100} {
		v := vals(r, 2*n, -1, 1)
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(float64(v[2*i]), float64(v[2*i+1]))
		}
		ct := tensor.FromSlice(v, n, 2)
		want := dft64(x, false)
		flat := func(c []complex128, scale float64) []float64 {
			out := make([]float64, 0, 2*len(c))
			for _, e := range c {
				out = append(out, real(e)*scale, imag(e)*scale)
			}
			return out
		}
		tol := 1e-5 * math.Sqrt(float64(n))
		closeTo(t, "fft", data[float32](t, ct.FFT(0)), flat(want, 1), tol)
		closeTo(t, "ifft", data[float32](t, ct.IFFT(-1)), flat(dft64(x, true), 1/float64(n)), tol)
		re := make([]complex128, n)
		for i := range re {
			re[i] = complex(float64(v[i]), 0)
		}
		rt := tensor.FromSlice(v[:n], n)
		closeTo(t, "rfft", data[float32](t, rt.RFFT(0)), flat(dft64(re, false)[:n/2+1], 1), tol)
		// irfft of rfft is the signal again, for its length.
		closeTo(t, "irfft", data[float32](t, rt.RFFT(0).IRFFT(0, n)), widen(v[:n]), tol)
	}
	// Along an inner axis of a batch, with the default length.
	x := tensor.Randn(1, 3, 5, 4)
	y := x.RFFT(1).IRFFT(1, 5)
	closeTo(t, "irfft(rfft) of a batch", data[float32](t, y), widen(data[float32](t, x)), 1e-5)
	if got := x.RFFT(1).IRFFT(1, 0).Shape(); !slices.Equal(got, []int{3, 4, 4}) {
		t.Errorf("irfft default length gives %v, want [3 4 4]", got)
	}
}

func widen(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func f32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

// linalgGradCases are gradient cases of the factorizations, solves,
// sorts and transforms, on well-conditioned inputs and through functions
// of eigenvectors and singular vectors that do not depend on their signs.
func linalgGradCases() []gradCase {
	r := rand.New(rand.NewPCG(9, 10))
	u := func(lo, hi float32, shape ...int) ([]float32, []int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		return vals(r, n, lo, hi), shape
	}
	var cases []gradCase
	add := func(name string, f func(x []*tensor.Tensor) *tensor.Tensor, ins ...[]any) {
		c := gradCase{name: name, f: f}
		for _, in := range ins {
			c.in, c.dims = append(c.in, in[0].([]float32)), append(c.dims, in[1].([]int))
		}
		cases = append(cases, c)
	}
	in := func(lo, hi float32, shape ...int) []any { v, s := u(lo, hi, shape...); return []any{v, s} }
	shift := func(x *tensor.Tensor, by float32) *tensor.Tensor {
		n := x.Shape()[x.Rank()-1]
		return x.Add(eyeOf(n).Mul(tensor.Scalar(by)))
	}
	flat := func(ts ...*tensor.Tensor) *tensor.Tensor {
		parts := make([]*tensor.Tensor, len(ts))
		for i, t := range ts {
			parts[i] = t.Reshape(-1)
		}
		return tensor.Cat(0, parts...)
	}
	add("cholesky", func(x []*tensor.Tensor) *tensor.Tensor {
		return shift(x[0].MatMul(x[0].Transpose(-2, -1)), 2).Cholesky()
	}, in(-1, 1, 2, 3, 3))
	add("triangular_solve_lower", func(x []*tensor.Tensor) *tensor.Tensor {
		return tensor.TriangularSolve(shift(x[0], 3), x[1], false)
	}, in(-1, 1, 3, 3), in(-1, 1, 3, 2))
	add("triangular_solve_upper_broadcast", func(x []*tensor.Tensor) *tensor.Tensor {
		return tensor.TriangularSolve(shift(x[0], 3), x[1], true)
	}, in(-1, 1, 3, 3), in(-1, 1, 2, 3, 2))
	add("solve", func(x []*tensor.Tensor) *tensor.Tensor { return tensor.Solve(shift(x[0], 3), x[1]) },
		in(-1, 1, 2, 3, 3), in(-1, 1, 3, 2))
	add("solve_vector", func(x []*tensor.Tensor) *tensor.Tensor { return tensor.Solve(shift(x[0], 3), x[1]) },
		in(-1, 1, 3, 3), in(-1, 1, 3))
	add("inv", func(x []*tensor.Tensor) *tensor.Tensor { return shift(x[0], 3).Inv() }, in(-1, 1, 3, 3))
	add("det", func(x []*tensor.Tensor) *tensor.Tensor { return shift(x[0], 1).Det() }, in(-1, 1, 2, 3, 3))
	add("slogdet", func(x []*tensor.Tensor) *tensor.Tensor {
		_, l := x[0].SlogDet()
		return l
	}, in(-2, 2, 3, 3))
	add("qr_tall", func(x []*tensor.Tensor) *tensor.Tensor { q, r := x[0].QR(); return flat(q, r) }, in(-1, 1, 4, 3))
	add("qr_wide", func(x []*tensor.Tensor) *tensor.Tensor { q, r := x[0].QR(); return flat(q, r) }, in(-1, 1, 3, 4))
	add("svd_tall", func(x []*tensor.Tensor) *tensor.Tensor {
		u, s, vh := x[0].SVD()
		return flat(s, u.Mul(u), vh.Mul(vh))
	}, in(-1, 1, 4, 3))
	add("svd_wide", func(x []*tensor.Tensor) *tensor.Tensor {
		u, s, vh := x[0].SVD()
		return flat(s, u.Mul(u), vh.Mul(vh))
	}, in(-1, 1, 2, 4))
	add("svd_values", func(x []*tensor.Tensor) *tensor.Tensor { _, s, _ := x[0].SVD(); return s }, in(-1, 1, 3, 3))
	add("eigh", func(x []*tensor.Tensor) *tensor.Tensor {
		w, v := x[0].Add(x[0].Transpose(-2, -1)).Eigh()
		return flat(w, v.Mul(v))
	}, in(-2, 2, 3, 3))

	distinct := func(shape ...int) []any {
		v, s := u(0, 1, shape...)
		for i, p := range r.Perm(len(v)) {
			v[i] = -2 + 0.2*float32(p)
		}
		return []any{v, s}
	}
	add("sort", func(x []*tensor.Tensor) *tensor.Tensor { v, _ := x[0].Sort(1, false); return v }, distinct(2, 5))
	add("topk", func(x []*tensor.Tensor) *tensor.Tensor { v, _ := x[0].TopK(2, 0, true); return v }, distinct(5, 3))
	add("fft", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].FFT(1) }, in(-1, 1, 2, 4, 2))
	add("fft_bluestein", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].FFT(0) }, in(-1, 1, 17, 2))
	add("ifft", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].IFFT(-1) }, in(-1, 1, 2, 5, 2))
	add("rfft", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].RFFT(-1) }, in(-1, 1, 2, 6))
	add("irfft_even", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].IRFFT(0, 6) }, in(-1, 1, 4, 2))
	add("irfft_odd_padded", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].IRFFT(-1, 9) }, in(-1, 1, 2, 3, 2))
	return cases
}

func TestLinalgGradientsMatchFiniteDifferences(t *testing.T) {
	for _, c := range linalgGradCases() {
		t.Run(c.name, func(t *testing.T) { checkFiniteDifferences(t, c) })
	}
}
