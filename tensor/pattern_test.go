package tensor_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkPatternMatMul combines rows of 64 features over a random graph
// of 100,000 nodes and a million edges, as a graph convolution does.
func BenchmarkPatternMatMul(b *testing.B) {
	const nodes, edges, features = 100_000, 1_000_000, 64
	r := rand.New(rand.NewPCG(1, 2))
	row, col := randomEntries(r, nodes, nodes, edges)
	p := mustPattern(b, nodes, nodes, row, col)
	x := realized(b, 1, nodes, features)
	b.Run("sum", func(b *testing.B) {
		b.SetBytes(edges * features * 4)
		run(b, func() *tensor.Tensor { return p.MatMul(x) })
	})
	b.Run("max", func(b *testing.B) {
		b.SetBytes(edges * features * 4)
		run(b, func() *tensor.Tensor { return p.Max(x) })
	})
}

// randomEntries returns n random entries of a rows × cols pattern, with
// repeats.
func randomEntries(r *rand.Rand, rows, cols, n int) (row, col []int32) {
	for range n {
		row = append(row, int32(r.IntN(rows)))
		col = append(col, int32(r.IntN(cols)))
	}
	return row, col
}

func mustPattern(t testing.TB, rows, cols int, row, col []int32) *tensor.Pattern {
	t.Helper()
	p, err := tensor.NewPattern(rows, cols, row, col)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// denseOf returns the rows × cols matrix that counts the entries.
func denseOf(rows, cols int, row, col []int32) []float32 {
	a := make([]float32, rows*cols)
	for k := range row {
		a[int(row[k])*cols+int(col[k])]++
	}
	return a
}

func TestPatternProductMatchesDense(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	const rows, cols, d = 7, 9, 5
	row, col := randomEntries(r, rows, cols, 30)
	p := mustPattern(t, rows, cols, row, col)
	x := tensor.FromSlice(vals(r, cols*d, -1, 1), cols, d)
	a := tensor.FromSlice(denseOf(rows, cols, row, col), rows, cols)
	closeRows(t, data[float32](t, p.MatMul(x)), data[float32](t, a.MatMul(x)), 1e-5)
	// The transpose by the dense transpose.
	y := tensor.FromSlice(vals(r, rows*d, -1, 1), rows, d)
	closeRows(t, data[float32](t, p.T().MatMul(y)), data[float32](t, a.Transpose(0, 1).MatMul(y)), 1e-5)
	// Rows of more than one dimension, and the mean.
	z := tensor.FromSlice(vals(r, cols*6, -1, 1), cols, 2, 3)
	got := data[float32](t, p.Mean(z))
	want := data[float32](t, a.MatMul(z.Reshape(cols, 6)))
	deg := data[float32](t, p.Degrees())
	for i := range rows {
		for j := range 6 {
			want[i*6+j] /= max(deg[i], 1)
		}
	}
	closeRows(t, got, want, 1e-5)
	if rr, cc := p.Shape(); rr != rows || cc != cols || p.Entries() != 30 {
		t.Errorf("shape %d × %d with %d entries", rr, cc, p.Entries())
	}
}

func closeRows(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol*math.Max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("element %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestPatternMaxMatchesLoop(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	const rows, cols, d = 6, 8, 3
	row, col := randomEntries(r, rows, cols, 20)
	row = append(row, 2, 2) // a repeated entry
	col = append(col, 7, 7)
	p := mustPattern(t, rows, cols, row, col)
	xv := vals(r, cols*d, -1, 1)
	got := data[float32](t, p.Max(tensor.FromSlice(xv, cols, d)))
	want := make([]float32, rows*d)
	seen := make([]bool, rows)
	for k := range row {
		i, j := int(row[k]), int(col[k])
		for e := range d {
			v := xv[j*d+e]
			if !seen[i] || v > want[i*d+e] {
				want[i*d+e] = v
			}
		}
		seen[i] = true
	}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestPatternMaxSharesTiedGradients(t *testing.T) {
	// Row 0 takes the maximum of x[0], x[1] and x[2], of which two tie.
	p := mustPattern(t, 2, 3, []int32{0, 0, 0, 1}, []int32{0, 1, 2, 2})
	x := tensor.FromSlice([]float32{4, 4, 1}, 3, 1).RequiresGrad()
	if err := p.Max(x).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := data[float32](t, x.Grad()); !slices.Equal(got, []float32{0.5, 0.5, 1}) {
		t.Errorf("gradient %v, want [0.5 0.5 1]", got)
	}
}

func TestPatternHalfAndIntRows(t *testing.T) {
	p := mustPattern(t, 2, 3, []int32{0, 0, 1}, []int32{0, 2, 1})
	x := tensor.FromSlice([]int32{1, 2, 3, 4, 5, 6}, 3, 2)
	if got := data[int32](t, p.MatMul(x)); !slices.Equal(got, []int32{6, 8, 3, 4}) {
		t.Errorf("int sums %v", got)
	}
	if got := data[int32](t, p.Max(x)); !slices.Equal(got, []int32{5, 6, 3, 4}) {
		t.Errorf("int maxima %v", got)
	}
	h := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 3, 2).Cast(tensor.BFloat16)
	got := p.MatMul(h)
	if got.DType() != tensor.BFloat16 {
		t.Fatalf("dtype %s", got.DType())
	}
	if v := data[float32](t, got.Cast(tensor.Float32)); !slices.Equal(v, []float32{6, 8, 3, 4}) {
		t.Errorf("bfloat16 sums %v", v)
	}
}

func TestPatternEmptyRowsAreZero(t *testing.T) {
	p := mustPattern(t, 3, 2, []int32{1}, []int32{1})
	x := tensor.FromSlice([]float32{-1, -2, -3, -4}, 2, 2)
	if got := data[float32](t, p.Max(x)); !slices.Equal(got, []float32{0, 0, -3, -4, 0, 0}) {
		t.Errorf("maxima %v", got)
	}
	empty := mustPattern(t, 2, 4, nil, nil)
	if got := data[float32](t, empty.MatMul(tensor.Ones(tensor.Float32, 4, 3))); !slices.Equal(got, make([]float32, 6)) {
		t.Errorf("sums %v", got)
	}
}

func TestPatternErrors(t *testing.T) {
	for _, c := range []struct {
		rows, cols int
		row, col   []int32
		want       error
	}{
		{2, 2, []int32{0, 2}, []int32{0, 1}, tensor.ErrIndex},
		{2, 2, []int32{0, 1}, []int32{-1, 1}, tensor.ErrIndex},
		{2, 2, []int32{0}, []int32{0, 1}, tensor.ErrShape},
		{-1, 2, nil, nil, tensor.ErrShape},
	} {
		if _, err := tensor.NewPattern(c.rows, c.cols, c.row, c.col); !errors.Is(err, c.want) {
			t.Errorf("NewPattern(%d, %d, %v, %v): %v, want %v", c.rows, c.cols, c.row, c.col, err, c.want)
		}
	}
	p := mustPattern(t, 2, 3, []int32{0}, []int32{2})
	if err := p.MatMul(tensor.Ones(tensor.Float32, 2, 4)).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("product by the wrong rows: %v", err)
	}
	if err := p.Max(tensor.Scalar[float32](1)).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("maximum of a scalar: %v", err)
	}
	var nilPattern *tensor.Pattern
	if err := nilPattern.MatMul(tensor.Ones(tensor.Float32, 3, 4)).Err(); err == nil {
		t.Error("product by a nil pattern succeeded")
	}
}

func TestPatternSecondDerivativesAndVMap(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	const rows, cols, d = 4, 5, 3
	row, col := randomEntries(r, rows, cols, 12)
	p := mustPattern(t, rows, cols, row, col)
	xv := vals(r, cols*d, -1, 1)
	x := tensor.FromSlice(xv, cols, d)
	f := func(x ...*tensor.Tensor) *tensor.Tensor {
		h := p.MatMul(x[0].Mul(x[0]))
		return p.MatMul(p.T().Max(h).Tanh()).Sum()
	}
	// The Hessian-vector product by JVP of Grad matches differences of
	// gradients.
	v := tensor.FromSlice(vals(r, cols*d, -1, 1), cols, d)
	_, hv := tensor.JVP(func(x ...*tensor.Tensor) []*tensor.Tensor { return tensor.Grad(f)(x...) }, []*tensor.Tensor{x}, []*tensor.Tensor{v})
	got := data[float32](t, hv[0])
	const h = 1e-2
	plus := data[float32](t, tensor.Grad(f)(x.Add(v.Mul(tensor.Scalar[float32](h))))[0])
	minus := data[float32](t, tensor.Grad(f)(x.Sub(v.Mul(tensor.Scalar[float32](h))))[0])
	for i := range got {
		want := (plus[i] - minus[i]) / (2 * h)
		if math.Abs(float64(got[i]-want)) > 1e-2+1e-2*math.Abs(float64(want)) {
			t.Fatalf("element %d of the Hessian-vector product: %v, differences %v", i, got[i], want)
		}
	}
	// VMap over a batch of features matches a loop.
	batch := tensor.FromSlice(vals(r, 2*cols*d, -1, 1), 2, cols, d)
	mapped := tensor.VMap(func(x ...*tensor.Tensor) []*tensor.Tensor {
		return []*tensor.Tensor{p.Max(p.T().MatMul(p.MatMul(x[0])))}
	})(batch)[0]
	for b := range 2 {
		one := batch.Shrink([][2]int{{b, b + 1}, {0, cols}, {0, d}}).Reshape(cols, d)
		closeRows(t, data[float32](t, mapped.Shrink([][2]int{{b, b + 1}, {0, rows}, {0, d}}).Reshape(rows, d)),
			data[float32](t, p.Max(p.T().MatMul(p.MatMul(one)))), 1e-6)
	}
}
