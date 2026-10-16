package gen_test

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// TestGEMMReadsTheOperandsElements checks, for every way the kernel tests
// read operands, that a product planned for a matrix library reads each
// operand element from where the views put it.
func TestGEMMReadsTheOperandsElements(t *testing.T) {
	f := backend.Float32
	planned := map[string]bool{}
	for _, s := range []struct {
		batch   []int
		m, k, n int
	}{
		{nil, 5, 7, 3},
		{nil, 1, 7, 3},
		{nil, 5, 1, 3},
		{nil, 5, 7, 1},
		{[]int{2}, 4, 6, 5},
		{[]int{3, 2}, 4, 6, 5},
		{[]int{1, 3}, 4, 6, 5},
	} {
		xs := slices.Concat(s.batch, []int{s.m, s.k})
		ys := slices.Concat(s.batch, []int{s.k, s.n})
		out := slices.Concat(s.batch, []int{s.m, s.n})
		for xn, x := range kerneltest.Views(xs) {
			for yn, y := range kerneltest.Views(ys) {
				name := fmt.Sprintf("%v x %v/%s/%s", xs, ys, xn, yn)
				k := &backend.Kernel{Op: backend.OpMatMul, DType: f, Shape: out, In: []backend.View{x.V, y.V}}
				g, ok := gen.PlanGEMM(k, []int{x.N, y.N})
				if !ok {
					continue
				}
				planned[xn+"/"+yn] = true
				checkGEMM(t, name, g, s.batch, x.V, y.V, x.N, y.N)
			}
		}
	}
	for _, want := range []string{"contiguous/contiguous", "transposed/transposed", "padded/contiguous", "contiguous/broadcast", "broadcast/contiguous"} {
		if !planned[want] {
			t.Errorf("no product of %s operands planned", want)
		}
	}
	if planned["flipped/contiguous"] || planned["contiguous/flipped"] {
		t.Error("a reversed operand planned")
	}
}

func checkGEMM(t *testing.T, name string, g gen.GEMM, batch []int, vx, vy backend.View, nx, ny int) {
	t.Helper()
	nb, _ := backend.NumElements(batch)
	r := len(batch) + 2
	m, k, n := vx.Shape[r-2], vx.Shape[r-1], vy.Shape[r-1]
	if g.K != k || g.N != n || g.Batch*g.M != nb*m || g.ResultSize != g.M*n {
		t.Fatalf("%s: planned %d × [%d, %d, %d]", name, g.Batch, g.M, g.K, g.N)
	}
	for _, mat := range []struct {
		gen.Matrix
		len int
	}{{g.A, nx}, {g.B, ny}} {
		hi := mat.Offset + g.Batch*mat.Stride
		if mat.Stride == 0 {
			hi = mat.Offset + mat.Rows*mat.LD
		}
		if hi > mat.len || mat.Cols > mat.LD {
			t.Fatalf("%s: matrix %+v outside a buffer of %d", name, mat.Matrix, mat.len)
		}
	}
	addr := func(v backend.View, idx []int) int {
		a := v.Offset
		for d, i := range idx {
			a += i * v.Strides[d]
		}
		return a
	}
	for fb := range nb {
		coords := make([]int, len(batch))
		for d, rem := len(batch)-1, fb; d >= 0; d-- {
			coords[d], rem = rem%batch[d], rem/batch[d]
		}
		b, row := fb, 0
		if g.Batch == 1 {
			b, row = 0, fb*m
		}
		for i := range m {
			for p := range k {
				if got, want := g.A.At(b, row+i, p), addr(vx, append(slices.Clone(coords), i, p)); got != want {
					t.Fatalf("%s: left element %v at %d, want %d", name, append(coords, i, p), got, want)
				}
			}
		}
		for p := range k {
			for j := range n {
				if got, want := g.B.At(b, p, j), addr(vy, append(slices.Clone(coords), p, j)); got != want {
					t.Fatalf("%s: right element %v at %d, want %d", name, append(coords, p, j), got, want)
				}
			}
		}
	}
}

func TestGEMMLeavesOtherProductsToTheKernel(t *testing.T) {
	f := backend.Float32
	for name, k := range map[string]*backend.Kernel{
		"int32": {Op: backend.OpMatMul, DType: backend.Int32, Shape: []int{2, 2},
			In: []backend.View{backend.Contiguous([]int{2, 3}), backend.Contiguous([]int{3, 2})}},
		"batch sum": {Op: backend.OpMatMul, DType: f, Shape: []int{1, 2, 2}, Attrs: backend.Attrs{Axes: []int{0}},
			In: []backend.View{backend.Contiguous([]int{4, 2, 3}), backend.Contiguous([]int{4, 3, 2})}},
		"empty": {Op: backend.OpMatMul, DType: f, Shape: []int{2, 2},
			In: []backend.View{backend.Contiguous([]int{2, 0}), backend.Contiguous([]int{0, 2})}},
		"both broadcast": {Op: backend.OpMatMul, DType: f, Shape: []int{4, 2, 2},
			In: []backend.View{backend.Contiguous([]int{1, 2, 3}).Expand([]int{4, 2, 3}), backend.Contiguous([]int{1, 3, 2}).Expand([]int{4, 3, 2})}},
	} {
		lens := []int{1 << 10, 1 << 10}
		if _, ok := gen.PlanGEMM(k, lens); ok {
			t.Errorf("%s: planned", name)
		}
	}
}

// TestGEMMFoldsSumsIntoTheInnerDimension checks that a product summed over
// batch axes laid out outside the inner dimension, as a layer's weight
// gradient over a batch of sequences reads its operands, is planned as
// one product with a longer inner dimension that computes the sum.
func TestGEMMFoldsSumsIntoTheInnerDimension(t *testing.T) {
	f := backend.Float32
	rng := rand.New(rand.NewPCG(5, 6))
	for _, c := range []struct {
		name    string
		x, y    backend.View
		shape   []int
		axes    []int
		k, rows int // the folded inner dimension, and the rows of one product
	}{
		// [B, T, F]ᵀ · [B, T, O] summed over B.
		{"weight gradient", backend.Contiguous([]int{3, 5, 4}).Permute([]int{0, 2, 1}), backend.Contiguous([]int{3, 5, 2}),
			[]int{1, 4, 2}, []int{0}, 15, 4},
		// Two summed axes, and a batch axis kept.
		{"two sums", backend.Contiguous([]int{2, 3, 5, 4}).Permute([]int{0, 1, 3, 2}), backend.Contiguous([]int{2, 3, 5, 2}),
			[]int{1, 1, 4, 2}, []int{0, 1}, 30, 4},
		{"kept batch", backend.Contiguous([]int{2, 3, 5, 4}).Permute([]int{0, 1, 3, 2}), backend.Contiguous([]int{2, 3, 5, 2}),
			[]int{2, 1, 4, 2}, []int{1}, 15, 4},
	} {
		k := &backend.Kernel{Op: backend.OpMatMul, DType: f, Shape: c.shape, In: []backend.View{c.x, c.y}, Attrs: backend.Attrs{Axes: c.axes}}
		nx, _ := backend.NumElements(c.x.Shape)
		ny, _ := backend.NumElements(c.y.Shape)
		g, ok := gen.PlanGEMM(k, []int{nx, ny})
		if !ok {
			t.Errorf("%s: not planned", c.name)
			continue
		}
		if g.K != c.k || g.M != c.rows {
			t.Errorf("%s: planned %d × [%d, %d, %d], want an inner dimension of %d", c.name, g.Batch, g.M, g.K, g.N, c.k)
		}
		x, y := kerneltest.HostBuffer(t, rng, f, nx, -5, 5), kerneltest.HostBuffer(t, rng, f, ny, -5, 5)
		n, _ := backend.NumElements(c.shape)
		want := kerneltest.HostBuffer(t, nil, f, n, 0, 0)
		if err := kerneltest.Host.Run(t.Context(), k, []backend.Buffer{want}, []backend.Buffer{x, y}); err != nil {
			t.Fatal(err)
		}
		got := kerneltest.GEMM(g, kerneltest.Floats(t, x), kerneltest.Floats(t, y))
		for i, w := range kerneltest.Floats(t, want) {
			if d := got[i] - w; d > 1e-3 || d < -1e-3 {
				t.Fatalf("%s: element %d is %v, want %v", c.name, i, got[i], w)
			}
		}
	}
}
