package metal

import (
	"fmt"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// TestGEMMReadsTheOperandsElements checks, for every way the kernel tests
// read operands, that a product planned for Metal Performance Shaders
// reads each operand element from where the views put it.
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
		for xn, x := range views(xs) {
			for yn, y := range views(ys) {
				name := fmt.Sprintf("%v x %v/%s/%s", xs, ys, xn, yn)
				k := &backend.Kernel{Op: backend.OpMatMul, DType: f, Shape: out, In: []backend.View{x.v, y.v}}
				g, ok := planGEMM(k, []int{x.n, y.n})
				if !ok {
					continue
				}
				planned[xn+"/"+yn] = true
				checkGEMM(t, name, g, s.batch, x.v, y.v, x.n, y.n)
			}
		}
	}
	for _, want := range []string{"contiguous/contiguous", "transposed/transposed", "padded/contiguous", "contiguous/broadcast"} {
		if !planned[want] {
			t.Errorf("no product of %s operands ran through MPS", want)
		}
	}
	if planned["flipped/contiguous"] || planned["contiguous/flipped"] {
		t.Error("a reversed operand ran through MPS")
	}
}

func checkGEMM(t *testing.T, name string, g gemm, batch []int, vx, vy backend.View, nx, ny int) {
	t.Helper()
	nb, _ := backend.NumElements(batch)
	r := len(batch) + 2
	m, k, n := vx.Shape[r-2], vx.Shape[r-1], vy.Shape[r-1]
	if g.k != k || g.n != n || g.batch*g.m != nb*m || g.resultSize != g.m*n {
		t.Fatalf("%s: planned %d × [%d, %d, %d]", name, g.batch, g.m, g.k, g.n)
	}
	for _, mat := range []struct {
		matrix
		len int
	}{{g.a, nx}, {g.b, ny}} {
		if hi := mat.offset + g.batch*mat.stride; hi > mat.len || mat.cols > mat.ld {
			t.Fatalf("%s: matrix %+v outside a buffer of %d", name, mat.matrix, mat.len)
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
		if g.batch == 1 {
			b, row = 0, fb*m
		}
		for i := range m {
			for p := range k {
				if got, want := g.a.at(b, row+i, p), addr(vx, append(slices.Clone(coords), i, p)); got != want {
					t.Fatalf("%s: left element %v at %d, want %d", name, append(coords, i, p), got, want)
				}
			}
		}
		for p := range k {
			for j := range n {
				if got, want := g.b.at(b, p, j), addr(vy, append(slices.Clone(coords), p, j)); got != want {
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
		"broadcast left": {Op: backend.OpMatMul, DType: f, Shape: []int{4, 2, 2},
			In: []backend.View{backend.Contiguous([]int{1, 2, 3}).Expand([]int{4, 2, 3}), backend.Contiguous([]int{4, 3, 2})}},
	} {
		lens := []int{1 << 10, 1 << 10}
		if _, ok := planGEMM(k, lens); ok {
			t.Errorf("%s: planned for MPS", name)
		}
	}
}
