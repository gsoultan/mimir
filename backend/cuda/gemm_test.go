package cuda

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// cublas computes the call s as cuBLAS does, over the operand buffers
// ins: for each batch, the column-major result C = op(A)·op(B).
func cublas(s sgemm, ins [2][]float32) []float32 {
	c := make([]float32, s.batch*s.strideC)
	for b := range s.batch {
		for i := range s.m {
			for j := range s.n {
				var acc float32
				for p := range s.k {
					acc += ins[s.a.in][s.a.at(b, i, p)] * ins[s.b.in][s.b.at(b, p, j)]
				}
				c[b*s.strideC+j*s.ldc+i] = acc
			}
		}
	}
	return c
}

// TestSgemmComputesTheProduct checks, for every way the kernel tests read
// operands, that the cuBLAS call planned for a product computes what the
// product's own plan does.
func TestSgemmComputesTheProduct(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	f := backend.Float32
	planned := 0
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
				planned++
				xv := kerneltest.Floats(t, kerneltest.HostBuffer(t, rng, f, x.N, 0, 0))
				yv := kerneltest.Floats(t, kerneltest.HostBuffer(t, rng, f, y.N, 0, 0))
				want := kerneltest.GEMM(g, xv, yv)
				got := cublas(planSgemm(g), [2][]float32{xv, yv})
				if !slices.Equal(got, want) {
					t.Errorf("%s: cuBLAS computes %v, want %v", name, got, want)
				}
			}
		}
	}
	if planned < 20 {
		t.Errorf("only %d products planned", planned)
	}
}
