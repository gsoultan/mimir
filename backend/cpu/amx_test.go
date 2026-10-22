package cpu

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestAMXGemm checks products on AMX against float64 sums of the same
// products of operands rounded to bfloat16, over shapes with ragged edges,
// strided and transposed operands, batches and summed products.
func TestAMXGemm(t *testing.T) {
	if amxTile == nil {
		t.Skip("no AMX")
	}
	defer SetMatMulBFloat16(MatMulBFloat16())
	SetMatMulBFloat16(true)
	be := New()
	r := rand.New(rand.NewPCG(1, 2))
	for _, c := range []struct{ m, n, k, items, sums int }{
		{64, 64, 64, 1, 1}, {100, 70, 333, 1, 1}, {512, 3 * 32, 20, 1, 1}, {33, 129, 64, 3, 1}, {48, 40, 200, 4, 2},
	} {
		for _, transB := range []bool{false, true} {
			a := make([]float32, c.items*c.m*c.k)
			b := make([]float32, c.items*c.k*c.n)
			for i := range a {
				a[i] = float32(r.NormFloat64())
			}
			for i := range b {
				b[i] = float32(r.NormFloat64())
			}
			offA, offB := make([]int, c.items), make([]int, c.items)
			for i := range c.items {
				offA[i], offB[i] = i*c.m*c.k, i*c.k*c.n
			}
			ao := operand{data: a, rs: c.k, cs: 1}
			bo := operand{data: b, rs: c.n, cs: 1}
			if transB {
				bo = operand{data: b, rs: 1, cs: c.k}
			}
			out := make([]float32, c.items/c.sums*c.m*c.n)
			ok, err := be.amxGemm(t.Context(), out, ao, offA, bo, offB, c.m, c.n, c.k, c.sums, nil)
			if !ok || err != nil {
				t.Fatalf("%+v: ok %v, %v", c, ok, err)
			}
			rb := func(x float32) float64 { return float64(bf16ToF32(f32ToBF16(x))) }
			for o := range c.items / c.sums {
				for i := range c.m {
					for j := range c.n {
						var want, mag float64
						for s := range c.sums {
							it := o*c.sums + s
							for p := range c.k {
								v := rb(a[offA[it]+i*ao.rs+p*ao.cs]) * rb(b[offB[it]+p*bo.rs+j*bo.cs])
								want += v
								mag += math.Abs(v)
							}
						}
						got := float64(out[(o*c.m+i)*c.n+j])
						if math.Abs(got-want) > 1e-5*mag+1e-6 {
							t.Fatalf("%+v trans %v: out[%d][%d][%d] = %v, want %v", c, transB, o, i, j, got, want)
						}
					}
				}
			}
		}
	}
}
