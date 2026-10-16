package cpu

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestDots4(t *testing.T) {
	withRoutines(t, func(t *testing.T) {
		rng := rand.New(rand.NewPCG(5, 6))
		for _, k := range []int{1, 15, 16, 17, 40, 63, 64, 65, 200, 2048} {
			const stride = 2100
			x, y := make([]float32, k), make([]float32, 3*stride+k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			for i := range y {
				y[i] = float32(rng.NormFloat64())
			}
			var d [4]float32
			dots4(&d, x, y, stride)
			for r, got := range d {
				var want, scale float64
				for i, v := range x {
					want += float64(v) * float64(y[r*stride+i])
					scale += math.Abs(float64(v) * float64(y[r*stride+i]))
				}
				if math.Abs(float64(got)-want) > 1e-6*scale {
					t.Errorf("row %d of %d: got %v, want %v", r, k, got, want)
				}
				// Alone, the row gets the same sum.
				var one [4]float32
				dots4(&one, x, y[r*stride:], 0)
				if one != [4]float32{got, got, got, got} {
					t.Errorf("row %d of %d alone: got %v, want %v", r, k, one, got)
				}
			}
		}
	})
}

func TestGEMVMatchesReference(t *testing.T) {
	withRoutines(t, testGEMV)
}

func testGEMV(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, s := range []struct{ batch, m, n, k int }{
		{1, 1, 300, 16}, {1, 1, 5, 2048}, {1, 8, 70, 37}, {4, 2, 33, 100}, {2, 3, 1, 64}, {1, 2, 4097, 24},
	} {
		// The left operand is row-major, the right the transpose of a
		// row-major weight, as a Linear layer reads it, shared by the
		// batch or not.
		for _, shared := range []bool{true, false} {
			a := layout(rng, "row-major", s.batch*s.m, s.k)
			nb := s.n
			if !shared {
				nb *= s.batch
			}
			b := layout(rng, "column-major", s.k, nb)
			offA, offB := make([]int, s.batch), make([]int, s.batch)
			for i := range s.batch {
				offA[i] = a.off + i*s.m*s.k
				if !shared {
					offB[i] = i * s.n * s.k
				}
			}
			var first []float32
			for _, workers := range []int{1, 4} {
				be := New()
				be.workers = workers
				got := make([]float32, s.batch*s.m*s.n)
				ok, err := be.sgemv(t.Context(), got, a, offA, b, offB, s.m, s.n, s.k, 1, nil)
				if !ok || err != nil {
					t.Fatalf("%+v: computed %v, %v", s, ok, err)
				}
				for o := range s.batch {
					x, y := a, b
					x.off, y.off = offA[o], offB[o]
					want, scale := refGEMM(x, y, s.m, s.n, s.k)
					for i, w := range want {
						if g := float64(got[o*s.m*s.n+i]); math.Abs(g-w) > 1e-6*scale[i] {
							t.Fatalf("%+v shared %v: element %d of item %d = %v, want %v", s, shared, i, o, g, w)
						}
					}
				}
				if first == nil {
					first = got
				}
				for i := range got {
					if got[i] != first[i] {
						t.Fatalf("%+v: element %d differs between 1 and %d workers", s, i, workers)
					}
				}
			}
		}
	}
	// Products of more rows, or read otherwise, are left to the GEMM.
	be := New()
	a, b := layout(rng, "row-major", 9, 32), layout(rng, "column-major", 32, 4)
	if ok, _ := be.sgemv(t.Context(), make([]float32, 36), a, []int{a.off}, b, []int{0}, 9, 4, 32, 1, nil); ok {
		t.Error("9 rows computed as dot products")
	}
	b = layout(rng, "row-major", 32, 4)
	if ok, _ := be.sgemv(t.Context(), make([]float32, 4), a, []int{a.off}, b, []int{b.off}, 1, 4, 32, 1, nil); ok {
		t.Error("a row-major right operand computed as dot products")
	}
}
