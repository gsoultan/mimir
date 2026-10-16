package cpu

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestDot(t *testing.T) {
	withRoutines(t, func(t *testing.T) {
		rng := rand.New(rand.NewPCG(5, 6))
		for _, n := range []int{0, 1, 15, 16, 17, 63, 64, 65, 200, 2048} {
			a, b := make([]float32, n), make([]float32, n)
			var want, scale float64
			for i := range a {
				a[i], b[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
				want += float64(a[i]) * float64(b[i])
				scale += math.Abs(float64(a[i]) * float64(b[i]))
			}
			if got := float64(dot(a, b)); math.Abs(got-want) > 1e-6*scale {
				t.Errorf("dot of %d: got %v, want %v", n, got, want)
			}
		}
	})
}

func TestGEMVMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, s := range []struct{ batch, m, n, k int }{
		{1, 1, 300, 16}, {1, 1, 5, 2048}, {1, 8, 70, 37}, {4, 2, 33, 100}, {2, 3, 1, 64},
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
