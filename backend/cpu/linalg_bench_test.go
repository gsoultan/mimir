package cpu

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// BenchmarkLinalg factorizes one n×n matrix: symmetric positive definite
// for Cholesky and Eigh, random otherwise; TriangularSolve solves for n
// right-hand sides with the lower triangle of the same matrix.
func BenchmarkLinalg(b *testing.B) {
	for _, op := range []backend.Op{backend.OpCholesky, backend.OpLU, backend.OpTriangularSolve, backend.OpQR, backend.OpSVD, backend.OpEigh} {
		for _, n := range []int{256, 512, 1024} {
			b.Run(fmt.Sprintf("%s/%d", op, n), func(b *testing.B) {
				rng := rand.New(rand.NewPCG(1, uint64(n)))
				x := randMatrix(rng, n*n)
				if op == backend.OpCholesky || op == backend.OpEigh || op == backend.OpTriangularSolve {
					x = spd(rng, 1, n)
				}
				view := backend.Contiguous([]int{n, n})
				k := linalgKernel(op, []int{n, n}, view)
				in := []backend.Buffer{f32buf(x...)}
				if op == backend.OpTriangularSolve {
					k.In = append(k.In, view)
					in = append(in, f32buf(randMatrix(rng, n*n)...))
				}
				size, _ := backend.NumElements(k.Shape)
				be := New()
				out, err := be.Alloc(backend.Float32, size)
				if err != nil {
					b.Fatal(err)
				}
				for b.Loop() {
					if err := be.Run(b.Context(), k, []backend.Buffer{out}, in); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkArgsort(b *testing.B) {
	for _, s := range []struct{ lines, n, k int }{
		{1024, 1024, 1024},
		{1, 1 << 20, 1 << 20},
		{1024, 32000, 10},
		{1024, 32000, 1024},
	} {
		b.Run(fmt.Sprintf("%dx%d/k%d", s.lines, s.n, s.k), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))
			in := f32buf(randMatrix(rng, s.lines*s.n)...)
			k := &backend.Kernel{Op: backend.OpArgsort, DType: backend.Int32, Shape: []int{s.lines, s.k},
				In: []backend.View{backend.Contiguous([]int{s.lines, s.n})}, Attrs: backend.Attrs{Axis: 1}}
			be := New()
			out, _ := be.Alloc(backend.Int32, s.lines*s.k)
			for b.Loop() {
				if err := be.Run(b.Context(), k, []backend.Buffer{out}, []backend.Buffer{in}); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(s.lines*s.n)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Melem/s")
		})
	}
}

func BenchmarkFFT(b *testing.B) {
	for _, s := range []struct{ lines, n int }{
		{256, 1024},
		{256, 1000},
		{1, 1 << 20},
	} {
		b.Run(fmt.Sprintf("%dx%d", s.lines, s.n), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))
			shape := []int{s.lines, s.n, 2}
			in := f32buf(randMatrix(rng, 2*s.lines*s.n)...)
			k := &backend.Kernel{Op: backend.OpFFT, DType: backend.Float32, Shape: shape,
				In: []backend.View{backend.Contiguous(shape)}, Attrs: backend.Attrs{Axis: 1}}
			be := New()
			out, _ := be.Alloc(backend.Float32, 2*s.lines*s.n)
			for b.Loop() {
				if err := be.Run(b.Context(), k, []backend.Buffer{out}, []backend.Buffer{in}); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(s.lines*s.n)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Melem/s")
		})
	}
}
