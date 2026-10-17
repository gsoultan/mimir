package cpu

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

func TestFFTMatchesDefinition(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, n := range []int{1, 2, 3, 5, 6, 7, 8, 12, 16, 17, 31, 32, 100, 127, 128, 1000, 1024} {
		for _, inverse := range []bool{false, true} {
			t.Run(fmt.Sprintf("%d inverse %v", n, inverse), func(t *testing.T) {
				// Lines along the second axis of [3, n, 2, 2], read
				// transposed from [2, n, 3, 2].
				shape := []int{3, n, 2, 2}
				v := backend.Contiguous([]int{2, n, 3, 2}).Permute([]int{2, 1, 0, 3})
				x := make([]float32, 12*n)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				k := &backend.Kernel{Op: backend.OpFFT, DType: backend.Float32, Shape: shape, In: []backend.View{v}, Attrs: backend.Attrs{Axis: 1, Inverse: inverse}}
				got := runOn[float32](t, 2, k, f32buf(x...))
				sign := -1.0
				if inverse {
					sign = 1
				}
				tol := 4e-6 * math.Sqrt(float64(n))
				for a := range 3 {
					for c := range 2 {
						for f := range n {
							var want complex128
							for j := range n {
								re := float64(x[offsetOf(v, []int{a, j, c, 0})])
								im := float64(x[offsetOf(v, []int{a, j, c, 1})])
								want += complex(re, im) * cmplx.Rect(1, sign*2*math.Pi*float64(j*f%n)/float64(n))
							}
							o := ((a*n+f)*2 + c) * 2
							if g := complex(float64(got[o]), float64(got[o+1])); cmplx.Abs(g-want) > tol {
								t.Fatalf("line (%d, %d) element %d: got %v, want %v", a, c, f, g, want)
							}
						}
					}
				}
			})
		}
	}
}

func TestFFTRoundTrips(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{4096, 3000, 4097} {
		x := make([]float32, 2*n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		shape := []int{n, 2}
		k := &backend.Kernel{Op: backend.OpFFT, DType: backend.Float32, Shape: shape, In: []backend.View{backend.Contiguous(shape)}}
		y := runOn[float32](t, 1, k, f32buf(x...))
		k.Attrs.Inverse = true
		z := runOn[float32](t, 1, k, f32buf(y...))
		for i := range z {
			if d := math.Abs(float64(z[i])/float64(n) - float64(x[i])); d > 1e-5 {
				t.Fatalf("n %d element %d: round trip %v, want %v", n, i, z[i]/float32(n), x[i])
			}
		}
	}
}

// TestFFTOfALongLineSharedByTheWorkers checks that workers sharing a line
// compute what one computes alone.
func TestFFTOfALongLineSharedByTheWorkers(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, n := range []int{1 << 16, 3 * 5 << 13, 65537} {
		x := make([]float32, 2*n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		shape := []int{n, 2}
		k := &backend.Kernel{Op: backend.OpFFT, DType: backend.Float32, Shape: shape, In: []backend.View{backend.Contiguous(shape)}}
		one := runOn[float32](t, 1, k, f32buf(x...))
		four := runOn[float32](t, 4, k, f32buf(x...))
		if !slices.Equal(one, four) {
			t.Fatalf("n %d: four workers differ from one", n)
		}
	}
}
