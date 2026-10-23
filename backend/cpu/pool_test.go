package cpu

import (
	"context"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// TestPool2x2Vector checks the vector routines of pool2x2 against its Go
// loops, over rows of every width up to a few vectors, with ties, NaN
// and -0 among the values.
func TestPool2x2Vector(t *testing.T) {
	if poolMax2x2 == nil {
		t.Skip("no vector routines on this CPU")
	}
	rng := rand.New(rand.NewPCG(3, 4))
	be := New()
	for _, w := range []int{2, 4, 14, 16, 28, 30, 32, 34, 62, 64, 66, 100} {
		for _, planes := range []int{1, 3} {
			h := 6
			x := make([]float32, planes*h*w)
			for i := range x {
				switch r := rng.IntN(10); {
				case r == 0:
					x[i] = float32(math.NaN())
				case r == 1:
					x[i] = float32(math.Copysign(0, -1))
				case r < 5:
					x[i] = float32(rng.IntN(3)) // ties
				default:
					x[i] = rng.Float32()*4 - 2
				}
			}
			g := make([]float32, planes*h*w/4)
			for i := range g {
				g[i] = rng.Float32()
			}
			shape := []int{planes, h, w}
			k := &backend.Kernel{Op: backend.OpMaxPool, DType: backend.Float32, Shape: []int{planes, h / 2, w / 2},
				In:    []backend.View{{Shape: shape, Strides: backend.ContiguousStrides(shape)}},
				Attrs: backend.Attrs{Kernel: []int{2, 2}, Stride: []int{2, 2}, Dilation: []int{1, 1}, Pads: [][2]int{{}, {}}}}
			kg := *k
			kg.Op, kg.Shape = backend.OpMaxPoolGrad, shape
			kg.In = append(slices.Clone(k.In), backend.View{Shape: k.Shape, Strides: backend.ContiguousStrides(k.Shape)})
			run := func(vector bool) (out, dx []float32) {
				m, gr := poolMax2x2, poolGrad2x2
				if !vector {
					poolMax2x2, poolGrad2x2 = nil, nil
				}
				defer func() { poolMax2x2, poolGrad2x2 = m, gr }()
				out, dx = make([]float32, len(g)), make([]float32, len(x))
				if err := pool2x2(context.Background(), be, k, out, x, nil); err != nil {
					t.Fatal(err)
				}
				if err := pool2x2(context.Background(), be, &kg, dx, x, g); err != nil {
					t.Fatal(err)
				}
				return out, dx
			}
			wantOut, wantDX := run(false)
			gotOut, gotDX := run(true)
			same := func(a, b float32) bool { return a == b || a != a && b != b }
			for i := range wantOut {
				if !same(wantOut[i], gotOut[i]) {
					t.Fatalf("w=%d planes=%d out[%d]: vector %v, loops %v", w, planes, i, gotOut[i], wantOut[i])
				}
			}
			for i := range wantDX {
				if !same(wantDX[i], gotDX[i]) {
					t.Fatalf("w=%d planes=%d dx[%d]: vector %v, loops %v (x=%v)", w, planes, i, gotDX[i], wantDX[i], x[i])
				}
			}
		}
	}
}
