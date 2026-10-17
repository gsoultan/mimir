package kerneltest

import (
	"fmt"
	"math"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// specialTol is the error allowed on the special functions, which GPUs
// compute in float and the CPU in float64: a few float32 ulps of the
// result, or of the terms of a result near zero.
const specialTol = 4e-6

// specialFunctions returns cases of the special functions over their
// domains, at their poles and limits, in a fused program and on half
// elements.
func specialFunctions(t *testing.T) []Case {
	f := backend.Float32
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	grid := func(lo, hi float32, n int, extra ...float32) []float32 {
		v := make([]float32, 0, n+len(extra))
		for i := range n {
			v = append(v, lo+(hi-lo)*float32(i)/float32(n-1))
		}
		return append(v, extra...)
	}
	negZero := float32(math.Copysign(0, -1))
	gammas := grid(-6.07, 14.3, 97, 0, negZero, -1, -2, -5, 1, 2, 0.5, 1e-6, -1e-6, 30, 1e3, 1e30, inf, -inf, nan)
	domains := map[backend.Op][]float32{
		backend.OpLog1p:    grid(-0.999, 20, 90, -1, -2, 0, 1e-7, -1e-7, 1e-30, 1e10, inf, nan),
		backend.OpExpm1:    grid(-30, 30, 90, 0, 1e-7, -1e-7, 1e-30, 88, 89, 100, -inf, inf, nan),
		backend.OpErf:      grid(-6, 6, 97, 0, 0.4999, 0.5, -0.5, 1e-30, 10, inf, -inf, nan),
		backend.OpErfinv:   grid(-0.9999, 0.9999, 97, 0, 1, -1, 1.5, -2, 0.99999, -0.999999, 1e-30, nan),
		backend.OpLgamma:   gammas,
		backend.OpDigamma:  gammas,
		backend.OpTrigamma: gammas,
	}
	var cases []Case
	for op := backend.OpLog1p; op <= backend.OpTrigamma; op++ {
		x := domains[op]
		shape := []int{len(x)}
		cases = append(cases, Case{
			Name: fmt.Sprintf("%s/domain", op),
			K:    backend.Kernel{Op: op, DType: f, Shape: shape, In: []backend.View{backend.Contiguous(shape)}},
			Ins:  []backend.Buffer{floatBuffer(t, x)},
			Tol:  specialTol,
		})
		for _, dt := range halfTypes {
			cases = append(cases, Case{
				Name: fmt.Sprintf("%s/%s", op, dt),
				K:    backend.Kernel{Op: op, DType: dt, Shape: shape, In: []backend.View{backend.Contiguous(shape)}},
				Ins:  []backend.Buffer{cast(t, floatBuffer(t, x), dt)},
				Tol:  halfTol(dt),
			})
		}
	}
	// A program calling several special functions defines them once.
	x := grid(0.05, 9, 105)
	shape := []int{3, 5, 7}
	cases = append(cases, Case{Name: "fused special functions", K: backend.Kernel{Op: backend.OpFused, DType: f, Shape: shape,
		In: []backend.View{backend.Contiguous([]int{7, 5, 3}).Permute([]int{2, 1, 0})},
		Attrs: backend.Attrs{Program: []backend.Instr{
			{Op: backend.OpLoad, DType: f, Args: []int{0}},
			{Op: backend.OpLgamma, DType: f, Args: []int{0}},
			{Op: backend.OpDigamma, DType: f, Args: []int{0}},
			{Op: backend.OpErf, DType: f, Args: []int{0}},
			{Op: backend.OpMul, DType: f, Args: []int{2, 3}},
			{Op: backend.OpAdd, DType: f, Args: []int{1, 4}},
			{Op: backend.OpLog1p, DType: f, Args: []int{5}},
		}}},
		Ins: []backend.Buffer{floatBuffer(t, x)}, Tol: specialTol})
	return cases
}
