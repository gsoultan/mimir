package kerneltest

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// Scan returns cases of scans along short axes, which a thread scans
// line by line, and long ones, which a group scans block by block.
func Scan(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(11, 12))
	f, i := backend.Float32, backend.Int32
	combines := []backend.Combine{backend.CombineSum, backend.CombineMax, backend.CombineLogSumExp}
	// Float sums and log-sum-exps combine in another order than the
	// CPU's, in float rather than float64.
	tol := func(c backend.Combine, dt backend.DType) float64 {
		switch {
		case dt.Size() == 2:
			return halfTol(dt)
		case c == backend.CombineMax || dt == i:
			return 0
		}
		return 1e-4
	}
	var cases []Case
	add := func(name string, shape []int, x View, axis int, c backend.Combine, dt backend.DType) {
		cases = append(cases, Case{
			Name: fmt.Sprintf("%s/%s/%v/%d/%s", c, dt, shape, axis, name),
			K:    backend.Kernel{Op: backend.OpScan, DType: dt, Shape: shape, In: []backend.View{x.V}, Attrs: backend.Attrs{Axis: axis, Combine: c}},
			Ins:  []backend.Buffer{HostBuffer(t, rng, dt, x.N, -50, 50)},
			Tol:  tol(c, dt),
		})
	}
	// Every way Views reads a tensor for the short axes and for a long
	// axis of one block; one way for the others, whose groups take long
	// to emulate.
	for _, s := range []struct {
		shape []int
		axes  []int
		views []string // all if nil
	}{
		{[]int{3, 5, 7}, []int{0, 1, 2}, nil},
		{[]int{4, 64}, []int{1}, []string{"contiguous"}},      // the longest axis a thread scans
		{[]int{2, 700}, []int{1}, nil},                        // a group per line, one partial block
		{[]int{3000, 2}, []int{0}, []string{"contiguous"}},    // three blocks
		{[]int{2, 1100, 3}, []int{1}, []string{"transposed"}}, // lines apart
	} {
		for name, x := range Views(s.shape) {
			if s.views != nil && !slices.Contains(s.views, name) {
				continue
			}
			for _, axis := range s.axes {
				for _, c := range combines {
					add(name, s.shape, x, axis, c, f)
				}
				add(name, s.shape, x, axis, backend.CombineSum, i)
				add(name, s.shape, x, axis, backend.CombineMax, i)
			}
		}
	}
	// Half elements are combined in float and rounded once.
	for _, dt := range halfTypes {
		for _, s := range []struct {
			shape []int
			views []string
		}{
			{[]int{3, 5, 7}, []string{"contiguous", "transposed"}},
			{[]int{2, 700}, []string{"contiguous"}},
		} {
			for _, name := range s.views {
				for _, c := range combines {
					add(name, s.shape, Views(s.shape)[name], len(s.shape)-1, c, dt)
				}
			}
		}
	}
	cases = append(cases,
		Case{Name: "empty", K: backend.Kernel{Op: backend.OpScan, DType: f, Shape: []int{0, 3},
			In: []backend.View{backend.Contiguous([]int{0, 3})}, Attrs: backend.Attrs{Axis: 1, Combine: backend.CombineSum}},
			Ins: []backend.Buffer{HostBuffer(t, rng, f, 0, 0, 0)}},
	)
	return cases
}
