package kerneltest

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// Elementwise returns cases of elementwise kernels and fused programs.
func Elementwise(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(1, 2))
	f, i := backend.Float32, backend.Int32
	shape := []int{3, 5, 7}
	var cases []Case
	for name, x := range Views(shape) {
		for _, op := range []backend.Op{backend.OpNeg, backend.OpExp2, backend.OpSin, backend.OpContiguous} {
			cases = append(cases, Case{
				Name: fmt.Sprintf("%s/%s", op, name),
				K:    backend.Kernel{Op: op, DType: f, Shape: shape, In: []backend.View{x.V}},
				Ins:  []backend.Buffer{HostBuffer(t, rng, f, x.N, 0, 0)},
				Tol:  1e-6,
			})
		}
		for _, op := range []backend.Op{backend.OpAdd, backend.OpMul, backend.OpMax, backend.OpCmpLT, backend.OpCmpEQ, backend.OpMod} {
			for _, dt := range []backend.DType{f, i} {
				y := backend.Contiguous(shape)
				cases = append(cases, Case{
					Name: fmt.Sprintf("%s/%s/%s", op, dt, name),
					K:    backend.Kernel{Op: op, DType: dt, Shape: shape, In: []backend.View{x.V, y}},
					Ins:  []backend.Buffer{HostBuffer(t, rng, dt, x.N, -9, 9), HostBuffer(t, rng, dt, 105, 1, 5)},
				})
			}
		}
	}
	n := 105
	pos := HostBuffer(t, rng, f, n, 0, 0)
	cases = append(cases,
		Case{Name: "log2 sqrt recip", K: backend.Kernel{Op: backend.OpFused, DType: f, Shape: shape, In: []backend.View{backend.Contiguous(shape)},
			Attrs: backend.Attrs{Program: []backend.Instr{
				{Op: backend.OpLoad, DType: f, Args: []int{0}},
				{Op: backend.OpMul, DType: f, Args: []int{0, 0}},
				{Op: backend.OpLog2, DType: f, Args: []int{1}},
				{Op: backend.OpSqrt, DType: f, Args: []int{1}},
				{Op: backend.OpRecip, DType: f, Args: []int{3}},
				{Op: backend.OpAdd, DType: f, Args: []int{2, 4}},
			}}}, Ins: []backend.Buffer{pos}, Tol: 1e-6},
		Case{Name: "cast to int", K: backend.Kernel{Op: backend.OpCast, DType: i, Shape: shape, In: []backend.View{backend.Contiguous(shape)}},
			Ins: []backend.Buffer{pos}},
		Case{Name: "cast to float", K: backend.Kernel{Op: backend.OpCast, DType: f, Shape: shape, In: []backend.View{backend.Contiguous(shape)}},
			Ins: []backend.Buffer{HostBuffer(t, rng, i, n, -100, 100)}},
		Case{Name: "int mod by zero", K: backend.Kernel{Op: backend.OpMod, DType: i, Shape: shape, In: []backend.View{backend.Contiguous(shape), backend.Contiguous(shape)}},
			Ins: []backend.Buffer{HostBuffer(t, rng, i, n, -9, 9), HostBuffer(t, rng, i, n, -1, 2)}},
		Case{Name: "where", K: backend.Kernel{Op: backend.OpWhere, DType: f, Shape: shape, In: []backend.View{backend.Contiguous(shape), backend.Contiguous(shape), backend.Contiguous([]int{1, 1, 1}).Expand(shape)}},
			Ins: []backend.Buffer{HostBuffer(t, rng, i, n, 0, 2), pos, HostBuffer(t, rng, f, 1, 0, 0)}},
		Case{Name: "fused with stores and constants", K: backend.Kernel{Op: backend.OpFused, DType: f, Shape: shape,
			In: []backend.View{backend.Contiguous(shape), backend.Contiguous([]int{7, 5, 3}).Permute([]int{2, 1, 0})},
			Attrs: backend.Attrs{Program: []backend.Instr{
				{Op: backend.OpLoad, DType: f, Args: []int{0}},
				{Op: backend.OpLoad, DType: f, Args: []int{1}},
				{Op: backend.OpConst, DType: f, Value: 0.9},
				{Op: backend.OpMul, DType: f, Args: []int{0, 2}},
				{Op: backend.OpAdd, DType: f, Args: []int{3, 1}},
				{Op: backend.OpConst, DType: f, Value: math.Inf(-1)},
				{Op: backend.OpMax, DType: f, Args: []int{4, 5}},
				{Op: backend.OpCmpLT, DType: f, Args: []int{6, 2}},
				{Op: backend.OpWhere, DType: f, Args: []int{7, 3, 6}},
			}, Stores: []int{4, 6}}},
			Ins: []backend.Buffer{pos, HostBuffer(t, rng, f, n, 0, 0)}},
		Case{Name: "const", K: backend.Kernel{Op: backend.OpConst, DType: i, Shape: shape, Attrs: backend.Attrs{Value: -7}}},
		Case{Name: "empty", K: backend.Kernel{Op: backend.OpConst, DType: f, Shape: []int{0, 3}, Attrs: backend.Attrs{Value: 1}}},
	)
	return cases
}

// Random returns cases of random number kernels.
func Random(t *testing.T) []Case {
	var cases []Case
	for _, seed := range []uint64{0, 42, 1<<63 + 12345} {
		for _, n := range []int{1, 7, 1000} {
			cases = append(cases,
				Case{Name: fmt.Sprintf("uniform/%d/%d", seed, n), K: backend.Kernel{Op: backend.OpRandUniform, DType: backend.Float32, Shape: []int{n}, Attrs: backend.Attrs{Seed: seed}}},
				Case{Name: fmt.Sprintf("normal/%d/%d", seed, n), K: backend.Kernel{Op: backend.OpRandNormal, DType: backend.Float32, Shape: []int{n}, Attrs: backend.Attrs{Seed: seed}}, Tol: 1e-5},
			)
		}
	}
	return cases
}

// Reduce returns cases of reductions.
func Reduce(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(3, 4))
	var cases []Case
	for _, s := range []struct {
		shape []int
		axes  []int
	}{
		{[]int{4, 6, 5}, []int{1}},
		{[]int{4, 6, 5}, []int{0, 2}},
		{[]int{4, 6, 5}, []int{0, 1, 2}},
		{[]int{3, 700}, []int{1}},       // a threadgroup per output
		{[]int{2000, 3}, []int{0}},      // a threadgroup per output, strided
		{[]int{64, 3, 40}, []int{0, 2}}, // many outputs, many positions
	} {
		out := slices.Clone(s.shape)
		for _, a := range s.axes {
			out[a] = 1
		}
		for name, x := range Views(s.shape) {
			for _, op := range []backend.Op{backend.OpReduceSum, backend.OpReduceMax} {
				for _, dt := range []backend.DType{backend.Float32, backend.Int32} {
					cases = append(cases, Case{
						Name: fmt.Sprintf("%s/%s/%v%v/%s", op, dt, s.shape, s.axes, name),
						K:    backend.Kernel{Op: op, DType: dt, Shape: out, In: []backend.View{x.V}, Attrs: backend.Attrs{Axes: s.axes}},
						Ins:  []backend.Buffer{HostBuffer(t, rng, dt, x.N, -50, 50)},
						Tol:  1e-5,
					})
				}
			}
		}
	}
	return cases
}

// Movement returns cases of padding, gathers and scatters.
func Movement(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(5, 6))
	f, i := backend.Float32, backend.Int32
	var cases []Case
	for name, x := range Views([]int{3, 4, 5}) {
		cases = append(cases, Case{
			Name: "pad/" + name,
			K: backend.Kernel{Op: backend.OpPad, DType: f, Shape: []int{4, 4, 8}, In: []backend.View{x.V},
				Attrs: backend.Attrs{Pads: [][2]int{{1, 0}, {0, 0}, {2, 1}}, Value: -1.5}},
			Ins: []backend.Buffer{HostBuffer(t, rng, f, x.N, 0, 0)},
		})
		// Gather along each axis with an index of shape [3, 2, 5] that
		// fits the source outside the axis.
		for axis := range 3 {
			idx := []int{3, 4, 5}
			idx[(axis+1)%3] = 2
			n, _ := backend.NumElements(idx)
			cases = append(cases, Case{
				Name: fmt.Sprintf("gather/%d/%s", axis, name),
				K:    backend.Kernel{Op: backend.OpGather, DType: f, Shape: idx, In: []backend.View{x.V, backend.Contiguous(idx)}, Attrs: backend.Attrs{Axis: axis}},
				Ins:  []backend.Buffer{HostBuffer(t, rng, f, x.N, 0, 0), HostBuffer(t, rng, i, n, 0, []int{3, 4, 5}[axis])},
			})
			upd := slices.Clone(idx)
			upd[axis] = 7
			m, _ := backend.NumElements(upd)
			cases = append(cases, Case{
				Name: fmt.Sprintf("scatter add/%d/%s", axis, name),
				K: backend.Kernel{Op: backend.OpScatterAdd, DType: f, Shape: []int{3, 4, 5},
					In: []backend.View{x.V, backend.Contiguous(upd), backend.Contiguous(upd)}, Attrs: backend.Attrs{Axis: axis}},
				Ins: []backend.Buffer{HostBuffer(t, rng, f, x.N, 0, 0), HostBuffer(t, rng, i, m, 0, []int{3, 4, 5}[axis]), HostBuffer(t, rng, f, m, 0, 0)},
			})
		}
	}
	cases = append(cases,
		Case{Name: "gather out of range", K: backend.Kernel{Op: backend.OpGather, DType: i, Shape: []int{4}, In: []backend.View{backend.Contiguous([]int{4}), backend.Contiguous([]int{4})}},
			Ins: []backend.Buffer{HostBuffer(t, rng, i, 4, 0, 9), HostBuffer(t, rng, i, 4, 3, 5)}},
		Case{Name: "scatter add out of range", K: backend.Kernel{Op: backend.OpScatterAdd, DType: i, Shape: []int{4}, In: []backend.View{backend.Contiguous([]int{4}), backend.Contiguous([]int{6}), backend.Contiguous([]int{6})}},
			Ins: []backend.Buffer{HostBuffer(t, rng, i, 4, 0, 9), HostBuffer(t, rng, i, 6, -1, 1), HostBuffer(t, rng, i, 6, 0, 9)}},
	)
	return cases
}

// Window returns cases of unfolds and folds.
func Window(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(7, 8))
	f := backend.Float32
	var cases []Case
	for _, w := range []struct {
		name           string
		in             []int
		kernel, stride []int
		dilation       []int
		pads           [][2]int
		positions      []int
	}{
		{"1d", []int{2, 9}, []int{3}, []int{1}, []int{1}, [][2]int{{0, 0}}, []int{7}},
		{"2d strided padded", []int{2, 3, 7, 6}, []int{3, 2}, []int{2, 1}, []int{1, 2}, [][2]int{{1, 1}, {0, 2}}, []int{4, 6}},
		{"2d dilated", []int{1, 8, 8}, []int{2, 3}, []int{3, 2}, []int{3, 1}, [][2]int{{2, 0}, {1, 1}}, []int{4, 4}},
	} {
		attrs := backend.Attrs{Kernel: w.kernel, Stride: w.stride, Dilation: w.dilation, Pads: w.pads, Value: 0.5}
		s := len(w.kernel)
		lead := w.in[:len(w.in)-s]
		win := slices.Concat(lead, w.kernel, w.positions)
		for name, x := range Views(w.in) {
			cases = append(cases, Case{
				Name: "unfold/" + w.name + "/" + name,
				K:    backend.Kernel{Op: backend.OpUnfold, DType: f, Shape: win, In: []backend.View{x.V}, Attrs: attrs},
				Ins:  []backend.Buffer{HostBuffer(t, rng, f, x.N, 0, 0)},
			})
		}
		for name, x := range Views(win) {
			cases = append(cases, Case{
				Name: "fold/" + w.name + "/" + name,
				K:    backend.Kernel{Op: backend.OpFold, DType: f, Shape: w.in, In: []backend.View{x.V}, Attrs: attrs},
				Ins:  []backend.Buffer{HostBuffer(t, rng, f, x.N, 0, 0)},
			})
		}
	}
	return cases
}

// MatMul returns cases of matrix products, reading their operands in
// every way Views does, and of a product with an epilogue.
func MatMul(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(9, 10))
	f, i := backend.Float32, backend.Int32
	var cases []Case
	for _, s := range []struct {
		batch   []int
		m, k, n int
		axes    []int
	}{
		{nil, 5, 7, 3, nil},
		{[]int{2}, 4, 6, 5, nil},
		{[]int{3, 2}, 4, 6, 5, []int{0}},
		{[]int{3, 2}, 4, 6, 5, []int{0, 1}},
		{[]int{2}, 3, 0, 4, nil},
		{nil, 1, 9, 6, nil},
		{[]int{3}, 33, 70, 17, nil},
		{[]int{2}, 130, 70, 75, nil}, // several 64 × 64 tiles each way, the last ones partial
	} {
		xs := slices.Concat(s.batch, []int{s.m, s.k})
		ys := slices.Concat(s.batch, []int{s.k, s.n})
		out := slices.Concat(s.batch, []int{s.m, s.n})
		for _, a := range s.axes {
			out[a] = 1
		}
		for xn, x := range Views(xs) {
			for yn, y := range Views(ys) {
				// Every pair of views for float32, which a matrix library
				// may run, and a few for int32.
				dts := []backend.DType{f}
				if yn == "transposed" {
					dts = append(dts, i)
				}
				for _, dt := range dts {
					cases = append(cases, Case{
						Name: fmt.Sprintf("%s/%v x %v sum %v/%s/%s", dt, xs, ys, s.axes, xn, yn),
						K:    backend.Kernel{Op: backend.OpMatMul, DType: dt, Shape: out, In: []backend.View{x.V, y.V}, Attrs: backend.Attrs{Axes: s.axes}},
						Ins:  []backend.Buffer{HostBuffer(t, rng, dt, x.N, -5, 5), HostBuffer(t, rng, dt, y.N, -5, 5)},
						Tol:  1e-5,
					})
				}
			}
		}
	}
	// A bias and a ReLU as the epilogue of a product read as [2, 3, 4].
	bias := backend.Contiguous([]int{1, 1, 4}).Expand([]int{2, 3, 4})
	cases = append(cases, Case{
		Name: "epilogue",
		K: backend.Kernel{Op: backend.OpMatMul, DType: f, Shape: []int{6, 4},
			In: []backend.View{backend.Contiguous([]int{6, 5}), backend.Contiguous([]int{5, 4}), bias},
			Attrs: backend.Attrs{Program: []backend.Instr{
				{Op: backend.OpLoad, DType: f, Args: []int{0}},
				{Op: backend.OpLoad, DType: f, Args: []int{1}},
				{Op: backend.OpAdd, DType: f, Args: []int{0, 1}},
				{Op: backend.OpConst, DType: f},
				{Op: backend.OpMax, DType: f, Args: []int{2, 3}},
			}}},
		Ins: []backend.Buffer{HostBuffer(t, rng, f, 30, 0, 0), HostBuffer(t, rng, f, 20, 0, 0), HostBuffer(t, rng, f, 4, 0, 0)},
		Tol: 1e-5,
	})
	return cases
}
