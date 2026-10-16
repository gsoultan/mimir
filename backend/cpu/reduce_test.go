package cpu

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// refReduce reduces x, read through v, over axes with combine, starting
// every output element from its first reduced element.
func refReduce[T elem](x []T, v backend.View, axes []int, combine func(a, b T) T) []T {
	shape := slices.Clone(v.Shape)
	for _, a := range axes {
		shape[a] = 1
	}
	n, _ := backend.NumElements(shape)
	out := make([]T, n)
	seen := make([]bool, n)
	strides := backend.ContiguousStrides(shape)
	eachIndex(v.Shape, func(idx []int) {
		o := 0
		for d, i := range idx {
			if !slices.Contains(axes, d) {
				o += i * strides[d]
			}
		}
		x := x[offsetOf(v, idx)]
		if seen[o] {
			out[o] = combine(out[o], x)
		} else {
			out[o], seen[o] = x, true
		}
	})
	return out
}

// subsets returns every non-empty subset of the dimensions of a rank-r
// tensor.
func subsets(r int) [][]int {
	var all [][]int
	for mask := 1; mask < 1<<r; mask++ {
		var axes []int
		for d := range r {
			if mask&(1<<d) != 0 {
				axes = append(axes, d)
			}
		}
		all = append(all, axes)
	}
	return all
}

func TestReductionsMatchReference(t *testing.T) {
	shapes := [][]int{
		{7}, {3, 5}, {2, 3, 4}, {2, 1, 3, 5},
		{3, 2500},    // rows split into chunks
		{4, 3, 1100}, // a chunk shorter than the rest
		{1100, 3},    // short reduced runs, in rows split into chunks
	}
	for _, shape := range shapes {
		for _, sv := range stridedViews(shape) {
			xf := make([]float32, sv.size)
			xi := make([]int32, sv.size)
			for i := range xf {
				// Small integers, so that sums are exact in any order.
				xf[i] = float32((i*7)%23 - 11)
				xi[i] = int32((i*5)%19 - 9)
			}
			for _, axes := range subsets(len(shape)) {
				out := slices.Clone(shape)
				for _, a := range axes {
					out[a] = 1
				}
				for _, workers := range []int{1, 3} {
					name := fmt.Sprintf("%v %s axes %v workers %d", shape, sv.name, axes, workers)
					k := &backend.Kernel{Shape: out, In: []backend.View{sv.view}, Attrs: backend.Attrs{Axes: axes}}

					k.Op, k.DType = backend.OpReduceSum, backend.Float32
					got := runOn[float32](t, workers, k, f32buf(xf...))
					if want := refReduce(xf, sv.view, axes, func(a, b float32) float32 { return a + b }); !slices.Equal(got, want) {
						t.Fatalf("%s: float32 sum %v, want %v", name, got, want)
					}
					k.DType = backend.Int32
					goti := runOn[int32](t, workers, k, i32buf(xi...))
					if want := refReduce(xi, sv.view, axes, func(a, b int32) int32 { return a + b }); !slices.Equal(goti, want) {
						t.Fatalf("%s: int32 sum %v, want %v", name, goti, want)
					}
					k.Op, k.DType = backend.OpReduceMax, backend.Float32
					got = runOn[float32](t, workers, k, f32buf(xf...))
					if want := refReduce(xf, sv.view, axes, maxOf[float32]); !slices.Equal(got, want) {
						t.Fatalf("%s: max %v, want %v", name, got, want)
					}
				}
			}
		}
	}
}

func TestColumnSumsDoNotDependOnWorkers(t *testing.T) {
	// Few rows of sums over many positions, such as a bias gradient, sum
	// blocks of positions in parallel.
	x := make([]float32, 1000*5)
	for i := range x {
		x[i] = float32(math.Sin(float64(i))) * 1e3
	}
	k := &backend.Kernel{
		Op: backend.OpReduceSum, DType: backend.Float32, Shape: []int{1, 5},
		In: []backend.View{backend.Contiguous([]int{1000, 5})}, Attrs: backend.Attrs{Axes: []int{0}},
	}
	first := runOn[float32](t, 1, k, f32buf(x...))
	for j, got := range first {
		var want float64
		for i := range 1000 {
			want += float64(x[i*5+j])
		}
		if math.Abs(float64(got)-want) > 1e-6*math.Abs(want)+1e-3 {
			t.Errorf("column %d: sum %v, want %v", j, got, want)
		}
	}
	for _, workers := range []int{2, 4} {
		if got := runOn[float32](t, workers, k, f32buf(x...)); !slices.Equal(got, first) {
			t.Errorf("%d workers: sums %v, want %v as with one", workers, got, first)
		}
	}
}

func TestReduceSumAccumulatesFloat32InFloat64(t *testing.T) {
	// 2^24 + 1 is not a float32, so a float32 running sum drops every 1.
	x := []float32{1 << 24, 1, 1, 1, 1, 1, 1, 1, 1}
	for _, axis := range []int{0, 1} {
		// Summing along axis 0 of the transposed [9, 2] view combines rows.
		shape, v := []int{2, 9}, backend.Contiguous([]int{2, 9})
		buf := slices.Concat(x, x)
		if axis == 0 {
			shape, v = []int{9, 2}, backend.Contiguous([]int{2, 9}).Permute([]int{1, 0})
		}
		out := slices.Clone(shape)
		out[axis] = 1
		k := &backend.Kernel{
			Op: backend.OpReduceSum, DType: backend.Float32, Shape: out,
			In: []backend.View{v}, Attrs: backend.Attrs{Axes: []int{axis}},
		}
		got := runOn[float32](t, 1, k, f32buf(buf...))
		if want := float32(1<<24 + 8); got[0] != want || got[1] != want {
			t.Fatalf("axis %d: got %v, want %v twice", axis, got, want)
		}
	}
}

func TestReduceMaxPropagatesNaNInRows(t *testing.T) {
	// Reducing [3, 4] over axis 0 combines whole rows; NaN must win
	// wherever it appears, first row or not.
	nan := float32(math.NaN())
	k := &backend.Kernel{
		Op: backend.OpReduceMax, DType: backend.Float32, Shape: []int{1, 4},
		In: []backend.View{backend.Contiguous([]int{3, 4})}, Attrs: backend.Attrs{Axes: []int{0}},
	}
	got := runOn[float32](t, 1, k, f32buf(nan, 1, 2, 3, 5, nan, 6, 7, 1, 1, 9, nan))
	if !math.IsNaN(float64(got[0])) || !math.IsNaN(float64(got[1])) || got[2] != 9 || !math.IsNaN(float64(got[3])) {
		t.Fatalf("got %v, want [NaN NaN 9 NaN]", got)
	}
}
