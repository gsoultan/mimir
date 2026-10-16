package cpu

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// refScan scans x, read through v, along axis with combine, in float64
// for float elements, starting every line from its first element.
func refScan[T number](x []T, v backend.View, axis int, combine func(a, b float64) float64) []T {
	out := make([]T, v.Size())
	acc := make([]float64, len(out))
	strides := backend.ContiguousStrides(v.Shape)
	eachIndex(v.Shape, func(idx []int) {
		o := 0
		for d, i := range idx {
			o += i * strides[d]
		}
		x := float64(x[offsetOf(v, idx)])
		if idx[axis] == 0 {
			acc[o] = x
		} else {
			acc[o] = combine(acc[o-strides[axis]], x)
		}
		out[o] = T(acc[o])
		if dtypeOf[T]() == backend.Int32 {
			acc[o] = float64(out[o]) // wrap like int32
		}
	})
	return out
}

func scanKernel(dt backend.DType, shape []int, v backend.View, axis int, c backend.Combine) *backend.Kernel {
	return &backend.Kernel{Op: backend.OpScan, DType: dt, Shape: shape, In: []backend.View{v}, Attrs: backend.Attrs{Axis: axis, Combine: c}}
}

func TestScansMatchReference(t *testing.T) {
	shapes := []struct {
		shape []int
		axes  []int // every axis if nil
	}{
		{[]int{7}, nil}, {[]int{3, 5}, nil}, {[]int{2, 3, 4}, nil}, {[]int{2, 1, 3, 5}, nil}, {[]int{4, 1}, nil},
		{[]int{2, 40000}, []int{1}},    // few long lines, split into segments
		{[]int{40000, 3}, []int{0}},    // a long axis of short rows, split into segments
		{[]int{3, 1100, 2}, []int{1}},  // rows of two
		{[]int{2, 3, 1100}, []int{1}},  // rows split into chunks
		{[]int{1, 33000, 5}, []int{1}}, // segments of rows
	}
	sum := func(a, b float64) float64 { return a + b }
	mx := func(a, b float64) float64 { return max(a, b) }
	for _, s := range shapes {
		axes := s.axes
		for d := range s.shape {
			if s.axes == nil {
				axes = append(axes, d)
			}
		}
		for _, sv := range stridedViews(s.shape) {
			xf := make([]float32, sv.size)
			xi := make([]int32, sv.size)
			for i := range xf {
				// Small integers, so that sums are exact in any order.
				xf[i] = float32((i*7)%23 - 11)
				xi[i] = int32((i*5)%19 - 9)
			}
			for _, axis := range axes {
				sumF, maxF := refScan(xf, sv.view, axis, sum), refScan(xf, sv.view, axis, mx)
				sumI, maxI := refScan(xi, sv.view, axis, sum), refScan(xi, sv.view, axis, mx)
				for _, workers := range []int{1, 3} {
					name := fmt.Sprintf("%v %s axis %d workers %d", s.shape, sv.name, axis, workers)
					k := scanKernel(backend.Float32, s.shape, sv.view, axis, backend.CombineSum)
					if got := runOn[float32](t, workers, k, f32buf(xf...)); !slices.Equal(got, sumF) {
						t.Fatalf("%s: float32 sum differs", name)
					}
					k.Attrs.Combine = backend.CombineMax
					if got := runOn[float32](t, workers, k, f32buf(xf...)); !slices.Equal(got, maxF) {
						t.Fatalf("%s: float32 max differs", name)
					}
					k.DType = backend.Int32
					if got := runOn[int32](t, workers, k, i32buf(xi...)); !slices.Equal(got, maxI) {
						t.Fatalf("%s: int32 max differs", name)
					}
					k.Attrs.Combine = backend.CombineSum
					if got := runOn[int32](t, workers, k, i32buf(xi...)); !slices.Equal(got, sumI) {
						t.Fatalf("%s: int32 sum differs", name)
					}
				}
			}
		}
	}
}

func TestScanLogSumExpMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, shape := range [][]int{{5, 9}, {3, 40000}, {40000, 2}, {2, 300, 7}} {
		n, _ := backend.NumElements(shape)
		x := make([]float32, n)
		for i := range x {
			x[i] = rng.Float32()*40 - 20
		}
		for axis := range shape {
			k := scanKernel(backend.Float32, shape, backend.Contiguous(shape), axis, backend.CombineLogSumExp)
			got := runOn[float32](t, 3, k, f32buf(x...))
			want := refScan(x, backend.Contiguous(shape), axis, logAddExp)
			for i := range got {
				if d := math.Abs(float64(got[i] - want[i])); d > 1e-6*math.Max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("%v axis %d element %d: got %v, want %v", shape, axis, i, got[i], want[i])
				}
			}
		}
	}
}

func TestScanResultDoesNotDependOnWorkers(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, shape := range [][]int{{3, 70000}, {70000, 3}, {200, 300}} {
		n, _ := backend.NumElements(shape)
		x := make([]float32, n)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		for axis := range shape {
			for _, c := range []backend.Combine{backend.CombineSum, backend.CombineLogSumExp} {
				k := scanKernel(backend.Float32, shape, backend.Contiguous(shape), axis, c)
				one := runOn[float32](t, 1, k, f32buf(x...))
				for _, workers := range []int{2, 4, 7} {
					if got := runOn[float32](t, workers, k, f32buf(x...)); !slices.Equal(got, one) {
						t.Fatalf("%v axis %d %s: %d workers differ from one", shape, axis, c, workers)
					}
				}
			}
		}
	}
}

func TestScanSumAccumulatesInFloat64(t *testing.T) {
	// 2^24 + 1 is not a float32, so a float32 running sum drops every 1.
	x := []float32{1 << 24, 1, 1, 1, 1, 1, 1, 1, 1}
	k := scanKernel(backend.Float32, []int{9}, backend.Contiguous([]int{9}), 0, backend.CombineSum)
	got := runOn[float32](t, 1, k, f32buf(x...))
	if want := float32(1<<24 + 8); got[8] != want {
		t.Fatalf("got %v, want %v last", got, want)
	}
}

func TestScanEdgeValues(t *testing.T) {
	nan, inf, negZero := float32(math.NaN()), float32(math.Inf(1)), float32(math.Copysign(0, -1))
	for _, tt := range []struct {
		c       backend.Combine
		in, out []float32
	}{
		{backend.CombineMax, []float32{1, nan, 3}, []float32{1, nan, nan}},
		{backend.CombineSum, []float32{negZero, negZero}, []float32{negZero, negZero}},
		{backend.CombineLogSumExp, []float32{-inf, -inf, 0, inf, 1}, []float32{-inf, -inf, 0, inf, inf}},
		{backend.CombineLogSumExp, []float32{0, nan, 0}, []float32{0, nan, nan}},
	} {
		n := len(tt.in)
		k := scanKernel(backend.Float32, []int{n}, backend.Contiguous([]int{n}), 0, tt.c)
		got := runOn[float32](t, 1, k, f32buf(tt.in...))
		for i, w := range tt.out {
			if math.Float32bits(got[i]) != math.Float32bits(w) && !(math.IsNaN(float64(w)) && math.IsNaN(float64(got[i]))) {
				t.Fatalf("%s of %v: got %v, want %v", tt.c, tt.in, got, tt.out)
			}
		}
	}
}

func TestScanHalfRoundsOnce(t *testing.T) {
	// Every partial sum of 1 + 2^-9 + ... is rounded once: a bfloat16
	// running sum would stay 1.
	for _, dt := range []backend.DType{backend.BFloat16, backend.Float16} {
		x := []float32{1, 1.0 / 512, 1.0 / 512, 1.0 / 512, 1.0 / 512}
		bits := make([]uint16, len(x))
		for i, f := range x {
			bits[i] = f32ToHalf(dt, f)
		}
		k := scanKernel(dt, []int{1, 5}, backend.Contiguous([]int{1, 5}), 1, backend.CombineSum)
		b := New()
		out, err := b.Alloc(dt, 5)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Run(t.Context(), k, []backend.Buffer{out}, []backend.Buffer{halfBuf(dt, bits)}); err != nil {
			t.Fatal(err)
		}
		var last uint16
		switch o := out.(type) {
		case *buffer[bf16]:
			last = uint16(o.data[4])
		case *buffer[f16]:
			last = uint16(o.data[4])
		}
		if got, want := halfToF32(dt, last), halfToF32(dt, f32ToHalf(dt, 1+4.0/512)); got != want {
			t.Fatalf("%s: last sum %v, want %v", dt, got, want)
		}
	}
}

func TestScanRejectsLogSumExpOfInt32(t *testing.T) {
	k := scanKernel(backend.Int32, []int{3}, backend.Contiguous([]int{3}), 0, backend.CombineLogSumExp)
	if _, err := run[int32](t, k, i32buf(1, 2, 3)); err == nil {
		t.Fatal("no error")
	}
}

// BenchmarkScan scans 4M float32 elements along long and short axes,
// innermost and not.
func BenchmarkScan(b *testing.B) {
	for _, s := range []struct {
		name  string
		shape []int
		axis  int
		c     backend.Combine
	}{
		{"sum/one line of 4M", []int{1, 1 << 22}, 1, backend.CombineSum},
		{"sum/4096 lines of 1024", []int{4096, 1024}, 1, backend.CombineSum},
		{"sum/262144 lines of 16", []int{1 << 18, 16}, 1, backend.CombineSum},
		{"sum/outer 1024 of rows of 4096", []int{1024, 4096}, 0, backend.CombineSum},
		{"sum/outer 16 of rows of 262144", []int{16, 1 << 18}, 0, backend.CombineSum},
		{"sum/outer 1M of rows of 4", []int{1 << 20, 4}, 0, backend.CombineSum},
		{"sum/middle 64 of 4096x16", []int{4096, 64, 16}, 1, backend.CombineSum},
		{"max/4096 lines of 1024", []int{4096, 1024}, 1, backend.CombineMax},
		{"max/outer 1024 of rows of 4096", []int{1024, 4096}, 0, backend.CombineMax},
		{"logsumexp/4096 lines of 1024", []int{4096, 1024}, 1, backend.CombineLogSumExp},
		{"logsumexp/outer 1024 of rows of 4096", []int{1024, 4096}, 0, backend.CombineLogSumExp},
	} {
		b.Run(s.name, func(b *testing.B) {
			n, _ := backend.NumElements(s.shape)
			rng := rand.New(rand.NewPCG(1, 1))
			x := make([]float32, n)
			for i := range x {
				x[i] = rng.Float32()
			}
			be := New()
			in, out := f32buf(x...), f32buf(make([]float32, n)...)
			k := scanKernel(backend.Float32, s.shape, backend.Contiguous(s.shape), s.axis, s.c)
			b.SetBytes(int64(8 * n))
			for b.Loop() {
				if err := be.Run(b.Context(), k, []backend.Buffer{out}, []backend.Buffer{in}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
