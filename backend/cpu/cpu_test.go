package cpu

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

func f32buf(v ...float32) backend.Buffer { return &buffer[float32]{data: slices.Clone(v)} }
func i32buf(v ...int32) backend.Buffer   { return &buffer[int32]{data: slices.Clone(v)} }

func run[T elem](t *testing.T, k *backend.Kernel, ins ...backend.Buffer) ([]T, error) {
	t.Helper()
	n, _ := backend.NumElements(k.Shape)
	b := New()
	out, err := b.Alloc(k.DType, n)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Run(t.Context(), k, out, ins); err != nil {
		return nil, err
	}
	got, err := data[T](out)
	if err != nil {
		t.Fatal(err)
	}
	return got, nil
}

func TestPhiloxKnownAnswers(t *testing.T) {
	// Known-answer vectors from the Random123 distribution (kat_vectors).
	tests := []struct {
		ctr  [4]uint32
		key  [2]uint32
		want [4]uint32
	}{
		{[4]uint32{}, [2]uint32{}, [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{
			[4]uint32{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
			[2]uint32{0xffffffff, 0xffffffff},
			[4]uint32{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
		},
		{
			[4]uint32{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
			[2]uint32{0xa4093822, 0x299f31d0},
			[4]uint32{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1},
		},
	}
	for _, tt := range tests {
		if got := philox(tt.ctr, tt.key); got != tt.want {
			t.Errorf("philox(%x, %x) = %x, want %x", tt.ctr, tt.key, got, tt.want)
		}
	}
}

func TestRandIsDeterministicAndWellDistributed(t *testing.T) {
	const n = 1 << 18
	for _, normal := range []bool{false, true} {
		op, wantMean, wantVar := backend.OpRandUniform, 0.5, 1.0/12
		if normal {
			op, wantMean, wantVar = backend.OpRandNormal, 0, 1
		}
		k := &backend.Kernel{Op: op, DType: backend.Float32, Shape: []int{n}, Attrs: backend.Attrs{Seed: 42}}
		a, err := run[float32](t, k)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := run[float32](t, k)
		if !slices.Equal(a, b) {
			t.Fatalf("%s: same seed gave different values", op)
		}
		var sum, sq float64
		for _, v := range a {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || (!normal && (v < 0 || v >= 1)) {
				t.Fatalf("%s: value %v out of range", op, v)
			}
			sum += float64(v)
			sq += float64(v) * float64(v)
		}
		mean := sum / n
		variance := sq/n - mean*mean
		if math.Abs(mean-wantMean) > 0.01 || math.Abs(variance-wantVar) > 0.01*max(wantVar, 0.1) {
			t.Errorf("%s: mean %.4f var %.4f, want %.4f and %.4f", op, mean, variance, wantMean, wantVar)
		}
	}
}

func TestParallelForRecoversPanic(t *testing.T) {
	b := &Backend{workers: 4}
	err := b.parallelFor(t.Context(), 100, 1, func(lo, hi int) error {
		if lo > 0 {
			var s []int
			_ = s[lo]
		}
		return nil
	})
	if !errors.Is(err, backend.ErrInternal) {
		t.Fatalf("got %v, want ErrInternal", err)
	}
}

func TestParallelForStopsWhenCancelled(t *testing.T) {
	b := &Backend{workers: 4}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	ran := false
	err := b.parallelFor(ctx, 100, 1, func(lo, hi int) error { ran = true; return nil })
	if !errors.Is(err, context.Canceled) || ran {
		t.Fatalf("got err %v, ran %v; want context.Canceled without running", err, ran)
	}
}

func TestRunRejectsViewOutsideBuffer(t *testing.T) {
	k := &backend.Kernel{
		Op: backend.OpNeg, DType: backend.Float32, Shape: []int{4},
		In: []backend.View{{Shape: []int{4}, Strides: []int{1}, Offset: 1}},
	}
	if _, err := run[float32](t, k, f32buf(1, 2, 3, 4)); !errors.Is(err, backend.ErrInternal) {
		t.Fatalf("got %v, want ErrInternal", err)
	}
}

func TestMatMulMatchesNaiveOnStridedInputs(t *testing.T) {
	const batch, m, kk, n = 3, 5, 7, 4
	x := make([]float32, batch*m*kk)
	y := make([]float32, batch*n*kk) // stored as [batch, N, K] and read transposed
	for i := range x {
		x[i] = float32(i%11) - 5
	}
	for i := range y {
		y[i] = float32(i%7) - 3
	}
	k := &backend.Kernel{
		Op: backend.OpMatMul, DType: backend.Float32, Shape: []int{batch, m, n},
		In: []backend.View{
			backend.Contiguous([]int{batch, m, kk}),
			backend.Contiguous([]int{batch, n, kk}).Permute([]int{0, 2, 1}),
		},
	}
	got, err := run[float32](t, k, f32buf(x...), f32buf(y...))
	if err != nil {
		t.Fatal(err)
	}
	for bi := range batch {
		for i := range m {
			for j := range n {
				var want float32
				for p := range kk {
					want += x[bi*m*kk+i*kk+p] * y[bi*n*kk+j*kk+p]
				}
				if g := got[bi*m*n+i*n+j]; g != want {
					t.Fatalf("[%d,%d,%d] = %v, want %v", bi, i, j, g, want)
				}
			}
		}
	}
}

func TestReduceSumOverAxes(t *testing.T) {
	// [2, 3] summed over axis 1 and over both axes.
	in := f32buf(1, 2, 3, 4, 5, 6)
	k := &backend.Kernel{
		Op: backend.OpReduceSum, DType: backend.Float32, Shape: []int{2, 1},
		In: []backend.View{backend.Contiguous([]int{2, 3})}, Attrs: backend.Attrs{Axes: []int{1}},
	}
	if got, err := run[float32](t, k, in); err != nil || !slices.Equal(got, []float32{6, 15}) {
		t.Fatalf("axis 1: got %v, %v", got, err)
	}
	k.Shape, k.Attrs.Axes = []int{1, 1}, []int{0, 1}
	if got, err := run[float32](t, k, in); err != nil || !slices.Equal(got, []float32{21}) {
		t.Fatalf("all axes: got %v, %v", got, err)
	}
}

func TestReduceMaxPropagatesNaN(t *testing.T) {
	nan := float32(math.NaN())
	k := &backend.Kernel{
		Op: backend.OpReduceMax, DType: backend.Float32, Shape: []int{2, 1},
		In: []backend.View{backend.Contiguous([]int{2, 3})}, Attrs: backend.Attrs{Axes: []int{1}},
	}
	got, err := run[float32](t, k, f32buf(1, nan, 3, 4, 9, 6))
	if err != nil || !math.IsNaN(float64(got[0])) || got[1] != 9 {
		t.Fatalf("got %v, %v; want [NaN 9]", got, err)
	}
}

func TestPad(t *testing.T) {
	k := &backend.Kernel{
		Op: backend.OpPad, DType: backend.Int32, Shape: []int{3, 4},
		In:    []backend.View{backend.Contiguous([]int{1, 2})},
		Attrs: backend.Attrs{Pads: [][2]int{{1, 1}, {2, 0}}, Value: -1},
	}
	got, err := run[int32](t, k, i32buf(7, 8))
	want := []int32{-1, -1, -1, -1, -1, -1, 7, 8, -1, -1, -1, -1}
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("got %v, %v; want %v", got, err, want)
	}
}

func TestGatherAndScatterAdd(t *testing.T) {
	src := f32buf(10, 11, 12, 20, 21, 22) // [2, 3]
	idx := i32buf(2, 0, 1, 1)             // [2, 2]
	k := &backend.Kernel{
		Op: backend.OpGather, DType: backend.Float32, Shape: []int{2, 2},
		In:    []backend.View{backend.Contiguous([]int{2, 3}), backend.Contiguous([]int{2, 2})},
		Attrs: backend.Attrs{Axis: 1},
	}
	if got, err := run[float32](t, k, src, idx); err != nil || !slices.Equal(got, []float32{12, 10, 21, 21}) {
		t.Fatalf("gather: got %v, %v", got, err)
	}

	k = &backend.Kernel{
		Op: backend.OpScatterAdd, DType: backend.Float32, Shape: []int{2, 3},
		In: []backend.View{
			backend.Contiguous([]int{2, 3}), backend.Contiguous([]int{2, 2}), backend.Contiguous([]int{2, 2}),
		},
		Attrs: backend.Attrs{Axis: 1},
	}
	got, err := run[float32](t, k, f32buf(0, 0, 0, 0, 0, 0), idx, f32buf(1, 2, 3, 4))
	if err != nil || !slices.Equal(got, []float32{2, 0, 1, 0, 7, 0}) {
		t.Fatalf("scatter_add: got %v, %v", got, err)
	}
}

func TestIndexOutOfRangeIsAnError(t *testing.T) {
	k := &backend.Kernel{
		Op: backend.OpGather, DType: backend.Float32, Shape: []int{1},
		In:    []backend.View{backend.Contiguous([]int{3}), backend.Contiguous([]int{1})},
		Attrs: backend.Attrs{Axis: 0},
	}
	for _, j := range []int32{3, -1} {
		if _, err := run[float32](t, k, f32buf(1, 2, 3), i32buf(j)); !errors.Is(err, backend.ErrIndex) {
			t.Errorf("index %d: got %v, want ErrIndex", j, err)
		}
	}
}

func TestIntegerModByZeroIsAnError(t *testing.T) {
	k := &backend.Kernel{
		Op: backend.OpMod, DType: backend.Int32, Shape: []int{2},
		In: []backend.View{backend.Contiguous([]int{2}), backend.Contiguous([]int{2})},
	}
	if _, err := run[int32](t, k, i32buf(5, 6), i32buf(2, 0)); !errors.Is(err, backend.ErrDivideByZero) {
		t.Fatalf("got %v, want ErrDivideByZero", err)
	}
}

func TestFloatOnlyOpsRejectInt32(t *testing.T) {
	k := &backend.Kernel{
		Op: backend.OpSqrt, DType: backend.Int32, Shape: []int{1},
		In: []backend.View{backend.Contiguous([]int{1})},
	}
	if _, err := run[int32](t, k, i32buf(4)); !errors.Is(err, backend.ErrDType) {
		t.Fatalf("got %v, want ErrDType", err)
	}
}

func TestAllocTooLargeIsAnError(t *testing.T) {
	if _, err := New().Alloc(backend.Float32, math.MaxInt/2); !errors.Is(err, backend.ErrOutOfMemory) {
		t.Fatalf("got %v, want ErrOutOfMemory", err)
	}
}
