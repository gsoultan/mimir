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
	return runWith[T](t, New(), k, ins...)
}

// runWith runs k on b and returns its outputs end to end, the result
// first and then each stored value.
func runWith[T elem](t *testing.T, b *Backend, k *backend.Kernel, ins ...backend.Buffer) ([]T, error) {
	t.Helper()
	n, _ := backend.NumElements(k.Shape)
	outs := make([]backend.Buffer, 1+len(k.Attrs.Stores))
	for i := range outs {
		var err error
		if outs[i], err = b.Alloc(k.DType, n); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Run(t.Context(), k, outs, ins); err != nil {
		return nil, err
	}
	var got []T
	for _, out := range outs {
		d, err := data[T](out)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, d...)
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

func TestMatMulSumsOverBatchAxes(t *testing.T) {
	const m, kk, n = 5, 7, 4
	x := make([]float32, 2*3*m*kk)
	y := make([]float32, 2*3*kk*n)
	for i := range x {
		x[i] = float32(i%11) - 5
	}
	for i := range y {
		y[i] = float32(i%7) - 3
	}
	xi, yi := make([]int32, len(x)), make([]int32, len(y))
	for i, v := range x {
		xi[i] = int32(v)
	}
	for i, v := range y {
		yi[i] = int32(v)
	}
	for _, axes := range [][]int{{0}, {1}, {0, 1}} {
		shape := []int{2, 3, m, n}
		for _, a := range axes {
			shape[a] = 1
		}
		want := make([]float32, shape[0]*shape[1]*m*n)
		for b0 := range 2 {
			for b1 := range 3 {
				o0, o1 := min(b0, shape[0]-1), min(b1, shape[1]-1)
				out, item := want[(o0*shape[1]+o1)*m*n:], b0*3+b1
				for i := range m {
					for j := range n {
						for p := range kk {
							out[i*n+j] += x[(item*m+i)*kk+p] * y[(item*kk+p)*n+j]
						}
					}
				}
			}
		}
		k := &backend.Kernel{
			Op: backend.OpMatMul, DType: backend.Float32, Shape: shape,
			In:    []backend.View{backend.Contiguous([]int{2, 3, m, kk}), backend.Contiguous([]int{2, 3, kk, n})},
			Attrs: backend.Attrs{Axes: axes},
		}
		got, err := run[float32](t, k, f32buf(x...), f32buf(y...))
		if err != nil || !slices.Equal(got, want) {
			t.Fatalf("float32 over axes %v: got %v, %v; want %v", axes, got, err, want)
		}
		k.DType = backend.Int32
		goti, err := run[int32](t, k, i32buf(xi...), i32buf(yi...))
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range goti {
			if float32(v) != want[i] {
				t.Fatalf("int32 over axes %v: element %d = %d, want %v", axes, i, v, want[i])
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

func TestCoalesceKeepsElementOrder(t *testing.T) {
	tests := map[string]struct {
		shape    []int
		views    []backend.View
		wantRank int
	}{
		"contiguous": {[]int{2, 3, 4}, []backend.View{backend.Contiguous([]int{2, 3, 4})}, 1},
		"broadcast row": {[]int{2, 3, 4}, []backend.View{
			{Shape: []int{2, 3, 4}, Strides: []int{0, 4, 1}, Offset: 5},
			backend.Contiguous([]int{2, 3, 4}),
		}, 2},
		"transposed": {[]int{3, 2}, []backend.View{{Shape: []int{3, 2}, Strides: []int{1, 3}}}, 2},
		"unit dims":  {[]int{1, 3, 1, 2}, []backend.View{{Shape: []int{1, 3, 1, 2}, Strides: []int{9, 2, 7, 1}}}, 1},
		"all ones":   {[]int{1, 1}, []backend.View{{Shape: []int{1, 1}, Strides: []int{4, 2}, Offset: 3}}, 0},
		"empty":      {[]int{2, 0, 3}, []backend.View{backend.Contiguous([]int{2, 0, 3})}, 3},
		"windows":    {[]int{2, 2, 3}, []backend.View{{Shape: []int{2, 2, 3}, Strides: []int{8, 1, 4}, Offset: 1}}, 3},
	}
	for name, tt := range tests {
		shape, views := coalesce(tt.shape, tt.views)
		if len(shape) != tt.wantRank {
			t.Errorf("%s: coalesced %v to %v, want rank %d", name, tt.shape, shape, tt.wantRank)
		}
		n, _ := backend.NumElements(tt.shape)
		a, b := newCursor(tt.shape, tt.views, 0), newCursor(shape, views, 0)
		for i := range n {
			if !slices.Equal(a.offs, b.offs) {
				t.Errorf("%s: element %d at offsets %v, coalesced %v", name, i, a.offs, b.offs)
				break
			}
			a.next()
			b.next()
		}
	}
}

func TestStridedBinarySplitsLongRows(t *testing.T) {
	// x is broadcast along the first axis, so the add takes the strided
	// path. Its dimensions coalesce to [5, 21000], and the four chunks of
	// work start partway through a row.
	const rows, cols = 5, 3 * 7000
	x := make([]float32, cols)
	y := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(i)
	}
	for i := range y {
		y[i] = float32(i % 7)
	}
	k := &backend.Kernel{
		Op: backend.OpAdd, DType: backend.Float32, Shape: []int{rows, 3, cols / 3},
		In: []backend.View{
			{Shape: []int{rows, 3, cols / 3}, Strides: []int{0, cols / 3, 1}},
			backend.Contiguous([]int{rows, 3, cols / 3}),
		},
	}
	b := New()
	b.workers = 7
	out, err := b.Alloc(backend.Float32, rows*cols)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Run(t.Context(), k, []backend.Buffer{out}, []backend.Buffer{f32buf(x...), f32buf(y...)}); err != nil {
		t.Fatal(err)
	}
	got, err := data[float32](out)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if want := x[i%cols] + y[i]; v != want {
			t.Fatalf("element %d = %v, want %v", i, v, want)
		}
	}
}
