package tensor_test

import (
	"context"
	"errors"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

func data[T tensor.Elem](t *testing.T, x *tensor.Tensor) []T {
	t.Helper()
	v, err := x.Data[T](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestErrorsAreStickyAndNeverPanic(t *testing.T) {
	a := tensor.Ones(tensor.Float32, 2, 3)
	b := tensor.Ones(tensor.Float32, 4)
	y := a.Add(b).Exp().Sum(0).MatMul(a)
	if !errors.Is(y.Err(), tensor.ErrShape) {
		t.Fatalf("Err() = %v, want ErrShape", y.Err())
	}
	if y.Shape() != nil || y.Size() != 0 || y.DType() != 0 {
		t.Errorf("failed tensor reports shape %v size %d dtype %v", y.Shape(), y.Size(), y.DType())
	}
	if _, err := y.Data[float32](t.Context()); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("Data: %v, want ErrShape", err)
	}
	if err := tensor.Realize(t.Context(), a, y); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("Realize: %v, want ErrShape", err)
	}

	var null *tensor.Tensor
	if err := a.Add(null).Err(); !errors.Is(err, tensor.ErrNilTensor) {
		t.Errorf("nil operand: %v, want ErrNilTensor", err)
	}
	if err := null.Neg().Err(); !errors.Is(err, tensor.ErrNilTensor) {
		t.Errorf("nil receiver: %v, want ErrNilTensor", err)
	}
}

func TestBuildTimeErrors(t *testing.T) {
	f := tensor.Ones(tensor.Float32, 2, 3)
	i := tensor.Ones(tensor.Int32, 2, 3)
	tests := map[string]struct {
		got  *tensor.Tensor
		want error
	}{
		"mixed dtypes":       {f.Add(i), tensor.ErrDType},
		"int sqrt":           {i.Sqrt(), tensor.ErrDType},
		"int divide":         {i.Div(i), tensor.ErrDType},
		"bad reshape":        {f.Reshape(4, 2), tensor.ErrShape},
		"two inferred dims":  {f.Reshape(-1, -1), tensor.ErrShape},
		"axis out of range":  {f.Sum(2), tensor.ErrShape},
		"repeated axis":      {f.Sum(1, -1), tensor.ErrShape},
		"bad permutation":    {f.Permute(0, 0), tensor.ErrShape},
		"bad expand":         {f.Expand(4, 3), tensor.ErrShape},
		"negative pad":       {f.Pad([][2]int{{0, 0}, {-1, 0}}, 0), tensor.ErrShape},
		"shrink past end":    {f.Shrink([][2]int{{0, 2}, {1, 4}}), tensor.ErrShape},
		"matmul inner":       {f.MatMul(f), tensor.ErrShape},
		"float index":        {f.Gather(1, f), tensor.ErrDType},
		"squeeze non-1":      {f.Squeeze(0), tensor.ErrShape},
		"max of empty axis":  {tensor.Zeros(tensor.Float32, 2, 0).Max(1), tensor.ErrShape},
		"slice/shape length": {tensor.FromSlice([]float32{1, 2, 3}, 2, 2), tensor.ErrShape},
		"zero arange step":   {tensor.Arange[int32](0, 4, 0), tensor.ErrShape},
		"negative dimension": {tensor.Zeros(tensor.Float32, 2, -1), tensor.ErrShape},
	}
	for name, tt := range tests {
		if err := tt.got.Err(); !errors.Is(err, tt.want) {
			t.Errorf("%s: got %v, want %v", name, err, tt.want)
		}
	}
}

func TestRuntimeErrorsSurfaceFromData(t *testing.T) {
	eager := os.Getenv("MIMIR_EAGER") == "1"
	x := tensor.FromSlice([]float32{1, 2, 3})
	g := x.Gather(0, tensor.FromSlice([]int32{0, 3}))
	if !eager && g.Err() != nil {
		t.Fatalf("index values are only known when computed, got %v", g.Err())
	}
	if _, err := g.Data[float32](t.Context()); !errors.Is(err, tensor.ErrIndex) {
		t.Errorf("gather: %v, want ErrIndex", err)
	}
	m := tensor.FromSlice([]int32{5}).Fmod(tensor.FromSlice([]int32{0}))
	if _, err := m.Data[int32](t.Context()); !errors.Is(err, tensor.ErrDivideByZero) {
		t.Errorf("fmod: %v, want ErrDivideByZero", err)
	}
	if _, err := x.Data[int32](t.Context()); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("Data[int32] of float32: %v, want ErrDType", err)
	}
	if _, err := x.Item[float32](t.Context()); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("Item of 3 elements: %v, want ErrShape", err)
	}
}

func TestHugeTensorsFailWithoutPanicking(t *testing.T) {
	huge := tensor.Zeros(tensor.Float32, 1<<31, 1<<31) // a constant: one element of memory
	if err := huge.Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := huge.Data[float32](t.Context()); !errors.Is(err, tensor.ErrOutOfMemory) {
		t.Errorf("Data of 2^62 elements: %v, want ErrOutOfMemory", err)
	}
	if err := tensor.Arange[float32](0, 1e30, 1).Err(); !errors.Is(err, tensor.ErrShape) && !errors.Is(err, tensor.ErrOutOfMemory) {
		t.Errorf("Arange of 1e30 elements: %v, want ErrShape or ErrOutOfMemory", err)
	}
	if err := tensor.Zeros(tensor.Float32, 1<<40, 1<<40).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("shape overflowing int: %v, want ErrShape", err)
	}
}

func TestCancelledContextStopsWork(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	y := tensor.Rand(1, 1000).Exp()
	if _, err := y.Data[float32](ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	// The failed attempt leaves y intact.
	if len(data[float32](t, y)) != 1000 {
		t.Fatal("tensor unusable after a cancelled realize")
	}
}

func TestConstructors(t *testing.T) {
	if got := data[float32](t, tensor.Full(tensor.Float32, 2.5, 2, 2)); !slices.Equal(got, []float32{2.5, 2.5, 2.5, 2.5}) {
		t.Errorf("Full = %v", got)
	}
	if got := data[int32](t, tensor.Arange[int32](5, -1, -2)); !slices.Equal(got, []int32{5, 3, 1}) {
		t.Errorf("Arange = %v", got)
	}
	if got := data[float32](t, tensor.Arange[float32](0, 1, 0.25)); !slices.Equal(got, []float32{0, 0.25, 0.5, 0.75}) {
		t.Errorf("Arange = %v", got)
	}
	v, err := tensor.Scalar[int32](-7).Item[int32](t.Context())
	if err != nil || v != -7 {
		t.Errorf("Scalar Item = %v, %v", v, err)
	}
	src := []float32{1, 2}
	x := tensor.FromSlice(src)
	src[0] = 99
	if got := data[float32](t, x); got[0] != 1 {
		t.Error("FromSlice did not copy its input")
	}
	if got := data[float32](t, tensor.Zeros(tensor.Float32, 0, 3)); len(got) != 0 {
		t.Errorf("empty tensor has %d elements", len(got))
	}
}

func TestShapes(t *testing.T) {
	x := tensor.Zeros(tensor.Float32, 2, 3, 4)
	tests := map[string]struct {
		got  *tensor.Tensor
		want []int
	}{
		"inferred reshape": {x.Reshape(4, -1), []int{4, 6}},
		"transpose":        {x.Transpose(0, -1), []int{4, 3, 2}},
		"broadcast add":    {x.Add(tensor.Zeros(tensor.Float32, 3, 1)), []int{2, 3, 4}},
		"expand keep":      {tensor.Zeros(tensor.Float32, 3, 1).Expand(2, -1, 5), []int{2, 3, 5}},
		"unsqueeze end":    {x.Unsqueeze(-1), []int{2, 3, 4, 1}},
		"squeeze all":      {tensor.Zeros(tensor.Float32, 1, 3, 1).Squeeze(), []int{3}},
		"sum keeps rest":   {x.Sum(-1, 0), []int{3}},
		"dot":              {tensor.Ones(tensor.Float32, 3).MatMul(tensor.Ones(tensor.Float32, 3)), []int{}},
		"batched matmul":   {tensor.Zeros(tensor.Float32, 5, 1, 2, 3).MatMul(tensor.Zeros(tensor.Float32, 4, 3, 6)), []int{5, 4, 2, 6}},
		"softmax":          {x.Softmax(1), []int{2, 3, 4}},
	}
	for name, tt := range tests {
		if err := tt.got.Err(); err != nil {
			t.Errorf("%s: %v", name, err)
		} else if got := tt.got.Shape(); !slices.Equal(got, tt.want) {
			t.Errorf("%s: shape %v, want %v", name, got, tt.want)
		}
	}
}

func TestSplit(t *testing.T) {
	x := tensor.Arange[float32](0, 12, 1).Reshape(2, 6)
	parts := x.Split(-1, 1, 3, 2)
	want := [][]float32{{0, 6}, {1, 2, 3, 7, 8, 9}, {4, 5, 10, 11}}
	for i, p := range parts {
		got, err := p.Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want[i]) {
			t.Errorf("part %d = %v, want %v", i, got, want[i])
		}
	}
	for _, sizes := range [][]int{{1, 2}, {7, -1}, {}} {
		parts := x.Split(1, sizes...)
		if len(parts) != len(sizes) {
			t.Errorf("split into %v: %d parts", sizes, len(parts))
		}
		for _, p := range parts {
			if err := p.Err(); !errors.Is(err, tensor.ErrShape) {
				t.Errorf("split into %v: %v, want ErrShape", sizes, err)
			}
		}
	}
	if err := x.Split(2, 6)[0].Err(); err == nil {
		t.Error("split along a missing axis succeeded")
	}
}

func TestRealizeSharesWorkAndKeepsResults(t *testing.T) {
	x := tensor.Randn(3, 64, 64)
	h := x.MatMul(x).ReLU()
	a, b := h.Sum(), h.Max()
	if err := tensor.Realize(t.Context(), a, b); err != nil {
		t.Fatal(err)
	}
	first := data[float32](t, a)
	again := data[float32](t, a) // served from the realized result
	if !slices.Equal(first, again) {
		t.Fatalf("realized value changed: %v then %v", first, again)
	}
}

func TestDeepGraphDoesNotOverflowTheStack(t *testing.T) {
	y := tensor.Scalar[float32](0)
	one := tensor.Scalar[float32](1)
	for range 200_000 {
		y = y.Add(one)
	}
	v, err := y.Item[float32](t.Context())
	if err != nil || v != 200_000 {
		t.Fatalf("got %v, %v", v, err)
	}
}

func TestConcurrentUse(t *testing.T) {
	w := tensor.Randn(1, 32, 32)
	var wg sync.WaitGroup
	results := make([][]float32, 8)
	for g := range results {
		wg.Go(func() {
			x := tensor.Randn(2, 4, 32)
			y := x.MatMul(w).GELU().Softmax(-1)
			v, err := y.Data[float32](t.Context())
			if err != nil {
				t.Error(err)
			}
			results[g] = v
		})
	}
	wg.Wait()
	for _, r := range results[1:] {
		if !slices.Equal(r, results[0]) {
			t.Fatal("goroutines computed different results for the same inputs")
		}
	}
}
