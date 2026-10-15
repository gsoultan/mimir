package data_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/tensor"
)

// epoch returns the Y values of each batch of one pass over d.
func epoch(t *testing.T, d data.Dataset) [][]int32 {
	t.Helper()
	var out [][]int32
	for b, err := range d.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		y, err := b.Y.Data[int32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		x, err := b.X.Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range y {
			if x[2*i] != float32(v) || x[2*i+1] != -float32(v) {
				t.Fatalf("batch row %d: x %v does not match y %d", i, x[2*i:2*i+2], v)
			}
		}
		out = append(out, y)
	}
	return out
}

func examples(n int) (*tensor.Tensor, *tensor.Tensor) {
	x := make([]float32, 2*n)
	y := make([]int32, n)
	for i := range n {
		x[2*i], x[2*i+1], y[i] = float32(i), -float32(i), int32(i)
	}
	return tensor.FromSlice(x, n, 2), tensor.FromSlice(y)
}

func TestBatchesInOrder(t *testing.T) {
	x, y := examples(7)
	d, err := data.FromTensors(x, y, data.BatchSize(3))
	if err != nil {
		t.Fatal(err)
	}
	got := epoch(t, d)
	want := [][]int32{{0, 1, 2}, {3, 4, 5}, {6}}
	if !slices.EqualFunc(got, want, slices.Equal) || d.Batches() != 3 || d.Len() != 7 {
		t.Errorf("batches %v (%d), want %v", got, d.Batches(), want)
	}
	dl, _ := data.FromTensors(x, y, data.BatchSize(3), data.DropLast())
	if got := epoch(t, dl); len(got) != 2 || dl.Batches() != 2 {
		t.Errorf("DropLast gave %d batches", len(got))
	}
}

func TestShuffleIsAPermutationThatRepeats(t *testing.T) {
	x, y := examples(10)
	a, _ := data.FromTensors(x, y, data.BatchSize(4), data.Shuffle(7))
	b, _ := data.FromTensors(x, y, data.BatchSize(4), data.Shuffle(7))
	first, second := slices.Concat(epoch(t, a)...), slices.Concat(epoch(t, a)...)
	if slices.Equal(first, second) {
		t.Error("two epochs had the same order")
	}
	if sorted := slices.Sorted(slices.Values(first)); !slices.Equal(sorted, []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
		t.Errorf("epoch visited %v, want each example once", first)
	}
	b.SetEpoch(1)
	if got := slices.Concat(epoch(t, b)...); !slices.Equal(got, second) {
		t.Error("SetEpoch(1) did not reproduce the second epoch")
	}
}

func TestErrors(t *testing.T) {
	x, y := examples(4)
	if _, err := data.FromTensors(x, tensor.Zeros(tensor.Int32, 3)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("length mismatch: %v", err)
	}
	if _, err := data.FromTensors(x, y, data.BatchSize(0)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("batch size 0: %v", err)
	}
	d, _ := data.FromTensors(x, y)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	for _, err := range d.All(ctx) {
		if !errors.Is(err, context.Canceled) {
			t.Errorf("cancelled epoch: %v", err)
		}
	}
}
