package datasets

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

// writeExamples writes n examples to a Mimir Model file: x holds [i, -i]
// and y holds i.
func writeExamples(t *testing.T, dir string, n int, extra map[string]*tensor.Tensor) string {
	t.Helper()
	x := make([]float32, 2*n)
	y := make([]int32, n)
	for i := range n {
		x[2*i], x[2*i+1], y[i] = float32(i), -float32(i), int32(i)
	}
	tensors := map[string]*tensor.Tensor{
		"x": tensor.FromSlice(x, n, 2),
		"y": tensor.FromSlice(y, n),
	}
	for name, st := range extra {
		tensors[name] = st
	}
	path := filepath.Join(dir, "examples.mim")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := mim.WriteTensors(t.Context(), f, tensors, nil); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// labels returns the Y values of one pass over d, checking each batch's
// X against them.
func labels(t *testing.T, d data.Dataset) []int32 {
	t.Helper()
	var out []int32
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
				t.Fatalf("row %d: x %v does not match y %d", i, x[2*i:2*i+2], v)
			}
		}
		out = append(out, y...)
	}
	return out
}

func TestMappedReadsBatchesFromTheFile(t *testing.T) {
	path := writeExamples(t, t.TempDir(), 7, nil)
	d, err := Mapped(path, "x", "y", data.BatchSize(3))
	if err != nil {
		t.Fatal(err)
	}
	if d.Len() != 7 || d.Batches() != 3 {
		t.Errorf("%d examples in %d batches, want 7 in 3", d.Len(), d.Batches())
	}
	if got := labels(t, d); !slices.Equal(got, []int32{0, 1, 2, 3, 4, 5, 6}) {
		t.Errorf("labels in order = %v", got)
	}
	shuffled, err := Mapped(path, "x", "y", data.BatchSize(3), data.Shuffle(9))
	if err != nil {
		t.Fatal(err)
	}
	got := labels(t, shuffled)
	if slices.IsSorted(got) {
		t.Errorf("shuffled labels %v are in order", got)
	}
	if sorted := slices.Sorted(slices.Values(got)); !slices.Equal(sorted, []int32{0, 1, 2, 3, 4, 5, 6}) {
		t.Errorf("shuffled epoch visited %v, want each example once", got)
	}
}

func TestMappedRejectsBadFiles(t *testing.T) {
	dir := t.TempDir()
	if _, err := Mapped(filepath.Join(dir, "none.mim"), "x", "y"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("missing file: %v, want ErrNotExist", err)
	}
	path := writeExamples(t, dir, 3, map[string]*tensor.Tensor{
		"halves": tensor.FromSlice([]float32{1, 2, 3}, 3).Cast(tensor.BFloat16),
		"short":  tensor.FromSlice([]int32{0, 1}, 2),
	})
	if _, err := Mapped(path, "x", "halves"); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("bf16 targets: %v, want ErrDType", err)
	}
	if _, err := Mapped(path, "x", "labels"); !errors.Is(err, ErrFormat) {
		t.Errorf("missing tensor: %v, want ErrFormat", err)
	}
	if _, err := Mapped(path, "x", "short"); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("2 targets for 3 inputs: %v, want ErrShape", err)
	}
	// A new file: the tensors that the failed calls made may still hold
	// the mapping of the first, which Windows does not let a write change.
	corrupt := filepath.Join(dir, "corrupt.mim")
	if err := os.WriteFile(corrupt, []byte("not a Mimir Model file"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Mapped(corrupt, "x", "y"); !errors.Is(err, mim.ErrFormat) {
		t.Errorf("corrupt file: %v, want mim.ErrFormat", err)
	}
}

// TestMappedBatchesOutliveTheDataset checks that a batch holds the mapping
// it reads: the dataset may be garbage while a batch is still in use.
func TestMappedBatchesOutliveTheDataset(t *testing.T) {
	path := writeExamples(t, t.TempDir(), 4096, nil)
	batch := func() data.Batch {
		d, err := Mapped(path, "x", "y", data.BatchSize(4096))
		if err != nil {
			t.Fatal(err)
		}
		for b, err := range d.All(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
			return b
		}
		t.Fatal("no batch")
		return data.Batch{}
	}()
	for range 3 {
		runtime.GC()
	}
	y, err := batch.Y.Data[int32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 4096 || y[4095] != 4095 {
		t.Fatalf("batch read %d labels after the dataset was dropped", len(y))
	}
}
