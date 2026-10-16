package safetensors

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
	"unsafe"
)

// canMap reports whether Open maps files on this platform.
func canMap() bool {
	switch runtime.GOOS {
	case "plan9", "js", "wasip1":
		return false
	}
	return true
}

// mapsPath reports whether the process maps the file at path, where that
// can be seen.
func mapsPath(t *testing.T, path string) (mapped, known bool) {
	t.Helper()
	if runtime.GOOS != "linux" {
		return false, false
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	return strings.Contains(string(maps), path), true
}

func writeTestFile(t *testing.T, tensors map[string]Tensor) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "t.safetensors")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := Write(f, tensors, map[string]string{"k": "v"}); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestOpenMapsFilesWithAlignedTensors(t *testing.T) {
	in := map[string]Tensor{
		"w":     Float32Tensor([]int{3, 2}, []float32{1, -2, 3.5, 0, 7, 8}),
		"ids":   Int32Tensor([]int{3}, []int32{-1, 0, 1 << 30}),
		"empty": Float32Tensor([]int{0, 3}, nil),
	}
	path := writeTestFile(t, in)
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if f.Mapped() != canMap() {
		t.Errorf("Mapped() = %v on %s", f.Mapped(), runtime.GOOS)
	}
	if mapped, known := mapsPath(t, path); known && mapped != f.Mapped() {
		t.Errorf("the process maps the file: %v, Mapped() = %v", mapped, f.Mapped())
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(f.Names(), parsed.Names()) || f.Metadata["k"] != "v" {
		t.Errorf("names %v and metadata %v, want %v and k=v", f.Names(), f.Metadata, parsed.Names())
	}
	for _, name := range parsed.Names() {
		got, _ := f.Tensor(name)
		want, _ := parsed.Tensor(name)
		if got.DType != want.DType || !slices.Equal(got.Shape, want.Shape) || !bytes.Equal(got.Data, want.Data) {
			t.Errorf("%s = %+v, want %+v", name, got, want)
		}
		if len(got.Data) > 0 && uintptr(unsafe.Pointer(&got.Data[0]))%4 != 0 {
			t.Errorf("%s: data at %p is not aligned for its elements", name, &got.Data[0])
		}
	}
	// Write pads the header so the data section, and so the first tensor,
	// starts 8-byte aligned.
	first, _ := f.Tensor("ids")
	if uintptr(unsafe.Pointer(&first.Data[0]))%8 != 0 {
		t.Errorf("data section at %p, want 8-byte alignment", &first.Data[0])
	}

	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if _, ok := f.Tensor("w"); ok || len(f.Names()) != 0 {
		t.Error("a closed file still has tensors")
	}
	if mapped, known := mapsPath(t, path); known && mapped {
		t.Error("the file is still mapped after Close")
	}
}

func TestOpenReportsMissingAndCorruptFiles(t *testing.T) {
	dir := t.TempDir()
	if _, err := Open(filepath.Join(dir, "none.safetensors")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("missing file: %v, want ErrNotExist", err)
	}
	corrupt := map[string][]byte{
		"empty":     {},
		"short":     {1, 2, 3},
		"bad json":  file(`{"a":`),
		"bad sizes": file(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, 0, 0, 0, 0),
	}
	for name, b := range corrupt {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
		f, err := Open(path)
		if !errors.Is(err, ErrFormat) || f != nil {
			t.Errorf("%s: Open = %v, %v; want ErrFormat", name, f, err)
		}
		if mapped, known := mapsPath(t, path); known && mapped {
			t.Errorf("%s: the file stayed mapped after Open failed", name)
		}
	}
}

// TestHoldKeepsTheMappingPastClose checks that the mapping lasts while a
// reader that Hold added keeps it, and goes with the last of them.
func TestHoldKeepsTheMappingPastClose(t *testing.T) {
	path := writeTestFile(t, map[string]Tensor{"w": Float32Tensor([]int{2}, []float32{1.5, -2})})
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if !f.Mapped() {
		f.Hold()() // does nothing
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		t.Skip("files are not mapped on " + runtime.GOOS)
	}
	w, _ := f.Tensor("w")
	first, second := f.Hold(), f.Hold()
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if _, ok := f.Tensor("w"); ok {
		t.Error("a closed file still has tensors")
	}
	f.Hold()() // a closed file adds no hold, and letting go of it does nothing
	first()
	first() // lets go once
	if got, _ := w.Float32s(); !slices.Equal(got, []float32{1.5, -2}) {
		t.Errorf("held tensor reads %v after Close", got)
	}
	if mapped, known := mapsPath(t, path); known && !mapped {
		t.Error("the file was unmapped while a hold remained")
	}
	second()
	if mapped, known := mapsPath(t, path); known && mapped {
		t.Error("the file is still mapped after the last hold went")
	}
}

// TestDroppedFileIsUnmapped checks that a File that is never closed lets
// go of its mapping once it is garbage.
func TestDroppedFileIsUnmapped(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the mappings of the process are read on Linux")
	}
	path := writeTestFile(t, map[string]Tensor{"w": Float32Tensor([]int{1}, []float32{3})})
	if _, err := Open(path); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		runtime.GC()
		if mapped, _ := mapsPath(t, path); !mapped {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("a dropped File kept its mapping")
		}
		time.Sleep(time.Millisecond)
	}
}
