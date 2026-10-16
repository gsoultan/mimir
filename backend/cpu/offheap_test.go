package cpu

import (
	"encoding/binary"
	"errors"
	"math"
	"runtime"
	"testing"
	"time"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// aligned returns n bytes that start at an 8-byte boundary.
func aligned(n int) []byte {
	words := make([]uint64, (n+7)/8)
	return unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), n)
}

// settle runs the cleanups of the off-heap buffers that earlier tests
// dropped, so that the count of mapped bytes changes only with this
// test's own allocations. runtime.GC queues them once it has swept; the
// cleanup goroutines then run them, which the loop waits out.
func settle(t *testing.T) {
	t.Helper()
	runtime.GC()
	v := OffHeapBytes()
	deadline := time.Now().Add(10 * time.Second)
	for stable := 0; stable < 20; stable++ {
		time.Sleep(time.Millisecond)
		if w := OffHeapBytes(); w != v {
			v, stable = w, -1
		}
		if time.Now().After(deadline) {
			t.Fatalf("the count of mapped bytes did not settle: %d", v)
		}
	}
}

func TestAllocMapsLargeBuffersOffHeap(t *testing.T) {
	if !offHeap {
		t.Skipf("no anonymous mappings on %s", runtime.GOOS)
	}
	settle(t)
	b := New()
	before := OffHeapBytes()
	small, err := b.Alloc(backend.Float32, offHeapMin/4-1)
	if err != nil {
		t.Fatal(err)
	}
	if small.(*buffer[float32]).free != nil || OffHeapBytes() != before {
		t.Errorf("a buffer below offHeapMin was mapped off heap (%d bytes mapped)", OffHeapBytes()-before)
	}
	large, err := b.Alloc(backend.Int32, offHeapMin/4)
	if err != nil {
		t.Fatal(err)
	}
	lb := large.(*buffer[int32])
	if lb.free == nil || OffHeapBytes() != before+offHeapMin {
		t.Fatalf("a buffer of offHeapMin bytes stayed on the heap (%d bytes mapped)", OffHeapBytes()-before)
	}
	for i, v := range lb.data {
		if v != 0 {
			t.Fatalf("element %d of a new buffer = %d, want 0", i, v)
		}
	}
	b.Free(large)
	if OffHeapBytes() != before || lb.data != nil {
		t.Errorf("after Free %d bytes are mapped and the buffer holds %d elements", OffHeapBytes()-before, len(lb.data))
	}
	b.Free(large) // a second Free does nothing
	b.Free(small) // and so does freeing a heap buffer
	if OffHeapBytes() != before || small.Len() != offHeapMin/4-1 {
		t.Errorf("freeing again: %d bytes mapped, heap buffer of %d elements", OffHeapBytes()-before, small.Len())
	}
}

func TestKernelsRunOnOffHeapBuffers(t *testing.T) {
	if !offHeap {
		t.Skipf("no anonymous mappings on %s", runtime.GOOS)
	}
	b := New()
	const n = 2 * offHeapMin / 4
	in, err := b.Alloc(backend.Float32, n)
	if err != nil {
		t.Fatal(err)
	}
	if in.(*buffer[float32]).free == nil {
		t.Fatal("input buffer is not off heap")
	}
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%97) - 48
	}
	if err := b.Upload(t.Context(), in, asBytes(x)); err != nil {
		t.Fatal(err)
	}
	k := &backend.Kernel{
		Op: backend.OpNeg, DType: backend.Float32, Shape: []int{n},
		In: []backend.View{backend.Contiguous([]int{n})},
	}
	got, err := runWith[float32](t, b, k, in)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != -x[i] {
			t.Fatalf("element %d = %v, want %v", i, v, -x[i])
		}
	}
	b.Free(in)
}

func TestWrapReadsAndWritesBytesInPlace(t *testing.T) {
	raw := aligned(17)
	for i, v := range []float32{1, -2.5, 3, 4} {
		binary.NativeEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	in, err := Wrap(backend.Float32, raw[:16])
	if err != nil {
		t.Fatal(err)
	}
	if in.DType() != backend.Float32 || in.Len() != 4 {
		t.Fatalf("wrapped %s buffer of %d elements, want float32 of 4", in.DType(), in.Len())
	}
	// A write to the bytes shows in the buffer.
	binary.NativeEndian.PutUint32(raw[12:], math.Float32bits(40))
	if d, _ := data[float32](in); d[3] != 40 {
		t.Errorf("element 3 = %v after writing 40 to its bytes", d[3])
	}
	// A kernel reads a wrapped input and writes a wrapped output.
	outRaw := aligned(16)
	out, err := Wrap(backend.Float32, outRaw)
	if err != nil {
		t.Fatal(err)
	}
	k := &backend.Kernel{
		Op: backend.OpNeg, DType: backend.Float32, Shape: []int{4},
		In: []backend.View{backend.Contiguous([]int{4})},
	}
	if err := New().Run(t.Context(), k, []backend.Buffer{out}, []backend.Buffer{in}); err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{-1, 2.5, -3, -40} {
		if got := math.Float32frombits(binary.NativeEndian.Uint32(outRaw[4*i:])); got != want {
			t.Errorf("output byte %d = %v, want %v", 4*i, got, want)
		}
	}
	New().Free(out) // a wrapped buffer is the caller's: Free leaves it alone
	if d, _ := data[float32](out); len(d) != 4 {
		t.Error("Free emptied a wrapped buffer")
	}

	if _, err := Wrap(backend.Float32, raw[1:17]); !errors.Is(err, ErrUnaligned) {
		t.Errorf("wrapping bytes at an odd address: %v, want ErrUnaligned", err)
	}
	if _, err := Wrap(backend.Int32, raw[:6]); !errors.Is(err, backend.ErrShape) {
		t.Errorf("wrapping 6 bytes as int32: %v, want ErrShape", err)
	}
	if _, err := Wrap(backend.Invalid, raw[:16]); !errors.Is(err, backend.ErrDType) {
		t.Errorf("wrapping as an invalid dtype: %v, want ErrDType", err)
	}
	empty, err := Wrap(backend.Int32, nil)
	if err != nil || empty.Len() != 0 {
		t.Errorf("wrapping no bytes: %v, %v", empty, err)
	}
}
