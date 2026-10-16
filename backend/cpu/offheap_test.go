package cpu

import (
	"encoding/binary"
	"errors"
	"math"
	"runtime"
	"runtime/debug"
	"sync/atomic"
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
	const n = 8 * offHeapMin / 4
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

// TestKernelsKeepTheirBuffersAlive runs kernels over off-heap buffers
// that nothing but the call refers to, with the garbage collector running
// as often as it can: a buffer collected during a kernel would be
// unmapped under it.
func TestKernelsKeepTheirBuffersAlive(t *testing.T) {
	defer debug.SetGCPercent(debug.SetGCPercent(1))
	b := New()
	const n = 2 * offHeapMin / 4
	k := &backend.Kernel{
		Op: backend.OpNeg, DType: backend.Float32, Shape: []int{n},
		In: []backend.View{backend.Contiguous([]int{n})},
	}
	x := make([]float32, n)
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()
	for i := range 200 {
		in, err := b.Alloc(backend.Float32, n)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Upload(t.Context(), in, asBytes(x)); err != nil {
			t.Fatal(err)
		}
		out, err := b.Alloc(backend.Float32, n)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Run(t.Context(), k, []backend.Buffer{out}, []backend.Buffer{in}); err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
		if err := b.Download(t.Context(), asBytes(x), out); err != nil {
			t.Fatal(err)
		}
	}
}

// TestWrapFuncReleasesOnce checks that a buffer of WrapFunc runs its
// release once: when Free frees it, or once it is garbage.
func TestWrapFuncReleasesOnce(t *testing.T) {
	b := New()
	raw := aligned(64)
	var freed, collected atomic.Int32
	buf, err := WrapFunc(backend.Float32, raw, func() { freed.Add(1) })
	if err != nil {
		t.Fatal(err)
	}
	b.Free(buf)
	b.Free(buf)
	if n := freed.Load(); n != 1 || buf.Len() != 0 {
		t.Errorf("released %d times by Free, %d elements left", n, buf.Len())
	}
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
	if n := freed.Load(); n != 1 {
		t.Errorf("released %d times after Free and a collection", n)
	}

	func() {
		if _, err := WrapFunc(backend.Int32, raw, func() { collected.Add(1) }); err != nil {
			t.Fatal(err)
		}
	}()
	deadline := time.Now().Add(10 * time.Second)
	for collected.Load() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("the release of a garbage buffer never ran")
		}
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
	if _, err := WrapFunc(backend.Float32, raw[1:5], func() { t.Error("a failed WrapFunc ran release") }); !errors.Is(err, ErrUnaligned) {
		t.Errorf("unaligned WrapFunc: %v", err)
	}
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
	if n := collected.Load(); n != 1 {
		t.Errorf("released %d times once garbage", n)
	}
}

// TestAllocPacesTheCollector drops mapped buffers without freeing them,
// with the heap's own pacing off, as a training loop whose tensors hold
// little heap memory leaves it: Alloc's collections must keep the mapped
// bytes bounded rather than let them grow with every buffer.
func TestAllocPacesTheCollector(t *testing.T) {
	if !offHeap {
		t.Skip("no off-heap buffers on " + runtime.GOOS)
	}
	settle(t)
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	b := New()
	const size = 16 << 20
	base := OffHeapBytes()
	var peak int64
	for range 256 { // 4 GiB in all, never touched, so never resident
		if _, err := b.Alloc(backend.Float32, size/4); err != nil {
			t.Fatal(err)
		}
		peak = max(peak, OffHeapBytes()-base)
	}
	t.Logf("peak of %d MiB mapped", peak>>20)
	if peak > 1<<30 {
		t.Errorf("%d MiB mapped at once by buffers dropped as soon as made", peak>>20)
	}
}
