package tensor_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"runtime"
	"slices"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// f32bytes returns the bytes of v, which start at an 8-byte boundary.
func f32bytes(v []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v))
}

func TestFromBytesReadsAlignedBytesInPlace(t *testing.T) {
	v := []float32{1, 2, 3, 4, 5, 6}
	x := tensor.FromBytes(tensor.Float32, f32bytes(v), 2, 3)
	if err := x.Err(); err != nil {
		t.Fatal(err)
	}
	if !x.Device().Equal(tensor.CPU()) || !slices.Equal(x.Shape(), []int{2, 3}) {
		t.Fatalf("tensor on %v of shape %v, want the CPU and [2 3]", x.Device(), x.Shape())
	}
	v[4] = 50 // written after the tensor was made
	if got, err := x.Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{1, 2, 3, 4, 50, 6}) {
		t.Errorf("Data = %v, %v; want the bytes as they are now", got, err)
	}
	// Views read the same memory.
	v[0] = -1
	if got, err := x.Reshape(6).Shrink([][2]int{{0, 2}}).Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{-1, 2}) {
		t.Errorf("view = %v, %v; want [-1 2]", got, err)
	}

	i := []int32{7, -8}
	y := tensor.FromBytes(tensor.Int32, unsafe.Slice((*byte)(unsafe.Pointer(&i[0])), 8))
	if got, err := y.Data[int32](t.Context()); err != nil || !slices.Equal(got, i) || y.Rank() != 1 {
		t.Errorf("int32 tensor = %v, %v (rank %d); want %v", got, err, y.Rank(), i)
	}
	if empty := tensor.FromBytes(tensor.Float32, nil, 0, 3); empty.Err() != nil || empty.Size() != 0 {
		t.Errorf("empty tensor: %v", empty)
	}
}

func TestFromBytesCopiesUnalignedBytes(t *testing.T) {
	backing := f32bytes(make([]float32, 6))
	raw := backing[1:21] // five float32s starting one byte past alignment
	want := []float32{1.5, -2, 3, 4, 5}
	for i, v := range want {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	x := tensor.FromBytes(tensor.Float32, raw, 5)
	if err := x.Err(); err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(raw, math.Float32bits(100)) // after the copy
	if got, err := x.Data[float32](t.Context()); err != nil || !slices.Equal(got, want) {
		t.Errorf("Data = %v, %v; want the copy %v", got, err, want)
	}
}

func TestFromBytesRejectsBadLengthsAndDTypes(t *testing.T) {
	raw := f32bytes(make([]float32, 3))
	for name, x := range map[string]*tensor.Tensor{
		"partial element": tensor.FromBytes(tensor.Float32, raw[:6]),
		"shape too big":   tensor.FromBytes(tensor.Float32, raw, 4),
		"shape too small": tensor.FromBytes(tensor.Int32, raw, 1, 2),
		"negative dim":    tensor.FromBytes(tensor.Int32, raw, -3),
	} {
		if err := x.Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
	if err := tensor.FromBytes(tensor.DType(0), raw).Err(); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("invalid dtype: %v, want ErrDType", err)
	}
}

func TestOptimizerStepsWriteTheBytesOfAParam(t *testing.T) {
	v := []float32{1, 2}
	raw := f32bytes(v)
	p := tensor.NewParam(tensor.Zeros(tensor.Float32, 2))
	if err := p.Set(tensor.FromBytes(tensor.Float32, raw)); err != nil {
		t.Fatal(err)
	}
	// d/dp sum(p*p) = 2p, so one SGD step at rate 0.5 reaches zero.
	if err := p.Value().Mul(p.Value()).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	opt := optim.SGD([]*tensor.Param{p}, optim.LR(0.5))
	if err := opt.Step(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got, err := p.Value().Data[float32](t.Context()); err != nil || !slices.Equal(got, []float32{0, 0}) {
		t.Fatalf("parameter after the step = %v, %v; want [0 0]", got, err)
	}
	// The step wrote the new value over the old one's memory: the bytes.
	// Eager mode computes the new value before the step can, into memory
	// of its own, so the bytes keep the old value there.
	if !tensor.Eager() && !slices.Equal(v, []float32{0, 0}) {
		t.Errorf("bytes after the step = %v, want [0 0]", v)
	}
}

// TestFromBytesFuncReleases checks when FromBytesFunc lets go of its
// bytes: at once when it copies them or fails, and only once the tensor's
// memory is garbage when it reads them in place.
func TestFromBytesFuncReleases(t *testing.T) {
	backing := f32bytes(make([]float32, 6))
	released := 0
	release := func() { released++ }
	if x := tensor.FromBytesFunc(tensor.Float32, backing[1:21], release, 5); x.Err() != nil || released != 1 {
		t.Errorf("unaligned bytes: %v, released %d times, want once at once", x.Err(), released)
	}
	if x := tensor.FromBytesFunc(tensor.Float32, backing[:6], release); !errors.Is(x.Err(), tensor.ErrShape) || released != 2 {
		t.Errorf("partial element: %v, released %d times in all, want 2", x.Err(), released)
	}

	var inPlace atomic.Int32
	func() {
		x := tensor.FromBytesFunc(tensor.Float32, backing, func() { inPlace.Add(1) }, 6)
		y := x.Reshape(2, 3).Add(tensor.Scalar(float32(1)))
		if _, err := y.Data[float32](t.Context()); err != nil {
			t.Fatal(err)
		}
		runtime.GC()
		if n := inPlace.Load(); n != 0 {
			t.Errorf("released %d times while the tensor is in use", n)
		}
		runtime.KeepAlive(x)
	}()
	deadline := time.Now().Add(10 * time.Second)
	for inPlace.Load() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("bytes read in place were never released")
		}
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
}

func TestBytesReturnWhatFromBytesReads(t *testing.T) {
	ctx := context.Background()
	half := []byte{0x00, 0x3c, 0x00, 0xc0, 0x00, 0x00} // 1, -2, 0 in binary16
	q8 := make([]byte, tensor.Q8_0.BlockBytes())
	q8[0], q8[1], q8[2], q8[33] = 0x00, 0x3c, 7, 0xf9 // scale 1, quants 7 and -7
	for _, tc := range []struct {
		dt    tensor.DType
		data  []byte
		shape []int
	}{
		{tensor.Float32, f32bytes([]float32{1.5, -2, 3, 0}), []int{2, 2}},
		{tensor.Float16, half, []int{3}},
		{tensor.Q8_0, q8, []int{1, 32}},
	} {
		got, err := tensor.FromBytes(tc.dt, tc.data, tc.shape...).Bytes(ctx)
		if err != nil || !slices.Equal(got, tc.data) {
			t.Errorf("%s: bytes %v, %v; want %v", tc.dt, got, err, tc.data)
		}
	}
	// A computed tensor's bytes are its elements, as Data returns them.
	sum := tensor.FromSlice([]float32{1, 2}).Add(tensor.FromSlice([]float32{3, 4}))
	got, err := sum.Bytes(ctx)
	if err != nil || binary.LittleEndian.Uint32(got[4:]) != math.Float32bits(6) {
		t.Errorf("bytes of a sum: %v, %v", got, err)
	}
}
