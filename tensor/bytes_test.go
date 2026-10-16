package tensor_test

import (
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"
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
