// Package backend defines the contract between mimir's device-independent
// layers and the devices that execute kernels.
//
// Everything above this package (tensors, autograd, the scheduler) speaks in
// terms of DType, Op, View and Kernel. A Backend allocates device memory and
// runs one Kernel at a time; it never sees the graph.
package backend

import (
	"context"
	"errors"
	"fmt"
)

// Errors reported by graph construction and kernels. Callers test for them
// with errors.Is; the wrapping error carries the operation and shapes.
var (
	// ErrShape reports incompatible or invalid shapes, axes or ranges.
	ErrShape = errors.New("shape mismatch")
	// ErrDType reports an element type an operation does not accept.
	ErrDType = errors.New("unsupported dtype")
	// ErrIndex reports an index value outside the indexed dimension.
	ErrIndex = errors.New("index out of range")
	// ErrDivideByZero reports integer division or modulo by zero.
	ErrDivideByZero = errors.New("integer divide by zero")
	// ErrOutOfMemory reports an allocation the device cannot satisfy.
	ErrOutOfMemory = errors.New("out of memory")
	// ErrInternal reports a defect inside mimir, such as a recovered panic
	// in a kernel. It is never expected; please report it.
	ErrInternal = errors.New("internal error")
)

// DType is the element type of a tensor.
type DType uint8

// Supported element types.
const (
	Invalid DType = iota
	Float32
	Int32
)

// Size returns the size of one element in bytes.
func (d DType) Size() int {
	switch d {
	case Float32, Int32:
		return 4
	default:
		return 0
	}
}

// IsFloat reports whether d is a floating-point type.
func (d DType) IsFloat() bool { return d == Float32 }

func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Int32:
		return "int32"
	default:
		return fmt.Sprintf("DType(%d)", uint8(d))
	}
}

// Buffer is device memory holding Len elements of one DType. Buffers are
// created by the Backend that owns them and are only valid with it.
type Buffer interface {
	DType() DType
	Len() int
}

// Backend executes kernels on one device.
//
// Implementations must be safe for concurrent use by multiple goroutines on
// distinct buffers. A kernel's output buffer must not alias any input.
type Backend interface {
	// Name identifies the device, such as "cpu".
	Name() string
	// Alloc returns an uninitialized buffer of n elements.
	Alloc(dt DType, n int) (Buffer, error)
	// Free releases a buffer. The buffer must not be used afterwards.
	Free(b Buffer)
	// Upload copies host bytes into dst. len(src) must equal the buffer size
	// in bytes.
	Upload(ctx context.Context, dst Buffer, src []byte) error
	// Download copies the contents of src into host bytes. len(dst) must
	// equal the buffer size in bytes.
	Download(ctx context.Context, dst []byte, src Buffer) error
	// Run executes k, reading ins through k.In views and writing a
	// contiguous result of k.Shape into outs[0], and the values k stores
	// into the rest of outs. Each output holds exactly the elements of
	// k.Shape.
	//
	// An elementwise kernel (OpFused, a unary or binary op, OpCast,
	// OpWhere or OpContiguous) may write an output over an input it reads
	// through Contiguous(k.Shape): it reads every element of that input
	// before it writes any element of that output, as an in-place update
	// needs. No other kernel shares memory between its inputs and outputs.
	Run(ctx context.Context, k *Kernel, outs []Buffer, ins []Buffer) error
	// Sync waits until all submitted work has finished.
	Sync(ctx context.Context) error
}

// Kernel describes one operation for a Backend to execute.
type Kernel struct {
	Op    Op
	DType DType  // element type of the output
	Shape []int  // shape of the contiguous output
	In    []View // how each input buffer is read
	Attrs Attrs
}

// Attrs holds the operation-specific parameters of a Kernel or graph node.
// Each Op documents which fields it uses.
type Attrs struct {
	Axes   []int    // Reduce*: reduced axes. Permute: the permutation. Flip: flipped axes.
	Axis   int      // Gather, ScatterAdd: the indexed axis.
	Pads   [][2]int // Pad: elements added before and after each dimension. Unfold, Fold: each windowed dimension.
	Ranges [][2]int // Shrink: half-open [start, end) kept in each dimension.
	Value  float64  // Const, Pad, Unfold: the fill value.
	Seed   uint64   // RandUniform, RandNormal: the Philox key.

	// Unfold, Fold: the window size, step and spacing of its elements in
	// each windowed dimension.
	Kernel, Stride, Dilation []int

	// Fused: the program; its last instruction computes the output.
	Program []Instr
	// Fused: instructions whose values the kernel also writes, value
	// Stores[i] into output i+1. Stored values have the output's dtype.
	Stores []int
}

// Instr is one instruction of a fused program. Instruction i computes
// value i, one element per index of the kernel's output, from the values
// of earlier instructions. Values have the shape of the output, so
// elementwise ops apply to them index by index.
type Instr struct {
	// Op is OpLoad, which reads input Args[0] of the kernel, OpConst,
	// whose every element is Value, or an elementwise op (unary, binary,
	// OpCast or OpWhere) applied to the values Args.
	Op    Op
	DType DType // element type of the value
	Args  []int
	Value float64
}

// NumElements returns the number of elements in shape. It reports false if
// a dimension is negative or the product overflows int.
func NumElements(shape []int) (int, bool) {
	n := 1
	for _, d := range shape {
		if d < 0 {
			return 0, false
		}
		if d != 0 && n > maxInt/d {
			return 0, false
		}
		n *= d
	}
	return n, true
}

const maxInt = int(^uint(0) >> 1)
