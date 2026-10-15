// Package tensor is mimir's n-dimensional array.
//
// Operations are lazy: each call checks shapes and dtypes and records a node
// in a graph, and nothing is computed until Realize, Data or Item is called.
// Errors are sticky instead of panicking. An invalid operation returns a
// tensor carrying the error, every later operation on it carries the same
// error, and Err, Realize, Data and Item report it:
//
//	y := x.MatMul(w).Add(b).ReLU()
//	if err := y.Err(); err != nil { ... } // shape errors, at build time
//	out, err := y.Data[float32](ctx)       // runtime errors, such as ErrIndex
//
// Set MIMIR_EAGER=1 to compute every operation as soon as it is created,
// which makes runtime errors appear at the call that caused them.
//
// Tensors are immutable and safe to share between goroutines.
package tensor

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// DType is the element type of a tensor.
type DType = backend.DType

// Element types.
const (
	Float32 = backend.Float32
	Int32   = backend.Int32
)

// Elem is the set of Go types that tensors hold.
type Elem interface{ float32 | int32 }

// Errors carried by tensors. Test for them with errors.Is.
var (
	ErrShape        = backend.ErrShape
	ErrDType        = backend.ErrDType
	ErrIndex        = backend.ErrIndex
	ErrDivideByZero = backend.ErrDivideByZero
	ErrOutOfMemory  = backend.ErrOutOfMemory
	ErrInternal     = backend.ErrInternal
	ErrNilTensor    = errors.New("nil tensor")
)

// device runs every tensor. Milestone 1 ships the CPU backend only.
var device backend.Backend = cpu.New()

// eager computes each operation as soon as it is created.
var eager = os.Getenv("MIMIR_EAGER") == "1"

// Tensor is an immutable n-dimensional array, or the error that prevented
// building it.
type Tensor struct {
	node *graph.Node
	err  error
}

func fail(err error) *Tensor { return &Tensor{err: err} }

// wrap turns a graph constructor's result into a tensor.
func wrap(n *graph.Node, err error) *Tensor {
	if err != nil {
		return fail(err)
	}
	if eager {
		if err := sched.Realize(context.Background(), device, n); err != nil {
			return fail(err)
		}
	}
	return &Tensor{node: n}
}

// firstErr returns the error of the first nil or failed tensor.
func firstErr(ts ...*Tensor) error {
	for _, t := range ts {
		if t == nil {
			return fmt.Errorf("tensor: %w", ErrNilTensor)
		}
		if t.err != nil {
			return t.err
		}
	}
	return nil
}

func dtypeOf[T Elem]() DType {
	var zero T
	if _, ok := any(zero).(float32); ok {
		return Float32
	}
	return Int32
}

func asBytes[T Elem](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(s[0])))
}

// alloc returns a zeroed slice of n elements, or ErrOutOfMemory when the
// runtime cannot address that many.
func alloc[T Elem](n int) (s []T, err error) {
	defer func() {
		if r := recover(); r != nil {
			s, err = nil, fmt.Errorf("tensor: allocate %d elements: %v: %w", n, r, ErrOutOfMemory)
		}
	}()
	return make([]T, n), nil
}

// FromSlice returns a tensor holding a copy of data. With no shape it is
// one-dimensional; otherwise the shape must hold exactly len(data) elements.
func FromSlice[T Elem](data []T, shape ...int) *Tensor {
	if len(shape) == 0 {
		shape = []int{len(data)}
	}
	if n, ok := backend.NumElements(shape); !ok || n != len(data) {
		return fail(fmt.Errorf("tensor: %d elements for shape %v: %w", len(data), shape, ErrShape))
	}
	buf, err := device.Alloc(dtypeOf[T](), len(data))
	if err != nil {
		return fail(err)
	}
	if err := device.Upload(context.Background(), buf, asBytes(data)); err != nil {
		device.Free(buf)
		return fail(err)
	}
	return wrap(graph.Load(buf, shape))
}

// Full returns a tensor of shape whose elements all equal value. It uses one
// element of memory until it is combined with other tensors.
func Full(dt DType, value float64, shape ...int) *Tensor {
	return wrap(graph.Const(dt, shape, value))
}

// Zeros returns a tensor of zeros.
func Zeros(dt DType, shape ...int) *Tensor { return Full(dt, 0, shape...) }

// Ones returns a tensor of ones.
func Ones(dt DType, shape ...int) *Tensor { return Full(dt, 1, shape...) }

// Scalar returns a zero-dimensional tensor holding v.
func Scalar[T Elem](v T) *Tensor { return Full(dtypeOf[T](), float64(v), []int{}...) }

// Arange returns the values start, start+step, ... that are less than stop
// (greater than stop for a negative step), like torch.arange.
func Arange[T Elem](start, stop, step T) *Tensor {
	if step == 0 {
		return fail(fmt.Errorf("tensor: arange with step 0: %w", ErrShape))
	}
	span := (float64(stop) - float64(start)) / float64(step)
	if math.IsNaN(span) || span > 1<<62 {
		return fail(fmt.Errorf("tensor: arange(%v, %v, %v): %w", start, stop, step, ErrShape))
	}
	n := 0
	if span > 0 {
		n = int(math.Ceil(span))
	}
	data, err := alloc[T](n)
	if err != nil {
		return fail(err)
	}
	for i := range data {
		data[i] = T(float64(start) + float64(i)*float64(step))
	}
	return FromSlice(data)
}

// Rand returns uniform random values in [0, 1). The values depend only on
// seed and the element's position, on every device.
func Rand(seed uint64, shape ...int) *Tensor { return wrap(graph.Rand(shape, seed, false)) }

// Randn returns standard normal random values determined by seed.
func Randn(seed uint64, shape ...int) *Tensor { return wrap(graph.Rand(shape, seed, true)) }

// Err returns the error that prevented building t, if any.
func (t *Tensor) Err() error { return firstErr(t) }

// Shape returns a copy of t's dimensions, or nil if t failed.
func (t *Tensor) Shape() []int {
	if firstErr(t) != nil {
		return nil
	}
	return slices.Clone(t.node.Shape)
}

// Rank returns the number of dimensions, or 0 if t failed.
func (t *Tensor) Rank() int { return len(t.Shape()) }

// Size returns the number of elements, or 0 if t failed.
func (t *Tensor) Size() int {
	if firstErr(t) != nil {
		return 0
	}
	return t.node.Size()
}

// DType returns t's element type, or backend.Invalid if t failed.
func (t *Tensor) DType() DType {
	if firstErr(t) != nil {
		return backend.Invalid
	}
	return t.node.DType
}

func (t *Tensor) String() string {
	if err := firstErr(t); err != nil {
		return fmt.Sprintf("Tensor(error: %v)", err)
	}
	return fmt.Sprintf("Tensor(%s%v)", t.node.DType, t.node.Shape)
}

// Realize computes t now. Later reads reuse the result.
func (t *Tensor) Realize(ctx context.Context) error { return Realize(ctx, t) }

// Realize computes all of ts in one schedule, so work they share runs once.
func Realize(ctx context.Context, ts ...*Tensor) error {
	if err := firstErr(ts...); err != nil {
		return err
	}
	nodes := make([]*graph.Node, len(ts))
	for i, t := range ts {
		nodes[i] = t.node
	}
	return sched.Realize(ctx, device, nodes...)
}

// Data computes t and returns its elements in row-major order. T must match
// t's dtype.
func (t *Tensor) Data[T Elem](ctx context.Context) ([]T, error) {
	if err := firstErr(t); err != nil {
		return nil, err
	}
	if want := dtypeOf[T](); t.node.DType != want {
		return nil, fmt.Errorf("tensor: Data[%s] of %s tensor: %w", want, t.node.DType, ErrDType)
	}
	out, err := alloc[T](t.node.Size())
	if err != nil {
		return nil, err
	}
	if err := sched.Download(ctx, device, t.node, asBytes(out)); err != nil {
		return nil, err
	}
	return out, nil
}

// Item computes a tensor of one element and returns that element.
func (t *Tensor) Item[T Elem](ctx context.Context) (T, error) {
	if err := firstErr(t); err != nil {
		return 0, err
	}
	if n := t.node.Size(); n != 1 {
		return 0, fmt.Errorf("tensor: Item of %d elements: %w", n, ErrShape)
	}
	v, err := t.Data[T](ctx)
	if err != nil {
		return 0, err
	}
	return v[0], nil
}
