// Package graph is mimir's lazy intermediate representation: a DAG of
// primitive operations whose shapes and dtypes are checked when each node is
// built, so errors surface at the line that caused them even though nothing
// runs until the graph is realized.
package graph

import (
	"errors"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// ErrOverwritten reports a read of a realized node whose memory now holds
// other values, such as a parameter's value from before an in-place
// optimizer step.
var ErrOverwritten = errors.New("tensor memory was overwritten")

// Node is one primitive operation. Op, DType, Shape and Attrs never change
// after construction. Mem, Ver, View and Inputs are owned by the scheduler:
// once a node is realized it holds its result and drops its inputs, so a
// realized tensor does not keep the graph that produced it alive.
type Node struct {
	Op    backend.Op
	DType backend.DType
	// Placeholder marks a node that no graph computes, set when it is
	// made: it stands for values computed elsewhere, as a tensor inside
	// tensor.VMap stands for one example of a batch.
	Placeholder bool
	Shape       []int
	Attrs       backend.Attrs
	Inputs      []*Node

	// Dev is the device that holds the node's inputs, and so computes
	// it, or nil if it is computed from constants alone and runs on any
	// device. A realized node's result lives on Mem.Dev.
	Dev backend.Backend

	Mem  *Storage     // set once realized
	Ver  int          // the version of Mem that holds the result
	View backend.View // how Shape is read from Mem.Buf

	// Mark is the scheduler's scratch space, which it uses under its lock
	// while it walks a graph.
	Mark Mark
}

// Mark numbers a node or storage within one walk of the scheduler over a
// graph: Num is meaningful only while Walk is the walk's number, so a new
// walk starts with every mark cleared without touching them.
type Mark struct {
	Walk uint64
	Num  int32
}

// Storage is a buffer that realized nodes read, each through its own view.
// Writing new values over the buffer, as an in-place update does, bumps
// Version, which makes the nodes realized with earlier versions stale.
type Storage struct {
	Buf     backend.Buffer
	Dev     backend.Backend // the backend that owns Buf
	Version int
	Refs    int // the nodes realized into the storage since Version changed

	Mark Mark // the scheduler's scratch space, like Node.Mark
}

// Realized reports whether the node holds its result.
func (n *Node) Realized() bool { return n.Mem != nil }

// Stale reports whether the node's memory has been overwritten since it
// was realized.
func (n *Node) Stale() bool { return n.Mem != nil && n.Ver != n.Mem.Version }

// Buf returns the buffer that holds a realized node's result.
func (n *Node) Buf() backend.Buffer { return n.Mem.Buf }

// Size returns the number of elements.
func (n *Node) Size() int {
	s, _ := backend.NumElements(n.Shape)
	return s
}

func errorf(op backend.Op, kind error, format string, args ...any) error {
	return fmt.Errorf("%s: %s: %w", op, fmt.Sprintf(format, args...), kind)
}

func checkShape(op backend.Op, shape []int) error {
	if _, ok := backend.NumElements(shape); !ok {
		return errorf(op, backend.ErrShape, "invalid shape %v", shape)
	}
	return nil
}

func checkDType(op backend.Op, dt backend.DType) error {
	if dt.Size() == 0 {
		return errorf(op, backend.ErrDType, "%s", dt)
	}
	return nil
}

// newNode completes n, whose inputs must share a device, with that
// device. Block-quantized inputs are refused except where n dequantizes
// them (see readsQuantized).
func newNode(n *Node) (*Node, error) {
	for i, in := range n.Inputs {
		if in.DType.IsQuantized() && !readsQuantized(n.Op, i) {
			return nil, errorf(n.Op, backend.ErrDType, "%s input %d", in.DType, i)
		}
	}
	for _, in := range n.Inputs {
		switch d := in.Dev; {
		case d == nil:
		case n.Dev == nil:
			n.Dev = d
		case d != n.Dev:
			return nil, errorf(n.Op, backend.ErrDevice, "inputs on %s and %s", n.Dev.Name(), d.Name())
		}
	}
	return n, nil
}

// readsQuantized reports whether input i of an op may be block-quantized:
// the input of a cast, which dequantizes it, the source of a gather, which
// dequantizes what it reads, and the weights of a quantized product.
func readsQuantized(op backend.Op, i int) bool {
	switch op {
	case backend.OpCast, backend.OpGather:
		return i == 0
	case backend.OpQMatMul:
		return i == 1
	}
	return false
}

// Load wraps a buffer of dev that already holds data of the given shape.
// A buffer of a block-quantized type holds whole blocks along the last
// dimension.
func Load(dev backend.Backend, buf backend.Buffer, shape []int) (*Node, error) {
	if err := checkShape(backend.OpLoad, shape); err != nil {
		return nil, err
	}
	if n, _ := backend.NumElements(shape); buf.Len() != n {
		return nil, errorf(backend.OpLoad, backend.ErrShape, "%d elements for shape %v", buf.Len(), shape)
	}
	if dt := buf.DType(); dt.IsQuantized() && (len(shape) == 0 || shape[len(shape)-1]%dt.BlockSize() != 0) {
		return nil, errorf(backend.OpLoad, backend.ErrShape, "%s of shape %v: rows of whole blocks of %d", dt, shape, dt.BlockSize())
	}
	mem := &Storage{Buf: buf, Dev: dev, Refs: 1}
	return &Node{Op: backend.OpLoad, DType: buf.DType(), Shape: slices.Clone(shape), Dev: dev, Mem: mem, View: backend.Contiguous(shape)}, nil
}

// Const is a tensor whose elements all equal value. It costs one element of
// memory however large its shape.
func Const(dt backend.DType, shape []int, value float64) (*Node, error) {
	if err := checkDType(backend.OpConst, dt); err != nil {
		return nil, err
	}
	if err := checkShape(backend.OpConst, shape); err != nil {
		return nil, err
	}
	return &Node{Op: backend.OpConst, DType: dt, Shape: slices.Clone(shape), Attrs: backend.Attrs{Value: value}}, nil
}

// Rand is a Float32 tensor of uniform [0, 1) or standard normal values
// determined by seed.
func Rand(shape []int, seed uint64, normal bool) (*Node, error) {
	op := backend.OpRandUniform
	if normal {
		op = backend.OpRandNormal
	}
	if err := checkShape(op, shape); err != nil {
		return nil, err
	}
	return &Node{Op: op, DType: backend.Float32, Shape: slices.Clone(shape), Attrs: backend.Attrs{Seed: seed}}, nil
}

// Unary applies an elementwise op. Exp2, Log2, Sin, Sqrt and Recip accept
// only float types.
func Unary(op backend.Op, x *Node) (*Node, error) {
	if !op.IsUnary() || op == backend.OpCast {
		return nil, errorf(op, backend.ErrInternal, "not a unary op")
	}
	if op != backend.OpNeg && !x.DType.IsFloat() {
		return nil, errorf(op, backend.ErrDType, "%s input", x.DType)
	}
	return newNode(&Node{Op: op, DType: x.DType, Shape: x.Shape, Inputs: []*Node{x}})
}

// Cast converts x to dt. Float to integer truncates toward zero; float to
// a narrower float rounds to nearest even. A block-quantized x is
// dequantized, to a float dt only.
func Cast(x *Node, dt backend.DType) (*Node, error) {
	if err := checkDType(backend.OpCast, dt); err != nil {
		return nil, err
	}
	if x.DType.IsQuantized() && !dt.IsFloat() {
		return nil, errorf(backend.OpCast, backend.ErrDType, "%s to %s", x.DType, dt)
	}
	return newNode(&Node{Op: backend.OpCast, DType: dt, Shape: x.Shape, Inputs: []*Node{x}})
}

// Binary applies an elementwise op to two inputs of the same shape and dtype.
func Binary(op backend.Op, x, y *Node) (*Node, error) {
	if !op.IsBinary() {
		return nil, errorf(op, backend.ErrInternal, "not a binary op")
	}
	if !slices.Equal(x.Shape, y.Shape) {
		return nil, errorf(op, backend.ErrShape, "%v and %v", x.Shape, y.Shape)
	}
	if x.DType != y.DType {
		return nil, errorf(op, backend.ErrDType, "%s and %s", x.DType, y.DType)
	}
	return newNode(&Node{Op: op, DType: x.DType, Shape: x.Shape, Inputs: []*Node{x, y}})
}

// Where selects x where cond is non-zero and y elsewhere. All three share a
// shape; x and y share a dtype.
func Where(cond, x, y *Node) (*Node, error) {
	if !slices.Equal(cond.Shape, x.Shape) || !slices.Equal(x.Shape, y.Shape) {
		return nil, errorf(backend.OpWhere, backend.ErrShape, "%v, %v and %v", cond.Shape, x.Shape, y.Shape)
	}
	if x.DType != y.DType {
		return nil, errorf(backend.OpWhere, backend.ErrDType, "%s and %s", x.DType, y.DType)
	}
	return newNode(&Node{Op: backend.OpWhere, DType: x.DType, Shape: x.Shape, Inputs: []*Node{cond, x, y}})
}

// Reduce reduces x over axes, keeping them with size 1. Axes must be valid,
// distinct and sorted. ReduceMax rejects empty axes, which have no maximum.
// A half input is reduced in float32, into a Float32 result.
func Reduce(op backend.Op, x *Node, axes []int) (*Node, error) {
	if !op.IsReduce() {
		return nil, errorf(op, backend.ErrInternal, "not a reduction")
	}
	shape := slices.Clone(x.Shape)
	for i, a := range axes {
		if a < 0 || a >= len(shape) || (i > 0 && a <= axes[i-1]) {
			return nil, errorf(op, backend.ErrShape, "axes %v for shape %v", axes, x.Shape)
		}
		if op == backend.OpReduceMax && shape[a] == 0 {
			return nil, errorf(op, backend.ErrShape, "empty axis %d of %v", a, x.Shape)
		}
		shape[a] = 1
	}
	dt := x.DType
	if dt.Size() == 2 {
		dt = backend.Float32
	}
	return newNode(&Node{Op: op, DType: dt, Shape: shape, Inputs: []*Node{x}, Attrs: backend.Attrs{Axes: slices.Clone(axes)}})
}

// Reshape gives x a new shape with the same number of elements.
func Reshape(x *Node, shape []int) (*Node, error) {
	if err := checkShape(backend.OpReshape, shape); err != nil {
		return nil, err
	}
	if n, _ := backend.NumElements(shape); n != x.Size() {
		return nil, errorf(backend.OpReshape, backend.ErrShape, "%v to %v", x.Shape, shape)
	}
	return newNode(&Node{Op: backend.OpReshape, DType: x.DType, Shape: slices.Clone(shape), Inputs: []*Node{x}})
}

// Permute reorders dimensions: output dimension i is input dimension perm[i].
func Permute(x *Node, perm []int) (*Node, error) {
	if len(perm) != len(x.Shape) {
		return nil, errorf(backend.OpPermute, backend.ErrShape, "permutation %v for shape %v", perm, x.Shape)
	}
	seen := make([]bool, len(perm))
	shape := make([]int, len(perm))
	for i, p := range perm {
		if p < 0 || p >= len(perm) || seen[p] {
			return nil, errorf(backend.OpPermute, backend.ErrShape, "permutation %v for shape %v", perm, x.Shape)
		}
		seen[p] = true
		shape[i] = x.Shape[p]
	}
	return newNode(&Node{Op: backend.OpPermute, DType: x.DType, Shape: shape, Inputs: []*Node{x}, Attrs: backend.Attrs{Axes: slices.Clone(perm)}})
}

// Expand broadcasts size-1 dimensions of x to shape, which has the same rank.
func Expand(x *Node, shape []int) (*Node, error) {
	if err := checkShape(backend.OpExpand, shape); err != nil {
		return nil, err
	}
	if len(shape) != len(x.Shape) {
		return nil, errorf(backend.OpExpand, backend.ErrShape, "%v to %v", x.Shape, shape)
	}
	for i, d := range shape {
		if x.Shape[i] != d && x.Shape[i] != 1 {
			return nil, errorf(backend.OpExpand, backend.ErrShape, "%v to %v", x.Shape, shape)
		}
	}
	return newNode(&Node{Op: backend.OpExpand, DType: x.DType, Shape: slices.Clone(shape), Inputs: []*Node{x}})
}

// Pad adds pads[d][0] elements before and pads[d][1] after dimension d,
// filled with value.
func Pad(x *Node, pads [][2]int, value float64) (*Node, error) {
	if len(pads) != len(x.Shape) {
		return nil, errorf(backend.OpPad, backend.ErrShape, "pads %v for shape %v", pads, x.Shape)
	}
	shape := make([]int, len(pads))
	for i, p := range pads {
		if p[0] < 0 || p[1] < 0 {
			return nil, errorf(backend.OpPad, backend.ErrShape, "negative pad %v", p)
		}
		shape[i] = x.Shape[i] + p[0] + p[1]
	}
	if err := checkShape(backend.OpPad, shape); err != nil {
		return nil, err
	}
	return newNode(&Node{Op: backend.OpPad, DType: x.DType, Shape: shape, Inputs: []*Node{x}, Attrs: backend.Attrs{Pads: slices.Clone(pads), Value: value}})
}

// Shrink keeps the half-open range [r[0], r[1]) of each dimension.
func Shrink(x *Node, ranges [][2]int) (*Node, error) {
	if len(ranges) != len(x.Shape) {
		return nil, errorf(backend.OpShrink, backend.ErrShape, "ranges %v for shape %v", ranges, x.Shape)
	}
	shape := make([]int, len(ranges))
	for i, r := range ranges {
		if r[0] < 0 || r[0] > r[1] || r[1] > x.Shape[i] {
			return nil, errorf(backend.OpShrink, backend.ErrShape, "range %v for dimension %d of %v", r, i, x.Shape)
		}
		shape[i] = r[1] - r[0]
	}
	return newNode(&Node{Op: backend.OpShrink, DType: x.DType, Shape: shape, Inputs: []*Node{x}, Attrs: backend.Attrs{Ranges: slices.Clone(ranges)}})
}

// Flip reverses x along distinct axes.
func Flip(x *Node, axes []int) (*Node, error) {
	seen := make([]bool, len(x.Shape))
	for _, a := range axes {
		if a < 0 || a >= len(x.Shape) || seen[a] {
			return nil, errorf(backend.OpFlip, backend.ErrShape, "axes %v for shape %v", axes, x.Shape)
		}
		seen[a] = true
	}
	return newNode(&Node{Op: backend.OpFlip, DType: x.DType, Shape: x.Shape, Inputs: []*Node{x}, Attrs: backend.Attrs{Axes: slices.Clone(axes)}})
}

// MatMul multiplies [..., M, K] by [..., K, N] with equal batch dimensions.
// Products are Float32 or Int32: the tensor package multiplies half types
// in Float32.
func MatMul(x, y *Node) (*Node, error) {
	r := len(x.Shape)
	if r < 2 || len(y.Shape) != r || !slices.Equal(x.Shape[:r-2], y.Shape[:r-2]) || x.Shape[r-1] != y.Shape[r-2] {
		return nil, errorf(backend.OpMatMul, backend.ErrShape, "%v and %v", x.Shape, y.Shape)
	}
	if x.DType != y.DType || x.DType.Size() == 2 {
		return nil, errorf(backend.OpMatMul, backend.ErrDType, "%s and %s", x.DType, y.DType)
	}
	shape := slices.Concat(x.Shape[:r-1], y.Shape[r-1:])
	if err := checkShape(backend.OpMatMul, shape); err != nil {
		return nil, err
	}
	return newNode(&Node{Op: backend.OpMatMul, DType: x.DType, Shape: shape, Inputs: []*Node{x, y}})
}

// checkIndex validates an Int32 index of the same rank as x that fits x
// outside axis.
func checkIndex(op backend.Op, x, idx *Node, axis int) error {
	if idx.DType != backend.Int32 {
		return errorf(op, backend.ErrDType, "%s index", idx.DType)
	}
	if axis < 0 || axis >= len(x.Shape) || len(idx.Shape) != len(x.Shape) {
		return errorf(op, backend.ErrShape, "axis %d, index %v for %v", axis, idx.Shape, x.Shape)
	}
	for d := range x.Shape {
		if d != axis && idx.Shape[d] > x.Shape[d] {
			return errorf(op, backend.ErrShape, "index %v does not fit %v outside axis %d", idx.Shape, x.Shape, axis)
		}
	}
	return nil
}

// Gather reads x along axis at Int32 positions idx, like torch.gather. The
// result has the shape of idx. Positions are checked when the graph runs.
// A gather from a block-quantized x dequantizes what it reads into a
// Float32 result.
func Gather(x, idx *Node, axis int) (*Node, error) {
	if err := checkIndex(backend.OpGather, x, idx, axis); err != nil {
		return nil, err
	}
	dt := x.DType
	if dt.IsQuantized() {
		dt = backend.Float32
	}
	return newNode(&Node{Op: backend.OpGather, DType: dt, Shape: idx.Shape, Inputs: []*Node{x, idx}, Attrs: backend.Attrs{Axis: axis}})
}

// QMatMul multiplies x, Float32 [..., K], by the transpose of w, a
// block-quantized [N, K], into a Float32 [..., N]; see backend.OpQMatMul.
func QMatMul(x, w *Node) (*Node, error) {
	if x.DType != backend.Float32 || !w.DType.IsQuantized() {
		return nil, errorf(backend.OpQMatMul, backend.ErrDType, "%s by %s", x.DType, w.DType)
	}
	r := len(x.Shape)
	if r == 0 || len(w.Shape) != 2 || x.Shape[r-1] != w.Shape[1] {
		return nil, errorf(backend.OpQMatMul, backend.ErrShape, "%v by the transpose of %v", x.Shape, w.Shape)
	}
	shape := append(slices.Clone(x.Shape[:r-1]), w.Shape[0])
	return newNode(&Node{Op: backend.OpQMatMul, DType: backend.Float32, Shape: shape, Inputs: []*Node{x, w}})
}

// ScatterAdd returns x with src added at Int32 positions idx along axis,
// like torch.Tensor.scatter_add. idx and src share a shape.
func ScatterAdd(x, idx, src *Node, axis int) (*Node, error) {
	if err := checkIndex(backend.OpScatterAdd, x, idx, axis); err != nil {
		return nil, err
	}
	if !slices.Equal(idx.Shape, src.Shape) {
		return nil, errorf(backend.OpScatterAdd, backend.ErrShape, "index %v and source %v", idx.Shape, src.Shape)
	}
	if src.DType != x.DType {
		return nil, errorf(backend.OpScatterAdd, backend.ErrDType, "%s and %s", x.DType, src.DType)
	}
	return newNode(&Node{Op: backend.OpScatterAdd, DType: x.DType, Shape: x.Shape, Inputs: []*Node{x, idx, src}, Attrs: backend.Attrs{Axis: axis}})
}

// Window describes sliding windows over the trailing len(Kernel)
// dimensions of a tensor; see backend.OpUnfold.
type Window struct {
	Kernel, Stride, Dilation []int
	Pads                     [][2]int
}

// positions returns the number of window positions along each windowed
// dimension of size spatial[d].
func (w Window) positions(op backend.Op, spatial []int) ([]int, error) {
	s := len(w.Kernel)
	if s == 0 || len(w.Stride) != s || len(w.Dilation) != s || len(w.Pads) != s || len(spatial) != s {
		return nil, errorf(op, backend.ErrShape, "window %+v over %v", w, spatial)
	}
	out := make([]int, s)
	for d := range s {
		k, st, dil, p := w.Kernel[d], w.Stride[d], w.Dilation[d], w.Pads[d]
		if k < 1 || st < 1 || dil < 1 || p[0] < 0 || p[1] < 0 {
			return nil, errorf(op, backend.ErrShape, "window %+v", w)
		}
		span := (k-1)*dil + 1
		if spatial[d]+p[0]+p[1] < span {
			return nil, errorf(op, backend.ErrShape, "window %+v larger than %v", w, spatial)
		}
		out[d] = (spatial[d]+p[0]+p[1]-span)/st + 1
	}
	return out, nil
}

func (w Window) attrs(value float64) backend.Attrs {
	return backend.Attrs{
		Kernel:   slices.Clone(w.Kernel),
		Stride:   slices.Clone(w.Stride),
		Dilation: slices.Clone(w.Dilation),
		Pads:     slices.Clone(w.Pads),
		Value:    value,
	}
}

// Unfold returns the sliding windows w of x, filling the padding with
// value: [B..., D...] becomes [B..., K..., O...].
func Unfold(x *Node, w Window, value float64) (*Node, error) {
	s := len(w.Kernel)
	if len(x.Shape) < s {
		return nil, errorf(backend.OpUnfold, backend.ErrShape, "window %+v over %v", w, x.Shape)
	}
	lead, spatial := x.Shape[:len(x.Shape)-s], x.Shape[len(x.Shape)-s:]
	pos, err := w.positions(backend.OpUnfold, spatial)
	if err != nil {
		return nil, err
	}
	shape := slices.Concat(lead, w.Kernel, pos)
	if err := checkShape(backend.OpUnfold, shape); err != nil {
		return nil, err
	}
	return newNode(&Node{Op: backend.OpUnfold, DType: x.DType, Shape: shape, Inputs: []*Node{x}, Attrs: w.attrs(value)})
}

// Fold is the adjoint of Unfold: it sums the windows x, shaped
// [B..., K..., O...], into a [B..., spatial...] result.
func Fold(x *Node, spatial []int, w Window) (*Node, error) {
	s := len(w.Kernel)
	if len(x.Shape) < 2*s {
		return nil, errorf(backend.OpFold, backend.ErrShape, "window %+v from %v", w, x.Shape)
	}
	pos, err := w.positions(backend.OpFold, spatial)
	if err != nil {
		return nil, err
	}
	lead, win := x.Shape[:len(x.Shape)-2*s], x.Shape[len(x.Shape)-2*s:]
	if !slices.Equal(win, slices.Concat(w.Kernel, pos)) {
		return nil, errorf(backend.OpFold, backend.ErrShape, "windows %v for window %+v over %v", x.Shape, w, spatial)
	}
	shape := slices.Concat(lead, spatial)
	if err := checkShape(backend.OpFold, shape); err != nil {
		return nil, err
	}
	return newNode(&Node{Op: backend.OpFold, DType: x.DType, Shape: shape, Inputs: []*Node{x}, Attrs: w.attrs(0)})
}

// Scan is the inclusive scan of x along axis by combine; see
// backend.OpScan. The result has x's shape and dtype. CombineLogSumExp
// accepts only float types.
func Scan(x *Node, axis int, combine backend.Combine) (*Node, error) {
	if axis < 0 || axis >= len(x.Shape) {
		return nil, errorf(backend.OpScan, backend.ErrShape, "axis %d for shape %v", axis, x.Shape)
	}
	switch combine {
	case backend.CombineSum, backend.CombineMax:
	case backend.CombineLogSumExp:
		if !x.DType.IsFloat() {
			return nil, errorf(backend.OpScan, backend.ErrDType, "%s of %s", combine, x.DType)
		}
	default:
		return nil, errorf(backend.OpScan, backend.ErrInternal, "no combine %s", combine)
	}
	return newNode(&Node{Op: backend.OpScan, DType: x.DType, Shape: x.Shape, Inputs: []*Node{x}, Attrs: backend.Attrs{Axis: axis, Combine: combine}})
}

// Cat joins xs along axis, each after the ones before it. They share a
// dtype, a rank and every dimension but axis; there are 1 to
// backend.MaxCatInputs of them.
func Cat(xs []*Node, axis int) (*Node, error) {
	if len(xs) == 0 || len(xs) > backend.MaxCatInputs {
		return nil, errorf(backend.OpCat, backend.ErrInternal, "%d inputs", len(xs))
	}
	first := xs[0]
	if axis < 0 || axis >= len(first.Shape) {
		return nil, errorf(backend.OpCat, backend.ErrShape, "axis %d of %v", axis, first.Shape)
	}
	shape := slices.Clone(first.Shape)
	shape[axis] = 0
	for _, x := range xs {
		if x.DType != first.DType {
			return nil, errorf(backend.OpCat, backend.ErrDType, "%s and %s", first.DType, x.DType)
		}
		if len(x.Shape) != len(shape) || !slices.Equal(x.Shape[:axis], shape[:axis]) || !slices.Equal(x.Shape[axis+1:], shape[axis+1:]) {
			return nil, errorf(backend.OpCat, backend.ErrShape, "%v and %v along axis %d", first.Shape, x.Shape, axis)
		}
		shape[axis] += x.Shape[axis]
	}
	if err := checkShape(backend.OpCat, shape); err != nil {
		return nil, err
	}
	return newNode(&Node{Op: backend.OpCat, DType: first.DType, Shape: shape, Inputs: slices.Clone(xs), Attrs: backend.Attrs{Axis: axis}})
}

// AttentionAttrs configures an attention: the factor applied to the
// scores, and whether to hide key j from query i when j > i + Offset.
type AttentionAttrs struct {
	Scale  float64
	Causal bool
	Offset int
}

func (a AttentionAttrs) attrs() backend.Attrs {
	return backend.Attrs{Scale: a.Scale, Causal: a.Causal, CausalOffset: a.Offset}
}

// attentionShape checks the float inputs of an attention, and its mask
// unless it is nil, and returns its geometry.
func attentionShape(op backend.Op, q, k, v, mask *Node) (backend.AttentionShape, error) {
	ins := []*Node{q, k, v}
	var m []int
	if mask != nil {
		ins, m = append(ins, mask), mask.Shape
	}
	for _, in := range ins {
		if !in.DType.IsFloat() {
			return backend.AttentionShape{}, errorf(op, backend.ErrDType, "%s input", in.DType)
		}
	}
	s, err := backend.NewAttentionShape(q.Shape, k.Shape, v.Shape, m)
	if err != nil {
		return s, fmt.Errorf("%s: %w", op, err)
	}
	return s, nil
}

// Attention computes softmax(scale·q·kᵀ + mask)·v over queries
// [B..., H, Q, D], keys [B..., HK, K, D] and values [B..., HK, K, DV] of
// any float dtypes, adding mask, [B..., H, Q, K], unless it is nil. The
// Float32 result is one-dimensional: the attention's result, followed by
// the logsumexp of each query's scores (see backend.OpAttention).
func Attention(q, k, v, mask *Node, a AttentionAttrs) (*Node, error) {
	s, err := attentionShape(backend.OpAttention, q, k, v, mask)
	if err != nil {
		return nil, err
	}
	ins := []*Node{q, k, v}
	if mask != nil {
		ins = append(ins, mask)
	}
	return newNode(&Node{Op: backend.OpAttention, DType: backend.Float32, Shape: []int{s.Size()}, Inputs: ins, Attrs: a.attrs()})
}

// AttentionGrad computes the gradients of the attention of q, k and v
// with mask and a, from its result out, its logsumexp lse and the
// gradient of its result gout, which have float dtypes. The Float32
// result is one-dimensional: the gradients of q, k and v in turn.
func AttentionGrad(q, k, v, mask, out, lse, gout *Node, a AttentionAttrs) (*Node, error) {
	s, err := attentionShape(backend.OpAttentionGrad, q, k, v, mask)
	if err != nil {
		return nil, err
	}
	if !slices.Equal(out.Shape, s.OutShape()) || !slices.Equal(gout.Shape, s.OutShape()) || !slices.Equal(lse.Shape, s.LSEShape()) {
		return nil, errorf(backend.OpAttentionGrad, backend.ErrShape, "result %v, logsumexp %v and gradient %v for %v", out.Shape, lse.Shape, gout.Shape, s.OutShape())
	}
	if !out.DType.IsFloat() || !gout.DType.IsFloat() || lse.DType != backend.Float32 {
		return nil, errorf(backend.OpAttentionGrad, backend.ErrDType, "result %s, logsumexp %s and gradient %s", out.DType, lse.DType, gout.DType)
	}
	ins := []*Node{q, k, v, out, lse, gout}
	if mask != nil {
		ins = append(ins, mask)
	}
	return newNode(&Node{Op: backend.OpAttentionGrad, DType: backend.Float32, Shape: []int{s.GradSize()}, Inputs: ins, Attrs: a.attrs()})
}

// SliceScatter returns x with the block ranges, half-open in each
// dimension, replaced by src, which has the block's shape and x's dtype.
func SliceScatter(x, src *Node, ranges [][2]int) (*Node, error) {
	if len(ranges) != len(x.Shape) || len(src.Shape) != len(x.Shape) {
		return nil, errorf(backend.OpSliceScatter, backend.ErrShape, "%v into %v at %v", src.Shape, x.Shape, ranges)
	}
	for i, r := range ranges {
		if r[0] < 0 || r[0] > r[1] || r[1] > x.Shape[i] || src.Shape[i] != r[1]-r[0] {
			return nil, errorf(backend.OpSliceScatter, backend.ErrShape, "%v into %v at %v", src.Shape, x.Shape, ranges)
		}
	}
	if x.DType != src.DType {
		return nil, errorf(backend.OpSliceScatter, backend.ErrDType, "%s into %s", src.DType, x.DType)
	}
	return newNode(&Node{Op: backend.OpSliceScatter, DType: x.DType, Shape: x.Shape, Inputs: []*Node{x, src}, Attrs: backend.Attrs{Ranges: slices.Clone(ranges)}})
}

// Argsort is the Int32 positions along axis of the first k elements of
// the stable sort of x along it, ascending or descending; see
// backend.OpArgsort. k is x's length along axis, or at most
// backend.MaxTopK of it.
func Argsort(x *Node, axis, k int, descending bool) (*Node, error) {
	if axis < 0 || axis >= len(x.Shape) {
		return nil, errorf(backend.OpArgsort, backend.ErrShape, "axis %d for shape %v", axis, x.Shape)
	}
	if n := x.Shape[axis]; k < 0 || k > n || k < n && k > backend.MaxTopK {
		return nil, errorf(backend.OpArgsort, backend.ErrShape, "%d of %d positions along axis %d", k, n, axis)
	}
	if err := checkDType(backend.OpArgsort, x.DType); err != nil {
		return nil, err
	}
	shape := slices.Clone(x.Shape)
	shape[axis] = k
	return newNode(&Node{Op: backend.OpArgsort, DType: backend.Int32, Shape: shape, Inputs: []*Node{x}, Attrs: backend.Attrs{Axis: axis, Descending: descending}})
}

// FFT is the discrete Fourier transform of the complex Float32 x,
// [..., 2], along axis, which is not its last; see backend.OpFFT.
func FFT(x *Node, axis int, inverse bool) (*Node, error) {
	r := len(x.Shape)
	if r < 2 || x.Shape[r-1] != 2 || axis < 0 || axis >= r-1 {
		return nil, errorf(backend.OpFFT, backend.ErrShape, "axis %d of complex %v", axis, x.Shape)
	}
	if x.DType != backend.Float32 {
		return nil, errorf(backend.OpFFT, backend.ErrDType, "%s input", x.DType)
	}
	return newNode(&Node{Op: backend.OpFFT, DType: backend.Float32, Shape: x.Shape, Inputs: []*Node{x}, Attrs: backend.Attrs{Axis: axis, Inverse: inverse}})
}

// checkMatrices validates Float32 matrices of shape [B..., m, n], square
// if square is set.
func checkMatrices(op backend.Op, x *Node, square bool) error {
	r := len(x.Shape)
	if r < 2 || square && x.Shape[r-2] != x.Shape[r-1] {
		return errorf(op, backend.ErrShape, "matrices of shape %v", x.Shape)
	}
	if x.DType != backend.Float32 {
		return errorf(op, backend.ErrDType, "%s input", x.DType)
	}
	return nil
}

// Factor factorizes the Float32 matrices x, [B..., m, n], by op:
// OpCholesky, whose result is shaped like x, or OpLU, OpQR, OpSVD or
// OpEigh, whose one-dimensional result holds the parts that
// backend.FactorParts describes.
func Factor(op backend.Op, x *Node) (*Node, error) {
	if op == backend.OpCholesky {
		if err := checkMatrices(op, x, true); err != nil {
			return nil, err
		}
		return newNode(&Node{Op: op, DType: backend.Float32, Shape: x.Shape, Inputs: []*Node{x}})
	}
	if err := checkMatrices(op, x, op == backend.OpLU || op == backend.OpEigh); err != nil {
		return nil, err
	}
	_, size, err := backend.FactorParts(op, x.Shape)
	if err != nil {
		return nil, err
	}
	return newNode(&Node{Op: op, DType: backend.Float32, Shape: []int{size}, Inputs: []*Node{x}})
}

// TriangularSolve is X of a·X = b for the triangular Float32 matrices a,
// [B..., n, n], and b, [B..., n, k]; see backend.OpTriangularSolve.
func TriangularSolve(a, b *Node, upper, unit bool) (*Node, error) {
	if err := checkMatrices(backend.OpTriangularSolve, a, true); err != nil {
		return nil, err
	}
	if err := checkMatrices(backend.OpTriangularSolve, b, false); err != nil {
		return nil, err
	}
	r := len(a.Shape)
	if len(b.Shape) != r || !slices.Equal(a.Shape[:r-1], b.Shape[:r-1]) {
		return nil, errorf(backend.OpTriangularSolve, backend.ErrShape, "%v by %v", a.Shape, b.Shape)
	}
	return newNode(&Node{Op: backend.OpTriangularSolve, DType: backend.Float32, Shape: b.Shape, Inputs: []*Node{a, b}, Attrs: backend.Attrs{Upper: upper, Unit: unit}})
}
