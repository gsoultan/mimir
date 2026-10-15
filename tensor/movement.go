package tensor

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/internal/graph"
)

// Reshape returns t with a new shape of the same size. One dimension may be
// -1 and is inferred. Reshaping never copies unless t is a strided view.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	shape = slices.Clone(shape)
	infer, known := -1, 1
	for i, d := range shape {
		switch {
		case d == -1 && infer < 0:
			infer = i
		case d < 0:
			return fail(fmt.Errorf("tensor: reshape %v to %v: %w", t.node.Shape, shape, ErrShape))
		default:
			known *= d
		}
	}
	if infer >= 0 {
		if known == 0 || t.node.Size()%known != 0 {
			return fail(fmt.Errorf("tensor: reshape %v to %v: %w", t.node.Shape, shape, ErrShape))
		}
		shape[infer] = t.node.Size() / known
	}
	if slices.Equal(shape, t.node.Shape) {
		return t
	}
	return wrap(graph.Reshape(t.node, shape))
}

// Permute reorders dimensions: dimension i of the result is dimension
// perm[i] of t. Negative entries count from the end.
func (t *Tensor) Permute(perm ...int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	p := make([]int, len(perm))
	for i, a := range perm {
		var err error
		if p[i], err = normAxis(a, len(t.node.Shape)); err != nil {
			return fail(err)
		}
	}
	return wrap(graph.Permute(t.node, p))
}

// Transpose swaps dimensions a and b.
func (t *Tensor) Transpose(a, b int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	r := len(t.node.Shape)
	a, errA := normAxis(a, r)
	b, errB := normAxis(b, r)
	if errA != nil || errB != nil {
		return fail(fmt.Errorf("tensor: transpose axes for rank %d: %w", r, ErrShape))
	}
	perm := make([]int, r)
	for i := range perm {
		perm[i] = i
	}
	perm[a], perm[b] = b, a
	return wrap(graph.Permute(t.node, perm))
}

// Expand broadcasts t to shape like torch.Tensor.expand: new dimensions are
// added on the left, size-1 dimensions repeat, and -1 keeps a dimension.
// No memory is copied.
func (t *Tensor) Expand(shape ...int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	shape = slices.Clone(shape)
	extra := len(shape) - len(t.node.Shape)
	if extra < 0 {
		return fail(fmt.Errorf("tensor: expand %v to %v: %w", t.node.Shape, shape, ErrShape))
	}
	for i := extra; i < len(shape); i++ {
		if shape[i] == -1 {
			shape[i] = t.node.Shape[i-extra]
		}
	}
	return wrap(expandTo(t.node, shape))
}

// Pad adds pads[d][0] elements before and pads[d][1] after each dimension d,
// filled with value. pads has one entry per dimension.
func (t *Tensor) Pad(pads [][2]int, value float64) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return wrap(graph.Pad(t.node, pads, value))
}

// Shrink keeps the half-open range [r[0], r[1]) of each dimension. It is a
// view; no memory is copied.
func (t *Tensor) Shrink(ranges [][2]int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return wrap(graph.Shrink(t.node, ranges))
}

// Flip reverses the order of elements along axes. It is a view.
func (t *Tensor) Flip(axes ...int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	ax := make([]int, len(axes))
	for i, a := range axes {
		var err error
		if ax[i], err = normAxis(a, len(t.node.Shape)); err != nil {
			return fail(err)
		}
	}
	return wrap(graph.Flip(t.node, ax))
}

// Unsqueeze inserts a dimension of size 1 at axis, which may be in
// [-rank-1, rank].
func (t *Tensor) Unsqueeze(axis int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	a, err := normAxis(axis, len(t.node.Shape)+1)
	if err != nil {
		return fail(err)
	}
	return wrap(graph.Reshape(t.node, slices.Insert(slices.Clone(t.node.Shape), a, 1)))
}

// Squeeze removes the given dimensions, which must have size 1. With no
// axes it removes every dimension of size 1.
func (t *Tensor) Squeeze(axes ...int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	shape := t.node.Shape
	var drop []int
	if len(axes) == 0 {
		for d, s := range shape {
			if s == 1 {
				drop = append(drop, d)
			}
		}
	} else {
		var err error
		if drop, err = normAxes(axes, len(shape)); err != nil {
			return fail(err)
		}
		for _, d := range drop {
			if shape[d] != 1 {
				return fail(fmt.Errorf("tensor: squeeze axis %d of %v: %w", d, shape, ErrShape))
			}
		}
	}
	return t.dropAxes(drop)
}

// MatMul returns the matrix product with NumPy semantics: 1-D operands are
// promoted to matrices and the added dimension removed afterwards, and
// batch dimensions broadcast.
func (t *Tensor) MatMul(u *Tensor) *Tensor {
	if err := firstErr(t, u); err != nil {
		return fail(err)
	}
	x, y := t.node, u.node
	if len(x.Shape) == 0 || len(y.Shape) == 0 {
		return fail(fmt.Errorf("tensor: matmul of scalar: %w", ErrShape))
	}
	vecX, vecY := len(x.Shape) == 1, len(y.Shape) == 1
	var err error
	if vecX {
		if x, err = graph.Reshape(x, []int{1, x.Shape[0]}); err != nil {
			return fail(err)
		}
	}
	if vecY {
		if y, err = graph.Reshape(y, []int{y.Shape[0], 1}); err != nil {
			return fail(err)
		}
	}
	rx, ry := len(x.Shape), len(y.Shape)
	batch, err := broadcastShape(x.Shape[:rx-2], y.Shape[:ry-2])
	if err != nil {
		return fail(fmt.Errorf("tensor: matmul %v and %v: %w", t.node.Shape, u.node.Shape, ErrShape))
	}
	if x, err = expandTo(x, slices.Concat(batch, x.Shape[rx-2:])); err != nil {
		return fail(err)
	}
	if y, err = expandTo(y, slices.Concat(batch, y.Shape[ry-2:])); err != nil {
		return fail(err)
	}
	out, err := graph.MatMul(x, y)
	if err != nil {
		return fail(fmt.Errorf("tensor: matmul %v and %v: %w", t.node.Shape, u.node.Shape, ErrShape))
	}
	if !vecX && !vecY {
		return wrap(out, nil)
	}
	shape := slices.Clone(out.Shape)
	r := len(shape)
	switch {
	case vecX && vecY:
		shape = shape[:r-2]
	case vecX:
		shape = slices.Delete(shape, r-2, r-1)
	default:
		shape = shape[:r-1]
	}
	return wrap(graph.Reshape(out, shape))
}

// Gather reads t along axis at the Int32 positions in idx, like
// torch.gather. The result has idx's shape, which must fit t outside axis.
// Positions outside the axis fail with ErrIndex when computed.
func (t *Tensor) Gather(axis int, idx *Tensor) *Tensor {
	if err := firstErr(t, idx); err != nil {
		return fail(err)
	}
	a, err := normAxis(axis, len(t.node.Shape))
	if err != nil {
		return fail(err)
	}
	return wrap(graph.Gather(t.node, idx.node, a))
}

// ScatterAdd returns a copy of t with src added at the Int32 positions in
// idx along axis, like torch.Tensor.scatter_add. idx and src share a shape.
// Repeated positions accumulate.
func (t *Tensor) ScatterAdd(axis int, idx, src *Tensor) *Tensor {
	if err := firstErr(t, idx, src); err != nil {
		return fail(err)
	}
	a, err := normAxis(axis, len(t.node.Shape))
	if err != nil {
		return fail(err)
	}
	return wrap(graph.ScatterAdd(t.node, idx.node, src.node, a))
}
