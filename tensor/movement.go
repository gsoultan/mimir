package tensor

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
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
	return t.reshape(shape)
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
	return t.permute(p)
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
	return t.permute(perm)
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
	return t.broadcastTo(shape)
}

// Pad adds pads[d][0] elements before and pads[d][1] after each dimension d,
// filled with value. pads has one entry per dimension.
func (t *Tensor) Pad(pads [][2]int, value float64) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return t.pad(pads, value)
}

// Shrink keeps the half-open range [r[0], r[1]) of each dimension. It is a
// view; no memory is copied.
func (t *Tensor) Shrink(ranges [][2]int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return t.shrink(ranges)
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
	return t.flip(ax)
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
	return t.reshape(slices.Insert(slices.Clone(t.node.Shape), a, 1))
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
// batch dimensions broadcast. Half operands are multiplied in Float32 and
// the product returned in their dtype, or in Float32 when their dtypes
// differ.
func (t *Tensor) MatMul(u *Tensor) *Tensor {
	if err := firstErr(t, u); err != nil {
		return fail(err)
	}
	x, y := t, u
	if x.Rank() == 0 || y.Rank() == 0 {
		return fail(fmt.Errorf("tensor: matmul of scalar: %w", ErrShape))
	}
	vecX, vecY := x.Rank() == 1, y.Rank() == 1
	if vecX {
		x = x.reshape([]int{1, x.node.Shape[0]})
	}
	if vecY {
		y = y.reshape([]int{y.node.Shape[0], 1})
	}
	rx, ry := len(x.node.Shape), len(y.node.Shape)
	batch, err := broadcastShape(x.node.Shape[:rx-2], y.node.Shape[:ry-2])
	if err != nil {
		return fail(fmt.Errorf("tensor: matmul %v and %v: %w", t.node.Shape, u.node.Shape, ErrShape))
	}
	x = x.broadcastTo(slices.Concat(batch, x.node.Shape[rx-2:]))
	y = y.broadcastTo(slices.Concat(batch, y.node.Shape[ry-2:]))
	out := product(x, y)
	if out.err != nil {
		return fail(fmt.Errorf("tensor: matmul %v and %v: %w", t.node.Shape, u.node.Shape, ErrShape))
	}
	if !vecX && !vecY {
		return out
	}
	shape := slices.Clone(out.node.Shape)
	r := len(shape)
	switch {
	case vecX && vecY:
		shape = shape[:r-2]
	case vecX:
		shape = slices.Delete(shape, r-2, r-1)
	default:
		shape = shape[:r-1]
	}
	return out.reshape(shape)
}

// product multiplies matrices with equal batch dimensions, half operands
// in Float32, and returns the product in the operands' common dtype.
func product(x, y *Tensor) *Tensor {
	if err := firstErr(x, y); err != nil {
		return fail(err)
	}
	dx, dy := x.node.DType, y.node.DType
	if dx.Size() != 2 && dy.Size() != 2 || !dx.IsFloat() || !dy.IsFloat() {
		return matmulPrim(x, y)
	}
	out := matmulPrim(x.Cast(Float32), y.Cast(Float32))
	if dx != dy {
		return out
	}
	return out.Cast(dx)
}

// matmulPrim multiplies matrices with equal batch dimensions.
func matmulPrim(x, y *Tensor) *Tensor {
	if err := firstErr(x, y); err != nil {
		return fail(err)
	}
	out := wrap(graph.MatMul(x.node, y.node))
	a, b := x.detached(), y.detached()
	return track(out, func(g *Tensor) []*Tensor {
		return grads(g.MatMul(b.Transpose(-1, -2)), a.Transpose(-1, -2).MatMul(g))
	}, x, y)
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
	out := wrap(graph.Gather(t.node, idx.node, a))
	i, shape := idx.detached(), t.node.Shape
	return track(out, func(g *Tensor) []*Tensor {
		return grads(Zeros(g.node.DType, shape...).ScatterAdd(a, i, g), nil)
	}, t, idx)
}

// ScatterAdd returns a copy of t with src added at the Int32 positions in
// idx along axis, like torch.Tensor.scatter_add. idx and src share a shape.
// Repeated positions accumulate. t and src of different float dtypes are
// promoted to Float32.
func (t *Tensor) ScatterAdd(axis int, idx, src *Tensor) *Tensor {
	if err := firstErr(t, idx, src); err != nil {
		return fail(err)
	}
	a, err := normAxis(axis, len(t.node.Shape))
	if err != nil {
		return fail(err)
	}
	x, s := promote(t, src)
	out := wrap(graph.ScatterAdd(x.node, idx.node, s.node, a))
	i := idx.detached()
	return track(out, func(g *Tensor) []*Tensor { return grads(g, nil, g.Gather(a, i)) }, x, idx, s)
}

// The movement primitives below take validated arguments and record their
// gradients, which are the inverse movement.

func (t *Tensor) reshape(shape []int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if slices.Equal(shape, t.node.Shape) {
		return t
	}
	from := t.node.Shape
	out := wrap(graph.Reshape(t.node, shape))
	return track(out, func(g *Tensor) []*Tensor { return grads(g.reshape(from)) }, t)
}

// expand broadcasts size-1 dimensions to shape, which has t's rank.
// Its gradient sums over the broadcast dimensions.
func (t *Tensor) expand(shape []int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if slices.Equal(shape, t.node.Shape) {
		return t
	}
	out := wrap(graph.Expand(t.node, shape))
	if !tracked(t) || out.err != nil {
		return out
	}
	var axes []int
	for i, d := range t.node.Shape {
		if d == 1 && shape[i] != 1 {
			axes = append(axes, i)
		}
	}
	return track(out, func(g *Tensor) []*Tensor { return grads(g.reduce(backend.OpReduceSum, axes)) }, t)
}

func (t *Tensor) permute(perm []int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	out := wrap(graph.Permute(t.node, perm))
	if !tracked(t) || out.err != nil {
		return out
	}
	inv := make([]int, len(perm))
	for i, p := range perm {
		inv[p] = i
	}
	return track(out, func(g *Tensor) []*Tensor { return grads(g.permute(inv)) }, t)
}

func (t *Tensor) pad(pads [][2]int, value float64) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	out := wrap(graph.Pad(t.node, pads, value))
	if !tracked(t) || out.err != nil {
		return out
	}
	keep := make([][2]int, len(pads))
	for i, p := range pads {
		keep[i] = [2]int{p[0], p[0] + t.node.Shape[i]}
	}
	return track(out, func(g *Tensor) []*Tensor { return grads(g.shrink(keep)) }, t)
}

func (t *Tensor) shrink(ranges [][2]int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	out := wrap(graph.Shrink(t.node, ranges))
	if !tracked(t) || out.err != nil {
		return out
	}
	pads := make([][2]int, len(ranges))
	for i, r := range ranges {
		pads[i] = [2]int{r[0], t.node.Shape[i] - r[1]}
	}
	return track(out, func(g *Tensor) []*Tensor { return grads(g.pad(pads, 0)) }, t)
}

func (t *Tensor) flip(axes []int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	out := wrap(graph.Flip(t.node, axes))
	return track(out, func(g *Tensor) []*Tensor { return grads(g.flip(axes)) }, t)
}

// Flatten merges dimensions start through end, inclusive, into one, like
// torch.flatten. Negative axes count from the end.
func (t *Tensor) Flatten(start, end int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if len(t.node.Shape) == 0 {
		return t.reshape([]int{1})
	}
	s, errS := normAxis(start, len(t.node.Shape))
	e, errE := normAxis(end, len(t.node.Shape))
	if errS != nil || errE != nil || s > e {
		return fail(fmt.Errorf("tensor: flatten %d through %d of %v: %w", start, end, t.node.Shape, ErrShape))
	}
	n, _ := backend.NumElements(t.node.Shape[s : e+1])
	return t.reshape(slices.Concat(t.node.Shape[:s], []int{n}, t.node.Shape[e+1:]))
}

// Cat joins tensors along axis, like torch.cat. They must share a dtype and
// all dimensions but axis.
func Cat(axis int, ts ...*Tensor) *Tensor {
	if len(ts) == 0 {
		return fail(fmt.Errorf("tensor: cat of no tensors: %w", ErrShape))
	}
	if err := firstErr(ts...); err != nil {
		return fail(err)
	}
	first := ts[0].node
	a, err := normAxis(axis, len(first.Shape))
	if err != nil {
		return fail(err)
	}
	for _, t := range ts {
		s := t.node.Shape
		if len(s) != len(first.Shape) || t.node.DType != first.DType ||
			!slices.Equal(s[:a], first.Shape[:a]) || !slices.Equal(s[a+1:], first.Shape[a+1:]) {
			return fail(fmt.Errorf("tensor: cat %s%v and %s%v along %d: %w", first.DType, first.Shape, t.node.DType, s, a, ErrShape))
		}
	}
	return cat(a, ts)
}

// cat joins tensors that fit along axis a. A kernel joins at most
// backend.MaxCatInputs, so more are joined in groups, and the groups
// joined in turn: every element is copied once per level, and a level
// joins MaxCatInputs times as many as the one before.
func cat(a int, ts []*Tensor) *Tensor {
	for len(ts) > backend.MaxCatInputs {
		groups := (len(ts) + backend.MaxCatInputs - 1) / backend.MaxCatInputs
		joined := make([]*Tensor, 0, groups)
		for part := range slices.Chunk(ts, (len(ts)+groups-1)/groups) {
			joined = append(joined, catPrim(a, part))
		}
		ts = joined
	}
	return catPrim(a, ts)
}

// catPrim is the join primitive: each input's gradient is a view of the
// result's.
func catPrim(a int, ts []*Tensor) *Tensor {
	if err := firstErr(ts...); err != nil {
		return fail(err)
	}
	if len(ts) == 1 {
		return ts[0]
	}
	nodes := make([]*graph.Node, len(ts))
	for i, t := range ts {
		nodes[i] = t.node
	}
	out := wrap(graph.Cat(nodes, a))
	if !tracked(ts...) || out.err != nil {
		return out
	}
	shape := out.node.Shape
	ranges := make([][2]int, len(ts))
	at := 0
	for i, t := range ts {
		ranges[i] = [2]int{at, at + t.node.Shape[a]}
		at = ranges[i][1]
	}
	return track(out, func(g *Tensor) []*Tensor {
		gs := make([]*Tensor, len(ranges))
		for i, r := range ranges {
			gs[i] = g.shrink(rangeAxis(shape, a, r[0], r[1]))
		}
		return gs
	}, ts...)
}

// split cuts t along the axis a into parts of the given sizes, which add
// up to its length there, like torch.split. The parts are views of t.
// Backward joins their gradients into t's with one Cat, where the
// gradient of each view on its own would be padded to t's shape: a loop
// over the parts would pay for that once per part.
func (t *Tensor) split(a int, sizes []int) []*Tensor {
	parts := make([]*Tensor, len(sizes))
	if err := firstErr(t); err != nil {
		for i := range parts {
			parts[i] = fail(err)
		}
		return parts
	}
	x := t.detached()
	var s *splitting
	if t.tape != nil {
		s = &splitting{axis: a, shapes: make([][]int, len(sizes))}
	}
	at := 0
	for i, n := range sizes {
		p := x.shrink(rangeAxis(t.node.Shape, a, at, at+n))
		at += n
		if s != nil && p.err == nil {
			s.shapes[i] = p.node.Shape
			p = &Tensor{node: p.node, tape: newTape(p.node.Shape, []*tape{t.tape}, nil)}
			p.tape.part = &part{of: s, i: i}
		}
		parts[i] = p
	}
	return parts
}

// splitting is a tensor split into parts of shapes along axis.
type splitting struct {
	axis   int
	shapes [][]int
}

// part is part i of a splitting.
type part struct {
	of *splitting
	i  int
}

// joining collects the gradients of the parts of a splitting during
// propagate: grads[i] is part i's, or nil if it received none.
type joining struct {
	of    *splitting
	grads []*Tensor
}

// join returns the gradient of the split tensor: the parts' gradients
// joined, with zeros for the parts that received none. Gradients of
// different float dtypes join in Float32.
func (j *joining) join() *Tensor {
	var dt DType
	for _, g := range j.grads {
		switch {
		case g == nil:
		case dt == backend.Invalid:
			dt = g.node.DType
		case dt != g.node.DType:
			dt = Float32
		}
	}
	parts := make([]*Tensor, len(j.grads))
	for i, g := range j.grads {
		if g == nil {
			g = Zeros(dt, j.of.shapes[i]...)
		}
		parts[i] = g.Cast(dt)
	}
	return cat(j.of.axis, parts)
}

// Stack joins tensors of one shape along a new axis, like torch.stack.
func Stack(axis int, ts ...*Tensor) *Tensor {
	parts := make([]*Tensor, len(ts))
	for i, t := range ts {
		parts[i] = t.Unsqueeze(axis)
	}
	return Cat(axis, parts...)
}

// Unfold returns every slice of size elements along axis, starting step
// elements apart, like torch.Tensor.unfold: axis becomes the number of
// slices and a new last dimension of length size holds each slice.
//
// It copies t once, about size/step times over, and then reads the
// windows as views of the copy, so a convolution built on it is a matrix
// multiplication.
func (t *Tensor) Unfold(axis, size, step int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	a, err := normAxis(axis, len(t.node.Shape))
	if err != nil {
		return fail(err)
	}
	shape := t.node.Shape
	n := shape[a]
	if size < 1 || step < 1 || size > n {
		return fail(fmt.Errorf("tensor: unfold %v along %d with size %d and step %d: %w", shape, a, size, step, ErrShape))
	}
	o := (n-size)/step + 1
	before, after := shape[:a], shape[a+1:]
	at := func(dims ...int) []int { return slices.Concat(before, dims, after) }
	var w *Tensor // [before, size, o, after]
	if size <= step {
		// Windows do not overlap: cut the axis into o chunks of step and
		// keep the first size elements of each.
		x := t
		if o*step > n {
			x = x.pad(padAxis(len(shape), a, 0, o*step-n), 0)
		} else {
			x = x.shrink(rangeAxis(shape, a, 0, o*step))
		}
		w = x.reshape(at(o, step)).shrink(rangeAxis(at(o, step), a+1, 0, size)).
			permute(swapNext(len(shape)+1, a))
	} else {
		// Windows overlap. Repeat the axis r times and read the copy as
		// rows of n+1 elements: row j then starts at element j, so column i
		// of row j is element j+i, and window i is column i*step.
		r := (size*(n+1) + n - 1) / n
		x := t.reshape(at(1, n)).expand(at(r, n)).reshape(at(r * n))
		x = x.shrink(rangeAxis(at(r*n), a, 0, size*(n+1))).reshape(at(size, n+1))
		w = x.shrink(rangeAxis(at(size, n+1), a+1, 0, o*step))
		if step > 1 {
			w = w.reshape(at(size, o, step)).shrink(rangeAxis(at(size, o, step), a+2, 0, 1)).reshape(at(size, o))
		}
	}
	// [before, size, o, after] -> [before, o, after, size]
	perm := make([]int, 0, len(shape)+1)
	for d := range a {
		perm = append(perm, d)
	}
	perm = append(perm, a+1)
	for d := a + 2; d <= len(shape); d++ {
		perm = append(perm, d)
	}
	return w.permute(append(perm, a))
}

// padAxis returns pads for a rank-r tensor that pad only axis.
func padAxis(r, axis, before, after int) [][2]int {
	pads := make([][2]int, r)
	pads[axis] = [2]int{before, after}
	return pads
}

// rangeAxis returns shrink ranges for shape that keep [lo, hi) of axis and
// everything of the other dimensions.
func rangeAxis(shape []int, axis, lo, hi int) [][2]int {
	ranges := make([][2]int, len(shape))
	for i, d := range shape {
		ranges[i] = [2]int{0, d}
	}
	ranges[axis] = [2]int{lo, hi}
	return ranges
}

// swapNext returns the rank-r permutation that swaps axis and axis+1.
func swapNext(r, axis int) []int {
	perm := make([]int, r)
	for i := range perm {
		perm[i] = i
	}
	perm[axis], perm[axis+1] = axis+1, axis
	return perm
}
