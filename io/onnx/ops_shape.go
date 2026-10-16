package onnx

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// maxIndices bounds the positions along one axis that an operator lists
// on the host to gather, as Resize and Pad do.
const maxIndices = 1 << 24

// strides returns the row-major strides of shape.
func strides(shape []int) []int {
	s := make([]int, len(shape))
	n := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = n
		n *= shape[i]
	}
	return s
}

// forEach calls f with each position of shape in row-major order.
func forEach(shape []int, f func(pos []int)) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	pos := make([]int, len(shape))
	for range n {
		f(pos)
		for d := len(shape) - 1; d >= 0; d-- {
			if pos[d]++; pos[d] < shape[d] {
				break
			}
			pos[d] = 0
		}
	}
}

// pick returns the elements of host value x at the flat positions idx,
// shaped shape.
func pick(x *value, shape []int, idx []int) *value {
	o := &value{typ: x.typ, shape: shape, host: true}
	if x.typ.IsFloat() {
		o.floats = make([]float64, len(idx))
		for i, j := range idx {
			o.floats[i] = x.floats[j]
		}
	} else {
		o.ints = make([]int64, len(idx))
		for i, j := range idx {
			o.ints[i] = x.ints[j]
		}
	}
	return o
}

// reshaped returns x with another shape of as many elements.
func reshaped(x *value, shape []int) *value {
	if x.host {
		return &value{typ: x.typ, shape: shape, host: true, ints: x.ints, floats: x.floats, constant: x.constant}
	}
	return out(x.typ, x.t.Reshape(shape...))
}

// reshape returns c's input 0 reshaped to shape, on the host or not.
func (c *call) reshape(shape []int) []*value {
	x := c.in[0]
	if x.host {
		return one1(reshaped(x, shape))
	}
	return one1(out(x.typ, c.t(0).Reshape(shape...)))
}

func shapeOp(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	shape := c.in[0].shape
	r := len(shape)
	lo, hi := c.int("start", 0), c.int("end", int64(r))
	clamp := func(a int64) int {
		if a < 0 {
			a += int64(r)
		}
		return int(min(max(a, 0), int64(r)))
	}
	s, e := clamp(lo), clamp(hi)
	dims := []int64{}
	for _, d := range shape[s:max(s, e)] {
		dims = append(dims, int64(d))
	}
	return one1(hostInts(Int64, []int{len(dims)}, dims)), nil
}

func sizeOp(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	return one1(hostInts(Int64, []int{}, []int64{int64(c.in[0].size())})), nil
}

func constant(c *call) ([]*value, error) {
	for _, a := range c.n.attrs {
		switch a.name {
		case "value":
			if a.typ != attrTensor || a.t == nil {
				return nil, fmt.Errorf("%w: Constant value of attribute type %d", ErrFormat, a.typ)
			}
			v, err := c.m.constant(a.t, c.m.dir, false)
			if err != nil {
				return nil, err
			}
			return one1(v), nil
		case "value_float":
			return one1(hostFloats(Float, []int{}, []float64{c.float(a.name, 0)})), nil
		case "value_floats":
			f, _ := c.floats(a.name)
			o := make([]float64, len(f))
			for i, x := range f {
				o[i] = float64(x)
			}
			return one1(hostFloats(Float, []int{len(o)}, o)), nil
		case "value_int":
			return one1(hostInts(Int64, []int{}, []int64{c.int(a.name, 0)})), nil
		case "value_ints":
			v, _ := c.ints(a.name)
			return one1(hostInts(Int64, []int{len(v)}, slices.Clone(v))), nil
		case "sparse_value", "value_string", "value_strings":
			return nil, c.unsupported("attribute %s", a.name)
		}
	}
	return nil, fmt.Errorf("%w: Constant without a value", ErrFormat)
}

// shapeArg returns input i, a list of dimensions, which must be at least
// 0 and hold at most limit elements together. Dimensions of -1 and 0
// are let through for Reshape to resolve if loose.
func (c *call) shapeArg(i int, loose bool) ([]int, error) {
	list, ok, err := c.intList(i)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("%w: %s without a shape", ErrFormat, c.n.op)
	}
	shape := make([]int, len(list))
	for j, d := range list {
		if d < 0 && !(loose && d == -1) || d > math.MaxInt32 {
			return nil, fmt.Errorf("onnx: %s to dimensions %v: %w", c.n.op, list, tensor.ErrShape)
		}
		shape[j] = int(d)
	}
	return shape, nil
}

func constantOfShape(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	shape, err := c.shapeArg(0, false)
	if err != nil {
		return nil, err
	}
	fill := hostFloats(Float, []int{1}, []float64{0})
	if a := c.attr("value", attrTensor); a != nil && a.t != nil {
		if fill, err = c.m.constant(a.t, "", false); err != nil {
			return nil, err
		}
		if fill.size() != 1 || !fill.host {
			return nil, fmt.Errorf("%w: ConstantOfShape value of shape %v", ErrFormat, fill.shape)
		}
	}
	n, ok := numElements(shape, hostLimit)
	if !ok {
		dt, err := dtypeOf(fill.typ)
		if err != nil {
			return nil, err
		}
		return one1(&value{typ: fill.typ, shape: shape, t: tensor.Full(dt, fill.float(0), shape...)}), nil
	}
	return one1(pick(fill, shape, make([]int, n))), nil
}

func reshape(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	shape, err := c.shapeArg(1, true)
	if err != nil {
		return nil, err
	}
	in := c.in[0].shape
	allowZero := c.int("allowzero", 0) == 1
	infer, known := -1, 1
	for i, d := range shape {
		switch {
		case d == 0 && !allowZero:
			if i >= len(in) {
				return nil, fmt.Errorf("onnx: reshape %v to %v: %w", in, shape, tensor.ErrShape)
			}
			shape[i] = in[i]
			known *= in[i]
		case d == -1:
			if infer >= 0 {
				return nil, fmt.Errorf("onnx: reshape %v to %v: %w", in, shape, tensor.ErrShape)
			}
			infer = i
		default:
			if d != 0 && known > math.MaxInt32/d {
				return nil, fmt.Errorf("onnx: reshape %v to %v: %w", in, shape, tensor.ErrShape)
			}
			known *= d
		}
	}
	size := c.in[0].size()
	if infer >= 0 {
		if known == 0 || size%known != 0 {
			return nil, fmt.Errorf("onnx: reshape %v to %v: %w", in, shape, tensor.ErrShape)
		}
		shape[infer] = size / known
	} else if known != size {
		return nil, fmt.Errorf("onnx: reshape %v to %v: %w", in, shape, tensor.ErrShape)
	}
	return c.reshape(shape), nil
}

func flatten(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	in := c.in[0].shape
	axis := c.int("axis", 1)
	if axis < -int64(len(in)) || axis > int64(len(in)) {
		return nil, fmt.Errorf("onnx: flatten axis %d of %v: %w", axis, in, tensor.ErrShape)
	}
	if axis < 0 {
		axis += int64(len(in))
	}
	a, b := 1, 1
	for i, d := range in {
		if i < int(axis) {
			a *= d
		} else {
			b *= d
		}
	}
	return c.reshape([]int{a, b}), nil
}

// axesArg returns the axes of a node, from input i in newer opsets or
// from the attribute "axes" in older ones.
func (c *call) axesArg(i int) ([]int64, bool, error) {
	if a, ok := c.ints("axes"); ok {
		return a, true, nil
	}
	return c.intList(i)
}

func squeeze(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	in := c.in[0].shape
	axes, given, err := c.axesArg(1)
	if err != nil {
		return nil, err
	}
	drop := make([]bool, len(in))
	if !given {
		for d, s := range in {
			drop[d] = s == 1
		}
	} else {
		ax, err := normAxes(axes, len(in))
		if err != nil {
			return nil, err
		}
		for _, d := range ax {
			if in[d] != 1 {
				return nil, fmt.Errorf("onnx: squeeze axis %d of %v: %w", d, in, tensor.ErrShape)
			}
			drop[d] = true
		}
	}
	shape := []int{}
	for d, s := range in {
		if !drop[d] {
			shape = append(shape, s)
		}
	}
	return c.reshape(shape), nil
}

func unsqueeze(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	in := c.in[0].shape
	axes, given, err := c.axesArg(1)
	if err != nil {
		return nil, err
	}
	if !given {
		return nil, fmt.Errorf("%w: Unsqueeze without axes", ErrFormat)
	}
	rank := len(in) + len(axes)
	ax, err := normAxes(axes, rank)
	if err != nil {
		return nil, err
	}
	shape := make([]int, rank)
	for _, d := range ax {
		shape[d] = 1
	}
	j := 0
	for d := range shape {
		if !slices.Contains(ax, d) {
			shape[d] = in[j]
			j++
		}
	}
	return c.reshape(shape), nil
}

func transpose(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	r := len(x.shape)
	perm64, ok := c.ints("perm")
	if !ok {
		for i := r - 1; i >= 0; i-- {
			perm64 = append(perm64, int64(i))
		}
	}
	if len(perm64) != r {
		return nil, fmt.Errorf("onnx: transpose of %v by %v: %w", x.shape, perm64, tensor.ErrShape)
	}
	perm, err := normAxes(perm64, r)
	if err != nil {
		return nil, err
	}
	if !x.host {
		return one1(out(x.typ, c.t(0).Permute(perm...))), nil
	}
	shape := make([]int, r)
	for i, p := range perm {
		shape[i] = x.shape[p]
	}
	st := strides(x.shape)
	idx := make([]int, 0, x.size())
	forEach(shape, func(pos []int) {
		off := 0
		for i, p := range perm {
			off += pos[i] * st[p]
		}
		idx = append(idx, off)
	})
	return one1(pick(x, shape, idx)), nil
}

func concat(c *call) ([]*value, error) {
	if len(c.in) == 0 {
		return nil, fmt.Errorf("%w: Concat of no inputs", ErrFormat)
	}
	if err := c.need(len(c.in)); err != nil {
		return nil, err
	}
	first := c.in[0]
	axis, err := normAxis(c.int("axis", 0), len(first.shape))
	if err != nil {
		return nil, err
	}
	shape := slices.Clone(first.shape)
	shape[axis] = 0
	for _, v := range c.in {
		if len(v.shape) != len(shape) {
			return nil, fmt.Errorf("onnx: concat of ranks %d and %d: %w", len(first.shape), len(v.shape), tensor.ErrShape)
		}
		for d := range shape {
			if d != axis && v.shape[d] != first.shape[d] {
				return nil, fmt.Errorf("onnx: concat of %v and %v on axis %d: %w", first.shape, v.shape, axis, tensor.ErrShape)
			}
		}
		shape[axis] += v.shape[axis]
	}
	if _, ok := numElements(shape, hostLimit); ok && c.allHost() {
		o := &value{typ: first.typ, shape: shape, host: true}
		outer := 1
		for _, d := range shape[:axis] {
			outer *= d
		}
		for i := range outer {
			for _, v := range c.in {
				chunk := v.size() / max(outer, 1)
				if v.typ.IsFloat() {
					o.floats = append(o.floats, v.floats[i*chunk:(i+1)*chunk]...)
				} else {
					o.ints = append(o.ints, v.ints[i*chunk:(i+1)*chunk]...)
				}
			}
		}
		if first.typ.IsFloat() && o.floats == nil {
			o.floats = []float64{}
		}
		return one1(o), nil
	}
	var ts []*tensor.Tensor
	for i, v := range c.in {
		if v.shape[axis] > 0 {
			ts = append(ts, c.t(i))
		}
	}
	if len(ts) == 0 {
		return one1(out(first.typ, c.t(0))), nil
	}
	return one1(out(first.typ, tensor.Cat(axis, ts...))), nil
}

func split(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	axis, err := normAxis(c.int("axis", 0), len(x.shape))
	if err != nil {
		return nil, err
	}
	d := x.shape[axis]
	sizes64, ok := c.ints("split")
	if !ok {
		if sizes64, ok, err = c.intList(1); err != nil {
			return nil, err
		}
	}
	var sizes []int
	if ok {
		total := int64(0)
		for _, s := range sizes64 {
			if s < 0 || s > int64(d) {
				return nil, fmt.Errorf("onnx: split %d into %v: %w", d, sizes64, tensor.ErrShape)
			}
			total += s
			sizes = append(sizes, int(s))
		}
		if total != int64(d) {
			return nil, fmt.Errorf("onnx: split %d into %v: %w", d, sizes64, tensor.ErrShape)
		}
	} else {
		k := int(c.int("num_outputs", int64(len(c.n.outputs))))
		if k <= 0 || k > max(d, 1) {
			return nil, fmt.Errorf("onnx: split %d into %d: %w", d, k, tensor.ErrShape)
		}
		// Equal parts, the last smaller if they do not divide d.
		part := (d + k - 1) / k
		if c.int("num_outputs", 0) == 0 && d%k != 0 {
			return nil, fmt.Errorf("onnx: split %d into %d equal parts: %w", d, k, tensor.ErrShape)
		}
		for i := range k {
			sizes = append(sizes, max(min(part, d-i*part), 0))
		}
	}
	outs := make([]*value, len(sizes))
	lo := 0
	for i, s := range sizes {
		v, err := sliceAxis(c, x, axis, lo, lo+s)
		if err != nil {
			return nil, err
		}
		outs[i] = v
		lo += s
	}
	return outs, nil
}

// sliceAxis returns x's elements lo to hi along axis.
func sliceAxis(c *call, x *value, axis, lo, hi int) (*value, error) {
	idx := make([]int, hi-lo)
	for i := range idx {
		idx[i] = lo + i
	}
	return take(c, x, axis, idx)
}

// take returns the elements of x at positions idx, which lie in the
// axis, along axis. On a device a range of positions is a view; others
// gather.
func take(c *call, x *value, axis int, idx []int) (*value, error) {
	shape := slices.Clone(x.shape)
	shape[axis] = len(idx)
	if n, ok := numElements(shape, hostLimit); ok && x.host {
		st := strides(x.shape)
		pos := make([]int, 0, n)
		forEach(shape, func(p []int) {
			off := 0
			for d, i := range p {
				if d == axis {
					i = idx[i]
				}
				off += i * st[d]
			}
			pos = append(pos, off)
		})
		return pick(x, shape, pos), nil
	}
	t, err := takeTensor(c, c.tensorOf(x), axis, idx)
	if err != nil {
		return nil, err
	}
	return out(x.typ, t), nil
}

// tensorOf returns v as a tensor on the run's device, or a tensor that
// carries the error that prevented it.
func (c *call) tensorOf(v *value) *tensor.Tensor {
	t, err := c.tensor(v)
	if err != nil {
		return tensor.Failed(err)
	}
	return t
}

// takeTensor returns the elements of t at positions idx along axis.
func takeTensor(c *call, t *tensor.Tensor, axis int, idx []int) (*tensor.Tensor, error) {
	shape := t.Shape()
	contiguous := true
	for i := 1; i < len(idx); i++ {
		contiguous = contiguous && idx[i] == idx[i-1]+1
	}
	if contiguous && len(idx) > 0 {
		r := whole(shape)
		r[axis] = [2]int{idx[0], idx[0] + len(idx)}
		return t.Shrink(r), nil
	}
	reversed := len(idx) > 0
	for i := 1; i < len(idx); i++ {
		reversed = reversed && idx[i] == idx[i-1]-1
	}
	if reversed {
		r := whole(shape)
		r[axis] = [2]int{idx[len(idx)-1], idx[0] + 1}
		return t.Shrink(r).Flip(axis), nil
	}
	ints := make([]int64, len(idx))
	for i, j := range idx {
		ints[i] = int64(j)
	}
	iv := hostInts(Int64, []int{len(ints)}, ints)
	iv.key = c.key("take")
	it, err := c.tensor(iv)
	if err != nil {
		return nil, err
	}
	return gatherAxis(t, it, axis), nil
}

// key returns a name for a value a node computes for itself, such as the
// indices of a gather, unique among the values of the graph.
func (c *call) key(what string) string {
	c.keys++
	return fmt.Sprintf("%s\x00%s%d", c.n.outputs[0], what, c.keys)
}

// whole returns shrink ranges that keep every element of shape.
func whole(shape []int) [][2]int {
	r := make([][2]int, len(shape))
	for i, d := range shape {
		r[i] = [2]int{0, d}
	}
	return r
}

// gatherAxis returns t's elements at the Int32 positions idx, of any
// shape, along axis, like numpy.take: idx's dimensions replace axis.
func gatherAxis(t, idx *tensor.Tensor, axis int) *tensor.Tensor {
	shape, ishape := t.Shape(), idx.Shape()
	p, q := 1, 1
	for d, s := range shape {
		if d < axis {
			p *= s
		} else if d > axis {
			q *= s
		}
	}
	n := idx.Size()
	x := t.Reshape(p, shape[axis], q)
	ix := idx.Reshape(1, n, 1).Expand(p, n, q)
	o := slices.Concat(shape[:axis], ishape, shape[axis+1:])
	return x.Gather(1, ix).Reshape(o...)
}

func slice(c *call) ([]*value, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	x := c.in[0]
	r := len(x.shape)
	starts, _, err := c.intList(1)
	if err != nil {
		return nil, err
	}
	ends, _, err := c.intList(2)
	if err != nil {
		return nil, err
	}
	axes, given, err := c.intList(3)
	if err != nil {
		return nil, err
	}
	if !given {
		for i := range starts {
			axes = append(axes, int64(i))
		}
	}
	steps, given, err := c.intList(4)
	if err != nil {
		return nil, err
	}
	if !given {
		steps = slices.Repeat([]int64{1}, len(starts))
	}
	if len(ends) != len(starts) || len(axes) != len(starts) || len(steps) != len(starts) {
		return nil, fmt.Errorf("%w: Slice of %d starts, %d ends, %d axes and %d steps", ErrFormat, len(starts), len(ends), len(axes), len(steps))
	}
	ax, err := normAxes(axes, r)
	if err != nil {
		return nil, err
	}
	v := x
	for i, a := range ax {
		d := int64(x.shape[a])
		lo, hi, step := starts[i], ends[i], steps[i]
		if step == 0 {
			return nil, fmt.Errorf("onnx: slice with step 0: %w", tensor.ErrShape)
		}
		if lo < 0 {
			lo += d
		}
		if hi < 0 {
			hi += d
		}
		var idx []int
		if step > 0 {
			lo, hi = min(max(lo, 0), d), min(max(hi, 0), d)
			for j := lo; j < hi; j += step {
				idx = append(idx, int(j))
			}
		} else {
			lo, hi = min(max(lo, 0), d-1), min(max(hi, -1), d-1)
			for j := lo; j > hi; j += step {
				idx = append(idx, int(j))
			}
		}
		if len(idx) == int(d) && step == 1 {
			continue
		}
		if v, err = take(c, v, a, idx); err != nil {
			return nil, err
		}
	}
	return one1(v), nil
}

func gather(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x, iv := c.in[0], c.in[1]
	axis, err := normAxis(c.int("axis", 0), len(x.shape))
	if err != nil {
		return nil, err
	}
	d := x.shape[axis]
	if iv.host {
		idx := make([]int, iv.size())
		for i := range idx {
			j := iv.int(i)
			if j < 0 {
				j += int64(d)
			}
			if j < 0 || j >= int64(d) {
				return nil, fmt.Errorf("onnx: gather index %d of %d: %w", iv.int(i), d, tensor.ErrIndex)
			}
			idx[i] = int(j)
		}
		v, err := take(c, x, axis, idx)
		if err != nil {
			return nil, err
		}
		// The indices' dimensions replace the axis.
		shape := slices.Concat(x.shape[:axis], iv.shape, x.shape[axis+1:])
		return one1(reshaped(v, shape)), nil
	}
	idx := c.t(1)
	idx = tensor.Where(idx.Less(zero(idx)), idx.Add(scalarOf(idx, float64(d))), idx)
	return one1(out(x.typ, gatherAxis(c.t(0), idx, axis))), nil
}

func gatherElements(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	axis, err := normAxis(c.int("axis", 0), len(x.shape))
	if err != nil {
		return nil, err
	}
	idx := c.t(1)
	idx = tensor.Where(idx.Less(zero(idx)), idx.Add(scalarOf(idx, float64(x.shape[axis]))), idx)
	return one1(out(x.typ, c.t(0).Gather(axis, idx))), nil
}

func expand(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	to, err := c.shapeArg(1, false)
	if err != nil {
		return nil, err
	}
	shape, err := broadcastShape(x.shape, to)
	if err != nil {
		return nil, err
	}
	if _, ok := numElements(shape, hostLimit); ok && x.host {
		return one1(pick(x, shape, broadcastIndex(x.shape, shape))), nil
	}
	return one1(out(x.typ, c.t(0).Expand(shape...))), nil
}

func tile(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	reps, err := c.shapeArg(1, false)
	if err != nil {
		return nil, err
	}
	if len(reps) != len(x.shape) {
		return nil, fmt.Errorf("onnx: tile %v by %v: %w", x.shape, reps, tensor.ErrShape)
	}
	// [d0, d1] tiled by [r0, r1] is [1, d0, 1, d1] expanded to
	// [r0, d0, r1, d1].
	split, expanded, shape := []int{}, []int{}, []int{}
	for i, d := range x.shape {
		split = append(split, 1, d)
		expanded = append(expanded, reps[i], d)
		if reps[i] != 0 && d > math.MaxInt32/reps[i] {
			return nil, fmt.Errorf("onnx: tile %v by %v: %w", x.shape, reps, tensor.ErrShape)
		}
		shape = append(shape, reps[i]*d)
	}
	if _, ok := numElements(shape, hostLimit); ok && x.host {
		return one1(pick(x, shape, broadcastIndex(split, expanded))), nil
	}
	return one1(out(x.typ, c.t(0).Reshape(split...).Expand(expanded...).Reshape(shape...))), nil
}

func rangeOp(c *call) ([]*value, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	var s [3]*value
	for i := range s {
		v, err := c.host(i)
		if err != nil {
			return nil, err
		}
		if v.size() != 1 {
			return nil, fmt.Errorf("onnx: range bound of shape %v: %w", v.shape, tensor.ErrShape)
		}
		s[i] = v
	}
	typ := s[0].typ
	start, limit, delta := s[0].float(0), s[1].float(0), s[2].float(0)
	if delta == 0 {
		return nil, fmt.Errorf("onnx: range with delta 0: %w", tensor.ErrShape)
	}
	span := math.Ceil((limit - start) / delta)
	if math.IsNaN(span) || span > math.MaxInt32 {
		return nil, fmt.Errorf("onnx: range(%v, %v, %v): %w", start, limit, delta, tensor.ErrShape)
	}
	n := max(int(span), 0)
	if n > hostLimit {
		// A long range is computed where it is used, as a running sum of
		// ones: start + i·delta.
		i := tensor.Full(tensor.Int32, 1, n).CumSum(0).Sub(tensor.Full(tensor.Int32, 1))
		dt, err := dtypeOf(typ)
		if err != nil {
			return nil, err
		}
		if !typ.IsFloat() {
			if start < math.MinInt32 || start > math.MaxInt32 || delta < math.MinInt32 || delta > math.MaxInt32 {
				return nil, fmt.Errorf("onnx: range(%v, %v, %v) of int32s: %w", start, limit, delta, tensor.ErrDType)
			}
			return one1(out(typ, i.Mul(tensor.Full(dt, delta)).Add(tensor.Full(dt, start)))), nil
		}
		return one1(out(typ, i.Cast(dt).Mul(tensor.Full(dt, delta)).Add(tensor.Full(dt, start)))), nil
	}
	if typ.IsFloat() {
		o := make([]float64, n)
		for i := range o {
			o[i] = start + float64(i)*delta
		}
		return one1(hostFloats(typ, []int{n}, o)), nil
	}
	o := make([]int64, n)
	a, b := s[0].int(0), s[2].int(0)
	for i := range o {
		o[i] = a + int64(i)*b
	}
	return one1(hostInts(typ, []int{n}, o)), nil
}

func pad(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	r := len(x.shape)
	pads, _, err := c.intList(1)
	if err != nil {
		return nil, err
	}
	axes, given, err := c.intList(3)
	if err != nil {
		return nil, err
	}
	if !given {
		for i := range r {
			axes = append(axes, int64(i))
		}
	}
	ax, err := normAxes(axes, r)
	if err != nil {
		return nil, err
	}
	if len(pads) != 2*len(ax) {
		return nil, fmt.Errorf("onnx: %d pads for %d axes: %w", len(pads), len(ax), tensor.ErrShape)
	}
	p := make([][2]int, r)
	for i, a := range ax {
		lo, hi := pads[i], pads[i+len(ax)]
		if lo < -int64(x.shape[a]) || hi < -int64(x.shape[a]) || lo > math.MaxInt32 || hi > math.MaxInt32 {
			return nil, fmt.Errorf("onnx: pads %v of %v: %w", pads, x.shape, tensor.ErrShape)
		}
		p[a] = [2]int{int(lo), int(hi)}
	}
	fill := 0.0
	if v := c.arg(2); v != nil {
		h, err := c.download(v)
		if err != nil {
			return nil, err
		}
		if h.size() > 0 {
			fill = h.float(0)
		}
	}
	mode := c.str("mode", "constant")
	if mode != "constant" && mode != "edge" && mode != "reflect" {
		return nil, c.unsupported("mode %q", mode)
	}
	v := x
	for a, pa := range p {
		if pa == [2]int{} {
			continue
		}
		v, err = padAxis(c, v, a, pa, mode, fill)
		if err != nil {
			return nil, err
		}
	}
	return one1(v), nil
}

// padAxis pads x along axis by pa, in the manner of mode. Negative pads
// remove elements.
func padAxis(c *call, x *value, axis int, pa [2]int, mode string, fill float64) (*value, error) {
	d := x.shape[axis]
	if d+pa[0]+pa[1] < 0 {
		return nil, fmt.Errorf("onnx: pads %v of %d elements: %w", pa, d, tensor.ErrShape)
	}
	if mode == "constant" && !x.host {
		t := c.tensorOf(x)
		if pa[0] < 0 || pa[1] < 0 {
			r := whole(x.shape)
			r[axis] = [2]int{max(-pa[0], 0), d - max(-pa[1], 0)}
			if r[axis][1] < r[axis][0] {
				return nil, fmt.Errorf("onnx: pads %v of %d elements: %w", pa, d, tensor.ErrShape)
			}
			t = t.Shrink(r)
		}
		pads := make([][2]int, len(x.shape))
		pads[axis] = [2]int{max(pa[0], 0), max(pa[1], 0)}
		return out(x.typ, t.Pad(pads, fill)), nil
	}
	if mode != "constant" && d+pa[0]+pa[1] > maxIndices {
		return nil, fmt.Errorf("onnx: %s pad of %d elements by %v: %w", mode, d, pa, tensor.ErrShape)
	}
	if d == 0 && mode != "constant" {
		return nil, fmt.Errorf("onnx: %s pad of an empty axis: %w", mode, tensor.ErrShape)
	}
	// Edge and reflect pads read the positions they copy; a constant
	// pad on the host reads position -1, the fill.
	idx := make([]int, 0, d+pa[0]+pa[1])
	for i := -pa[0]; i < d+pa[1]; i++ {
		j := i
		switch {
		case i >= 0 && i < d:
		case mode == "edge":
			j = min(max(i, 0), d-1)
		case mode == "reflect":
			if d == 1 {
				j = 0
				break
			}
			period := 2 * (d - 1)
			j = ((i % period) + period) % period
			if j >= d {
				j = period - j
			}
		default:
			j = -1
		}
		idx = append(idx, j)
	}
	if mode != "constant" {
		return take(c, x, axis, idx)
	}
	shape := slices.Clone(x.shape)
	shape[axis] = len(idx)
	st := strides(x.shape)
	o := &value{typ: x.typ, shape: shape, host: true}
	forEach(shape, func(p []int) {
		off := 0
		for dd, i := range p {
			if dd == axis {
				if i = idx[i]; i < 0 {
					off = -1
					break
				}
			}
			off += i * st[dd]
		}
		if x.typ.IsFloat() {
			f := fill
			if off >= 0 {
				f = x.floats[off]
			}
			o.floats = append(o.floats, f)
		} else {
			n := toInt(fill)
			if off >= 0 {
				n = x.ints[off]
			}
			o.ints = append(o.ints, n)
		}
	})
	return o, nil
}

func trilu(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	r := len(x.shape)
	if r < 2 {
		return nil, fmt.Errorf("onnx: trilu of %v: %w", x.shape, tensor.ErrShape)
	}
	k := int64(0)
	if c.arg(1) != nil {
		h, err := c.host(1)
		if err != nil {
			return nil, err
		}
		if h.size() != 1 {
			return nil, fmt.Errorf("onnx: trilu k of shape %v: %w", h.shape, tensor.ErrShape)
		}
		k = h.int(0)
	}
	upper := c.int("upper", 1) == 1
	n, m := x.shape[r-2], x.shape[r-1]
	keep := func(i, j int) bool {
		if upper {
			return int64(j)-int64(i) >= k
		}
		return int64(j)-int64(i) <= k
	}
	if x.host {
		o := &value{typ: x.typ, shape: x.shape, host: true}
		o.ints, o.floats = slices.Clone(x.ints), slices.Clone(x.floats)
		for i := range x.size() {
			if !keep(i/m%n, i%m) {
				if x.typ.IsFloat() {
					o.floats[i] = 0
				} else {
					o.ints[i] = 0
				}
			}
		}
		return one1(o), nil
	}
	// The mask compares positions, which take n+m elements on the host.
	rows, cols := make([]int64, n), make([]int64, m)
	for i := range rows {
		rows[i] = int64(i)
	}
	for j := range cols {
		cols[j] = int64(j)
	}
	rv, cv := hostInts(Int32, []int{n, 1}, rows), hostInts(Int32, []int{1, m}, cols)
	rv.key, cv.key = c.key("rows"), c.key("cols")
	rt, err := c.tensor(rv)
	if err != nil {
		return nil, err
	}
	ct, err := c.tensor(cv)
	if err != nil {
		return nil, err
	}
	diff := ct.Sub(rt) // j - i
	kt := tensor.Full(tensor.Int32, float64(min(max(k, -math.MaxInt32), math.MaxInt32)))
	var mask *tensor.Tensor
	if upper {
		mask = notT(diff.Less(kt))
	} else {
		mask = notT(diff.Greater(kt))
	}
	t := c.t(0)
	return one1(out(x.typ, tensor.Where(mask, t, zero(t)))), nil
}

func nonZero(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x, err := c.host(0)
	if err != nil {
		return nil, err
	}
	var pos [][]int
	i := 0
	forEach(x.shape, func(p []int) {
		if x.float(i) != 0 {
			pos = append(pos, slices.Clone(p))
		}
		i++
	})
	r := len(x.shape)
	o := make([]int64, r*len(pos))
	for j, p := range pos {
		for d := range r {
			o[d*len(pos)+j] = int64(p[d])
		}
	}
	return one1(hostInts(Int64, []int{r, len(pos)}, o)), nil
}
