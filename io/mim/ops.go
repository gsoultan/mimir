package mim

import (
	"fmt"
	"math"
	"slices"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/tensor"
)

// call is one op being run.
type call struct {
	r   *run
	op  *mimfile.Op
	in  []*tensor.Tensor
	err error // the first attribute error
}

// op runs op and stores its outputs, checked against their declared
// dtypes and shapes.
func (r *run) op(op *mimfile.Op) error {
	c := &call{r: r, op: op, in: make([]*tensor.Tensor, len(op.Inputs))}
	for i, vi := range op.Inputs {
		if c.in[i] = r.env[vi]; c.in[i] == nil {
			return fmt.Errorf("%w: %s reads %q before it is computed", ErrFormat, opName(op), valueName(r.m.g.Values[vi], vi))
		}
	}
	outs, err := c.eval()
	if err == nil {
		err = c.err
	}
	if err != nil {
		return c.wrap(err)
	}
	if len(outs) != len(op.Outputs) {
		return c.wrap(fmt.Errorf("%w: %d outputs, want %d", ErrFormat, len(op.Outputs), len(outs)))
	}
	for i, vi := range op.Outputs {
		v := r.m.g.Values[vi]
		t := outs[i]
		if err := t.Err(); err != nil {
			return c.wrap(err)
		}
		if dt, _ := storage(v.DType); t.DType() != dt {
			t = t.Cast(dt)
		}
		if err := r.bind(vi, t.Shape()); err != nil {
			return c.wrap(fmt.Errorf("output %q: %w", valueName(v, vi), err))
		}
		r.env[vi] = t
	}
	return nil
}

func (c *call) wrap(err error) error {
	out := ""
	if len(c.op.Outputs) > 0 {
		vi := c.op.Outputs[0]
		out = fmt.Sprintf(" (value %q)", valueName(c.r.m.g.Values[vi], vi))
	}
	return fmt.Errorf("mim: %s%s: %w", opName(c.op), out, err)
}

// opName names op, with the composite a call calls.
func opName(op *mimfile.Op) string {
	if op.Opcode != mimfile.OpCall {
		return op.Opcode.String()
	}
	name, version := "?", int64(0)
	for _, a := range op.Attrs {
		switch a.Key {
		case "name":
			name = a.String
		case "version":
			version = a.Int
		}
	}
	return fmt.Sprintf("call %s.v%d", name, version)
}

// dt returns the declared dtype of input i.
func (c *call) dt(i int) mimfile.DType { return c.r.m.g.Values[c.op.Inputs[i]].DType }

// outDT returns the declared dtype of output i.
func (c *call) outDT(i int) mimfile.DType { return c.r.m.g.Values[c.op.Outputs[i]].DType }

// outShape returns the declared shape of output i, with -1 for symbolic
// dimensions not yet bound.
func (c *call) outShape(i int) []int { return c.r.declared(c.op.Outputs[i]) }

// attr returns the attribute key, or nil.
func (c *call) attr(key string) *mimfile.Attr {
	for i := range c.op.Attrs {
		if c.op.Attrs[i].Key == key {
			return &c.op.Attrs[i]
		}
	}
	return nil
}

// int returns the int attribute key, or def if it is absent.
func (c *call) int(key string, def int64) int64 {
	a := c.attr(key)
	switch {
	case a == nil:
		return def
	case a.Type != mimfile.AttrInt:
		c.fail(fmt.Errorf("%w: attribute %s of type %d, want an int", ErrFormat, key, a.Type))
		return def
	}
	return a.Int
}

// ints returns the ints attribute key, or def if it is absent.
func (c *call) ints(key string, def []int64) []int64 {
	a := c.attr(key)
	switch {
	case a == nil:
		return def
	case a.Type != mimfile.AttrInts:
		c.fail(fmt.Errorf("%w: attribute %s of type %d, want ints", ErrFormat, key, a.Type))
		return def
	}
	return a.Ints
}

// float returns the float attribute key, which may be written as an int,
// or def if it is absent.
func (c *call) float(key string, def float64) float64 {
	a := c.attr(key)
	switch {
	case a == nil:
		return def
	case a.Type == mimfile.AttrInt:
		return float64(a.Int)
	case a.Type != mimfile.AttrFloat:
		c.fail(fmt.Errorf("%w: attribute %s of type %d, want a float", ErrFormat, key, a.Type))
		return def
	}
	return a.Float
}

// bool returns the bool attribute key, an int of 0 or 1, or def if it is
// absent.
func (c *call) bool(key string, def bool) bool {
	d := int64(0)
	if def {
		d = 1
	}
	switch c.int(key, d) {
	case 0:
		return false
	case 1:
		return true
	}
	c.fail(fmt.Errorf("%w: attribute %s is not 0 or 1", ErrFormat, key))
	return def
}

// str returns the string attribute key, or def if it is absent.
func (c *call) str(key, def string) string {
	a := c.attr(key)
	switch {
	case a == nil:
		return def
	case a.Type != mimfile.AttrString:
		c.fail(fmt.Errorf("%w: attribute %s of type %d, want a string", ErrFormat, key, a.Type))
		return def
	}
	return a.String
}

func (c *call) fail(err error) {
	if c.err == nil {
		c.err = err
	}
}

// axis returns the axis attribute key of a composite, which may count
// from the end, as an axis of a tensor of rank r.
func (c *call) axis(key string, def int64, r int) int {
	a := c.int(key, def)
	if a < 0 {
		a += int64(r)
	}
	if a < 0 || a >= int64(r) {
		c.fail(fmt.Errorf("%w: %s %d of a tensor of rank %d", ErrFormat, key, c.int(key, def), r))
		return 0
	}
	return int(a)
}

// full returns a tensor of value on the run's device.
func (c *call) full(dt tensor.DType, value float64, shape ...int) *tensor.Tensor {
	return tensor.Full(dt, value, shape...).To(c.r.dev)
}

// scalar returns value as a tensor of t's dtype and rank, of size 1, on
// the run's device.
func (c *call) scalar(t *tensor.Tensor, value float64) *tensor.Tensor {
	return c.full(t.DType(), value, slices.Repeat([]int{1}, t.Rank())...)
}

func (c *call) unsupported(detail string) error {
	return &UnsupportedError{Op: opName(c.op), Detail: detail}
}

// arity holds the number of inputs and outputs of each primitive.
var arity = map[mimfile.Opcode][2]int{
	mimfile.OpNeg: {1, 1}, mimfile.OpRecip: {1, 1}, mimfile.OpSqrt: {1, 1}, mimfile.OpExp2: {1, 1},
	mimfile.OpLog2: {1, 1}, mimfile.OpSin: {1, 1}, mimfile.OpTrunc: {1, 1}, mimfile.OpCast: {1, 1},
	mimfile.OpBitcast: {1, 1}, mimfile.OpAdd: {2, 1}, mimfile.OpMul: {2, 1}, mimfile.OpMax: {2, 1},
	mimfile.OpPow: {2, 1}, mimfile.OpIdiv: {2, 1}, mimfile.OpImod: {2, 1}, mimfile.OpShl: {2, 1},
	mimfile.OpShr: {2, 1}, mimfile.OpAnd: {2, 1}, mimfile.OpOr: {2, 1}, mimfile.OpXor: {2, 1},
	mimfile.OpCmplt: {2, 1}, mimfile.OpCmpeq: {2, 1}, mimfile.OpWhere: {3, 1}, mimfile.OpReduce: {1, 1},
	mimfile.OpReshape: {1, 1}, mimfile.OpExpand: {1, 1}, mimfile.OpPermute: {1, 1}, mimfile.OpPad: {1, 1},
	mimfile.OpShrink: {1, 1}, mimfile.OpGather: {2, 1}, mimfile.OpScatter: {3, 1}, mimfile.OpConst: {0, 1},
	mimfile.OpDequantize: {1, 1},
}

// check reports whether mimir runs op, before any run.
func (m *Model) check(op *mimfile.Op) error {
	ins, outs := len(op.Inputs), len(op.Outputs)
	if a, ok := arity[op.Opcode]; ok && (ins != a[0] || outs != a[1]) {
		return fmt.Errorf("%w: %s with %d inputs and %d outputs, want %d and %d", ErrFormat, op.Opcode, ins, outs, a[0], a[1])
	}
	switch op.Opcode {
	case mimfile.OpWhile:
		if ins != outs {
			return fmt.Errorf("%w: while with %d inputs and %d outputs", ErrFormat, ins, outs)
		}
	case mimfile.OpIf:
		if ins == 0 {
			return fmt.Errorf("%w: if without a condition", ErrFormat)
		}
	}
	switch op.Opcode {
	case mimfile.OpBitcast, mimfile.OpShl, mimfile.OpShr:
		return &UnsupportedError{Op: op.Opcode.String()}
	case mimfile.OpAnd, mimfile.OpOr, mimfile.OpXor:
		if dt := m.g.Values[op.Inputs[0]].DType; dt != mimfile.Bool {
			return &UnsupportedError{Op: op.Opcode.String(), Detail: "bitwise operations on " + dt.String()}
		}
	case mimfile.OpCast:
		for _, a := range op.Attrs {
			if a.Key == "to" && (a.DType == mimfile.F8E4M3 || a.DType == mimfile.F8E5M2) {
				return &UnsupportedError{Op: "cast", Detail: "to " + a.DType.String()}
			}
		}
	case mimfile.OpCall:
		name := opName(op)[len("call "):]
		if _, ok := composites[name]; ok {
			return nil
		}
		if _, ok := m.locals[name]; !ok {
			return &UnsupportedError{Op: opName(op)}
		}
	}
	return nil
}

// eval computes the op's outputs.
func (c *call) eval() ([]*tensor.Tensor, error) {
	in := c.in
	switch c.op.Opcode {
	case mimfile.OpNeg:
		return one(in[0].Neg())
	case mimfile.OpRecip:
		return one(in[0].Recip())
	case mimfile.OpSqrt:
		return one(in[0].Sqrt())
	case mimfile.OpExp2:
		return one(in[0].Exp2())
	case mimfile.OpLog2:
		return one(in[0].Log2())
	case mimfile.OpSin:
		return one(in[0].Sin())
	case mimfile.OpTrunc:
		if !isFloat(c.dt(0)) {
			return one(in[0])
		}
		return one(c.trunc(in[0]))
	case mimfile.OpCast:
		a := c.attr("to")
		if a == nil || a.Type != mimfile.AttrDType {
			return nil, fmt.Errorf("%w: cast without a dtype to cast to", ErrFormat)
		}
		return one(c.cast(in[0], c.dt(0), a.DType))
	case mimfile.OpAdd:
		return one(in[0].Add(in[1]))
	case mimfile.OpMul:
		return one(in[0].Mul(in[1]))
	case mimfile.OpMax:
		return one(in[0].Maximum(in[1]))
	case mimfile.OpPow:
		return one(c.pow(in[0], in[1]))
	case mimfile.OpIdiv, mimfile.OpImod:
		return one(c.idiv(in[0], in[1], c.op.Opcode == mimfile.OpImod))
	case mimfile.OpAnd:
		return one(in[0].Mul(in[1]))
	case mimfile.OpOr:
		return one(in[0].Maximum(in[1]))
	case mimfile.OpXor:
		return one(c.not(in[0].Equal(in[1])))
	case mimfile.OpCmplt:
		return one(in[0].Less(in[1]).Cast(tensor.Int32))
	case mimfile.OpCmpeq:
		return one(in[0].Equal(in[1]).Cast(tensor.Int32))
	case mimfile.OpWhere:
		return one(tensor.Where(in[0], in[1], in[2]))
	case mimfile.OpReduce:
		return one(c.reduce(in[0]))
	case mimfile.OpReshape:
		return one(in[0].Reshape(c.outShape(0)...))
	case mimfile.OpExpand:
		return one(in[0].Expand(c.outShape(0)...))
	case mimfile.OpPermute:
		return one(in[0].Permute(toInts(c.ints("perm", nil))...))
	case mimfile.OpPad:
		pads := c.ints("pads", nil)
		if len(pads) != 2*in[0].Rank() {
			return nil, fmt.Errorf("%w: %d pads for rank %d", ErrFormat, len(pads), in[0].Rank())
		}
		p := make([][2]int, in[0].Rank())
		for i := range p {
			p[i] = [2]int{int(pads[2*i]), int(pads[2*i+1])}
		}
		return one(in[0].Pad(p, c.float("value", 0)))
	case mimfile.OpShrink:
		begin, end := c.ints("begin", nil), c.ints("end", nil)
		shape := in[0].Shape()
		if len(begin) != len(shape) || len(end) != len(shape) {
			return nil, fmt.Errorf("%w: shrink of %d begins and %d ends for rank %d", ErrFormat, len(begin), len(end), len(shape))
		}
		ranges := make([][2]int, len(shape))
		for i := range ranges {
			e := int(end[i])
			if end[i] == -1 {
				e = shape[i]
			}
			ranges[i] = [2]int{int(begin[i]), e}
		}
		return one(in[0].Shrink(ranges))
	case mimfile.OpGather:
		return one(c.take(in[0], in[1], int(c.int("axis", 0))))
	case mimfile.OpScatter:
		return one(c.scatter(in[0], in[1], in[2], int(c.int("axis", 0))))
	case mimfile.OpConst:
		a := c.attr("value")
		if a == nil {
			return nil, fmt.Errorf("%w: const without a value", ErrFormat)
		}
		v := a.Float
		if a.Type == mimfile.AttrInt {
			v = float64(a.Int)
			if lo, hi := intRange(c.outDT(0)); isInt(c.outDT(0)) && (v < lo || v > hi) {
				return nil, &UnsupportedError{Op: "const", Detail: fmt.Sprintf("%s value %d does not fit in an int32", c.outDT(0), a.Int)}
			}
		}
		dt, _ := storage(c.outDT(0))
		return one(c.full(dt, v, c.outShape(0)...))
	case mimfile.OpDequantize:
		return one(in[0].Cast(tensor.Float32))
	case mimfile.OpCall:
		return c.composite()
	case mimfile.OpWhile:
		return c.while()
	case mimfile.OpIf:
		return c.ifElse()
	}
	return nil, c.unsupported("")
}

func one(t *tensor.Tensor) ([]*tensor.Tensor, error) { return []*tensor.Tensor{t}, t.Err() }

func toInts(v []int64) []int {
	out := make([]int, len(v))
	for i, x := range v {
		out[i] = int(x)
	}
	return out
}

// not returns 1 where the 0 or 1 of t is 0, as an Int32 bool.
func (c *call) not(t *tensor.Tensor) *tensor.Tensor {
	return c.scalar(t, 1).Sub(t).Cast(tensor.Int32)
}

// trunc rounds the floats of t toward zero. Floats of 2^23 and above are
// integers already, as are the infinities.
func (c *call) trunc(t *tensor.Tensor) *tensor.Tensor {
	big := t.Abs().Less(c.scalar(t, 1<<23)).Equal(c.scalar(t, 0))
	return tensor.Where(big, t, t.Sub(t.Fmod(c.scalar(t, 1))))
}

// cast converts t, of dtype from, to dtype to by the rules of the cast
// primitive.
func (c *call) cast(t *tensor.Tensor, from, to mimfile.DType) *tensor.Tensor {
	dt, ok := storage(to)
	switch {
	case !ok || to.IsBlock():
		return tensor.Failed(c.unsupported("to " + to.String()))
	case to == mimfile.Bool:
		return c.not(t.Equal(c.scalar(t, 0)))
	case from == mimfile.Bool, isFloat(to):
		return t.Cast(dt)
	case isFloat(from):
		// To an integer: toward zero, saturated, and NaN to 0.
		lo, hi := intRange(to)
		x := c.trunc(t.Cast(tensor.Float32))
		x = x.Maximum(c.scalar(x, lo)).Minimum(c.scalar(x, hi))
		x = tensor.Where(x.Equal(x), x, c.scalar(x, 0))
		return x.Cast(tensor.Int32)
	}
	// From an integer to an integer: modulo 2^w of the narrower types.
	var w float64
	switch to {
	case mimfile.I8, mimfile.U8:
		w = 1 << 8
	case mimfile.I16, mimfile.U16:
		w = 1 << 16
	default:
		return t
	}
	lo, _ := intRange(to)
	x := t.Sub(c.scalar(t, lo)).Fmod(c.scalar(t, w))
	x = tensor.Where(x.Less(c.scalar(x, 0)), x.Add(c.scalar(x, w)), x)
	return x.Add(c.scalar(x, lo))
}

// pow returns a to the power b. An exponent that a const op computes, as
// in x², is applied by products where it can.
func (c *call) pow(a, b *tensor.Tensor) *tensor.Tensor {
	if e, ok := c.r.m.constOf(c.op.Inputs[1]); ok && slices.Equal(b.Shape(), slices.Repeat([]int{1}, b.Rank())) {
		switch e {
		case 1:
			return a
		case 2:
			return a.Mul(a)
		case 3:
			return a.Mul(a).Mul(a)
		case 0.5:
			return a.Sqrt()
		case -0.5:
			return a.Sqrt().Recip()
		case -1:
			return a.Recip()
		case -2:
			return a.Mul(a).Recip()
		}
	}
	// |a|^b, with the sign of a for odd integers b, NaN for negative a and
	// other b, and 1 where b is 0 or a is 1, unless a is NaN.
	zero, one := c.scalar(a, 0), c.scalar(a, 1)
	p := a.Abs().Log().Mul(b).Exp()
	integral := b.Fmod(one).Equal(zero)
	odd := b.Abs().Fmod(c.scalar(b, 2)).Equal(one)
	neg := tensor.Where(integral, tensor.Where(odd, p.Neg(), p), c.scalar(a, math.NaN()))
	p = tensor.Where(a.Less(zero), neg, p)
	unit := b.Equal(zero).Add(a.Equal(one)).Mul(a.Equal(a))
	return tensor.Where(unit, one, p)
}

// idiv returns a / b toward zero, or with rem the remainder, which has the
// sign of a. Division by zero gives 0. The quotient is exact for values
// below 2^24.
func (c *call) idiv(a, b *tensor.Tensor, rem bool) *tensor.Tensor {
	zero := c.scalar(b, 0)
	byZero := b.Equal(zero)
	b = tensor.Where(byZero, c.scalar(b, 1), b)
	r := a.Fmod(b)
	if !rem {
		// a - r is a multiple of b: round the float quotient to it.
		q := a.Sub(r).Cast(tensor.Float32).Div(b.Cast(tensor.Float32))
		half := tensor.Where(q.Less(c.scalar(q, 0)), c.scalar(q, -0.5), c.scalar(q, 0.5))
		r = c.trunc(q.Add(half)).Cast(tensor.Int32)
	}
	return tensor.Where(byZero, c.scalar(r, 0), r)
}

// reduce runs the reduce primitive: the reduced axes stay, of size 1.
func (c *call) reduce(x *tensor.Tensor) *tensor.Tensor {
	axes := toInts(c.ints("axes", nil))
	if len(axes) == 0 {
		return x
	}
	keep := x.Shape()
	for _, a := range axes {
		if a < 0 || a >= len(keep) {
			return tensor.Failed(fmt.Errorf("%w: reduce over axis %d of rank %d", ErrFormat, a, len(keep)))
		}
		keep[a] = 1
	}
	switch mode := c.str("mode", ""); mode {
	case "add":
		return x.Sum(axes...).Reshape(keep...)
	case "max":
		return x.Max(axes...).Reshape(keep...)
	case "mul":
		for _, a := range axes {
			x = c.prod(x, a)
		}
		return x
	default:
		return tensor.Failed(fmt.Errorf("%w: reduce mode %q", ErrFormat, mode))
	}
}

// prod multiplies the elements of x along axis a, keeping it with size 1,
// by halving it.
func (c *call) prod(x *tensor.Tensor, a int) *tensor.Tensor {
	shape := x.Shape()
	if shape[a] == 0 {
		shape[a] = 1
		return c.full(x.DType(), 1, shape...)
	}
	for n := shape[a]; n > 1; n = (n + 1) / 2 {
		if n%2 == 1 {
			p := make([][2]int, len(shape))
			p[a][1] = 1
			x = x.Pad(p, 1)
		}
		h := (n + 1) / 2
		lo, hi := whole(x.Shape()), whole(x.Shape())
		lo[a], hi[a] = [2]int{0, h}, [2]int{h, 2 * h}
		x = x.Shrink(lo).Mul(x.Shrink(hi))
	}
	return x
}

// whole returns the ranges that keep every element of a tensor of shape.
func whole(shape []int) [][2]int {
	r := make([][2]int, len(shape))
	for i, d := range shape {
		r[i] = [2]int{0, d}
	}
	return r
}

// take returns the slices of t along axis at the positions idx, of any
// shape, like numpy.take: idx's dimensions replace the axis. Positions out
// of range, whose results the primitive leaves unspecified, read the
// nearest end.
func (c *call) take(t, idx *tensor.Tensor, axis int) *tensor.Tensor {
	shape := t.Shape()
	if axis < 0 || axis >= len(shape) {
		return tensor.Failed(fmt.Errorf("%w: gather along axis %d of rank %d", ErrFormat, axis, len(shape)))
	}
	p, q := 1, 1
	for d, s := range shape {
		if d < axis {
			p *= s
		} else if d > axis {
			q *= s
		}
	}
	n := idx.Size()
	out := slices.Concat(shape[:axis], idx.Shape(), shape[axis+1:])
	if axis == 0 && len(shape) == 2 {
		// Rows of a table, which may be quantized and so not reshaped.
		ix := c.clamp(idx, shape[0]).Reshape(n, 1).Expand(n, shape[1])
		return t.Gather(0, ix).Reshape(out...)
	}
	ix := c.clamp(idx, shape[axis]).Reshape(1, n, 1).Expand(p, n, q)
	return t.Reshape(p, shape[axis], q).Gather(1, ix).Reshape(out...)
}

// clamp limits the positions idx to [0, n).
func (c *call) clamp(idx *tensor.Tensor, n int) *tensor.Tensor {
	return idx.Maximum(c.scalar(idx, 0)).Minimum(c.scalar(idx, float64(max(n-1, 0))))
}

// scatter returns data with the slices along axis at the positions idx
// replaced by those of updates. Of repeated positions, the last in
// row-major order wins.
func (c *call) scatter(data, idx, updates *tensor.Tensor, axis int) *tensor.Tensor {
	shape := data.Shape()
	if axis < 0 || axis >= len(shape) {
		return tensor.Failed(fmt.Errorf("%w: scatter along axis %d of rank %d", ErrFormat, axis, len(shape)))
	}
	p, q := 1, 1
	for d, s := range shape {
		if d < axis {
			p *= s
		} else if d > axis {
			q *= s
		}
	}
	n := idx.Size()
	pos := c.clamp(idx, shape[axis]).Reshape(n)
	// keep marks the positions no later position repeats.
	later := pos.Reshape(n, 1).Equal(pos.Reshape(1, n)).Mul(c.upper(n))
	keep := c.not(later.Max(1)).Reshape(1, n, 1).Expand(p, n, q)
	ix := pos.Reshape(1, n, 1).Expand(p, n, q)
	upd := updates.Reshape(p, n, q)
	upd = tensor.Where(keep, upd, c.scalar(upd, 0))
	hit := c.full(tensor.Int32, 0, p, shape[axis], q).ScatterAdd(1, ix, keep)
	sum := c.full(data.DType(), 0, p, shape[axis], q).ScatterAdd(1, ix, upd)
	return tensor.Where(hit, sum.Cast(data.DType()), data.Reshape(p, shape[axis], q)).Reshape(shape...)
}

// upper returns the Int32 [n, n] matrix of 1 above the diagonal and 0
// elsewhere.
func (c *call) upper(n int) *tensor.Tensor {
	i := tensor.Arange[int32](0, int32(n), 1).To(c.r.dev)
	return i.Reshape(n, 1).Less(i.Reshape(1, n))
}

// while runs the while op: the body runs on the loop-carried values for
// as long as the condition holds.
func (c *call) while() ([]*tensor.Tensor, error) {
	cond, err := c.sub("cond")
	if err != nil {
		return nil, err
	}
	body, err := c.sub("body")
	if err != nil {
		return nil, err
	}
	v := slices.Clone(c.in)
	for {
		ok, err := c.truth(cond, v)
		if err != nil || !ok {
			return v, err
		}
		if v, err = c.r.subgraph(body, v); err != nil {
			return nil, err
		}
		// Realizing the carried values keeps each iteration's graph small.
		if err := tensor.Realize(c.r.ctx, v...); err != nil {
			return nil, err
		}
	}
}

// ifElse runs one of the branches of the if op on its operands.
func (c *call) ifElse() ([]*tensor.Tensor, error) {
	then, err := c.sub("then")
	if err != nil {
		return nil, err
	}
	els, err := c.sub("else")
	if err != nil {
		return nil, err
	}
	pred, err := c.in[0].Item[int32](c.r.ctx)
	if err != nil {
		return nil, err
	}
	if pred != 0 {
		return c.r.subgraph(then, c.in[1:])
	}
	return c.r.subgraph(els, c.in[1:])
}

// truth runs the subgraph sg, whose one output is a bool scalar, on v.
func (c *call) truth(sg *mimfile.Subgraph, v []*tensor.Tensor) (bool, error) {
	out, err := c.r.subgraph(sg, v)
	if err != nil {
		return false, err
	}
	if len(out) != 1 {
		return false, fmt.Errorf("%w: a condition of %d outputs", ErrFormat, len(out))
	}
	b, err := out[0].Item[int32](c.r.ctx)
	return b != 0, err
}

// sub returns the subgraph attribute key.
func (c *call) sub(key string) (*mimfile.Subgraph, error) {
	a := c.attr(key)
	if a == nil || a.Type != mimfile.AttrSubgraph || int(a.Subgraph) >= len(c.r.m.g.Subgraphs) {
		return nil, fmt.Errorf("%w: no subgraph %s", ErrFormat, key)
	}
	return &c.r.m.g.Subgraphs[a.Subgraph], nil
}

// constOf returns the value of value vi if a const op computes it.
func (m *Model) constOf(vi uint32) (float64, bool) {
	op := m.producer[vi]
	if op == nil || op.Opcode != mimfile.OpConst {
		return 0, false
	}
	for _, a := range op.Attrs {
		if a.Key == "value" {
			if a.Type == mimfile.AttrInt {
				return float64(a.Int), true
			}
			return a.Float, true
		}
	}
	return 0, false
}
