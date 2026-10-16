package onnx

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/tensor"
)

// opFunc computes a node's outputs from its inputs.
type opFunc func(c *call) ([]*value, error)

// ops holds the operators of the default domain that mimir runs.
var ops map[string]opFunc

func init() {
	ops = map[string]opFunc{
		// Elementwise.
		"Identity":    identity,
		"Neg":         unary(neg, (*tensor.Tensor).Neg),
		"Abs":         unary(math.Abs, (*tensor.Tensor).Abs),
		"Sqrt":        unary(math.Sqrt, (*tensor.Tensor).Sqrt),
		"Exp":         unary(math.Exp, (*tensor.Tensor).Exp),
		"Log":         unary(math.Log, (*tensor.Tensor).Log),
		"Reciprocal":  unary(func(x float64) float64 { return 1 / x }, (*tensor.Tensor).Recip),
		"Sin":         unary(math.Sin, (*tensor.Tensor).Sin),
		"Cos":         unary(math.Cos, (*tensor.Tensor).Cos),
		"Erf":         unary(math.Erf, (*tensor.Tensor).Erf),
		"Floor":       unary(math.Floor, floor),
		"Ceil":        unary(math.Ceil, func(x *tensor.Tensor) *tensor.Tensor { return floor(x.Neg()).Neg() }),
		"Sign":        unary(sign, func(x *tensor.Tensor) *tensor.Tensor { return x.Greater(zero(x)).Sub(x.Less(zero(x))) }),
		"Relu":        unary(func(x float64) float64 { return max(x, 0) }, (*tensor.Tensor).ReLU),
		"Sigmoid":     unary(func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }, (*tensor.Tensor).Sigmoid),
		"Tanh":        unary(math.Tanh, (*tensor.Tensor).Tanh),
		"Softplus":    unary(nil, softplus),
		"Softsign":    unary(nil, func(x *tensor.Tensor) *tensor.Tensor { return x.Div(x.Abs().Add(one(x))) }),
		"LeakyRelu":   leakyRelu,
		"Elu":         elu,
		"HardSigmoid": hardSigmoid,
		"HardSwish":   hardSwish,
		"PRelu":       prelu,
		"Gelu":        gelu,
		"Not":         not,
		"Add":         elementwise(arith{add, func(a, b int64) int64 { return a + b }}, (*tensor.Tensor).Add),
		"Sub":         elementwise(arith{sub, func(a, b int64) int64 { return a - b }}, (*tensor.Tensor).Sub),
		"Mul":         elementwise(arith{mul, func(a, b int64) int64 { return a * b }}, (*tensor.Tensor).Mul),
		"Div":         div,
		"Pow":         pow,
		"Mod":         mod,
		"Max":         variadic(arith{math.Max, func(a, b int64) int64 { return max(a, b) }}, (*tensor.Tensor).Maximum, false),
		"Min":         variadic(arith{math.Min, func(a, b int64) int64 { return min(a, b) }}, (*tensor.Tensor).Minimum, false),
		"Sum":         variadic(arith{add, func(a, b int64) int64 { return a + b }}, (*tensor.Tensor).Add, false),
		"Mean":        variadic(arith{add, nil}, (*tensor.Tensor).Add, true),
		"Equal":       compare(func(a, b float64) bool { return a == b }, (*tensor.Tensor).Equal),
		"Less":        compare(func(a, b float64) bool { return a < b }, (*tensor.Tensor).Less),
		"Greater":     compare(func(a, b float64) bool { return a > b }, (*tensor.Tensor).Greater),
		"LessOrEqual": compare(func(a, b float64) bool { return a <= b }, func(a, b *tensor.Tensor) *tensor.Tensor {
			return notT(a.Greater(b))
		}),
		"GreaterOrEqual": compare(func(a, b float64) bool { return a >= b }, func(a, b *tensor.Tensor) *tensor.Tensor {
			return notT(a.Less(b))
		}),
		"And":      logical(func(a, b bool) bool { return a && b }, (*tensor.Tensor).Mul),
		"Or":       logical(func(a, b bool) bool { return a || b }, (*tensor.Tensor).Maximum),
		"Xor":      logical(func(a, b bool) bool { return a != b }, func(a, b *tensor.Tensor) *tensor.Tensor { return notT(a.Equal(b)) }),
		"Where":    where,
		"Cast":     cast,
		"CastLike": castLike,
		"Clip":     clip,
		"Dropout":  dropout,

		// Products and reductions.
		"MatMul":          matmul,
		"Gemm":            gemm,
		"ReduceSum":       reduce("sum"),
		"ReduceMean":      reduce("mean"),
		"ReduceMax":       reduce("max"),
		"ReduceMin":       reduce("min"),
		"ReduceProd":      reduce("prod"),
		"ReduceSumSquare": reduce("sumsquare"),
		"ReduceL1":        reduce("l1"),
		"ReduceL2":        reduce("l2"),
		"ReduceLogSumExp": reduce("logsumexp"),
		"ArgMax":          argmax(false),
		"ArgMin":          argmax(true),
		"CumSum":          cumsum,

		// Shapes, joining and indexing.
		"Constant":        constant,
		"ConstantOfShape": constantOfShape,
		"Shape":           shapeOp,
		"Size":            sizeOp,
		"Reshape":         reshape,
		"Flatten":         flatten,
		"Squeeze":         squeeze,
		"Unsqueeze":       unsqueeze,
		"Transpose":       transpose,
		"Concat":          concat,
		"Split":           split,
		"Slice":           slice,
		"Gather":          gather,
		"GatherElements":  gatherElements,
		"Expand":          expand,
		"Tile":            tile,
		"Range":           rangeOp,
		"Pad":             pad,
		"Trilu":           trilu,
		"NonZero":         nonZero,

		// Neural network layers.
		"Softmax":               softmax(false),
		"LogSoftmax":            softmax(true),
		"Conv":                  conv,
		"ConvTranspose":         convTranspose,
		"MaxPool":               pool(true),
		"AveragePool":           pool(false),
		"GlobalAveragePool":     globalPool(false),
		"GlobalMaxPool":         globalPool(true),
		"BatchNormalization":    batchNorm,
		"InstanceNormalization": instanceNorm,
		"LayerNormalization":    layerNorm,
		"Resize":                resize,
	}
}

// call is the evaluation of one node. Reading a malformed attribute sets
// err, which the node's result reports.
type call struct {
	*run
	n    *node
	in   []*value
	err  error
	keys int // values the node named with key
}

// arg returns input i, or nil if the node does not give it.
func (c *call) arg(i int) *value {
	if i < len(c.in) {
		return c.in[i]
	}
	return nil
}

// need checks that the first k inputs are given.
func (c *call) need(k int) error {
	for i := range k {
		if c.arg(i) == nil {
			return fmt.Errorf("%w: input %d of %s is missing", ErrFormat, i, c.n.op)
		}
	}
	return nil
}

func (c *call) unsupported(format string, args ...any) error {
	return &UnsupportedError{Node: c.n.name, Op: c.n.op, Detail: fmt.Sprintf(format, args...)}
}

// attr returns the attribute named name, or nil. An attribute of another
// type than typ sets c.err.
func (c *call) attr(name string, typ attrType) *attribute {
	for _, a := range c.n.attrs {
		if a.name != name {
			continue
		}
		if a.typ != typ {
			if c.err == nil {
				c.err = fmt.Errorf("%w: attribute %s of type %d, want %d", ErrFormat, name, a.typ, typ)
			}
			return nil
		}
		return a
	}
	return nil
}

func (c *call) int(name string, def int64) int64 {
	if a := c.attr(name, attrInt); a != nil {
		return a.i
	}
	return def
}

func (c *call) ints(name string) ([]int64, bool) {
	if a := c.attr(name, attrInts); a != nil {
		return a.ints, true
	}
	return nil, false
}

func (c *call) float(name string, def float64) float64 {
	if a := c.attr(name, attrFloat); a != nil {
		return float64(a.f)
	}
	return def
}

func (c *call) floats(name string) ([]float32, bool) {
	if a := c.attr(name, attrFloats); a != nil {
		return a.floats, true
	}
	return nil, false
}

func (c *call) str(name, def string) string {
	if a := c.attr(name, attrString); a != nil {
		return string(a.s)
	}
	return def
}

// t returns input i as a tensor on the run's device, or a tensor that
// carries the error that prevented it.
func (c *call) t(i int) *tensor.Tensor {
	v := c.arg(i)
	if v == nil {
		return tensor.Failed(fmt.Errorf("%w: input %d of %s is missing", ErrFormat, i, c.n.op))
	}
	t, err := c.tensor(v)
	if err != nil {
		return tensor.Failed(err)
	}
	return t
}

// host returns input i on the host, computing it on its device first if
// it is a tensor.
func (c *call) host(i int) (*value, error) {
	v := c.arg(i)
	if v == nil {
		return nil, fmt.Errorf("%w: input %d of %s is missing", ErrFormat, i, c.n.op)
	}
	return c.download(v)
}

// intList returns the elements of input i, a list of integers, on the
// host; ok is false if the node does not give the input.
func (c *call) intList(i int) (list []int64, ok bool, err error) {
	if c.arg(i) == nil {
		return nil, false, nil
	}
	v, err := c.host(i)
	if err != nil {
		return nil, true, err
	}
	if len(v.shape) > 1 {
		return nil, true, fmt.Errorf("%w: input %d of %s has shape %v, want a list", tensor.ErrShape, i, c.n.op, v.shape)
	}
	list = make([]int64, v.size())
	for j := range list {
		list[j] = v.int(j)
	}
	return list, true, nil
}

// allHost reports whether every given input is on the host.
func (c *call) allHost() bool {
	for _, v := range c.in {
		if v != nil && !v.host {
			return false
		}
	}
	return true
}

// out returns a node's output tensor of type typ, with the half types
// and float32 told apart by the tensor's dtype.
func out(typ DataType, t *tensor.Tensor) *value {
	if typ.IsFloat() {
		switch t.DType() {
		case tensor.Float16:
			typ = Float16
		case tensor.BFloat16:
			typ = BFloat16
		case tensor.Float32:
			if typ != Double {
				typ = Float
			}
		}
	}
	return &value{typ: typ, shape: t.Shape(), t: t}
}

func one1(v *value) []*value { return []*value{v} }

// normAxis returns axis, which may count from the end, of a rank-rank
// shape.
func normAxis(axis int64, rank int) (int, error) {
	if axis < -int64(rank) || axis >= int64(max(rank, 1)) {
		return 0, fmt.Errorf("onnx: axis %d of rank %d: %w", axis, rank, tensor.ErrShape)
	}
	if axis < 0 {
		axis += int64(rank)
	}
	return int(axis), nil
}

// normAxes returns the axes of a rank-rank shape, sorted, which must not
// repeat.
func normAxes(axes []int64, rank int) ([]int, error) {
	out := make([]int, len(axes))
	seen := make([]bool, rank)
	for i, a := range axes {
		d, err := normAxis(a, rank)
		if err != nil {
			return nil, err
		}
		if seen[d] {
			return nil, fmt.Errorf("onnx: axis %d repeated: %w", a, tensor.ErrShape)
		}
		seen[d] = true
		out[i] = d
	}
	return out, nil
}

// scalarOf returns a tensor holding v with t's dtype, which broadcasts
// against t.
func scalarOf(t *tensor.Tensor, v float64) *tensor.Tensor {
	dt := t.DType()
	if dt == backend.Invalid {
		return t
	}
	return tensor.Full(dt, v)
}

func zero(t *tensor.Tensor) *tensor.Tensor { return scalarOf(t, 0) }
func one(t *tensor.Tensor) *tensor.Tensor  { return scalarOf(t, 1) }

func neg(x float64) float64 { return -x }

func sign(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	}
	return x
}

// floor rounds t down: x - fmod(x, 1), less one where the remainder is
// negative.
func floor(t *tensor.Tensor) *tensor.Tensor {
	if !t.DType().IsFloat() {
		return t
	}
	r := t.Fmod(one(t))
	return t.Sub(r).Sub(r.Less(zero(t)))
}

func softplus(t *tensor.Tensor) *tensor.Tensor {
	// log(1 + e^x) = max(x, 0) + log(1 + e^-|x|), which never overflows.
	return t.ReLU().Add(t.Abs().Neg().Exp().Add(one(t)).Log())
}

// notT returns 1 - t, the negation of a bool tensor.
func notT(t *tensor.Tensor) *tensor.Tensor { return one(t).Sub(t) }

// asBool returns a comparison's result as the Int32 0 or 1 of a bool.
func asBool(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.Int32) }

func identity(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	return one1(c.in[0]), nil
}

// dropout is the identity of inference, with a mask of ones.
func dropout(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	if v := c.arg(2); v != nil {
		h, err := c.download(v)
		if err != nil {
			return nil, err
		}
		if h.size() > 0 && h.int(0) != 0 {
			return nil, c.unsupported("training mode")
		}
	}
	x := c.in[0]
	if len(c.n.outputs) < 2 {
		return one1(x), nil
	}
	mask := tensor.Full(tensor.Int32, 1, x.shape...)
	return []*value{x, {typ: Bool, shape: x.shape, t: mask}}, nil
}

// unary returns an elementwise operator: hostFn computes it on the host,
// where it is not nil, and devFn on a device.
func unary(hostFn func(float64) float64, devFn func(*tensor.Tensor) *tensor.Tensor) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(1); err != nil {
			return nil, err
		}
		x := c.in[0]
		if x.host && hostFn != nil {
			return one1(mapHost(x, x.typ, hostFn)), nil
		}
		return one1(out(x.typ, devFn(c.t(0)))), nil
	}
}

// mapHost applies f to each element of host value x, giving a value of
// type typ.
func mapHost(x *value, typ DataType, f func(float64) float64) *value {
	n := x.size()
	if typ.IsFloat() {
		o := make([]float64, n)
		for i := range o {
			o[i] = f(x.float(i))
		}
		return hostFloats(typ, x.shape, o)
	}
	o := make([]int64, n)
	for i := range o {
		y := f(x.float(i))
		if typ == Bool && y != 0 {
			y = 1
		}
		o[i] = toInt(y)
	}
	return hostInts(typ, x.shape, o)
}

// toInt truncates f toward zero, saturating.
func toInt(f float64) int64 {
	switch {
	case math.IsNaN(f):
		return 0
	case f >= math.MaxInt64:
		return math.MaxInt64
	case f <= math.MinInt64:
		return math.MinInt64
	}
	return int64(f)
}

func leakyRelu(c *call) ([]*value, error) {
	alpha := c.float("alpha", 0.01)
	return unary(func(x float64) float64 {
		if x < 0 {
			return alpha * x
		}
		return x
	}, func(t *tensor.Tensor) *tensor.Tensor {
		return tensor.Where(t.Less(zero(t)), t.Mul(scalarOf(t, alpha)), t)
	})(c)
}

func elu(c *call) ([]*value, error) {
	alpha := c.float("alpha", 1)
	return unary(nil, func(t *tensor.Tensor) *tensor.Tensor {
		return tensor.Where(t.Less(zero(t)), t.Exp().Sub(one(t)).Mul(scalarOf(t, alpha)), t)
	})(c)
}

func hardSigmoid(c *call) ([]*value, error) {
	alpha, beta := c.float("alpha", 0.2), c.float("beta", 0.5)
	return unary(nil, func(t *tensor.Tensor) *tensor.Tensor {
		return t.Mul(scalarOf(t, alpha)).Add(scalarOf(t, beta)).Maximum(zero(t)).Minimum(one(t))
	})(c)
}

func hardSwish(c *call) ([]*value, error) {
	return unary(nil, func(t *tensor.Tensor) *tensor.Tensor {
		return t.Mul(t.Mul(scalarOf(t, 1.0/6)).Add(scalarOf(t, 0.5)).Maximum(zero(t)).Minimum(one(t)))
	})(c)
}

func prelu(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x, slope := c.t(0), c.t(1)
	return one1(out(c.in[0].typ, tensor.Where(x.Less(zero(x)), x.Mul(slope), x))), nil
}

func gelu(c *call) ([]*value, error) {
	approx := c.str("approximate", "none")
	return unary(nil, func(t *tensor.Tensor) *tensor.Tensor {
		switch approx {
		case "tanh":
			return t.GELU()
		case "none":
			// x/2 (1 + erf(x/√2))
			h := t.Mul(scalarOf(t, 0.5))
			return h.Add(h.Mul(t.Mul(scalarOf(t, 1/math.Sqrt2)).Erf()))
		}
		return tensor.Failed(c.unsupported("approximate %q", approx))
	})(c)
}

func not(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	if x.host {
		return one1(mapHost(x, Bool, func(f float64) float64 { return b2f(f == 0) })), nil
	}
	return one1(&value{typ: Bool, shape: x.shape, t: asBool(notT(c.t(0)))}), nil
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// arith is an elementwise function of two values, computed in float64,
// or, where i is not nil, exactly in int64 for two integers.
type arith struct {
	f func(a, b float64) float64
	i func(a, b int64) int64
}

func add(a, b float64) float64 { return a + b }
func sub(a, b float64) float64 { return a - b }
func mul(a, b float64) float64 { return a * b }

// elementwise returns a broadcasting elementwise operator of two inputs of
// one type.
func elementwise(fn arith, devFn func(a, b *tensor.Tensor) *tensor.Tensor) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(2); err != nil {
			return nil, err
		}
		a := c.in[0]
		if c.allHost() {
			v, err := zipHost(a.typ, c.in[0], c.in[1], fn)
			if err != nil || v != nil {
				return one1(v), err
			}
		}
		return one1(out(a.typ, devFn(c.t(0), c.t(1)))), nil
	}
}

// zipHost combines host values a and b, broadcast, with fn into a value
// of type typ, or returns nil if the result is too large for the host.
func zipHost(typ DataType, a, b *value, fn arith) (*value, error) {
	shape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	if _, ok := numElements(shape, hostLimit); !ok {
		return nil, nil
	}
	ia, ib := broadcastIndex(a.shape, shape), broadcastIndex(b.shape, shape)
	if typ.IsFloat() {
		o := make([]float64, len(ia))
		for i := range o {
			o[i] = fn.f(a.float(ia[i]), b.float(ib[i]))
		}
		return hostFloats(typ, shape, o), nil
	}
	exact := fn.i != nil && !a.typ.IsFloat() && !b.typ.IsFloat()
	o := make([]int64, len(ia))
	for i := range o {
		if exact {
			o[i] = fn.i(a.ints[ia[i]], b.ints[ib[i]])
		} else {
			o[i] = toInt(fn.f(a.float(ia[i]), b.float(ib[i])))
		}
	}
	return hostInts(typ, shape, o), nil
}

func div(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	a, b := c.in[0], c.in[1]
	if c.allHost() {
		if !a.typ.IsFloat() {
			for _, y := range b.ints {
				if y == 0 {
					return nil, fmt.Errorf("onnx: integer division by zero: %w", tensor.ErrDivideByZero)
				}
			}
		}
		v, err := zipHost(a.typ, a, b, arith{func(x, y float64) float64 { return x / y }, func(x, y int64) int64 { return x / y }})
		if err != nil || v != nil {
			return one1(v), err
		}
	}
	x, y := c.t(0), c.t(1)
	if a.typ.IsFloat() {
		return one1(out(a.typ, x.Div(y))), nil
	}
	q := x.Cast(tensor.Float32).Div(y.Cast(tensor.Float32))
	return one1(out(a.typ, q.Cast(tensor.Int32))), nil
}

func pow(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	a, b := c.in[0], c.in[1]
	if c.allHost() {
		v, err := zipHost(a.typ, a, b, arith{f: math.Pow})
		if err != nil || v != nil {
			return one1(v), err
		}
	}
	x := c.t(0)
	if !x.DType().IsFloat() {
		return nil, c.unsupported("integer bases")
	}
	if b.host {
		if e, ok := b.uniform(); ok && len(b.shape) <= len(a.shape) {
			switch e {
			case 1:
				return one1(a), nil
			case 2:
				return one1(out(a.typ, x.Mul(x))), nil
			case 3:
				return one1(out(a.typ, x.Mul(x).Mul(x))), nil
			case 0.5:
				return one1(out(a.typ, x.Sqrt())), nil
			case -0.5:
				return one1(out(a.typ, x.Sqrt().Recip())), nil
			case -1:
				return one1(out(a.typ, x.Recip())), nil
			}
		}
	}
	y := c.t(1)
	if y.DType() != x.DType() {
		y = y.Cast(x.DType())
	}
	// |x|^y, negated where x < 0 and y is odd: cos(πy) is ±1 at integers.
	p := x.Abs().Log().Mul(y).Exp()
	signs := tensor.Where(x.Less(zero(x)), y.Mul(scalarOf(y, math.Pi)).Cos(), one(x))
	return one1(out(a.typ, p.Mul(signs))), nil
}

func mod(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	a, b := c.in[0], c.in[1]
	fmod := c.int("fmod", 0) == 1
	if c.allHost() {
		if !a.typ.IsFloat() {
			for _, y := range b.ints {
				if y == 0 {
					return nil, fmt.Errorf("onnx: integer modulo by zero: %w", tensor.ErrDivideByZero)
				}
			}
		}
		v, err := zipHost(a.typ, a, b, arith{f: func(x, y float64) float64 {
			r := math.Mod(x, y)
			if !fmod && r != 0 && (r < 0) != (y < 0) {
				r += y
			}
			return r
		}, i: func(x, y int64) int64 {
			r := x % y
			if !fmod && r != 0 && (r < 0) != (y < 0) {
				r += y
			}
			return r
		}})
		if err != nil || v != nil {
			return one1(v), err
		}
	}
	x, y := c.t(0), c.t(1)
	r := x.Fmod(y)
	if !fmod {
		// Python's remainder takes the divisor's sign.
		r = r.Add(tensor.Where(r.Mul(y).Less(zero(r)), y, zero(r)))
	}
	return one1(out(a.typ, r)), nil
}

// variadic returns an operator that folds any number of inputs with f,
// dividing by their number if mean.
func variadic(fn arith, devFn func(a, b *tensor.Tensor) *tensor.Tensor, mean bool) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(len(c.in)); err != nil || len(c.in) == 0 {
			return nil, fmt.Errorf("%w: %s of no inputs", ErrFormat, c.n.op)
		}
		typ := c.in[0].typ
		if c.allHost() {
			acc := c.in[0]
			for _, v := range c.in[1:] {
				var err error
				if acc, err = zipHost(typ, acc, v, fn); err != nil || acc == nil {
					break
				}
			}
			if acc != nil && mean {
				n := float64(len(c.in))
				acc = mapHost(acc, typ, func(x float64) float64 { return x / n })
			}
			if acc != nil {
				return one1(acc), nil
			}
		}
		acc := c.t(0)
		for i := 1; i < len(c.in); i++ {
			acc = devFn(acc, c.t(i))
		}
		if mean {
			acc = acc.Div(scalarOf(acc, float64(len(c.in))))
		}
		return one1(out(typ, acc)), nil
	}
}

// compare returns a comparison operator, whose result is a bool.
func compare(hostFn func(a, b float64) bool, devFn func(a, b *tensor.Tensor) *tensor.Tensor) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(2); err != nil {
			return nil, err
		}
		if c.allHost() {
			v, err := zipHost(Bool, c.in[0], c.in[1], arith{f: func(x, y float64) float64 { return b2f(hostFn(x, y)) }})
			if err != nil || v != nil {
				return one1(v), err
			}
		}
		t := devFn(c.t(0), c.t(1))
		return one1(&value{typ: Bool, shape: t.Shape(), t: asBool(t)}), nil
	}
}

// logical returns a logical operator of two bools.
func logical(hostFn func(a, b bool) bool, devFn func(a, b *tensor.Tensor) *tensor.Tensor) opFunc {
	return compare(func(a, b float64) bool { return hostFn(a != 0, b != 0) }, devFn)
}

func where(c *call) ([]*value, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	cond, x, y := c.in[0], c.in[1], c.in[2]
	if c.allHost() {
		shape, err := broadcastShape(cond.shape, x.shape, y.shape)
		if err != nil {
			return nil, err
		}
		if n, ok := numElements(shape, hostLimit); ok {
			ic, ix, iy := broadcastIndex(cond.shape, shape), broadcastIndex(x.shape, shape), broadcastIndex(y.shape, shape)
			o := &value{typ: x.typ, shape: shape, host: true}
			if x.typ.IsFloat() {
				o.floats = make([]float64, n)
			} else {
				o.ints = make([]int64, n)
			}
			for i := range n {
				src, j := y, iy[i]
				if cond.float(ic[i]) != 0 {
					src, j = x, ix[i]
				}
				if x.typ.IsFloat() {
					o.floats[i] = src.float(j)
				} else {
					o.ints[i] = src.int(j)
				}
			}
			return one1(o), nil
		}
	}
	return one1(out(x.typ, tensor.Where(c.t(0), c.t(1), c.t(2)))), nil
}

func cast(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	return castTo(c, c.in[0], DataType(c.int("to", 0)))
}

func castLike(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	return castTo(c, c.in[0], c.in[1].typ)
}

func castTo(c *call, x *value, to DataType) ([]*value, error) {
	dt, ok := to.DType()
	if !ok {
		return nil, c.unsupported("cast to %s", to)
	}
	if x.host {
		round := func(f float64) float64 { return f }
		switch {
		case to == Bool:
			round = func(f float64) float64 { return b2f(f != 0) }
		case !to.IsFloat():
			round = func(f float64) float64 { return float64(wrapInt(to, toInt(f))) }
		}
		if !to.IsFloat() && !x.typ.IsFloat() {
			o := make([]int64, x.size())
			for i, v := range x.ints {
				if to == Bool {
					o[i] = int64(b2f(v != 0))
				} else {
					o[i] = wrapInt(to, v)
				}
			}
			return one1(hostInts(to, x.shape, o)), nil
		}
		return one1(mapHost(x, to, round)), nil
	}
	t := c.t(0)
	if to == Bool {
		return one1(&value{typ: Bool, shape: x.shape, t: asBool(notT(t.Equal(zero(t))))}), nil
	}
	return one1(&value{typ: to, shape: x.shape, t: t.Cast(dt)}), nil
}

// wrapInt returns x wrapped to the range of the integer type to, as a
// cast in C does.
func wrapInt(to DataType, x int64) int64 {
	switch to {
	case Int8:
		return int64(int8(x))
	case Uint8:
		return int64(uint8(x))
	case Int16:
		return int64(int16(x))
	case Uint16:
		return int64(uint16(x))
	case Int32:
		return int64(int32(x))
	case Uint32:
		return int64(uint32(x))
	}
	return x
}

func clip(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	lo, hi := c.arg(1), c.arg(2)
	if c.allHost() {
		v := x
		var err error
		if lo != nil {
			v, err = zipHost(x.typ, v, lo, arith{math.Max, func(a, b int64) int64 { return max(a, b) }})
		}
		if err == nil && v != nil && hi != nil {
			v, err = zipHost(x.typ, v, hi, arith{math.Min, func(a, b int64) int64 { return min(a, b) }})
		}
		if err != nil || v != nil {
			return one1(v), err
		}
	}
	t := c.t(0)
	if lo != nil {
		t = t.Maximum(c.t(1))
	}
	if hi != nil {
		t = t.Minimum(c.t(2))
	}
	return one1(out(x.typ, t)), nil
}
