package tensor

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// broadcastShape returns the NumPy broadcast of shapes: dimensions align on
// the right and each pair must match or contain a 1.
func broadcastShape(shapes ...[]int) ([]int, error) {
	rank := 0
	for _, s := range shapes {
		rank = max(rank, len(s))
	}
	out := make([]int, rank)
	for i := range out {
		out[i] = 1
	}
	for _, s := range shapes {
		for i, d := range s {
			o := &out[rank-len(s)+i]
			switch {
			case d == *o || d == 1:
			case *o == 1:
				*o = d
			default:
				return nil, fmt.Errorf("tensor: cannot broadcast %v: %w", shapes, ErrShape)
			}
		}
	}
	return out, nil
}

// broadcastTo expands t to shape, adding leading dimensions as needed.
func (t *Tensor) broadcastTo(shape []int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if extra := len(shape) - len(t.node.Shape); extra > 0 {
		t = t.reshape(slices.Concat(slices.Repeat([]int{1}, extra), t.node.Shape))
	}
	return t.expand(shape)
}

// broadcast expands every tensor to their common shape.
func broadcast(ts ...*Tensor) ([]*Tensor, error) {
	if err := firstErr(ts...); err != nil {
		return nil, err
	}
	shapes := make([][]int, len(ts))
	for i, t := range ts {
		shapes[i] = t.node.Shape
	}
	shape, err := broadcastShape(shapes...)
	if err != nil {
		return nil, err
	}
	out := make([]*Tensor, len(ts))
	for i, t := range ts {
		if out[i] = t.broadcastTo(shape); out[i].err != nil {
			return nil, out[i].err
		}
	}
	return out, nil
}

func (t *Tensor) binary(op backend.Op, u *Tensor) *Tensor {
	b, err := broadcast(t, u)
	if err != nil {
		return fail(err)
	}
	return binaryPrim(op, b[0], b[1])
}

// promote returns x and y cast to the dtype a binary operation on them
// computes in: Float32 when both are float and of different types. Other
// pairs are returned as they are, equal or for the operation to reject.
// A cast operand's gradient is cast back to its type.
func promote(x, y *Tensor) (*Tensor, *Tensor) {
	if firstErr(x, y) != nil || x.node.DType == y.node.DType || !x.node.DType.IsFloat() || !y.node.DType.IsFloat() {
		return x, y
	}
	return x.Cast(Float32), y.Cast(Float32)
}

// inFloat32 returns f applied to t computed in Float32: a half t is cast
// to Float32 and f's result back to t's dtype, so a composite of several
// primitives rounds once, as a primitive does.
func (t *Tensor) inFloat32(f func(x *Tensor) *Tensor) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	dt := t.node.DType
	if dt.Size() != 2 {
		return f(t)
	}
	return f(t.Cast(Float32)).Cast(dt)
}

// inFloat32Of returns f applied to t and u computed in Float32 when
// either is a half tensor: the result is cast to their common dtype, their
// own when equal.
func inFloat32Of(f func(x, y *Tensor) *Tensor, t, u *Tensor) *Tensor {
	if err := firstErr(t, u); err != nil {
		return fail(err)
	}
	dt, du := t.node.DType, u.node.DType
	if dt.Size() != 2 && du.Size() != 2 || !dt.IsFloat() || !du.IsFloat() {
		return f(t, u)
	}
	out := f(t.Cast(Float32), u.Cast(Float32))
	if dt != du {
		return out
	}
	return out.Cast(dt)
}

// binaryPrim applies a binary primitive to tensors of one shape and records
// its gradient. Comparisons have none. Operands of different float
// dtypes are promoted to Float32.
func binaryPrim(op backend.Op, x, y *Tensor) *Tensor {
	x, y = promote(x, y)
	if err := firstErr(x, y); err != nil {
		return fail(err)
	}
	var out *Tensor
	if l := batchLevel(x, y); l != 0 {
		out = batchBinary(l, op, x, y)
	} else {
		out = wrap(graph.Binary(op, x.node, y.node))
	}
	if r, ok := binaryRule(op); ok {
		return record(out, r, x, y)
	}
	return out
}

// binaryRule returns the rule of a binary primitive. Comparisons have
// none.
func binaryRule(op backend.Op) (rule, bool) {
	switch op {
	case backend.OpAdd:
		return rule{
			vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(g, g) },
			jvp: func(t, _ []*Tensor, _ *Tensor) *Tensor { return addGrad(t[0], t[1]) },
		}, true
	case backend.OpMul:
		return rule{
			vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return grads(g.Mul(in[1]), g.Mul(in[0])) },
			jvp: func(t, in []*Tensor, _ *Tensor) *Tensor {
				return addGrad(mulTangent(t[0], in[1]), mulTangent(t[1], in[0]))
			},
		}, true
	case backend.OpMax:
		// Like torch.maximum: ties split the gradient evenly.
		return rule{
			vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor {
				wa, wb := maxWeights(in[0], in[1])
				return grads(g.Mul(wa), g.Mul(wb))
			},
			jvp: func(t, in []*Tensor, _ *Tensor) *Tensor {
				wa, wb := maxWeights(in[0], in[1])
				return addGrad(mulTangent(t[0], wa), mulTangent(t[1], wb))
			},
		}, true
	case backend.OpMod:
		// fmod(a, b) = a - trunc(a/b)*b, and trunc(a/b) = (a - fmod(a, b))/b.
		return rule{
			vjp: func(g *Tensor, in []*Tensor, out *Tensor) []*Tensor {
				return grads(g, g.Mul(in[0].Sub(out).Div(in[1])).Neg())
			},
			jvp: func(t, in []*Tensor, out *Tensor) *Tensor {
				return addGrad(t[0], mulTangent(t[1], in[0].Sub(out).Div(in[1])).Neg())
			},
		}, true
	}
	return rule{}, false
}

// mulTangent returns t*w, or nil for a nil tangent t.
func mulTangent(t, w *Tensor) *Tensor {
	if t == nil {
		return nil
	}
	return t.Mul(w)
}

// maxWeights returns the share of the derivative of max(a, b) that goes
// to a and to b: all of it to the larger, half each when they tie.
func maxWeights(a, b *Tensor) (wa, wb *Tensor) {
	one, tie := a.scalar(1), a.Equal(b).Mul(a.scalar(0.5))
	return one.Sub(a.Less(b)).Sub(tie), one.Sub(b.Less(a)).Sub(tie)
}

func (t *Tensor) unary(op backend.Op) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	var out *Tensor
	if l := batchLevel(t); l != 0 {
		out = batchUnary(l, op, t)
	} else {
		out = wrap(graph.Unary(op, t.node))
	}
	if r, ok := unaryRule(op); ok {
		return record(out, r, t)
	}
	return out
}

// unaryRule returns the rule of a unary primitive. Each multiplies the
// gradient, or the tangent, by the derivative at the input x, whose
// result is y.
func unaryRule(op backend.Op) (rule, bool) {
	switch op {
	case backend.OpNeg:
		return rule{
			vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(g.Neg()) },
			jvp: func(t, _ []*Tensor, _ *Tensor) *Tensor { return t[0].Neg() },
		}, true
	case backend.OpExp2:
		return rule{
			vjp: func(g *Tensor, in []*Tensor, y *Tensor) []*Tensor { return grads(byExp2(g, y)) },
			jvp: func(t, in []*Tensor, y *Tensor) *Tensor { return byExp2(t[0], y) },
		}, true
	case backend.OpLog2:
		return rule{
			vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return grads(byLog2(g, in[0])) },
			jvp: func(t, in []*Tensor, _ *Tensor) *Tensor { return byLog2(t[0], in[0]) },
		}, true
	case backend.OpSin:
		return rule{
			vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return grads(g.Mul(in[0].Cos())) },
			jvp: func(t, in []*Tensor, _ *Tensor) *Tensor { return t[0].Mul(in[0].Cos()) },
		}, true
	case backend.OpSqrt:
		return rule{
			vjp: func(g *Tensor, _ []*Tensor, y *Tensor) []*Tensor { return grads(bySqrt(g, y)) },
			jvp: func(t, _ []*Tensor, y *Tensor) *Tensor { return bySqrt(t[0], y) },
		}, true
	case backend.OpRecip:
		return rule{
			vjp: func(g *Tensor, _ []*Tensor, y *Tensor) []*Tensor { return grads(byRecip(g, y)) },
			jvp: func(t, _ []*Tensor, y *Tensor) *Tensor { return byRecip(t[0], y) },
		}, true
	}
	return rule{}, false
}

// The derivatives of the unary primitives, times g.

// byExp2 returns g times the derivative of y = 2^x: y·ln 2.
func byExp2(g, y *Tensor) *Tensor { return g.Mul(y).Mul(g.scalar(math.Ln2)) }

// byLog2 returns g times the derivative of log2(x): 1/(x·ln 2).
func byLog2(g, x *Tensor) *Tensor { return g.Div(x.Mul(g.scalar(math.Ln2))) }

// bySqrt returns g times the derivative of y = √x: 1/(2y).
func bySqrt(g, y *Tensor) *Tensor { return g.Div(y.Mul(g.scalar(2))) }

// byRecip returns g times the derivative of y = 1/x: -1/x² = -y².
func byRecip(g, y *Tensor) *Tensor { return g.Mul(y).Mul(y).Neg() }

// scalar returns a zero-dimensional constant with t's dtype.
func (t *Tensor) scalar(v float64) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return Full(t.node.DType, v)
}

// Add returns t + u, broadcasting like NumPy. Like every binary
// operation, it promotes operands of different float dtypes to Float32.
func (t *Tensor) Add(u *Tensor) *Tensor { return t.binary(backend.OpAdd, u) }

// Sub returns t - u.
func (t *Tensor) Sub(u *Tensor) *Tensor { return t.Add(u.Neg()) }

// Mul returns t * u.
func (t *Tensor) Mul(u *Tensor) *Tensor { return t.binary(backend.OpMul, u) }

// Div returns t / u. Float types only.
func (t *Tensor) Div(u *Tensor) *Tensor {
	return inFloat32Of(func(x, y *Tensor) *Tensor { return x.Mul(y.Recip()) }, t, u)
}

// Maximum returns the elementwise maximum. NaN propagates.
func (t *Tensor) Maximum(u *Tensor) *Tensor { return t.binary(backend.OpMax, u) }

// Minimum returns the elementwise minimum. NaN propagates.
func (t *Tensor) Minimum(u *Tensor) *Tensor { return t.Neg().Maximum(u.Neg()).Neg() }

// Less returns 1 where t < u and 0 elsewhere, in the operands' common
// dtype.
func (t *Tensor) Less(u *Tensor) *Tensor { return t.binary(backend.OpCmpLT, u) }

// Greater returns 1 where t > u and 0 elsewhere, in the operands' common
// dtype.
func (t *Tensor) Greater(u *Tensor) *Tensor { return u.Less(t) }

// Equal returns 1 where t == u and 0 elsewhere, in the operands' common
// dtype.
func (t *Tensor) Equal(u *Tensor) *Tensor { return t.binary(backend.OpCmpEQ, u) }

// Fmod returns the remainder of t / u with the sign of t, like torch.fmod.
// Integer division by zero is ErrDivideByZero; float division by zero is NaN.
func (t *Tensor) Fmod(u *Tensor) *Tensor { return t.binary(backend.OpMod, u) }

// Where returns x where cond is non-zero and y elsewhere, broadcasting all
// three. cond may have any dtype; x and y of different float dtypes are
// promoted to Float32.
func Where(cond, x, y *Tensor) *Tensor {
	b, err := broadcast(cond, x, y)
	if err != nil {
		return fail(err)
	}
	a, d := promote(b[1], b[2])
	return wherePrim(b[0], a, d)
}

// wherePrim is the selection primitive, over tensors of one shape with x
// and y of one dtype. cond has no gradient.
func wherePrim(cond, x, y *Tensor) *Tensor {
	if err := firstErr(cond, x, y); err != nil {
		return fail(err)
	}
	c := cond.detached()
	var out *Tensor
	if l := batchLevel(c, x, y); l != 0 {
		out = batchWhere(l, c, x, y)
	} else {
		out = wrap(graph.Where(c.node, x.node, y.node))
	}
	return record(out, whereRule(), c, x, y)
}

// whereRule is the rule of the selection primitive, whose condition has
// no derivative.
func whereRule() rule {
	return rule{
		vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor {
			zero := g.scalar(0)
			return grads(nil, Where(in[0], g, zero), Where(in[0], zero, g))
		},
		jvp: func(t, in []*Tensor, out *Tensor) *Tensor {
			zero := Zeros(out.node.DType)
			a, b := t[1], t[2]
			if a == nil {
				a = zero
			}
			if b == nil {
				b = zero
			}
			return Where(in[0], a, b)
		},
	}
}

// Cast converts t to dt. Float to integer truncates toward zero; values
// outside the integer range are undefined. Float to a narrower float
// rounds to nearest even, keeping NaN and infinities and overflowing to
// infinity. Between float types the gradient flows through, cast back to
// t's dtype; a cast to or from Int32 has none.
func (t *Tensor) Cast(dt DType) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	from := t.node.DType
	if from == dt {
		return t
	}
	var out *Tensor
	if l := batchLevel(t); l != 0 {
		out = batchCast(l, t, dt)
	} else {
		out = wrap(graph.Cast(t.node, dt))
	}
	if !from.IsFloat() || !dt.IsFloat() {
		return out
	}
	return record(out, castRule(), t)
}

// castRule casts each derivative to the dtype of what it is the
// derivative of.
func castRule() rule {
	return rule{
		vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return grads(g.Cast(in[0].node.DType)) },
		jvp: func(t, _ []*Tensor, out *Tensor) *Tensor { return t[0].Cast(out.node.DType) },
	}
}

// Clone returns a copy of t in memory of its own, like torch.Tensor.clone.
// Computing the copy reads t, so realize it before writing over t's memory
// (see RealizeInto) to keep t's values. Gradients flow through it
// unchanged.
func (t *Tensor) Clone() *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	var out *Tensor
	if l := batchLevel(t); l != 0 {
		out = batchClone(l, t)
	} else {
		out = wrap(graph.Cast(t.node, t.node.DType))
	}
	return record(out, identity, t)
}

// Neg returns -t.
func (t *Tensor) Neg() *Tensor { return t.unary(backend.OpNeg) }

// Exp2 returns 2^t.
func (t *Tensor) Exp2() *Tensor { return t.unary(backend.OpExp2) }

// Log2 returns log2(t).
func (t *Tensor) Log2() *Tensor { return t.unary(backend.OpLog2) }

// Sin returns sin(t).
func (t *Tensor) Sin() *Tensor { return t.unary(backend.OpSin) }

// Sqrt returns the square root of t.
func (t *Tensor) Sqrt() *Tensor { return t.unary(backend.OpSqrt) }

// Recip returns 1/t.
func (t *Tensor) Recip() *Tensor { return t.unary(backend.OpRecip) }

// The composites below compute half tensors in Float32 and round once.

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor { return x.Mul(x.scalar(math.Log2E)).Exp2() })
}

// Log returns the natural logarithm of t.
func (t *Tensor) Log() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor { return x.Log2().Mul(x.scalar(math.Ln2)) })
}

// Cos returns cos(t).
func (t *Tensor) Cos() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor { return x.Add(x.scalar(math.Pi / 2)).Sin() })
}

// Abs returns |t|.
func (t *Tensor) Abs() *Tensor { return t.Maximum(t.Neg()) }

// ReLU returns max(t, 0). Its gradient is 0 at 0, like PyTorch's.
func (t *Tensor) ReLU() *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	var out *Tensor
	if l := batchLevel(t); l != 0 {
		out = batchReLU(l, t)
	} else {
		x := t.detached()
		out = x.Maximum(x.scalar(0))
	}
	return record(out, reluRule(), t)
}

// reluRule is ReLU's rule, whose derivative is 0 at 0, like PyTorch's.
func reluRule() rule {
	return rule{
		vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return grads(g.Mul(in[0].Greater(g.scalar(0)))) },
		jvp: func(t, in []*Tensor, _ *Tensor) *Tensor { return t[0].Mul(in[0].Greater(t[0].scalar(0))) },
	}
}

// The activations below have gradient rules of their own, computed from
// the input: the rules their compositions give would keep every
// intermediate value of the forward pass for the backward pass, each
// written to memory and read back, where these recompute what they need
// in the one kernel that reads the input and the gradient. The
// derivative is written with tensor operations on the input the rule
// receives, so it can be differentiated and batched in turn.

// activation returns f(x) with the rule that multiplies the gradient, or
// the tangent, by df at the input.
func activation(x *Tensor, f, df func(*Tensor) *Tensor) *Tensor {
	return record(f(x.detached()), rule{
		vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return grads(g.Mul(df(in[0]))) },
		jvp: func(t, in []*Tensor, _ *Tensor) *Tensor { return t[0].Mul(df(in[0])) },
	}, x)
}

// Sigmoid returns 1 / (1 + e^-t).
func (t *Tensor) Sigmoid() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor { return activation(x, sigmoid, dSigmoid) })
}

// sigmoid is 1 / (1 + 2^(-x·log2 e)), which is 0 or 1, not NaN, where the
// power overflows.
func sigmoid(x *Tensor) *Tensor {
	one := x.scalar(1)
	return one.Add(x.Mul(x.scalar(-math.Log2E)).Exp2()).Recip()
}

// dSigmoid is sigmoid's derivative, s·(1 - s).
func dSigmoid(x *Tensor) *Tensor {
	s := sigmoid(x)
	return s.Mul(s.scalar(1).Sub(s))
}

// tanh is 2·sigmoid(2x) - 1.
func tanh(x *Tensor) *Tensor {
	two := x.scalar(2)
	return two.Mul(sigmoid(x.Mul(two))).Sub(x.scalar(1))
}

// dTanh is tanh's derivative, 1 - tanh².
func dTanh(x *Tensor) *Tensor {
	th := tanh(x)
	return th.scalar(1).Sub(th.Mul(th))
}

// Tanh returns the hyperbolic tangent of t, computed as 2*sigmoid(2t) - 1.
func (t *Tensor) Tanh() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor { return activation(x, tanh, dTanh) })
}

// GELU's tanh approximation is x/2 · (1 + tanh(c·(x + a·x³))).
const (
	geluA = 0.044715
	geluC = 0.7978845608028654 // √(2/π)
)

// geluTanh returns tanh(c·(x + a·x³)).
func geluTanh(x *Tensor) *Tensor {
	inner := x.Add(x.scalar(geluA).Mul(x).Mul(x).Mul(x)).Mul(x.scalar(geluC))
	return tanh(inner)
}

// gelu is x/2 · (1 + tanh(c·(x + a·x³))).
func gelu(x *Tensor) *Tensor {
	return x.scalar(0.5).Mul(x).Mul(x.scalar(1).Add(geluTanh(x)))
}

// dGELU is gelu's derivative,
// (1 + th)/2 + x/2 · (1 - th²) · c·(1 + 3a·x²).
func dGELU(x *Tensor) *Tensor {
	th := geluTanh(x)
	half, one := x.scalar(0.5), x.scalar(1)
	slope := x.scalar(geluC).Mul(one.Add(x.scalar(3 * geluA).Mul(x).Mul(x)))
	return half.Mul(one.Add(th)).Add(half.Mul(x).Mul(one.Sub(th.Mul(th))).Mul(slope))
}

// GELU returns the Gaussian error linear unit with the tanh approximation,
// like torch.nn.functional.gelu(x, approximate="tanh").
func (t *Tensor) GELU() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor { return activation(x, gelu, dGELU) })
}

// silu is x·sigmoid(x).
func silu(x *Tensor) *Tensor { return x.Mul(sigmoid(x)) }

// dSiLU is silu's derivative, s + x·s·(1 - s) = s·(1 + x·(1 - s)).
func dSiLU(x *Tensor) *Tensor {
	s := sigmoid(x)
	one := x.scalar(1)
	return s.Mul(one.Add(x.Mul(one.Sub(s))))
}

// SiLU returns t * sigmoid(t), also called swish.
func (t *Tensor) SiLU() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor { return activation(x, silu, dSiLU) })
}
