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
	out := wrap(graph.Binary(op, x.node, y.node))
	if !tracked(x, y) || out.err != nil {
		return out
	}
	a, b := x.detached(), y.detached()
	var vjp func(g *Tensor) []*Tensor
	switch op {
	case backend.OpAdd:
		vjp = func(g *Tensor) []*Tensor { return grads(g, g) }
	case backend.OpMul:
		vjp = func(g *Tensor) []*Tensor { return grads(g.Mul(b), g.Mul(a)) }
	case backend.OpMax:
		// Like torch.maximum: ties split the gradient evenly.
		vjp = func(g *Tensor) []*Tensor {
			one, tie := g.scalar(1), a.Equal(b).Mul(g.scalar(0.5))
			return grads(g.Mul(one.Sub(a.Less(b)).Sub(tie)), g.Mul(one.Sub(b.Less(a)).Sub(tie)))
		}
	case backend.OpMod:
		// fmod(a, b) = a - trunc(a/b)*b, and trunc(a/b) = (a - fmod(a, b))/b.
		r := out.detached()
		vjp = func(g *Tensor) []*Tensor { return grads(g, g.Mul(a.Sub(r).Div(b)).Neg()) }
	default:
		return out
	}
	return track(out, vjp, x, y)
}

func (t *Tensor) unary(op backend.Op) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	out := wrap(graph.Unary(op, t.node))
	if !tracked(t) || out.err != nil {
		return out
	}
	x, y := t.detached(), out.detached()
	var dx func(g *Tensor) *Tensor
	switch op {
	case backend.OpNeg:
		dx = func(g *Tensor) *Tensor { return g.Neg() }
	case backend.OpExp2:
		dx = func(g *Tensor) *Tensor { return g.Mul(y).Mul(g.scalar(math.Ln2)) }
	case backend.OpLog2:
		dx = func(g *Tensor) *Tensor { return g.Div(x.Mul(g.scalar(math.Ln2))) }
	case backend.OpSin:
		dx = func(g *Tensor) *Tensor { return g.Mul(x.Cos()) }
	case backend.OpSqrt:
		dx = func(g *Tensor) *Tensor { return g.Div(y.Mul(g.scalar(2))) }
	case backend.OpRecip:
		dx = func(g *Tensor) *Tensor { return g.Mul(y).Mul(y).Neg() }
	default:
		return out
	}
	return track(out, func(g *Tensor) []*Tensor { return grads(dx(g)) }, t)
}

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
	c, a, d := b[0].detached(), b[1], b[2]
	a, d = promote(a, d)
	out := wrap(graph.Where(c.node, a.node, d.node))
	return track(out, func(g *Tensor) []*Tensor {
		zero := g.scalar(0)
		return grads(nil, Where(c, g, zero), Where(c, zero, g))
	}, b[0], a, d)
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
	out := wrap(graph.Cast(t.node, dt))
	if !from.IsFloat() || !dt.IsFloat() {
		return out
	}
	return track(out, func(g *Tensor) []*Tensor { return grads(g.Cast(from)) }, t)
}

// Clone returns a copy of t in memory of its own, like torch.Tensor.clone.
// Computing the copy reads t, so realize it before writing over t's memory
// (see RealizeInto) to keep t's values. Gradients flow through it
// unchanged.
func (t *Tensor) Clone() *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	out := wrap(graph.Cast(t.node, t.node.DType))
	return track(out, func(g *Tensor) []*Tensor { return grads(g) }, t)
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
	x := t.detached()
	out := x.Maximum(x.scalar(0))
	return track(out, func(g *Tensor) []*Tensor { return grads(g.Mul(x.Greater(g.scalar(0)))) }, t)
}

// Sigmoid returns 1 / (1 + e^-t).
func (t *Tensor) Sigmoid() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor {
		one := x.scalar(1)
		return one.Div(one.Add(x.Neg().Exp()))
	})
}

// Tanh returns the hyperbolic tangent of t, computed as 2*sigmoid(2t) - 1.
func (t *Tensor) Tanh() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor {
		two := x.scalar(2)
		return two.Mul(x.Mul(two).Sigmoid()).Sub(x.scalar(1))
	})
}

// GELU returns the Gaussian error linear unit with the tanh approximation,
// like torch.nn.functional.gelu(x, approximate="tanh").
func (t *Tensor) GELU() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor {
		inner := x.Add(x.scalar(0.044715).Mul(x).Mul(x).Mul(x)).Mul(x.scalar(math.Sqrt(2 / math.Pi)))
		return x.scalar(0.5).Mul(x).Mul(x.scalar(1).Add(inner.Tanh()))
	})
}

// SiLU returns t * sigmoid(t), also called swish.
func (t *Tensor) SiLU() *Tensor {
	return t.inFloat32(func(x *Tensor) *Tensor { return x.Mul(x.Sigmoid()) })
}
