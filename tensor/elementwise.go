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

// expandTo broadcasts n to shape, adding leading dimensions as needed.
func expandTo(n *graph.Node, shape []int) (*graph.Node, error) {
	if slices.Equal(n.Shape, shape) {
		return n, nil
	}
	if extra := len(shape) - len(n.Shape); extra > 0 {
		ones := slices.Repeat([]int{1}, extra)
		var err error
		if n, err = graph.Reshape(n, slices.Concat(ones, n.Shape)); err != nil {
			return nil, err
		}
	}
	return graph.Expand(n, shape)
}

// broadcast expands every tensor to their common shape.
func broadcast(ts ...*Tensor) ([]*graph.Node, error) {
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
	nodes := make([]*graph.Node, len(ts))
	for i, t := range ts {
		if nodes[i], err = expandTo(t.node, shape); err != nil {
			return nil, err
		}
	}
	return nodes, nil
}

func (t *Tensor) binary(op backend.Op, u *Tensor) *Tensor {
	n, err := broadcast(t, u)
	if err != nil {
		return fail(err)
	}
	return wrap(graph.Binary(op, n[0], n[1]))
}

func (t *Tensor) unary(op backend.Op) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return wrap(graph.Unary(op, t.node))
}

// scalar returns a zero-dimensional constant with t's dtype.
func (t *Tensor) scalar(v float64) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return Full(t.node.DType, v)
}

// Add returns t + u, broadcasting like NumPy.
func (t *Tensor) Add(u *Tensor) *Tensor { return t.binary(backend.OpAdd, u) }

// Sub returns t - u.
func (t *Tensor) Sub(u *Tensor) *Tensor { return t.Add(u.Neg()) }

// Mul returns t * u.
func (t *Tensor) Mul(u *Tensor) *Tensor { return t.binary(backend.OpMul, u) }

// Div returns t / u. Float32 only.
func (t *Tensor) Div(u *Tensor) *Tensor { return t.Mul(u.Recip()) }

// Maximum returns the elementwise maximum. NaN propagates.
func (t *Tensor) Maximum(u *Tensor) *Tensor { return t.binary(backend.OpMax, u) }

// Minimum returns the elementwise minimum. NaN propagates.
func (t *Tensor) Minimum(u *Tensor) *Tensor { return t.Neg().Maximum(u.Neg()).Neg() }

// Less returns 1 where t < u and 0 elsewhere, in t's dtype.
func (t *Tensor) Less(u *Tensor) *Tensor { return t.binary(backend.OpCmpLT, u) }

// Greater returns 1 where t > u and 0 elsewhere, in t's dtype.
func (t *Tensor) Greater(u *Tensor) *Tensor { return u.Less(t) }

// Equal returns 1 where t == u and 0 elsewhere, in t's dtype.
func (t *Tensor) Equal(u *Tensor) *Tensor { return t.binary(backend.OpCmpEQ, u) }

// Fmod returns the remainder of t / u with the sign of t, like torch.fmod.
// Integer division by zero is ErrDivideByZero; float division by zero is NaN.
func (t *Tensor) Fmod(u *Tensor) *Tensor { return t.binary(backend.OpMod, u) }

// Where returns x where cond is non-zero and y elsewhere, broadcasting all
// three. cond may have any dtype.
func Where(cond, x, y *Tensor) *Tensor {
	n, err := broadcast(cond, x, y)
	if err != nil {
		return fail(err)
	}
	return wrap(graph.Where(n[0], n[1], n[2]))
}

// Cast converts t to dt. Float to integer truncates toward zero; values
// outside the integer range are undefined.
func (t *Tensor) Cast(dt DType) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if t.node.DType == dt {
		return t
	}
	return wrap(graph.Cast(t.node, dt))
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

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor { return t.Mul(t.scalar(math.Log2E)).Exp2() }

// Log returns the natural logarithm of t.
func (t *Tensor) Log() *Tensor { return t.Log2().Mul(t.scalar(math.Ln2)) }

// Cos returns cos(t).
func (t *Tensor) Cos() *Tensor { return t.Add(t.scalar(math.Pi / 2)).Sin() }

// Abs returns |t|.
func (t *Tensor) Abs() *Tensor { return t.Maximum(t.Neg()) }

// ReLU returns max(t, 0).
func (t *Tensor) ReLU() *Tensor { return t.Maximum(t.scalar(0)) }

// Sigmoid returns 1 / (1 + e^-t).
func (t *Tensor) Sigmoid() *Tensor {
	one := t.scalar(1)
	return one.Div(one.Add(t.Neg().Exp()))
}

// Tanh returns the hyperbolic tangent of t, computed as 2*sigmoid(2t) - 1.
func (t *Tensor) Tanh() *Tensor {
	two := t.scalar(2)
	return two.Mul(t.Mul(two).Sigmoid()).Sub(t.scalar(1))
}

// GELU returns the Gaussian error linear unit with the tanh approximation,
// like torch.nn.functional.gelu(x, approximate="tanh").
func (t *Tensor) GELU() *Tensor {
	inner := t.Add(t.scalar(0.044715).Mul(t).Mul(t).Mul(t)).Mul(t.scalar(math.Sqrt(2 / math.Pi)))
	return t.scalar(0.5).Mul(t).Mul(t.scalar(1).Add(inner.Tanh()))
}

// SiLU returns t * sigmoid(t), also called swish.
func (t *Tensor) SiLU() *Tensor { return t.Mul(t.Sigmoid()) }
