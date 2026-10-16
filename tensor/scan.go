package tensor

import (
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// CumSum returns the running sums of t along axis, like torch.cumsum:
// element i along axis is the sum of elements 0 through i. Float32
// elements are summed in float64 on the CPU; half elements in float32,
// rounded once into t's dtype; Int32 sums wrap.
func (t *Tensor) CumSum(axis int) *Tensor {
	x, a, back := t.scanAxis(axis)
	return back(x.cumsum(a))
}

// CumMax returns the running maxima of t along axis and, as Int32, the
// position along axis each was found at, like torch.cummax. NaN
// propagates; a tie and a NaN take the latest position. The positions
// cost a second scan when computed, and nothing unless they are used.
// The gradient of the maxima flows to those positions.
func (t *Tensor) CumMax(axis int) (values, indices *Tensor) {
	x, a, back := t.scanAxis(axis)
	if x.err != nil {
		return x, x
	}
	v := wrap(graph.Scan(x.node, a, backend.CombineMax))
	idx := cumArgmax(x.detached(), v.detached(), a)
	shape := x.node.Shape
	v = track(v, func(g *Tensor) []*Tensor {
		return grads(Zeros(g.node.DType, shape...).ScatterAdd(a, idx, g))
	}, x)
	return back(v), back(idx)
}

// LogCumSumExp returns log(cumsum(exp(t))) along axis, like
// torch.logcumsumexp, without overflow for any finite elements. Float
// types only.
func (t *Tensor) LogCumSumExp(axis int) *Tensor {
	x, a, back := t.scanAxis(axis)
	return back(x.logCumSumExp(a))
}

// scanAxis returns t, a scalar t as one element, the normalized axis and
// the function that gives a scan of the result t's shape again.
func (t *Tensor) scanAxis(axis int) (*Tensor, int, func(*Tensor) *Tensor) {
	keep := func(s *Tensor) *Tensor { return s }
	if err := firstErr(t); err != nil {
		return fail(err), 0, keep
	}
	if len(t.node.Shape) == 0 && (axis == 0 || axis == -1) {
		return t.reshape([]int{1}), 0, func(s *Tensor) *Tensor { return s.reshape(nil) }
	}
	a, err := normAxis(axis, len(t.node.Shape))
	if err != nil {
		return fail(err), 0, keep
	}
	return t, a, keep
}

// cumsum is CumSum along a normalized axis. Its gradient is the running
// sum of the gradient from the end of the axis back.
func (t *Tensor) cumsum(axis int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	out := wrap(graph.Scan(t.node, axis, backend.CombineSum))
	return track(out, func(g *Tensor) []*Tensor { return grads(g.reverseScan(axis, (*Tensor).cumsum)) }, t)
}

// logCumSumExp is LogCumSumExp along a normalized axis.
func (t *Tensor) logCumSumExp(axis int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	out := wrap(graph.Scan(t.node, axis, backend.CombineLogSumExp))
	x, y := t.detached(), out.detached()
	return track(out, func(g *Tensor) []*Tensor {
		// dx[j] = sum over i >= j of g[i] exp(x[j] - y[i]), split by the
		// sign of g so that each part is a log-sum-exp from the end of
		// the axis back, like PyTorch's. Elements of g of the other sign
		// count as the lowest float, which adds nothing.
		return grads(x.inFloat32(func(x *Tensor) *Tensor {
			g, y := g.Cast(Float32), y.Cast(Float32)
			logAbs, lowest, zero := g.Abs().Log(), g.scalar(-math.MaxFloat32), g.scalar(0)
			part := func(sign *Tensor) *Tensor {
				return Where(sign, logAbs, lowest).Sub(y).reverseScan(axis, (*Tensor).logCumSumExp).Add(x).Exp()
			}
			return part(g.Greater(zero)).Sub(part(g.Less(zero)))
		}))
	}, t)
}

// reverseScan returns scan of t from the end of axis back to its start.
func (t *Tensor) reverseScan(axis int, scan func(t *Tensor, axis int) *Tensor) *Tensor {
	return scan(t.flip([]int{axis}), axis).flip([]int{axis})
}

// cumArgmax returns the position along axis of each running maximum v of
// x: the latest position i at or before it whose element is the maximum
// so far, or NaN, which v[i] equals or is NaN at.
func cumArgmax(x, v *Tensor, axis int) *Tensor {
	hit := x.Equal(v)
	if x.node.DType.IsFloat() {
		hit = hit.Maximum(x.scalar(1).Sub(x.Equal(x))) // NaN != NaN
	}
	// The positions along axis come from a scan of ones, which, unlike
	// a tensor of data, belongs to no device.
	shape := slices.Repeat([]int{1}, len(x.node.Shape))
	shape[axis] = x.node.Shape[axis]
	pos := Ones(Int32, shape...).cumsum(axis).Sub(Scalar[int32](1))
	return Where(hit, pos, Scalar[int32](0)).scan(axis, backend.CombineMax)
}

// scan is the scan primitive along a normalized axis, without a gradient.
func (t *Tensor) scan(axis int, c backend.Combine) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return wrap(graph.Scan(t.node, axis, c))
}
