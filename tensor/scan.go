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
	var v, idx *Tensor
	if l := batchLevel(x); l != 0 {
		v, idx = batchCumMax(l, x, a)
	} else {
		v = wrap(graph.Scan(x.node, a, backend.CombineMax))
		idx = cumArgmax(x.detached(), v.detached(), a)
	}
	if !tracked(x) || v.err != nil {
		return back(v), back(idx)
	}
	v = record(v, rule{
		vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor {
			return grads(Zeros(g.node.DType, in[0].node.Shape...).ScatterAdd(a, idx, g))
		},
		jvp: func(t, _ []*Tensor, _ *Tensor) *Tensor { return t[0].Gather(a, idx) },
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
	var out *Tensor
	if l := batchLevel(t); l != 0 {
		out = batchCumSum(l, t, axis)
	} else {
		out = wrap(graph.Scan(t.node, axis, backend.CombineSum))
	}
	if !tracked(t) || out.err != nil {
		return out
	}
	return record(out, rule{
		vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(g.reverseScan(axis, (*Tensor).cumsum)) },
		jvp: func(t, _ []*Tensor, _ *Tensor) *Tensor { return t[0].cumsum(axis) },
	}, t)
}

// logCumSumExp is LogCumSumExp along a normalized axis.
func (t *Tensor) logCumSumExp(axis int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	var out *Tensor
	if l := batchLevel(t); l != 0 {
		out = batchLogCumSumExp(l, t, axis)
	} else {
		out = wrap(graph.Scan(t.node, axis, backend.CombineLogSumExp))
	}
	if !tracked(t) || out.err != nil {
		return out
	}
	return record(out, rule{
		// dx[j] = sum over i >= j of g[i] exp(x[j] - y[i]).
		vjp: func(g *Tensor, in []*Tensor, out *Tensor) []*Tensor {
			return grads(in[0].inFloat32(func(x *Tensor) *Tensor {
				return expWeighted(g, x, out, axis, true, anyTracked(g, in[0], out))
			}))
		},
		// dy[i] = sum over j <= i of t[j] exp(x[j] - y[i]).
		jvp: func(t, in []*Tensor, out *Tensor) *Tensor {
			return in[0].inFloat32(func(x *Tensor) *Tensor {
				return expWeighted(t[0], x, out, axis, false, true)
			})
		},
	}, t)
}

// expWeighted returns the sums of w[k] exp(x[j] - y[i]), over k = i at
// each j and i >= j when back, the gradient of a log-sum-exp scan y of x
// along axis, and over k = j at each i and j <= i otherwise, its tangent.
// The sums run as log-sum-exp scans, so that no exponential overflows,
// of the logarithms of two positive parts of w whose difference is w.
//
// Like PyTorch, it parts w by sign, with log(|w|) where w has the sign
// and the lowest float elsewhere, which is exact but has derivative 0 in
// w wherever w is 0. When smooth, for sums that are differentiated in
// turn, the parts are w + |w| + d and |w| + d, which are never 0, with d
// the largest |w| along the axis: their logarithms have the derivative
// of w everywhere, at the cost of a rounding error of about 3d times the
// float epsilon.
func expWeighted(w, x, y *Tensor, axis int, back, smooth bool) *Tensor {
	w, y = w.Cast(Float32), y.Cast(Float32)
	sum := func(logPart *Tensor) *Tensor {
		if back {
			return logPart.Sub(y).reverseScan(axis, (*Tensor).logCumSumExp).Add(x).Exp()
		}
		return logPart.Add(x).logCumSumExp(axis).Sub(y).Exp()
	}
	abs := w.Abs()
	if smooth {
		d, _ := abs.stopGradient().reduceKeep(backend.OpReduceMax, []int{axis})
		d = d.Add(w.scalar(0x1p-126)) // the least normal float32, should w be all 0
		neg := abs.Add(d)
		return sum(w.Add(neg).Log()).Sub(sum(neg.Log()))
	}
	logAbs, lowest, zero := abs.Log(), w.scalar(-math.MaxFloat32), w.scalar(0)
	part := func(sign *Tensor) *Tensor { return sum(Where(sign, logAbs, lowest)) }
	return part(w.Greater(zero)).Sub(part(w.Less(zero)))
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
	if l := batchLevel(t); l != 0 {
		return batchScan(l, t, axis, c)
	}
	return wrap(graph.Scan(t.node, axis, c))
}
