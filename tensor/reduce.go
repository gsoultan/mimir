package tensor

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// normAxis maps a possibly negative axis into [0, rank).
func normAxis(axis, rank int) (int, error) {
	if axis < -rank || axis >= rank {
		return 0, fmt.Errorf("tensor: axis %d for rank %d: %w", axis, rank, ErrShape)
	}
	if axis < 0 {
		axis += rank
	}
	return axis, nil
}

// normAxes normalizes and sorts axes, rejecting duplicates. No axes means
// every axis.
func normAxes(axes []int, rank int) ([]int, error) {
	if len(axes) == 0 {
		out := make([]int, rank)
		for i := range out {
			out[i] = i
		}
		return out, nil
	}
	out := make([]int, len(axes))
	for i, a := range axes {
		var err error
		if out[i], err = normAxis(a, rank); err != nil {
			return nil, err
		}
	}
	slices.Sort(out)
	if len(slices.Compact(slices.Clone(out))) != len(out) {
		return nil, fmt.Errorf("tensor: repeated axis in %v: %w", axes, ErrShape)
	}
	return out, nil
}

// reduceKeep reduces over axes and keeps them with size 1.
func (t *Tensor) reduceKeep(op backend.Op, axes []int) (*Tensor, []int) {
	if err := firstErr(t); err != nil {
		return fail(err), nil
	}
	ax, err := normAxes(axes, len(t.node.Shape))
	if err != nil {
		return fail(err), nil
	}
	return t.reduce(op, ax), ax
}

// reduce is the reduction primitive over normalized axes, keeping them with
// size 1. A maximum's gradient is split evenly between tied elements, like
// torch.amax's.
func (t *Tensor) reduce(op backend.Op, axes []int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if len(axes) == 0 {
		return t
	}
	var out *Tensor
	if l := batchLevel(t); l != 0 {
		out = batchReduce(l, op, t, axes)
	} else {
		out = wrap(graph.Reduce(op, t.node, axes))
	}
	if op == backend.OpReduceSum {
		return record(out, sumRule(), t)
	}
	return record(out, maxReduceRule(), t)
}

// reducedAxes returns the axes that a keep-dims reduction of in into out
// reduced, but for those of length 1, whose reduction changes nothing.
func reducedAxes(in, out *Tensor) []int {
	var axes []int
	for i, d := range in.node.Shape {
		if d != out.node.Shape[i] {
			axes = append(axes, i)
		}
	}
	return axes
}

// sumRule is the rule of a sum.
func sumRule() rule {
	return rule{
		vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return grads(g.expand(in[0].node.Shape)) },
		jvp: func(t, in []*Tensor, out *Tensor) *Tensor {
			return t[0].reduce(backend.OpReduceSum, reducedAxes(in[0], out))
		},
	}
}

// maxReduceRule is the rule of a maximum, whose derivative is shared
// evenly between tied elements.
func maxReduceRule() rule {
	return rule{
		vjp: func(g *Tensor, in []*Tensor, out *Tensor) []*Tensor {
			hit := in[0].Equal(out)
			return grads(hit.Mul(g.Div(hit.reduce(backend.OpReduceSum, reducedAxes(in[0], out)))))
		},
		jvp: func(t, in []*Tensor, out *Tensor) *Tensor {
			axes := reducedAxes(in[0], out)
			hit := in[0].Equal(out)
			return hit.Mul(t[0]).reduce(backend.OpReduceSum, axes).Div(hit.reduce(backend.OpReduceSum, axes))
		},
	}
}

// dropAxes removes the size-1 axes left by a keep-dims reduction.
func (t *Tensor) dropAxes(axes []int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	shape := make([]int, 0, len(t.node.Shape))
	for d, s := range t.node.Shape {
		if !slices.Contains(axes, d) {
			shape = append(shape, s)
		}
	}
	return t.reshape(shape)
}

// Sum adds the elements over axes, which are removed from the shape. With
// no axes it sums everything into a scalar.
func (t *Tensor) Sum(axes ...int) *Tensor {
	r, ax := t.reduceKeep(backend.OpReduceSum, axes)
	return r.dropAxes(ax)
}

// Max returns the largest element over axes. NaN propagates.
func (t *Tensor) Max(axes ...int) *Tensor {
	r, ax := t.reduceKeep(backend.OpReduceMax, axes)
	return r.dropAxes(ax)
}

// Min returns the smallest element over axes. NaN propagates.
func (t *Tensor) Min(axes ...int) *Tensor { return t.Neg().Max(axes...).Neg() }

// count returns the number of elements reduced into each output element.
func (t *Tensor) count(axes []int) float64 {
	n := 1.0
	for _, a := range axes {
		n *= float64(t.node.Shape[a])
	}
	return n
}

// meanKeep returns the mean over axes, keeping them with size 1.
func (t *Tensor) meanKeep(axes []int) (*Tensor, []int) {
	s, ax := t.reduceKeep(backend.OpReduceSum, axes)
	if s.err != nil {
		return s, nil
	}
	return s.Div(t.scalar(t.count(ax))), ax
}

// Mean returns the arithmetic mean over axes. Float32 only.
func (t *Tensor) Mean(axes ...int) *Tensor {
	m, ax := t.meanKeep(axes)
	return m.dropAxes(ax)
}

// Var returns the sample variance over axes, dividing by N-1 like
// torch.var. Float32 only.
func (t *Tensor) Var(axes ...int) *Tensor {
	m, ax := t.meanKeep(axes)
	if m.err != nil {
		return m
	}
	d := t.Sub(m)
	s, _ := d.Mul(d).reduceKeep(backend.OpReduceSum, ax)
	return s.Div(t.scalar(t.count(ax) - 1)).dropAxes(ax)
}

// Softmax returns exp(t) normalized to sum to 1 along axis.
func (t *Tensor) Softmax(axis int) *Tensor {
	e := t.shiftedByMax(axis).Exp()
	s, _ := e.reduceKeep(backend.OpReduceSum, []int{axis})
	return e.Div(s)
}

// LogSoftmax returns log(softmax(t)) along axis, computed stably.
func (t *Tensor) LogSoftmax(axis int) *Tensor {
	z := t.shiftedByMax(axis)
	s, _ := z.Exp().reduceKeep(backend.OpReduceSum, []int{axis})
	return z.Sub(s.Log())
}

// shiftedByMax subtracts the maximum along axis so exponentials cannot
// overflow. The result of softmax does not depend on the shift, so no
// gradient flows through the maximum.
func (t *Tensor) shiftedByMax(axis int) *Tensor {
	m, _ := t.stopGradient().reduceKeep(backend.OpReduceMax, []int{axis})
	return t.Sub(m)
}

// Argmax returns the Int32 position of the largest element along axis,
// which is removed from the shape. Ties return the first position and NaN
// counts as largest, like torch.argmax. It has no gradient.
func (t *Tensor) Argmax(axis int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	a, err := normAxis(axis, len(t.node.Shape))
	if err != nil {
		return fail(err)
	}
	x := t.detached()
	m, _ := x.reduceKeep(backend.OpReduceMax, []int{a})
	hit := x.Equal(m)
	if x.node.DType.IsFloat() {
		hit = hit.Maximum(x.scalar(1).Sub(x.Equal(x))) // NaN != NaN
	}
	n := x.node.Shape[a]
	shape := slices.Repeat([]int{1}, len(x.node.Shape))
	shape[a] = n
	// The positions come from a scan of ones, which, unlike a tensor of
	// data, belongs to no device, so t may be on any.
	pos := Ones(Int32, shape...).cumsum(a).Sub(Scalar[int32](1))
	return Where(hit, pos, Scalar(int32(n))).Min(a)
}
