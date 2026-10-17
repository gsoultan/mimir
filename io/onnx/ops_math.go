package onnx

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

func matmul(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	return one1(out(c.in[0].typ, c.t(0).MatMul(c.t(1)))), nil
}

func gemm(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	alpha, beta := c.float("alpha", 1), c.float("beta", 1)
	a, b := c.t(0), c.t(1)
	if a.Rank() != 2 || b.Rank() != 2 {
		return nil, fmt.Errorf("onnx: gemm of %v and %v: %w", a.Shape(), b.Shape(), tensor.ErrShape)
	}
	if c.int("transA", 0) == 1 {
		a = a.Transpose(0, 1)
	}
	var y *tensor.Tensor
	if c.int("transB", 0) == 1 {
		y = a.MatMulT(b) // b is [N, K], as the weights of a linear layer
	} else {
		y = a.MatMul(b)
	}
	if alpha != 1 {
		y = y.Mul(scalarOf(y, alpha))
	}
	if c.arg(2) != nil && beta != 0 {
		bias := c.t(2)
		if beta != 1 {
			bias = bias.Mul(scalarOf(bias, beta))
		}
		y = y.Add(bias)
	}
	return one1(out(c.in[0].typ, y)), nil
}

// reduceAxes returns the axes a reduction of x reduces, and whether it is
// the identity.
func (c *call) reduceAxes(x *value) ([]int, bool, error) {
	axes, given, err := c.axesArg(1)
	if err != nil {
		return nil, false, err
	}
	if !given || len(axes) == 0 {
		if c.int("noop_with_empty_axes", 0) == 1 {
			return nil, true, nil
		}
		all := make([]int, len(x.shape))
		for i := range all {
			all[i] = i
		}
		return all, false, nil
	}
	ax, err := normAxes(axes, len(x.shape))
	slices.Sort(ax)
	return ax, false, err
}

// keptShape returns shape with the axes reduced to 1, or removed.
func keptShape(shape, axes []int, keep bool) []int {
	var o []int
	for d, s := range shape {
		switch {
		case !slices.Contains(axes, d):
			o = append(o, s)
		case keep:
			o = append(o, 1)
		}
	}
	if o == nil {
		o = []int{}
	}
	return o
}

// reduce returns a reduction operator of the given kind.
func reduce(kind string) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(1); err != nil {
			return nil, err
		}
		x := c.in[0]
		axes, noop, err := c.reduceAxes(x)
		if err != nil {
			return nil, err
		}
		if noop {
			return one1(x), nil
		}
		keep := c.int("keepdims", 1) == 1
		shape := keptShape(x.shape, axes, keep)
		if x.host {
			return one1(reduceHost(kind, x, axes, shape)), nil
		}
		t := c.t(0)
		var y *tensor.Tensor
		switch kind {
		case "sum":
			y = t.Sum(axes...)
		case "mean":
			y = t.Mean(axes...)
		case "max":
			y = t.Max(axes...)
		case "min":
			y = t.Min(axes...)
		case "sumsquare":
			y = t.Mul(t).Sum(axes...)
		case "l1":
			y = t.Abs().Sum(axes...)
		case "l2":
			y = t.Mul(t).Sum(axes...).Sqrt()
		case "logsumexp":
			m := t.Max(axes...).Reshape(keptShape(x.shape, axes, true)...)
			y = t.Sub(m).Exp().Sum(axes...).Log().Add(m.Reshape(keptShape(x.shape, axes, false)...))
		default:
			return nil, c.unsupported("on a device")
		}
		return one1(out(x.typ, y.Reshape(shape...))), nil
	}
}

// reduceHost reduces host value x over axes into shape.
func reduceHost(kind string, x *value, axes, shape []int) *value {
	keepShape := slices.Clone(x.shape)
	for _, a := range axes {
		keepShape[a] = 1
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	acc := make([]float64, n)
	count := make([]int, n)
	init := map[string]float64{"max": math.Inf(-1), "min": math.Inf(1), "prod": 1}[kind]
	for i := range acc {
		acc[i] = init
	}
	st := strides(keepShape)
	i := 0
	forEach(x.shape, func(p []int) {
		o := 0
		for d, q := range p {
			if keepShape[d] != 1 {
				o += q * st[d]
			}
		}
		v := x.float(i)
		i++
		count[o]++
		switch kind {
		case "max":
			acc[o] = math.Max(acc[o], v)
		case "min":
			acc[o] = math.Min(acc[o], v)
		case "prod":
			acc[o] *= v
		case "sumsquare", "l2":
			acc[o] += v * v
		case "l1":
			acc[o] += math.Abs(v)
		case "logsumexp":
			acc[o] += math.Exp(v)
		default:
			acc[o] += v
		}
	})
	for j := range acc {
		switch kind {
		case "mean":
			acc[j] /= float64(count[j])
		case "l2":
			acc[j] = math.Sqrt(acc[j])
		case "logsumexp":
			acc[j] = math.Log(acc[j])
		}
	}
	if x.typ.IsFloat() {
		return hostFloats(x.typ, shape, acc)
	}
	ints := make([]int64, n)
	for j, f := range acc {
		ints[j] = toInt(f)
	}
	return hostInts(x.typ, shape, ints)
}

// argmax returns ArgMax, or ArgMin if least.
func argmax(least bool) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(1); err != nil {
			return nil, err
		}
		x := c.in[0]
		axis, err := normAxis(c.int("axis", 0), len(x.shape))
		if err != nil {
			return nil, err
		}
		if c.int("select_last_index", 0) == 1 {
			return nil, c.unsupported("select_last_index")
		}
		shape := keptShape(x.shape, []int{axis}, c.int("keepdims", 1) == 1)
		t := c.t(0)
		if least {
			t = t.Neg()
		}
		return one1(&value{typ: Int64, shape: shape, t: t.Argmax(axis).Reshape(shape...)}), nil
	}
}

func cumsum(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	a, err := c.host(1)
	if err != nil {
		return nil, err
	}
	if a.size() != 1 {
		return nil, fmt.Errorf("onnx: cumsum axis of shape %v: %w", a.shape, tensor.ErrShape)
	}
	axis, err := normAxis(a.int(0), len(x.shape))
	if err != nil {
		return nil, err
	}
	exclusive, reverse := c.int("exclusive", 0) == 1, c.int("reverse", 0) == 1
	t := c.t(0)
	if reverse {
		t = t.Flip(axis)
	}
	s := t.CumSum(axis)
	if exclusive {
		// Shift the sums one place along the axis.
		pads := make([][2]int, len(x.shape))
		pads[axis][0] = 1
		r := whole(x.shape)
		r[axis][1] = x.shape[axis]
		s = s.Pad(pads, 0).Shrink(r)
	}
	if reverse {
		s = s.Flip(axis)
	}
	return one1(out(x.typ, s)), nil
}
