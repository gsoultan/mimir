package onnx

import (
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// maskKind is how a fused attention's mask hides keys.
type maskKind int

const (
	noMask       maskKind = iota
	additiveMask          // added to the scores: -Inf hides a key
	keepWhere             // a bool, non-zero where a query may attend
	hideWhere             // a bool, non-zero where a query may not attend
)

// attentionSpec is attention that a graph composes from MatMul and
// Softmax, which fuseAttention found:
//
//	MatMul(Softmax(mask(MatMul(q, kᵀ)·scale)), v)
//
// with kᵀ a Transpose of k or a value of its own, the scale a constant
// that multiplies or divides the scores or q and kᵀ, and the mask an Add
// or a Where that fills hidden scores with -Inf, or none.
type attentionSpec struct {
	scale      float64
	transposed bool // the graph gives kᵀ, not k
	axis       int64
	mask       maskKind
}

// fuseAttention replaces attention composed of MatMul and Softmax with
// one node that runs tensor.ScaledDotProductAttention, which never holds
// the scores. A value the pattern computes may have no other reader, so
// the graph computes the same outputs, save that a query whose keys are
// all hidden gets zeros rather than NaN.
func (m *Model) fuseAttention() {
	producer := map[string]*node{}
	readers := map[string]int{}
	for _, n := range m.nodes {
		for _, o := range n.outputs {
			producer[o] = n
		}
		for _, in := range n.inputs {
			readers[in]++
		}
	}
	for _, o := range m.outputs {
		readers[o.Name] += 2 // never absorbed
	}
	// only returns the node that computes name if it is op and name has no
	// other reader.
	only := func(name, op string) *node {
		n := producer[name]
		if n == nil || n.op != op || readers[name] != 1 || len(n.inputs) < 1 {
			return nil
		}
		return n
	}
	// scalar returns the value of a constant of one element.
	scalar := func(name string) (float64, bool) {
		v, ok := m.consts[name]
		if !ok || !v.host || v.size() != 1 || !v.typ.IsFloat() {
			return 0, false
		}
		return v.floats[0], true
	}
	// scaled returns x, less a multiplication by a constant, and the
	// constant.
	scaled := func(x string) (string, float64, *node) {
		n := only(x, "Mul")
		if n == nil || len(n.inputs) != 2 {
			return x, 1, nil
		}
		for i, in := range n.inputs {
			if s, ok := scalar(in); ok {
				return n.inputs[1-i], s, n
			}
		}
		return x, 1, nil
	}
	// scores parses s as q·kᵀ·scale, returning q, k and the nodes that
	// compute s from them.
	scores := func(s string, spec *attentionSpec) (q, k string, absorbed []*node, ok bool) {
		spec.scale, spec.transposed = 1, true
		if n := only(s, "Div"); n != nil && len(n.inputs) == 2 {
			d, isConst := scalar(n.inputs[1])
			if !isConst || d == 0 {
				return "", "", nil, false
			}
			spec.scale, s = 1/d, n.inputs[0]
			absorbed = append(absorbed, n)
		} else if x, c, n := scaled(s); n != nil {
			spec.scale, s = c, x
			absorbed = append(absorbed, n)
		}
		qk := only(s, "MatMul")
		if qk == nil || len(qk.inputs) != 2 {
			return "", "", nil, false
		}
		absorbed = append(absorbed, qk)
		q, k = qk.inputs[0], qk.inputs[1]
		for _, x := range []*string{&q, &k} {
			if y, c, n := scaled(*x); n != nil {
				*x, spec.scale = y, spec.scale*c
				absorbed = append(absorbed, n)
			}
		}
		if t := only(k, "Transpose"); t != nil && swapsLast(t) {
			k, spec.transposed = t.inputs[0], false
			absorbed = append(absorbed, t)
		}
		return q, k, absorbed, true
	}
	fused := map[*node]*node{} // final MatMul to its replacement
	gone := map[*node]bool{}
	for _, out := range m.nodes {
		if out.op != "MatMul" || len(out.inputs) != 2 || m.opset < 13 {
			continue
		}
		sm := only(out.inputs[0], "Softmax")
		if sm == nil {
			continue
		}
		spec := attentionSpec{axis: -1}
		for _, a := range sm.attrs {
			if a.name == "axis" {
				spec.axis = a.i
			}
		}
		absorbed := []*node{sm}
		s, mask := sm.inputs[0], ""
		if n := only(s, "Where"); n != nil && len(n.inputs) == 3 {
			if f, ok := scalar(n.inputs[1]); ok && math.IsInf(f, -1) {
				spec.mask, mask, s = hideWhere, n.inputs[0], n.inputs[2]
			} else if f, ok := scalar(n.inputs[2]); ok && math.IsInf(f, -1) {
				spec.mask, mask, s = keepWhere, n.inputs[0], n.inputs[1]
			} else {
				continue
			}
			absorbed = append(absorbed, n)
		} else if n := only(s, "Add"); n != nil && len(n.inputs) == 2 {
			// The scores are the side that parses as them, which must be
			// one side alone.
			var s0, s1 attentionSpec
			_, _, _, ok0 := scores(n.inputs[0], &s0)
			_, _, _, ok1 := scores(n.inputs[1], &s1)
			if ok0 == ok1 {
				continue
			}
			i := 0
			if ok1 {
				i = 1
			}
			spec.mask, mask, s = additiveMask, n.inputs[1-i], n.inputs[i]
			absorbed = append(absorbed, n)
		}
		q, k, more, ok := scores(s, &spec)
		if !ok {
			continue
		}
		absorbed = append(absorbed, more...)
		if slices.ContainsFunc(absorbed, func(n *node) bool { return gone[n] || fused[n] != nil }) {
			continue
		}
		for _, n := range absorbed {
			gone[n] = true
		}
		f := &node{nodeProto: &nodeProto{name: out.name, op: "Attention", outputs: out.outputs, inputs: []string{q, k, out.inputs[1], mask}}}
		f.fn = attention(spec)
		fused[out] = f
	}
	if len(fused) == 0 {
		return
	}
	kept := m.nodes[:0]
	for _, n := range m.nodes {
		switch {
		case fused[n] != nil:
			kept = append(kept, fused[n])
		case !gone[n]:
			kept = append(kept, n)
		}
	}
	m.nodes = kept
}

// swapsLast reports whether Transpose node t swaps the last two axes of
// its input and keeps the others.
func swapsLast(t *node) bool {
	for _, a := range t.attrs {
		if a.name != "perm" {
			continue
		}
		p := a.ints
		r := len(p)
		if r < 2 || p[r-2] != int64(r-1) || p[r-1] != int64(r-2) {
			return false
		}
		for i := range r - 2 {
			if p[i] != int64(i) {
				return false
			}
		}
		return true
	}
	return false
}

// attention returns the operator of a fused attention, whose inputs are
// q, k (or kᵀ), v and the mask, if any.
func attention(spec attentionSpec) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(3); err != nil {
			return nil, err
		}
		q, k, v := c.t(0), c.t(1), c.t(2)
		r := q.Rank()
		kt := k
		if spec.transposed {
			if k.Rank() >= 2 {
				k = k.Transpose(-1, -2)
			}
		} else if k.Rank() >= 2 {
			kt = k.Transpose(-1, -2)
		}
		axis := spec.axis
		if axis < 0 {
			axis += int64(r)
		}
		fits := r >= 2 && k.Rank() == r && v.Rank() == r && axis == int64(r-1) &&
			q.DType().IsFloat() && k.DType() == q.DType() && v.DType() == q.DType()
		var mask *tensor.Tensor
		var opts []tensor.AttentionOption
		if spec.mask != noMask {
			mv := c.arg(3)
			if mv == nil {
				return nil, c.need(4)
			}
			if fits && mv.host {
				if off, ok := causalOffset(mv, spec.mask, q.Shape()[r-2], k.Shape()[r-2]); ok {
					opts = append(opts, tensor.CausalOffset(off))
				}
			}
			if len(opts) == 0 {
				mask = c.tensorOf(mv)
			}
		}
		if !fits {
			// Not attention that the fused kernel computes: the graph's
			// own operations.
			s := q.MatMul(kt)
			if spec.scale != 1 {
				s = s.Mul(scalarOf(s, spec.scale))
			}
			switch spec.mask {
			case additiveMask:
				s = s.Add(mask)
			case keepWhere:
				s = tensor.Where(mask, s, scalarOf(s, math.Inf(-1)))
			case hideWhere:
				s = tensor.Where(mask, scalarOf(s, math.Inf(-1)), s)
			}
			a, err := normAxis(spec.axis, s.Rank())
			if err != nil {
				return nil, err
			}
			return one1(out(c.in[0].typ, s.Softmax(a).MatMul(v))), nil
		}
		opts = append(opts, tensor.AttentionScale(spec.scale))
		if mask != nil {
			switch spec.mask {
			case additiveMask:
				opts = append(opts, tensor.AdditiveMask(mask))
			case keepWhere:
				opts = append(opts, tensor.BoolMask(mask))
			case hideWhere:
				opts = append(opts, tensor.BoolMask(notT(mask.Cast(tensor.Int32))))
			}
		}
		return one1(out(c.in[0].typ, tensor.ScaledDotProductAttention(q, k, v, opts...))), nil
	}
}

// causalOffset reports whether host mask v hides exactly the keys j >
// i + offset from query i, for nq queries and nk keys, and returns the
// offset.
func causalOffset(v *value, kind maskKind, nq, nk int) (int, bool) {
	r := len(v.shape)
	if r < 2 || v.shape[r-2] != nq || v.shape[r-1] != nk || v.size() != nq*nk || nq == 0 || nk == 0 {
		return 0, false
	}
	attends := func(i int) bool {
		x := v.float(i)
		switch kind {
		case additiveMask:
			return !math.IsInf(x, -1)
		case keepWhere:
			return x != 0
		}
		return x == 0
	}
	// Row 0 attends to keys 0 to offset.
	off := -1
	for j := range nk {
		if !attends(j) {
			break
		}
		off = j
	}
	if off < 0 {
		return 0, false
	}
	for i := range nq {
		for j := range nk {
			x := v.float(i*nk + j)
			if attends(i*nk+j) != (j <= i+off) || kind == additiveMask && x != 0 && !math.IsInf(x, -1) {
				return 0, false
			}
		}
	}
	return off, true
}
