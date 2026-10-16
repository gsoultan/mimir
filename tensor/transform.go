package tensor

import (
	"errors"
	"fmt"
	"maps"
	"slices"
)

// ErrNotDifferentiable reports a derivative that cannot be computed: a
// gradient differentiated again through an operation whose gradient rule
// gives first derivatives only, such as a Func not marked Differentiable.
var ErrNotDifferentiable = errors.New("tensor: derivative not supported")

// Gradients returns the gradients of y, which has one element, with
// respect to each of xs, like torch.autograd.grad with create_graph set.
// Each x must require gradients: it comes from RequiresGrad or a
// Param's Value, or was computed from such a tensor, and y was computed
// from it. The gradient with respect to an x that y does not depend on is
// zeros.
//
// Unlike Backward, Gradients changes no accumulated gradient and releases
// nothing. The gradients are lazy tensors like any other, computed when
// realized, and they require gradients themselves when what they were
// computed from does, so they can be differentiated again: Gradients of a
// gradient is a second derivative. Effects (see WithEffects) are not
// computed.
func Gradients(y *Tensor, xs ...*Tensor) []*Tensor {
	if err := firstErr(y); err != nil {
		return failAll(len(xs), err)
	}
	if n := y.node.Size(); n != 1 {
		return failAll(len(xs), fmt.Errorf("tensor: gradients of %d elements, want 1: %w", n, ErrShape))
	}
	return pullback([]*Tensor{y}, xs, []*Tensor{Ones(y.node.DType, y.node.Shape...)})
}

// pullback returns the gradients of ys with respect to xs, given the
// gradients gs of ys: the sum over ys of the vector-Jacobian products.
// They are differentiable (see Gradients).
func pullback(ys, xs, gs []*Tensor) []*Tensor {
	out := make([]*Tensor, len(xs))
	if err := firstErr(slices.Concat(ys, xs, gs)...); err != nil {
		return failAll(len(xs), err)
	}
	var roots []*tape
	var seeds []*Tensor
	for i, y := range ys {
		g := gs[i]
		if !slices.Equal(g.node.Shape, y.node.Shape) {
			return failAll(len(xs), fmt.Errorf("tensor: gradient of shape %v for a result of shape %v: %w", g.node.Shape, y.node.Shape, ErrShape))
		}
		if y.tape != nil {
			roots, seeds = append(roots, y.tape), append(seeds, g)
		}
	}
	r := reverse{create: true, want: map[*tape]bool{}}
	for i, x := range xs {
		if x.tape == nil {
			out[i] = fail(fmt.Errorf("tensor: gradient with respect to a tensor that is not tracked: %w", ErrNoGrad))
			continue
		}
		r.want[x.tape] = true
	}
	if len(roots) > 0 {
		if _, _, _, err := r.propagate(roots, seeds, nil); err != nil {
			return failAll(len(xs), err)
		}
	}
	for i, x := range xs {
		if out[i] == nil {
			if out[i] = r.result(r.got[x.tape]); out[i] == nil {
				out[i] = Zeros(x.node.DType, x.node.Shape...)
			}
		}
	}
	return out
}

// standIns returns a tensor standing for each of xs (see standIn), and
// their tapes.
func standIns(xs []*Tensor) ([]*Tensor, []*tape) {
	args, ts := make([]*Tensor, len(xs)), make([]*tape, len(xs))
	for i, x := range xs {
		args[i] = standIn(x)
		ts[i] = args[i].tape
	}
	return args, ts
}

// checkArgs returns the error of the first failed argument of a
// transformation, or reports one that is not a float tensor.
func checkArgs(what string, xs []*Tensor) error {
	if err := firstErr(xs...); err != nil {
		return err
	}
	for _, x := range xs {
		if !x.node.DType.IsFloat() {
			return fmt.Errorf("tensor: %s with respect to a %s argument: %w", what, x.node.DType, ErrDType)
		}
	}
	return nil
}

// Grad returns the gradient of f, like jax.grad: a function that returns
// the gradients of f(x...), which has one element, with respect to each
// of the float arguments x. The arguments need not require gradients. The
// gradients are lazy, and differentiable with respect to any argument
// that requires gradients and anything else f reads that does, such as
// parameters, so that Grad of Grad, or JVP of Grad, gives second
// derivatives (see Hessian).
func Grad(f func(x ...*Tensor) *Tensor) func(x ...*Tensor) []*Tensor {
	return func(x ...*Tensor) []*Tensor {
		if err := checkArgs("gradient", x); err != nil {
			return failAll(len(x), err)
		}
		args, _ := standIns(x)
		y := f(args...)
		if y == nil {
			return failAll(len(x), fmt.Errorf("tensor: Grad of a function that returned nil: %w", ErrNilTensor))
		}
		return Gradients(y, args...)
	}
}

// VJP returns f(x...) and its vector-Jacobian product, like jax.vjp: a
// function that maps gradients of the results, one per result and shaped
// like it, to the gradients of the sum of their products with the
// results, with respect to each argument x. f runs once, and the function
// may be called any number of times, also inside VMap, which computes
// many products at once. Its gradients are lazy and differentiable, as
// those of Grad are.
func VJP(f func(x ...*Tensor) []*Tensor, x ...*Tensor) (y []*Tensor, vjp func(g ...*Tensor) []*Tensor) {
	if err := checkArgs("VJP", x); err != nil {
		return failedCall(f, len(x), err), func(...*Tensor) []*Tensor { return failAll(len(x), err) }
	}
	args, _ := standIns(x)
	y = f(args...)
	if err := firstErr(y...); err != nil {
		return failAll(len(y), err), func(...*Tensor) []*Tensor { return failAll(len(x), err) }
	}
	return y, func(g ...*Tensor) []*Tensor {
		if len(g) != len(y) {
			return failAll(len(x), fmt.Errorf("tensor: VJP of %d gradients for %d results: %w", len(g), len(y), ErrShape))
		}
		return pullback(y, args, g)
	}
}

// JVP returns f(x...) and the derivative of f at x applied to the
// tangents t, one per argument and shaped like it: the Jacobian-vector
// product, or directional derivative, like jax.jvp. It runs f once,
// recording what it computes, and pushes the tangents forward through
// it, each operation's tangent computed from those of its inputs, so that
// a JVP costs about two evaluations of f and keeps no gradients. The
// tangents of results that do not depend on the arguments are zeros.
//
// The tangents are lazy, and differentiable as the gradients of Grad
// are: Grad of a JVP, or JVP of Grad, gives second derivatives. Inside
// VMap, a batch of tangents gives many directional derivatives at once
// (see Jacobian).
func JVP(f func(x ...*Tensor) []*Tensor, x, t []*Tensor) (y, ty []*Tensor) {
	err := checkArgs("JVP", x)
	switch {
	case err != nil:
	case len(t) != len(x):
		err = fmt.Errorf("tensor: JVP of %d tangents for %d arguments: %w", len(t), len(x), ErrShape)
	default:
		err = firstErr(t...)
	}
	for i := range t {
		if err == nil && (!slices.Equal(t[i].node.Shape, x[i].node.Shape) || !t[i].node.DType.IsFloat()) {
			err = fmt.Errorf("tensor: JVP of a %s%v tangent for a %s%v argument: %w", t[i].node.DType, t[i].node.Shape, x[i].node.DType, x[i].node.Shape, ErrShape)
		}
	}
	if err != nil {
		y = failedCall(f, len(x), err)
		return y, failAll(len(y), err)
	}
	return jvpCall(f, x, t, nil)
}

// failedCall returns what a transformation of f returns when its
// arguments carry err: as many tensors carrying err as f has results,
// which f, called on n tensors carrying err, tells. A panic in that call,
// as of a function that indexes more arguments than it was given, leaves
// no results.
func failedCall(f func(x ...*Tensor) []*Tensor, n int, err error) (y []*Tensor) {
	defer func() {
		if recover() != nil {
			y = nil
		}
	}()
	return failAll(len(f(failAll(n, err)...)), err)
}

// jvpCall is JVP after its arguments are checked, with the tangents
// reached, if not nil, of tapes made before the call that f reaches
// besides its arguments.
func jvpCall(f func(x ...*Tensor) []*Tensor, x, t []*Tensor, reached map[*tape]*Tensor) (y, ty []*Tensor) {
	since := tapes.Load()
	args, stops := standIns(x)
	seeds := maps.Clone(reached)
	if seeds == nil {
		seeds = map[*tape]*Tensor{}
	}
	for i, ti := range t {
		seeds[stops[i]] = ti
	}
	y = f(args...)
	if err := firstErr(y...); err != nil {
		return failAll(len(y), err), failAll(len(y), err)
	}
	ty, err := pushforward(y, seeds, since, anyTracked(x...) || anyTracked(t...))
	if err != nil {
		return y, failAll(len(y), err)
	}
	return y, ty
}

// pushforward returns the tangents of ys, given the tangents seeds of
// some tapes: it applies the forward rules of the tapes reachable from ys
// that were made after since tapes existed, from the seeds on, and takes
// every other tape for a constant. A result that no seed reaches has
// zeros. The rules receive tracked tensors, so that the tangents are
// differentiable, with create or when the tapes walked lead to tapes that
// require gradients other than the seeds; otherwise detached ones.
func pushforward(ys []*Tensor, seeds map[*tape]*Tensor, since uint64, create bool) ([]*Tensor, error) {
	roots := make([]*tape, len(ys))
	for i, y := range ys {
		roots[i] = y.tape
	}
	stop := func(t *tape) bool { _, ok := seeds[t]; return ok }
	w, err := walkTapes(roots, stop, since, true)
	if err != nil {
		return nil, err
	}
	// The tangents are differentiable when what they are computed from
	// is: the arguments or their tangents, or tensors f read that
	// require gradients, such as parameters.
	for i := 0; !create && i < len(w.tapes); i++ {
		v := w.at(i)
		if _, ok := seeds[w.tapes[i]]; ok {
			continue
		}
		create = v.leaf
		for k := range v.deps() {
			if d, _ := v.dep(k); d != nil {
				if _, ok := w.pos[d]; !ok {
					create = true
				}
			}
		}
	}
	tan := make([]*Tensor, len(w.tapes))
	// The inputs of the first-order rules walked in a pass that creates
	// differentiable tangents (see reverse.result).
	var firstOrder []*Tensor
	for i, t := range w.tapes {
		v := w.at(i)
		if s, ok := seeds[t]; ok {
			tan[i] = s
			continue
		}
		if v.leaf {
			continue
		}
		if create && v.rule.firstOrder {
			firstOrder = append(firstOrder, v.ins...)
		}
		ts := make([]*Tensor, len(v.ins))
		some := false
		for j := range v.ins {
			if d, _ := v.dep(j); d != nil {
				if k, ok := w.pos[d]; ok {
					ts[j] = tan[k]
					some = some || ts[j] != nil
				}
			}
		}
		// A checkpoint's tangent is also that of what its function
		// reached besides its inputs.
		var reached map[*tape]*Tensor
		if v.ckpt != nil {
			for _, d := range v.ckpt.reaches {
				if k, ok := w.pos[d]; ok && tan[k] != nil {
					if reached == nil {
						reached = map[*tape]*Tensor{}
					}
					reached[d] = tan[k]
				}
			}
			some = some || reached != nil
		}
		if !some {
			continue
		}
		in, out := v.ins, &Tensor{node: v.node, tape: t}
		if !create {
			in, out = detachAll(v.ins), out.detached()
		}
		var dt *Tensor
		switch {
		case v.part != nil:
			dt = ts[0].shrink(v.part.ranges(ts[0].node.Shape))
		case v.ckpt != nil:
			dt, err = v.ckpt.pushforward(v.ins, ts, reached, create)
		case v.rule.jvp != nil:
			dt = v.rule.jvp(ts, in, out)
		case v.rule.firstOrder && create:
			if dt, err = transposed(v.rule, detachAll(ts), detachAll(v.ins), out.detached(), false); err == nil {
				dt = firstOrderOnly(dt, slices.Concat(ts, v.ins))
			}
		default:
			dt, err = transposed(v.rule, ts, in, out, create)
		}
		if err != nil {
			return nil, err
		}
		if dt != nil {
			if err := dt.Err(); err != nil {
				return nil, fmt.Errorf("tensor: JVP: %w", err)
			}
			if !slices.Equal(dt.node.Shape, v.node.Shape) {
				return nil, fmt.Errorf("tensor: JVP: tangent of shape %v for a result of shape %v: %w", dt.node.Shape, v.node.Shape, ErrInternal)
			}
		}
		tan[i] = dt
	}
	out := make([]*Tensor, len(ys))
	for i, y := range ys {
		if k, ok := w.pos[y.tape]; ok && y.tape != nil {
			out[i] = tan[k]
		}
		if out[i] != nil && firstOrder != nil {
			out[i] = firstOrderOnly(out[i], append([]*Tensor{out[i]}, firstOrder...))
		}
		if out[i] == nil {
			out[i] = Zeros(y.node.DType, y.node.Shape...)
		}
	}
	return out, nil
}

// detachAll returns ts detached.
func detachAll(ts []*Tensor) []*Tensor {
	out := make([]*Tensor, len(ts))
	for i, t := range ts {
		out[i] = t.detached()
	}
	return out
}

// pushforward is the forward rule of the checkpoint c over the inputs
// ins, whose tangents are ts, and the tapes it reached, whose tangents
// are reached: it recomputes the function and pushes the tangents
// through it.
func (c *checkpointCall) pushforward(ins, ts []*Tensor, reached map[*tape]*Tensor, create bool) (*Tensor, error) {
	for _, rd := range c.reads {
		if rd.p.Value().node != rd.node {
			return nil, fmt.Errorf("tensor: JVP: recompute of a checkpointed function: a parameter it read changed after the call: %w", ErrOverwritten)
		}
	}
	f := func(x ...*Tensor) []*Tensor { return []*Tensor{c.f(x...)} }
	_, ty := jvpCall(f, ins, ts, reached)
	return ty[0], ty[0].Err()
}

// transposed applies the forward derivative of the operation that r
// differentiates, from its inputs in to its result out, to the tangents
// ts. r's vjp maps g linearly to gᵀJ, so the tangent J·t is the gradient
// of the sum of the products of vjp(u) and t with respect to u, whatever
// u is.
func transposed(r rule, ts, in []*Tensor, out *Tensor, create bool) (*Tensor, error) {
	u := Zeros(out.node.DType, out.node.Shape...).RequiresGrad()
	gs := r.vjp(u, in, out)
	var s *Tensor
	for j, g := range gs {
		if j < len(ts) && ts[j] != nil && g != nil {
			s = addGrad(s, g.Mul(ts[j]).Sum())
		}
	}
	if s == nil {
		return nil, nil
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	switch {
	case s.tape == nil && r.firstOrder:
		// A first-order rule may compute its gradients with operations
		// that record nothing, which leaves nothing to transpose.
		return nil, fmt.Errorf("tensor: JVP through a gradient rule that cannot be transposed: %w", ErrNotDifferentiable)
	case s.tape == nil:
		return nil, nil // the tangents do not reach the result
	}
	var stops []*tape
	for _, x := range in {
		if x != nil && x.tape != nil {
			stops = append(stops, x.tape)
		}
	}
	rr := reverse{create: create, want: map[*tape]bool{u.tape: true}, fenced: true}
	if _, _, _, err := rr.propagate([]*tape{s.tape}, []*Tensor{Ones(s.node.DType)}, stops); err != nil {
		return nil, err
	}
	return rr.result(rr.got[u.tape]), nil
}

// vjpOf returns the gradients with respect to ins of f(ins), given the
// gradient g of the result, as tracked tensors. It is the gradient rule,
// to be differentiated in turn, of an operation that a kernel of its own
// computes and f composes from others.
func vjpOf(f func(in []*Tensor) *Tensor, ins []*Tensor, g *Tensor) []*Tensor {
	args, stops := standIns(ins)
	out := f(args)
	if err := firstErr(out); err != nil {
		return failAll(len(ins), err)
	}
	if out.tape == nil {
		return make([]*Tensor, len(ins))
	}
	r := reverse{create: true, want: map[*tape]bool{}}
	at, _, _, err := r.propagate([]*tape{out.tape}, []*Tensor{g}, stops)
	if err != nil {
		return failAll(len(ins), err)
	}
	for i := range at {
		at[i] = r.result(at[i])
	}
	return at
}

// jvpOf returns the tangent of f(ins) given the tangents ts of ins: the
// forward rule of an operation that a kernel of its own computes and f
// composes from others.
func jvpOf(f func(in []*Tensor) *Tensor, ins, ts []*Tensor) *Tensor {
	_, ty := jvpCall(func(x ...*Tensor) []*Tensor { return []*Tensor{f(x)} }, ins, ts, nil)
	return ty[0]
}

// Jacobian returns the Jacobian of f at x, shaped like f(x) followed by
// x: element [i..., j...] is the derivative of f(x)[i...] with respect to
// x[j...], like torch.autograd.functional.jacobian. It computes the
// Jacobian a column or a row at a time, all at once with VMap: by
// columns, with JVP, when x has no more elements than f(x), and otherwise
// by rows, with VJP, like jax.jacfwd and jax.jacrev. The result is lazy
// and differentiable.
func Jacobian(f func(x *Tensor) *Tensor, x *Tensor) *Tensor {
	if err := checkArgs("Jacobian", []*Tensor{x}); err != nil {
		return fail(err)
	}
	fs := func(x ...*Tensor) []*Tensor { return []*Tensor{f(x[0])} }
	y, pull := VJP(fs, x)
	if err := firstErr(y...); err != nil {
		return fail(err)
	}
	ys, xs := y[0].node.Shape, x.node.Shape
	n, m := x.node.Size(), y[0].node.Size()
	if n <= m {
		cols := VMap(func(t ...*Tensor) []*Tensor {
			_, ty := jvpCall(fs, []*Tensor{x}, t, nil)
			return ty
		})(eye(n).Cast(x.node.DType).reshape(slices.Concat([]int{n}, xs)))[0]
		return cols.Reshape(n, m).Transpose(0, 1).Reshape(slices.Concat(ys, xs)...)
	}
	rows := VMap(func(g ...*Tensor) []*Tensor { return pull(g...) })(eye(m).Cast(y[0].node.DType).reshape(slices.Concat([]int{m}, ys)))[0]
	return rows.Reshape(slices.Concat(ys, xs)...)
}

// Hessian returns the Hessian of f, whose result has one element, at x,
// shaped like x twice: element [i..., j...] is the second derivative of
// f(x) with respect to x[i...] and x[j...], like
// torch.autograd.functional.hessian. It is the Jacobian of Grad(f), in
// forward mode over reverse mode, like jax.hessian. For the product of
// the Hessian with a vector v without the matrix, use
// JVP(Grad(f), x, v).
func Hessian(f func(x *Tensor) *Tensor, x *Tensor) *Tensor {
	grad := Grad(func(x ...*Tensor) *Tensor { return f(x[0]) })
	return Jacobian(func(x *Tensor) *Tensor { return grad(x)[0] }, x)
}
