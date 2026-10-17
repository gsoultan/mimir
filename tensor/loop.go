package tensor

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// A ScanStep is the body of a loop that Scan runs: it maps the carry and
// the slices of the scanned tensors at one position to the next carry and
// the outputs at that position.
type ScanStep func(carry, x []*Tensor) (next, y []*Tensor)

// A ScanOption changes how Scan runs its loop.
type ScanOption func(*scanOptions)

type scanOptions struct {
	reverse bool
	every   int // steps per checkpoint, or 0
}

// ScanReverse runs the loop from the last position to the first, like
// jax.lax.scan(reverse=True). The outputs still line up with xs: slice i
// of ys[j] is what the step at slice i of xs returned.
func ScanReverse() ScanOption { return func(o *scanOptions) { o.reverse = true } }

// ScanCheckpoint runs the loop k steps at a time, each run of k steps
// through Checkpoint: Backward keeps the carry entering each run and
// recomputes the run's steps for their gradients, so a long sequence
// trains in the memory of k steps' activations and one carry per run
// instead of every step's activations. The carry and the outputs must
// then share a dtype, and step must read tracked tensors only through its
// arguments and parameters, as Checkpoint requires.
func ScanCheckpoint(k int) ScanOption { return func(o *scanOptions) { o.every = k } }

// Scan runs step along axis of the tensors xs, which share their length
// n there, carrying state from each step to the next, like jax.lax.scan:
//
//	carry := init
//	for i := range n {
//		carry, y[i] = step(carry, [x[i] for x in xs])
//	}
//	return carry, [stack(y[i][j] for i) for j]
//
// x[i] is the slice of x at position i without axis. step returns as many
// next tensors as init has, each with its init's shape and dtype, and the
// same number of outputs at every step, each with the same shape and
// dtype each time. ys[j] stacks the j-th outputs along axis, which counts
// as it does for xs and must fit their rank.
//
// Gradients flow through the carry and xs, and to every parameter step
// reads. The loop is built for the lazy graph: each step reads a view of
// xs and writes nothing of its own beyond what step computes, and the
// outputs are copied into ys once at the end. Backward joins the
// gradients of the steps' slices of xs with one copy too, instead of one
// per step, and frees each step's activations once its gradient is
// computed. ScanCheckpoint bounds the memory further.
//
// If init or xs carry an error, or the shapes do not fit, step runs once
// on tensors that carry the error, and every result carries it, so that
// Scan returns as many results as it otherwise would; if step panics on
// them, ys is empty. The same holds if
// a step returns a result that carries an error or does not fit. With
// n = 0, carry is init and each ys[j] is empty along axis; step runs once
// on zeros to tell the shapes of its outputs.
func Scan(step ScanStep, init, xs []*Tensor, axis int, opts ...ScanOption) (carry, ys []*Tensor) {
	var o scanOptions
	for _, opt := range opts {
		opt(&o)
	}
	if step == nil {
		err := fmt.Errorf("tensor: Scan(nil): %w", ErrInternal)
		return failAll(len(init), err), nil
	}
	n, a, err := scanLength(init, xs, axis, o)
	if err != nil {
		return scanFailed(step, init, xs, err)
	}
	switch {
	case n == 0:
		carry, ys, err = scanEmpty(step, init, xs, a)
	case o.every > 0:
		carry, ys, err = scanChunks(step, init, xs, a, n, o)
	default:
		carry, ys, err = scanSteps(step, init, xs, a, o.reverse)
	}
	switch {
	case err != nil && ys == nil:
		return scanFailed(step, init, xs, err) // no step told the outputs
	case err != nil:
		return failAll(len(init), err), failAll(len(ys), err)
	}
	return carry, ys
}

// scanLength checks the arguments of Scan and returns the length of the
// loop and the axis, normalized.
func scanLength(init, xs []*Tensor, axis int, o scanOptions) (n, a int, err error) {
	if err := firstErr(init...); err != nil {
		return 0, 0, err
	}
	if err := firstErr(xs...); err != nil {
		return 0, 0, err
	}
	if len(xs) == 0 {
		return 0, 0, fmt.Errorf("tensor: Scan over no tensors: %w", ErrShape)
	}
	if o.every < 0 {
		return 0, 0, fmt.Errorf("tensor: Scan with a checkpoint every %d steps: %w", o.every, ErrShape)
	}
	if a, err = normAxis(axis, len(xs[0].node.Shape)); err != nil {
		return 0, 0, err
	}
	n = xs[0].node.Shape[a]
	for _, x := range xs {
		if len(x.node.Shape) <= a || x.node.Shape[a] != n {
			return 0, 0, fmt.Errorf("tensor: Scan over %v and %v along axis %d: %w", xs[0].node.Shape, x.node.Shape, axis, ErrShape)
		}
	}
	return n, a, nil
}

// failAll returns n tensors carrying err.
func failAll(n int, err error) []*Tensor {
	ts := make([]*Tensor, n)
	for i := range ts {
		ts[i] = fail(err)
	}
	return ts
}

// scanFailed returns the results of a Scan that cannot run: step runs on
// tensors carrying err, to tell how many outputs it has.
// That run is Scan's own doing, so a panic in it, as of a step that
// indexes more tensors than it was given, leaves the outputs unknown
// rather than reaching the caller.
func scanFailed(step ScanStep, init, xs []*Tensor, err error) (carry, ys []*Tensor) {
	n := 0
	func() {
		defer func() { _ = recover() }()
		_, y := step(failAll(len(init), err), failAll(len(xs), err))
		n = len(y)
	}()
	return failAll(len(init), err), failAll(n, err)
}

// scanEmpty returns the results of a loop of no steps: the carry it
// starts with, and outputs empty along a, shaped and typed by running
// step once on zeros.
func scanEmpty(step ScanStep, init, xs []*Tensor, a int) (carry, ys []*Tensor, err error) {
	x := make([]*Tensor, len(xs))
	for i, t := range xs {
		x[i] = Zeros(t.node.DType, slices.Delete(slices.Clone(t.node.Shape), a, a+1)...)
	}
	_, y := step(init, x)
	ys = make([]*Tensor, len(y))
	for j, t := range y {
		if err := firstErr(t); err != nil {
			return nil, ys, err
		}
		if a > len(t.node.Shape) {
			return nil, ys, fmt.Errorf("tensor: Scan output %v stacked along axis %d: %w", t.node.Shape, a, ErrShape)
		}
		ys[j] = Zeros(t.node.DType, slices.Insert(slices.Clone(t.node.Shape), a, 0)...)
	}
	return init, ys, nil
}

// scanSteps runs step over every position along a of xs, which have
// passed scanLength, and stacks the outputs.
func scanSteps(step ScanStep, init, xs []*Tensor, a int, reverse bool) (carry, ys []*Tensor, err error) {
	n := xs[0].node.Shape[a]
	ones := slices.Repeat([]int{1}, n)
	at := make([][]*Tensor, len(xs)) // the slices of each tensor, by position
	for i, x := range xs {
		at[i] = x.split(a, ones)
		for t, s := range at[i] {
			at[i][t] = s.dropAxes([]int{a})
		}
	}
	outs := make([][]*Tensor, n) // by position, then output
	var first []*Tensor          // the outputs of the first step run
	carry = init
	for k := range n {
		t := k
		if reverse {
			t = n - 1 - k
		}
		x := make([]*Tensor, len(xs))
		for i := range xs {
			x[i] = at[i][t]
		}
		next, y := step(carry, x)
		if first == nil {
			first = y
		}
		if err := fits("carry", next, init); err != nil {
			return nil, failAll(len(y), err), err
		}
		if err := fits("output", y, first); err != nil {
			return nil, failAll(len(first), err), err
		}
		carry, outs[t] = next, y
	}
	ys = make([]*Tensor, len(first))
	for j, y := range first {
		if a > len(y.node.Shape) {
			err := fmt.Errorf("tensor: Scan output %v stacked along axis %d: %w", y.node.Shape, a, ErrShape)
			return nil, failAll(len(first), err), err
		}
		col := make([]*Tensor, n)
		for t := range n {
			col[t] = outs[t][j].reshape(slices.Insert(slices.Clone(y.node.Shape), a, 1))
		}
		ys[j] = cat(a, col)
	}
	return carry, ys, nil
}

// fits reports an error unless got, what a step returned for what, has
// one tensor without an error for each of want, of its shape and dtype.
func fits(what string, got, want []*Tensor) error {
	if len(got) != len(want) {
		return fmt.Errorf("tensor: Scan step returned %d tensors for a %s of %d: %w", len(got), what, len(want), ErrShape)
	}
	if err := firstErr(got...); err != nil {
		return err
	}
	for i, g := range got {
		w := want[i].node
		if !slices.Equal(g.node.Shape, w.Shape) || g.node.DType != w.DType {
			return fmt.Errorf("tensor: Scan step returned %s%v for a %s of %s%v: %w", g.node.DType, g.node.Shape, what, w.DType, w.Shape, ErrShape)
		}
	}
	return nil
}

// scanChunks runs the loop o.every steps at a time, each run of steps
// checkpointed. A run's carry and outputs travel through Checkpoint as
// one flat tensor, which split takes apart again.
func scanChunks(step ScanStep, init, xs []*Tensor, a, n int, o scanOptions) (carry, ys []*Tensor, err error) {
	var sizes []int
	for at := 0; at < n; at += o.every {
		sizes = append(sizes, min(o.every, n-at))
	}
	runs := make([][]*Tensor, len(xs)) // the runs of each tensor, by position
	for i, x := range xs {
		runs[i] = x.split(a, sizes)
	}
	outs := make([][]*Tensor, len(sizes)) // by run, then output
	carry = init
	for k := range sizes {
		r := k
		if o.reverse {
			r = len(sizes) - 1 - k
		}
		in := slices.Clone(carry)
		for i := range xs {
			in = append(in, runs[i][r])
		}
		var shapes [][]int // of the carry and the outputs, in the flat result
		flat := Checkpoint(func(in ...*Tensor) *Tensor {
			c, y, err := scanSteps(step, in[:len(init)], in[len(init):], a, o.reverse)
			if err != nil {
				return fail(err)
			}
			res := append(c, y...)
			if shapes == nil {
				for _, t := range res {
					shapes = append(shapes, t.node.Shape)
				}
			}
			return flatten(res)
		}, in...)
		if err := flat.Err(); err != nil {
			return nil, nil, err
		}
		parts := unflatten(flat, shapes)
		carry, outs[r] = parts[:len(init)], parts[len(init):]
	}
	ys = make([]*Tensor, len(outs[0]))
	for j := range ys {
		col := make([]*Tensor, len(outs))
		for r := range outs {
			col[r] = outs[r][j]
		}
		ys[j] = cat(a, col)
	}
	return carry, ys, nil
}

// flatten joins ts, which share a dtype, into one vector.
func flatten(ts []*Tensor) *Tensor {
	if len(ts) == 0 {
		return Zeros(Float32, 0)
	}
	flat := make([]*Tensor, len(ts))
	for i, t := range ts {
		if t.node.DType != ts[0].node.DType {
			return fail(fmt.Errorf("tensor: Scan with ScanCheckpoint of %s and %s results: %w", ts[0].node.DType, t.node.DType, ErrDType))
		}
		flat[i] = t.reshape([]int{t.node.Size()})
	}
	return cat(0, flat)
}

// unflatten takes apart a vector that flatten joined from tensors of
// shapes.
func unflatten(flat *Tensor, shapes [][]int) []*Tensor {
	sizes := make([]int, len(shapes))
	for i, s := range shapes {
		sizes[i], _ = backend.NumElements(s)
	}
	parts := flat.split(0, sizes)
	for i, p := range parts {
		parts[i] = p.reshape(shapes[i])
	}
	return parts
}
