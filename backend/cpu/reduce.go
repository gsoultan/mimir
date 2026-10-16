package cpu

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// A reduction reads its input as kept dimensions, which index the output,
// and reduced ones. When the innermost input dimension is kept, output
// elements adjacent along it are computed together as a row: every reduced
// position combines a strided run of the input into a row of accumulators,
// so the input streams in order. Otherwise each output element combines
// runs of the input along the innermost reduced dimension, unless the runs
// are shorter than shortRun: rows along the innermost kept dimension then
// cost less than starting a run per element. Either way, every output
// element visits its reduced positions in row-major order.

// reduction is the plan of a reduction.
type reduction struct {
	rows    backend.View // kept dimensions but the row's, with input strides
	rowLen  int          // output elements computed together
	rowStep int          // input stride between them
	red     backend.View // reduced dimensions, at offset 0
}

// rowChunk bounds the accumulators of a row, which a worker keeps in cache.
const rowChunk = 1024

// shortRun is the length of reduced runs below which a reduction works in
// rows.
const shortRun = 16

func planReduction(v backend.View, axes []int) reduction {
	kept := backend.View{Offset: v.Offset}
	var red backend.View
	last := -1 // innermost dimension with more than one element
	for d, n := range v.Shape {
		if slices.Contains(axes, d) {
			red.Shape = append(red.Shape, n)
			red.Strides = append(red.Strides, v.Strides[d])
		} else {
			kept.Shape = append(kept.Shape, n)
			kept.Strides = append(kept.Strides, v.Strides[d])
		}
		if n > 1 {
			last = d
		}
	}
	_, kv := backend.Coalesce(kept.Shape, []backend.View{kept})
	_, rv := backend.Coalesce(red.Shape, []backend.View{red})
	p := reduction{rows: kv[0], rowLen: 1, red: rv[0]}
	short := len(p.red.Shape) > 0 && p.red.Shape[len(p.red.Shape)-1] < shortRun
	if r := len(p.rows.Shape) - 1; r >= 0 && last >= 0 && (!slices.Contains(axes, last) || short) {
		p.rowLen, p.rowStep = p.rows.Shape[r], p.rows.Strides[r]
		p.rows.Shape, p.rows.Strides = p.rows.Shape[:r], p.rows.Strides[:r]
	}
	return p
}

// walker visits the elements of a view, which a reduction does for every
// output element or row. A view of more dimensions than one is walked from
// a list of its offsets if it is short enough, without setting up a
// cursor each time.
type walker struct {
	v    backend.View
	offs []int
}

// maxListed bounds the offsets a walker lists.
const maxListed = 1 << 16

func newWalker(v backend.View) walker {
	w := walker{v: v}
	if n, _ := backend.NumElements(v.Shape); len(v.Shape) > 1 && n <= maxListed {
		w.offs = make([]int, 0, n)
		walk(v, func(off int) { w.offs = append(w.offs, off) })
	}
	return w
}

// walk calls fn with the offset of every element of the view in row-major
// order.
func (w walker) walk(fn func(off int)) {
	if w.offs == nil {
		walk(w.v, fn)
		return
	}
	for _, off := range w.offs {
		fn(off)
	}
}

// walk calls fn with the buffer offset of every element of v in row-major
// order.
func walk(v backend.View, fn func(off int)) {
	switch len(v.Shape) {
	case 0:
		fn(v.Offset)
	case 1:
		off, s := v.Offset, v.Strides[0]
		for range v.Shape[0] {
			fn(off)
			off += s
		}
	default:
		c := newCursor(v.Shape, []backend.View{v}, 0)
		for range v.Size() {
			fn(c.offs[0])
			c.next()
		}
	}
}

// forReduction runs a reduction planned by p over n output elements:
// element(i, off) computes output element i, whose reduced elements start
// at input offset off, and row(i, off, acc) the len(acc) output elements of
// a row from i on, the first at input offset off.
func forReduction[A any](ctx context.Context, b *Backend, p reduction, n int, element func(i, off int), row func(i, off int, acc []A)) error {
	redSize, _ := backend.NumElements(p.red.Shape)
	if p.rowLen == 1 {
		return b.parallelFor(ctx, n, max(1, elementGrain/max(redSize, 1)), func(lo, hi int) error {
			c := newCursor(p.rows.Shape, []backend.View{p.rows}, lo)
			for i := lo; i < hi; i++ {
				element(i, c.offs[0])
				c.next()
			}
			return nil
		})
	}
	chunks := (p.rowLen + rowChunk - 1) / rowChunk
	width := min(p.rowLen, rowChunk)
	units := n / p.rowLen * chunks
	return b.parallelFor(ctx, units, max(1, elementGrain/max(width*redSize, 1)), func(lo, hi int) error {
		acc := make([]A, width)
		r := lo / chunks
		c := newCursor(p.rows.Shape, []backend.View{p.rows}, r)
		for u := lo; u < hi; u++ {
			for ; r < u/chunks; r++ {
				c.next()
			}
			j := u % chunks * rowChunk
			m := min(rowChunk, p.rowLen-j)
			row(r*p.rowLen+j, c.offs[0]+j*p.rowStep, acc[:m])
		}
		return nil
	})
}

// runReduce runs a reduction of in into out. Half inputs are reduced in
// float32, into a Float32 output.
func runReduce(ctx context.Context, b *Backend, k *backend.Kernel, out, in backend.Buffer) error {
	switch in := in.(type) {
	case *buffer[float32]:
		return reduce[float32](ctx, b, k, out, in.data)
	case *buffer[int32]:
		return reduce[int32](ctx, b, k, out, in.data)
	case *buffer[bf16]:
		return reduce[float32](ctx, b, k, out, in.data)
	case *buffer[f16]:
		return reduce[float32](ctx, b, k, out, in.data)
	default:
		return fmt.Errorf("foreign buffer %T: %w", in, backend.ErrDType)
	}
}

// reduce runs a reduction of the elements in, read as T, into the T
// output outBuf.
func reduce[T number, S elem](ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, in []S) error {
	out, err := data[T](outBuf)
	if err != nil {
		return err
	}
	if k.Op == backend.OpReduceMax {
		return reduceMax(ctx, b, k, out, in)
	}
	return reduceSum(ctx, b, k, out, in)
}

// reduceMax computes the maximum of the reduced elements, propagating NaN.
// The graph rejects reductions over empty axes, which have no maximum.
func reduceMax[T number, S elem](ctx context.Context, b *Backend, k *backend.Kernel, out []T, in []S) error {
	p := planReduction(k.In[0], k.Attrs.Axes)
	inner, step := 1, 0
	outer := p.red
	if r := len(p.red.Shape) - 1; r >= 0 {
		inner, step = p.red.Shape[r], p.red.Strides[r]
		outer = backend.View{Shape: p.red.Shape[:r], Strides: p.red.Strides[:r]}
	}
	var runs, red walker
	if p.rowLen == 1 {
		runs = newWalker(outer)
	} else {
		red = newWalker(p.red)
	}
	element := func(i, off int) {
		acc, first := T(0), true
		runs.walk(func(o int) {
			src := off + o
			if first {
				acc, first = conv[T](in[src]), false
				src += step
				for range inner - 1 {
					acc = maxOf(acc, conv[T](in[src]))
					src += step
				}
				return
			}
			for range inner {
				acc = maxOf(acc, conv[T](in[src]))
				src += step
			}
		})
		out[i] = acc
	}
	row := func(i, off int, acc []T) {
		first := true
		red.walk(func(o int) {
			src := off + o
			switch {
			case first:
				for j := range acc {
					acc[j] = conv[T](in[src])
					src += p.rowStep
				}
				first = false
			case p.rowStep == 1:
				for j, x := range in[src : src+len(acc)] {
					acc[j] = maxOf(acc[j], conv[T](x))
				}
			default:
				for j := range acc {
					acc[j] = maxOf(acc[j], conv[T](in[src]))
					src += p.rowStep
				}
			}
		})
		copy(out[i:], acc)
	}
	return forReduction(ctx, b, p, len(out), element, row)
}

// reduceSum accumulates float32 and half inputs in float64 and int32 in
// int32 (wrapping).
func reduceSum[T number, S elem](ctx context.Context, b *Backend, k *backend.Kernel, out []T, in []S) error {
	if dtypeOf[T]() == backend.Float32 {
		return sumInto[T, float64](ctx, b, k, out, in)
	}
	return sumInto[T, T](ctx, b, k, out, in)
}

// sumInto sums the reduced elements in accumulators of type A.
func sumInto[T number, A float64 | number, S elem](ctx context.Context, b *Backend, k *backend.Kernel, out []T, in []S) error {
	p := planReduction(k.In[0], k.Attrs.Axes)
	inner, step := 1, 0
	outer := p.red
	if r := len(p.red.Shape) - 1; r >= 0 {
		inner, step = p.red.Shape[r], p.red.Strides[r]
		outer = backend.View{Shape: p.red.Shape[:r], Strides: p.red.Strides[:r]}
	}
	var runs, red walker
	if p.rowLen == 1 {
		runs = newWalker(outer)
	} else {
		red = newWalker(p.red)
	}
	element := func(i, off int) {
		var acc A
		runs.walk(func(o int) {
			src := off + o
			if step == 1 {
				acc += sumRun[A](in[src : src+inner])
				return
			}
			for range inner {
				acc += conv[A](in[src])
				src += step
			}
		})
		out[i] = T(acc)
	}
	row := func(i, off int, acc []A) {
		clear(acc)
		red.walk(func(o int) {
			src := off + o
			if p.rowStep == 1 {
				for j, x := range in[src : src+len(acc)] {
					acc[j] += conv[A](x)
				}
				return
			}
			for j := range acc {
				acc[j] += conv[A](in[src])
				src += p.rowStep
			}
		})
		for j, a := range acc {
			out[i+j] = T(a)
		}
	}
	return forReduction(ctx, b, p, len(out), element, row)
}

// sumRun returns the sum of x. Four running sums let the additions
// overlap instead of each waiting for the one before.
func sumRun[A float64 | number, S elem](x []S) A {
	var a0, a1, a2, a3 A
	for len(x) >= 4 {
		a0 += conv[A](x[0])
		a1 += conv[A](x[1])
		a2 += conv[A](x[2])
		a3 += conv[A](x[3])
		x = x[4:]
	}
	for _, v := range x {
		a0 += conv[A](v)
	}
	return (a0 + a1) + (a2 + a3)
}
