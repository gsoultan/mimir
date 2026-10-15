package cpu

import (
	"context"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// A reduction reads its input as kept dimensions, which index the output,
// and reduced ones. When the innermost input dimension is kept, output
// elements adjacent along it are computed together as a row: every reduced
// position combines a strided run of the input into a row of accumulators,
// so the input streams in order. Otherwise each output element combines
// runs of the input along the innermost reduced dimension. Either way,
// every output element visits its reduced positions in row-major order.

// reduction is the plan of a reduction.
type reduction struct {
	rows    backend.View // kept dimensions but the row's, with input strides
	rowLen  int          // output elements computed together
	rowStep int          // input stride between them
	red     backend.View // reduced dimensions, at offset 0
}

// rowChunk bounds the accumulators of a row, which a worker keeps in cache.
const rowChunk = 1024

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
	_, kv := coalesce(kept.Shape, []backend.View{kept})
	_, rv := coalesce(red.Shape, []backend.View{red})
	p := reduction{rows: kv[0], rowLen: 1, red: rv[0]}
	if r := len(p.rows.Shape) - 1; last >= 0 && !slices.Contains(axes, last) && r >= 0 {
		p.rowLen, p.rowStep = p.rows.Shape[r], p.rows.Strides[r]
		p.rows.Shape, p.rows.Strides = p.rows.Shape[:r], p.rows.Strides[:r]
	}
	return p
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

// reduceMax computes the maximum of the reduced elements, propagating NaN.
// The graph rejects reductions over empty axes, which have no maximum.
func reduceMax[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T) error {
	p := planReduction(k.In[0], k.Attrs.Axes)
	inner, step := 1, 0
	outer := p.red
	if r := len(p.red.Shape) - 1; r >= 0 {
		inner, step = p.red.Shape[r], p.red.Strides[r]
		outer = backend.View{Shape: p.red.Shape[:r], Strides: p.red.Strides[:r]}
	}
	element := func(i, off int) {
		acc, first := T(0), true
		walk(outer, func(o int) {
			src := off + o
			if first {
				acc, first = in[src], false
				src += step
				for range inner - 1 {
					acc = maxOf(acc, in[src])
					src += step
				}
				return
			}
			for range inner {
				acc = maxOf(acc, in[src])
				src += step
			}
		})
		out[i] = acc
	}
	row := func(i, off int, acc []T) {
		first := true
		walk(p.red, func(o int) {
			src := off + o
			switch {
			case first:
				for j := range acc {
					acc[j] = in[src]
					src += p.rowStep
				}
				first = false
			case p.rowStep == 1:
				for j, x := range in[src : src+len(acc)] {
					acc[j] = maxOf(acc[j], x)
				}
			default:
				for j := range acc {
					acc[j] = maxOf(acc[j], in[src])
					src += p.rowStep
				}
			}
		})
		copy(out[i:], acc)
	}
	return forReduction(ctx, b, p, len(out), element, row)
}

// reduceSum accumulates float32 in float64 and int32 in int32 (wrapping).
func reduceSum[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T) error {
	if dtypeOf[T]() == backend.Float32 {
		return sumInto[T, float64](ctx, b, k, out, in)
	}
	return sumInto[T, T](ctx, b, k, out, in)
}

// sumInto sums the reduced elements in accumulators of type A.
func sumInto[T elem, A float64 | elem](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T) error {
	p := planReduction(k.In[0], k.Attrs.Axes)
	inner, step := 1, 0
	outer := p.red
	if r := len(p.red.Shape) - 1; r >= 0 {
		inner, step = p.red.Shape[r], p.red.Strides[r]
		outer = backend.View{Shape: p.red.Shape[:r], Strides: p.red.Strides[:r]}
	}
	element := func(i, off int) {
		var acc A
		walk(outer, func(o int) {
			src := off + o
			if step == 1 {
				acc += sumRun[T, A](in[src : src+inner])
				return
			}
			for range inner {
				acc += A(in[src])
				src += step
			}
		})
		out[i] = T(acc)
	}
	row := func(i, off int, acc []A) {
		clear(acc)
		walk(p.red, func(o int) {
			src := off + o
			if p.rowStep == 1 {
				for j, x := range in[src : src+len(acc)] {
					acc[j] += A(x)
				}
				return
			}
			for j := range acc {
				acc[j] += A(in[src])
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
func sumRun[T elem, A float64 | elem](x []T) A {
	var a0, a1, a2, a3 A
	for len(x) >= 4 {
		a0 += A(x[0])
		a1 += A(x[1])
		a2 += A(x[2])
		a3 += A(x[3])
		x = x[4:]
	}
	for _, v := range x {
		a0 += A(v)
	}
	return (a0 + a1) + (a2 + a3)
}
