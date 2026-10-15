package cpu

import (
	"context"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// reducedSpace returns the shape and strides of the reduced axes of v.
func reducedSpace(v backend.View, axes []int) backend.View {
	r := backend.View{Shape: make([]int, 0, len(axes)), Strides: make([]int, 0, len(axes))}
	for d := range v.Shape {
		if slices.Contains(axes, d) {
			r.Shape = append(r.Shape, v.Shape[d])
			r.Strides = append(r.Strides, v.Strides[d])
		}
	}
	return r
}

// forEachReduced calls visit(i, inner) for each output element i, where
// inner is the view of the input elements reduced into it. The output shape
// keeps reduced axes with size 1, so walking it with the input strides
// yields the offset of the first reduced element.
func forEachReduced(ctx context.Context, b *Backend, k *backend.Kernel, n int, visit func(i int, inner backend.View)) error {
	v := k.In[0]
	red := reducedSpace(v, k.Attrs.Axes)
	innerSize, _ := backend.NumElements(red.Shape)
	grain := max(1, elementGrain/max(innerSize, 1))
	base := backend.View{Shape: k.Shape, Strides: v.Strides, Offset: v.Offset}
	return b.parallelFor(ctx, n, grain, func(lo, hi int) error {
		c := newCursor(k.Shape, []backend.View{base}, lo)
		for i := lo; i < hi; i++ {
			visit(i, backend.View{Shape: red.Shape, Strides: red.Strides, Offset: c.offs[0]})
			c.next()
		}
		return nil
	})
}

// walk calls fn with the buffer offset of every element of v in row-major
// order. Reductions over a single axis, the common case, avoid the cursor.
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

// reduce folds the reduced elements with combine, starting from the first.
// The graph rejects reductions over empty axes, which have no first element.
func reduce[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T, combine func(acc, x T) T) error {
	return forEachReduced(ctx, b, k, len(out), func(i int, inner backend.View) {
		var acc T
		first := true
		walk(inner, func(off int) {
			if first {
				acc, first = in[off], false
				return
			}
			acc = combine(acc, in[off])
		})
		out[i] = acc
	})
}

// reduceSum accumulates float32 in float64 and int32 in int32 (wrapping).
func reduceSum[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T) error {
	if dtypeOf[T]() == backend.Float32 {
		return forEachReduced(ctx, b, k, len(out), func(i int, inner backend.View) {
			var acc float64
			walk(inner, func(off int) { acc += float64(in[off]) })
			out[i] = T(acc)
		})
	}
	return forEachReduced(ctx, b, k, len(out), func(i int, inner backend.View) {
		var acc T
		walk(inner, func(off int) { acc += in[off] })
		out[i] = acc
	})
}
