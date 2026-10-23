package cpu

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// gather reads out[m] = src[m with m[axis] = idx[m]]. The output has the
// shape of the index; the graph checks that it fits src outside axis.
func gather[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, src []T, idx []int32) error {
	axis := k.Attrs.Axis
	vs, vi := k.In[0], k.In[1]
	limit, step := vs.Shape[axis], vs.Strides[axis]
	// Walking the output shape with src strides, minus the indexed axis,
	// gives the offset of src at m with m[axis] = 0.
	base := backend.View{Shape: k.Shape, Strides: slices.Clone(vs.Strides), Offset: vs.Offset}
	base.Strides[axis] = 0
	return b.parallelFor(ctx, len(out), elementGrain, func(lo, hi int) error {
		c := newCursor(k.Shape, []backend.View{vi, base}, lo)
		for i := lo; i < hi; i++ {
			j := int(idx[c.offs[0]])
			if j < 0 || j >= limit {
				return fmt.Errorf("index %d for axis %d of size %d: %w", j, axis, limit, backend.ErrIndex)
			}
			out[i] = src[c.offs[1]+j*step]
			c.next()
		}
		return nil
	})
}

// scatterAdd copies dst into out, then adds each update into
// out[m with m[axis] = idx[m]]. Index and updates share a shape that fits
// dst outside axis. Colliding indices add in a fixed order, so results are
// reproducible. Half elements are summed in float32 and rounded once.
func scatterAdd[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, dst []T, idx []int32, upd []T) error {
	switch out := any(out).(type) {
	case []float32:
		return scatterAddInto(ctx, b, k, out, dst, idx, upd)
	case []int32:
		return scatterAddInto(ctx, b, k, out, dst, idx, upd)
	}
	sums := make([]float32, len(out))
	if err := scatterAddInto(ctx, b, k, sums, dst, idx, upd); err != nil {
		return err
	}
	for i, v := range sums {
		out[i] = ofF32[T](v)
	}
	return nil
}

// scatterAddInto computes a scatterAdd into out, as A. The updates of a
// line, which share their position outside axis, land in one line of out
// and no other line's do: workers add whole lines, each in order along
// axis, so every element adds its updates in row-major order of their
// positions however many workers there are.
func scatterAddInto[A number, S elem](ctx context.Context, b *Backend, k *backend.Kernel, out []A, dst []S, idx []int32, upd []S) error {
	if err := mapStrided(ctx, b, k.Shape, k.In[0], len(out), func(i, off int) { out[i] = conv[A](dst[off]) }); err != nil {
		return err
	}
	axis := k.Attrs.Axis
	vi, vu := k.In[1], k.In[2]
	limit, length := k.Shape[axis], vi.Shape[axis]
	strides := backend.ContiguousStrides(k.Shape)
	step, si, su := strides[axis], vi.Strides[axis], vu.Strides[axis]
	drop := func(s []int) []int { return slices.Delete(slices.Clone(s), axis, axis+1) }
	shape, views := backend.Coalesce(drop(vi.Shape), []backend.View{
		{Strides: drop(vi.Strides), Offset: vi.Offset},
		{Strides: drop(vu.Strides), Offset: vu.Offset},
		{Strides: drop(strides)},
	})
	lines, _ := backend.NumElements(shape)
	return b.parallelFor(ctx, lines, max(1, elementGrain/max(length, 1)), func(lo, hi int) error {
		c := newCursor(shape, views, lo)
		for range hi - lo {
			oi, ou, base := c.offs[0], c.offs[1], c.offs[2]
			for range length {
				j := int(idx[oi])
				if j < 0 || j >= limit {
					return fmt.Errorf("index %d for axis %d of size %d: %w", j, axis, limit, backend.ErrIndex)
				}
				out[base+j*step] += conv[A](upd[ou])
				oi += si
				ou += su
			}
			c.next()
		}
		return nil
	})
}

// scatter copies dst into out, unless out is dst's buffer already, and
// then writes each update over out[m with m[axis] = idx[m]]. Workers write
// whole lines, each in order along axis, so of the updates of an element
// the last along axis lands, as for scatterAddInto.
func scatter[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, dst []T, idx []int32, upd []T, inPlace bool) error {
	if !inPlace {
		err := forRows(ctx, b, k.Shape, []backend.View{k.In[0], backend.Contiguous(k.Shape)}, func(_ int, offs, steps []int, n int) {
			from, to := offs[0], offs[1]
			if steps[0] == 1 && steps[1] == 1 {
				copy(out[to:to+n], dst[from:from+n])
				return
			}
			for range n {
				out[to] = dst[from]
				from += steps[0]
				to += steps[1]
			}
		})
		if err != nil {
			return err
		}
	}
	axis := k.Attrs.Axis
	vi, vu := k.In[1], k.In[2]
	limit, length := k.Shape[axis], vi.Shape[axis]
	strides := backend.ContiguousStrides(k.Shape)
	step, si, su := strides[axis], vi.Strides[axis], vu.Strides[axis]
	drop := func(s []int) []int { return slices.Delete(slices.Clone(s), axis, axis+1) }
	shape, views := backend.Coalesce(drop(vi.Shape), []backend.View{
		{Strides: drop(vi.Strides), Offset: vi.Offset},
		{Strides: drop(vu.Strides), Offset: vu.Offset},
		{Strides: drop(strides)},
	})
	lines, _ := backend.NumElements(shape)
	return b.parallelFor(ctx, lines, max(1, elementGrain/max(length, 1)), func(lo, hi int) error {
		c := newCursor(shape, views, lo)
		for range hi - lo {
			oi, ou, base := c.offs[0], c.offs[1], c.offs[2]
			for range length {
				j := int(idx[oi])
				if j < 0 || j >= limit {
					return fmt.Errorf("index %d for axis %d of size %d: %w", j, axis, limit, backend.ErrIndex)
				}
				out[base+j*step] = upd[ou]
				oi += si
				ou += su
			}
			c.next()
		}
		return nil
	})
}
