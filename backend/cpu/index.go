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
// dst outside axis. It runs sequentially so that colliding indices add in a
// fixed order and results are reproducible.
func scatterAdd[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, dst []T, idx []int32, upd []T) error {
	if err := mapStrided(ctx, b, k.Shape, k.In[0], len(out), func(i, off int) { out[i] = dst[off] }); err != nil {
		return err
	}
	axis := k.Attrs.Axis
	vi, vu := k.In[1], k.In[2]
	limit := k.Shape[axis]
	strides := backend.ContiguousStrides(k.Shape)
	step := strides[axis]
	base := backend.View{Shape: vi.Shape, Strides: strides}
	base.Strides[axis] = 0

	c := newCursor(vi.Shape, []backend.View{vi, vu, base}, 0)
	for i := range vi.Size() {
		if i%elementGrain == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		j := int(idx[c.offs[0]])
		if j < 0 || j >= limit {
			return fmt.Errorf("index %d for axis %d of size %d: %w", j, axis, limit, backend.ErrIndex)
		}
		out[c.offs[2]+j*step] += upd[c.offs[1]]
		c.next()
	}
	return nil
}
