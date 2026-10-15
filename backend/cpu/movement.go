package cpu

import (
	"context"

	"github.com/gsoultan/mimir/backend"
)

// pad fills the output with Attrs.Value and copies the input into the
// interior given by Attrs.Pads. Pads are non-negative; the graph checks.
func pad[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T) error {
	fill := T(k.Attrs.Value)
	if err := b.parallelFor(ctx, len(out), elementGrain, func(lo, hi int) error {
		for i := lo; i < hi; i++ {
			out[i] = fill
		}
		return nil
	}); err != nil {
		return err
	}

	src := k.In[0]
	strides := backend.ContiguousStrides(k.Shape)
	dst := backend.View{Shape: src.Shape, Strides: strides}
	for d, p := range k.Attrs.Pads {
		dst.Offset += p[0] * strides[d]
	}
	return forRows(ctx, b, src.Shape, []backend.View{src, dst}, func(_ int, offs, steps []int, n int) {
		from, to := offs[0], offs[1]
		for range n {
			out[to] = in[from]
			from += steps[0]
			to += steps[1]
		}
	})
}
