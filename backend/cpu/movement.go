package cpu

import (
	"context"

	"github.com/gsoultan/mimir/backend"
)

// pad fills the output with Attrs.Value and copies the input into the
// interior given by Attrs.Pads. Pads are non-negative; the graph checks.
func pad[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T) error {
	fill := valueOf[T](k.Attrs.Value)
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

// sliceScatter copies x into out, unless out is x's buffer already, and
// then src over the block k.Attrs.Ranges of out.
func sliceScatter[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, x, src []T, inPlace bool) error {
	whole := backend.Contiguous(k.Shape)
	if !inPlace {
		err := forRows(ctx, b, k.Shape, []backend.View{k.In[0], whole}, func(_ int, offs, steps []int, n int) {
			from, to := offs[0], offs[1]
			if steps[0] == 1 && steps[1] == 1 {
				copy(out[to:to+n], x[from:from+n])
				return
			}
			for range n {
				out[to] = x[from]
				from += steps[0]
				to += steps[1]
			}
		})
		if err != nil {
			return err
		}
	}
	block := whole.Shrink(k.Attrs.Ranges)
	return forRows(ctx, b, k.In[1].Shape, []backend.View{k.In[1], block}, func(_ int, offs, steps []int, n int) {
		from, to := offs[0], offs[1]
		if steps[0] == 1 && steps[1] == 1 {
			copy(out[to:to+n], src[from:from+n])
			return
		}
		for range n {
			out[to] = src[from]
			from += steps[0]
			to += steps[1]
		}
	})
}
