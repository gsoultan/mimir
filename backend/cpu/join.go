package cpu

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/backend"
)

// cat copies each input into its slab of the output along Attrs.Axis. A
// join of large inputs copies each in parallel; one of many small inputs,
// as a loop over a sequence stacks its steps, copies several inputs per
// chunk of work instead, so that each chunk is worth its goroutine.
func cat[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out []T, ins [][]T) error {
	starts, err := catStarts(k)
	if err != nil {
		return err
	}
	if len(out) == 0 {
		return nil
	}
	strides := backend.ContiguousStrides(k.Shape)
	slab := func(i int) backend.View {
		return backend.View{Shape: k.In[i].Shape, Strides: strides, Offset: starts[i] * strides[k.Attrs.Axis]}
	}
	if len(out)/len(ins) >= elementGrain {
		for i, in := range ins {
			if err := forRows(ctx, b, k.In[i].Shape, []backend.View{k.In[i], slab(i)}, func(_ int, offs, steps []int, n int) {
				copyRow(out, in, offs, steps, n)
			}); err != nil {
				return err
			}
		}
		return nil
	}
	grain := max(1, elementGrain*len(ins)/len(out))
	return b.parallelFor(ctx, len(ins), grain, func(lo, hi int) error {
		for i := lo; i < hi; i++ {
			copyView(out, slab(i), ins[i], k.In[i])
		}
		return nil
	})
}

// catStarts checks the views of an OpCat kernel against its output and
// returns where each input starts along the axis.
func catStarts(k *backend.Kernel) ([]int, error) {
	axis, r := k.Attrs.Axis, len(k.Shape)
	if len(k.In) == 0 || len(k.In) > backend.MaxCatInputs || axis < 0 || axis >= r {
		return nil, fmt.Errorf("cat of %d inputs along axis %d of %v: %w", len(k.In), axis, k.Shape, backend.ErrInternal)
	}
	starts := make([]int, len(k.In))
	at := 0
	for i, v := range k.In {
		if len(v.Shape) != r {
			return nil, fmt.Errorf("cat input %v for output %v: %w", v.Shape, k.Shape, backend.ErrInternal)
		}
		for d, s := range v.Shape {
			if d != axis && s != k.Shape[d] {
				return nil, fmt.Errorf("cat input %v for output %v: %w", v.Shape, k.Shape, backend.ErrInternal)
			}
		}
		starts[i] = at
		at += v.Shape[axis]
	}
	if at != k.Shape[axis] {
		return nil, fmt.Errorf("cat inputs of %d along axis %d for output %v: %w", at, axis, k.Shape, backend.ErrInternal)
	}
	return starts, nil
}

// copyRow copies n elements of a row from in to out, starting at offs and
// stepping by steps: index 0 for the source, 1 for the destination.
func copyRow[T elem](out, in []T, offs, steps []int, n int) {
	from, to := offs[0], offs[1]
	if steps[0] == 1 && steps[1] == 1 {
		copy(out[to:to+n], in[from:from+n])
		return
	}
	for range n {
		out[to] = in[from]
		from += steps[0]
		to += steps[1]
	}
}

// copyView copies the elements that src reads in into the elements that
// dst writes in out, on the calling goroutine.
func copyView[T elem](out []T, dst backend.View, in []T, src backend.View) {
	shape, views := backend.Coalesce(dst.Shape, []backend.View{src, dst})
	if n, _ := backend.NumElements(shape); n == 0 {
		return
	}
	r := len(shape)
	if r == 0 {
		out[views[1].Offset] = in[views[0].Offset]
		return
	}
	outer := make([]backend.View, len(views))
	steps := make([]int, len(views))
	for v, view := range views {
		outer[v] = backend.View{Strides: view.Strides[:r-1], Offset: view.Offset}
		steps[v] = view.Strides[r-1]
	}
	rows, _ := backend.NumElements(shape[:r-1])
	c := newCursor(shape[:r-1], outer, 0)
	for range rows {
		copyRow(out, in, c.offs, steps, shape[r-1])
		c.next()
	}
}
