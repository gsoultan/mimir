package cpu

import (
	"context"
	"fmt"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// parallelFor runs fn over [0, n) split into contiguous chunks of at least
// grain elements, using up to b.workers goroutines. Chunks never overlap, so
// fn may write its own range of an output without synchronization, and
// their bounds depend only on n, grain and b.workers.
//
// It returns after every started chunk has finished. A panic in fn is
// recovered and reported as backend.ErrInternal. Cancellation is checked
// before each chunk starts; a running chunk is not interrupted.
func (b *Backend) parallelFor(ctx context.Context, n, grain int, fn func(lo, hi int) error) error {
	if n <= 0 {
		return nil
	}
	grain = max(grain, 1)
	chunks := min(b.workers, (n+grain-1)/grain)
	if chunks <= 1 {
		return runChunk(ctx, fn, 0, n)
	}
	size := (n + chunks - 1) / chunks
	if t := defaultTeam(); t != nil && t.mu.TryLock() {
		defer t.mu.Unlock()
		return t.run(ctx, n, size, fn)
	}
	// Another loop holds the team, so every processor is likely busy; let
	// the runtime share them between the loops.
	errs := make([]error, (n+size-1)/size)
	var wg sync.WaitGroup
	for c := range errs {
		lo := c * size
		hi := min(n, lo+size)
		wg.Go(func() { errs[c] = runChunk(ctx, fn, lo, hi) })
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

func runChunk(ctx context.Context, fn func(lo, hi int) error, lo, hi int) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("cpu: panic in kernel: %v: %w", r, backend.ErrInternal)
		}
	}()
	return fn(lo, hi)
}

// coalesce returns an equivalent walk of shape under views with fewer
// dimensions: it drops dimensions of size 1 and merges each dimension into
// the one before it when every view steps through the pair contiguously.
// Row-major order, and so the linear index of every element, is unchanged,
// and kernels take longer strides through fewer cursor steps.
func coalesce(shape []int, views []backend.View) ([]int, []backend.View) {
	dims := make([]int, 0, len(shape))
	out := make([]backend.View, len(views))
	for v, view := range views {
		out[v] = backend.View{Strides: make([]int, 0, len(shape)), Offset: view.Offset}
	}
	for d, n := range shape {
		if n == 1 {
			continue
		}
		last := len(dims) - 1
		merge := last >= 0 && n != 0 && dims[last] != 0
		for v := range views {
			merge = merge && out[v].Strides[last] == views[v].Strides[d]*n
		}
		if merge {
			dims[last] *= n
			for v := range views {
				out[v].Strides[last] = views[v].Strides[d]
			}
			continue
		}
		dims = append(dims, n)
		for v := range views {
			out[v].Strides = append(out[v].Strides, views[v].Strides[d])
		}
	}
	for v := range out {
		out[v].Shape = dims
	}
	return dims, out
}

// cursor walks a shape in row-major order and tracks, for each view, the
// buffer offset of the current element. All views share the walked shape.
type cursor struct {
	shape   []int
	idx     []int
	strides [][]int
	offs    []int
}

// newCursor positions a cursor at linear element start of shape.
func newCursor(shape []int, views []backend.View, start int) *cursor {
	c := &cursor{
		shape:   shape,
		idx:     make([]int, len(shape)),
		strides: make([][]int, len(views)),
		offs:    make([]int, len(views)),
	}
	rem := start
	for d := len(shape) - 1; d >= 0; d-- {
		if shape[d] > 0 {
			c.idx[d] = rem % shape[d]
			rem /= shape[d]
		}
	}
	for v, view := range views {
		c.strides[v] = view.Strides
		off := view.Offset
		for d, i := range c.idx {
			off += i * view.Strides[d]
		}
		c.offs[v] = off
	}
	return c
}

// next advances to the following element.
func (c *cursor) next() {
	for d := len(c.shape) - 1; d >= 0; d-- {
		c.idx[d]++
		for v := range c.offs {
			c.offs[v] += c.strides[v][d]
		}
		if c.idx[d] < c.shape[d] {
			return
		}
		for v := range c.offs {
			c.offs[v] -= c.strides[v][d] * c.shape[d]
		}
		c.idx[d] = 0
	}
}
