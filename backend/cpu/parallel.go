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
	c := new(cursor)
	c.reset(shape, views, start)
	return c
}

// reset positions c at linear element start of shape, reusing its memory.
func (c *cursor) reset(shape []int, views []backend.View, start int) {
	c.shape = shape
	c.idx = grow(c.idx, len(shape))
	c.strides = grow(c.strides, len(views))
	c.offs = grow(c.offs, len(views))
	rem := start
	for d := len(shape) - 1; d >= 0; d-- {
		c.idx[d] = 0
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
}

// next advances the cursor by one element in row-major order.
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

// forRows splits the output of shape into rows along the innermost
// dimension and calls fn(i, offs, steps, n) for each: the row covers output
// elements [i, i+n), offs holds where each view's row starts and steps each
// view's innermost stride. Strided and broadcast inputs then cost one
// cursor step per row instead of per element. Dimensions are coalesced
// first, so rows are as long as the views allow, and long rows are split
// between workers.
func forRows(ctx context.Context, b *Backend, shape []int, views []backend.View, fn func(i int, offs, steps []int, n int)) error {
	shape, views = backend.Coalesce(shape, views)
	r := len(shape)
	if r == 0 {
		offs := make([]int, len(views))
		for v, view := range views {
			offs[v] = view.Offset
		}
		fn(0, offs, make([]int, len(views)), 1)
		return nil
	}
	inner := shape[r-1]
	total, _ := backend.NumElements(shape)
	if total == 0 {
		return nil
	}
	outer := make([]backend.View, len(views))
	steps := make([]int, len(views))
	for v, view := range views {
		outer[v] = backend.View{Strides: view.Strides[:r-1], Offset: view.Offset}
		steps[v] = view.Strides[r-1]
	}
	return b.parallelFor(ctx, total, elementGrain, func(lo, hi int) error {
		c := newCursor(shape[:r-1], outer, lo/inner)
		offs := make([]int, len(views))
		for i := lo; i < hi; c.next() {
			col := i % inner
			n := min(inner-col, hi-i)
			for v := range offs {
				offs[v] = c.offs[v] + col*steps[v]
			}
			fn(i, offs, steps, n)
			i += n
		}
		return nil
	})
}

// mapStrided calls fn(i, off) for every output element i of shape, where off
// is the buffer offset of element i under view v.
func mapStrided(ctx context.Context, b *Backend, shape []int, v backend.View, n int, fn func(i, off int)) error {
	if v.IsContiguous() {
		return b.parallelFor(ctx, n, elementGrain, func(lo, hi int) error {
			for i := lo; i < hi; i++ {
				fn(i, v.Offset+i)
			}
			return nil
		})
	}
	return forRows(ctx, b, shape, []backend.View{v}, func(i int, offs, steps []int, n int) {
		off, s := offs[0], steps[0]
		for j := i; j < i+n; j++ {
			fn(j, off)
			off += s
		}
	})
}
