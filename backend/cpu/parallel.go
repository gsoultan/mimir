package cpu

import (
	"context"
	"fmt"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// parallelFor runs fn over [0, n) split into contiguous chunks of at least
// grain elements, using up to b.workers goroutines. Chunks never overlap, so
// fn may write its own range of an output without synchronization.
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
	errs := make([]error, chunks)
	var wg sync.WaitGroup
	for c := range chunks {
		lo := c * size
		hi := min(n, lo+size)
		if lo >= hi {
			break
		}
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
