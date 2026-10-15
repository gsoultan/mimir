package sched

import "github.com/gsoultan/mimir/backend"

// Buffers are reused. A schedule frees each intermediate buffer once its
// last reader has run, and a buffer freed into the pool serves the next
// output of the same dtype and length, later in the same schedule or in a
// later one. A training step runs the same kernels every time, so after
// the first step it takes every intermediate from the pool and allocates
// only the buffers that outlive it, such as new parameter values.
//
// A buffer that no schedule takes within poolAge schedules goes back to
// the backend, so memory that one phase of a program used is released when
// another phase follows.

// poolAge is how many schedules a pooled buffer waits before it is freed.
const poolAge = 16

// pool holds the free buffers of one backend. It is guarded by mu.
type pool struct {
	be    backend.Backend
	free  map[poolKey][]pooled
	clock int // schedules run on be
}

type poolKey struct {
	dt backend.DType
	n  int
}

type pooled struct {
	buf   backend.Buffer
	clock int // when the buffer was freed
}

var pools = map[backend.Backend]*pool{}

// poolOf returns the pool of be.
func poolOf(be backend.Backend) *pool {
	p := pools[be]
	if p == nil {
		p = &pool{be: be, free: map[poolKey][]pooled{}}
		pools[be] = p
	}
	return p
}

// alloc returns a buffer of n elements of dt with unspecified contents,
// the most recently freed one if the pool has one.
func (p *pool) alloc(dt backend.DType, n int) (backend.Buffer, error) {
	key := poolKey{dt, n}
	if bufs := p.free[key]; len(bufs) > 0 {
		p.free[key] = bufs[:len(bufs)-1]
		return bufs[len(bufs)-1].buf, nil
	}
	return p.be.Alloc(dt, n)
}

// release returns buf to the pool.
func (p *pool) release(buf backend.Buffer) {
	key := poolKey{buf.DType(), buf.Len()}
	p.free[key] = append(p.free[key], pooled{buf, p.clock})
}

// tick ends a schedule and frees the buffers that have waited poolAge
// schedules. Each list is in the order buffers were freed, oldest first.
func (p *pool) tick() {
	p.clock++
	for key, bufs := range p.free {
		old := 0
		for old < len(bufs) && p.clock-bufs[old].clock > poolAge {
			p.be.Free(bufs[old].buf)
			old++
		}
		switch {
		case old == len(bufs):
			delete(p.free, key)
		case old > 0:
			p.free[key] = append(bufs[:0], bufs[old:]...)
		}
	}
}

// pooledElements returns the number of elements the pool holds, for tests.
func (p *pool) pooledElements() int {
	n := 0
	for key, bufs := range p.free {
		n += key.n * len(bufs)
	}
	return n
}
