package sched

import (
	"reflect"
	"runtime"
	"sync"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// Buffers are reused. A schedule frees each intermediate buffer once its
// last reader has run, and a buffer freed into the pool serves the next
// output of the same dtype and length, later in the same schedule or in a
// later one. A training step runs the same kernels every time, so after
// the first step it takes every intermediate from the pool and allocates
// only the buffers that outlive it, such as the loss it returns; new
// parameter values go over the old ones (see RealizeInto).
//
// A buffer that no schedule takes within poolAge schedules goes back to
// the backend, so memory that one phase of a program used is released when
// another phase follows.
//
// The pool keeps account of the memory it has obtained from the backend
// that is still held: by results, by the pool itself, or by buffers that
// nothing references any more but the garbage collector has not reclaimed
// yet. A result that its tensor drops never comes back to the pool, so
// each buffer carries a cleanup that settles its account when the buffer
// is collected. A budget caps the memory held (see budget.go).

// poolAge is how many schedules a pooled buffer waits before it is freed.
const poolAge = 16

// pool holds the free buffers of one backend. It is guarded by mu, except
// for its account, which cleanups update outside mu.
type pool struct {
	be     backend.Backend
	free   map[poolKey][]pooled
	clock  int   // schedules run on be
	budget int64 // bytes the pool may hold from be, or 0 for no limit

	// For tests: the bytes obtained from be in all, and the bytes that
	// schedules predicted they would obtain. They are equal when every
	// prediction was exact.
	obtained, predicted int64

	acct      sync.Mutex
	allocated int64              // bytes obtained from be whose buffers are alive
	owned     map[uintptr]*entry // the alive buffers obtained from be, by address
}

type poolKey struct {
	dt backend.DType
	n  int
}

// bytes returns the size of a buffer of key.
func (k poolKey) bytes() int64 { return int64(k.n) * int64(k.dt.Size()) }

type pooled struct {
	buf   backend.Buffer
	clock int // when the buffer was freed
}

// entry is a buffer's account: its size, where the buffer lives and the
// cleanup that settles the account when the buffer is collected.
type entry struct {
	bytes int64
	addr  uintptr
	stop  runtime.Cleanup
}

var pools = map[backend.Backend]*pool{}

// poolOf returns the pool of be.
func poolOf(be backend.Backend) *pool {
	p := pools[be]
	if p == nil {
		p = &pool{be: be, free: map[poolKey][]pooled{}, owned: map[uintptr]*entry{}}
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
	buf, err := p.be.Alloc(dt, n)
	if err != nil {
		return nil, err
	}
	p.obtained += key.bytes()
	p.track(buf)
	return buf, nil
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
			old++
		}
		p.evict(key, old)
	}
}

// evict frees the n oldest pooled buffers of key.
func (p *pool) evict(key poolKey, n int) {
	bufs := p.free[key]
	for _, b := range bufs[:n] {
		p.untrack(b.buf)
		p.be.Free(b.buf)
	}
	switch {
	case n == len(bufs):
		delete(p.free, key)
	case n > 0:
		p.free[key] = append(bufs[:0], bufs[n:]...)
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

// track adds buf, just obtained from the backend, to the account, with a
// cleanup that settles it when nothing references buf any more, as
// happens to a result whose tensor is dropped. A buffer that is not a
// pointer cannot carry a cleanup and is left out of the account.
func (p *pool) track(buf backend.Buffer) {
	ptr := addressOf(buf)
	if ptr == nil {
		return
	}
	e := &entry{bytes: int64(buf.Len()) * int64(buf.DType().Size()), addr: uintptr(ptr)}
	p.acct.Lock()
	defer p.acct.Unlock()
	p.allocated += e.bytes
	p.owned[e.addr] = e
	e.stop = runtime.AddCleanup((*byte)(ptr), p.settle, e)
}

// untrack settles the account of buf as the pool frees it, if the pool
// obtained it: a buffer that reached the pool from elsewhere, such as an
// uploaded gradient that ZeroGrad released, was never counted.
func (p *pool) untrack(buf backend.Buffer) {
	ptr := addressOf(buf)
	if ptr == nil {
		return
	}
	p.acct.Lock()
	defer p.acct.Unlock()
	e := p.owned[uintptr(ptr)]
	if e == nil {
		return
	}
	e.stop.Stop()
	delete(p.owned, e.addr)
	p.allocated -= e.bytes
}

// settle is the cleanup of a buffer the pool obtained: the buffer has
// been collected, so its bytes are no longer held. Its address may hold a
// newer buffer by now, whose entry is left alone.
func (p *pool) settle(e *entry) {
	p.acct.Lock()
	defer p.acct.Unlock()
	if p.owned[e.addr] == e {
		delete(p.owned, e.addr)
	}
	p.allocated -= e.bytes
}

// held returns the bytes obtained from the backend whose buffers are
// alive.
func (p *pool) held() int64 {
	p.acct.Lock()
	defer p.acct.Unlock()
	return p.allocated
}

// addressOf returns the object that buf points to, or nil if buf is not a
// pointer.
func addressOf(buf backend.Buffer) unsafe.Pointer {
	v := reflect.ValueOf(buf)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return nil
	}
	return v.UnsafePointer()
}
