package sched

import (
	"reflect"
	"runtime"
	"sync"
	"unsafe"
	"weak"

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
// is collected. The account knows each buffer by a weak pointer to it,
// not by its address: a collected buffer's account may wait for its
// cleanup while a newer buffer lives at the same address, and the two
// must not be confused. A budget caps the memory held (see budget.go).

// poolAge is how many schedules a pooled buffer waits before it is freed.
const poolAge = 16

// pool holds the free buffers of one backend. It is guarded by mu, except
// for its account, which cleanups update outside mu. The buffers that
// running schedules hold are off its free lists (see claims in sched.go).
type pool struct {
	be     backend.Backend
	free   map[poolKey][]pooled
	clock  int   // schedules run on be
	budget int64 // bytes the pool may hold from be, or 0 for no limit

	// For tests: the bytes obtained from be in all, and the bytes that
	// schedules predicted they would obtain. They are equal when every
	// prediction was exact.
	obtained, predicted int64

	programs cache     // the programs of recent schedules (see cache.go)
	walkers  []*walker // walk the graphs of the schedules on be, one each

	loads map[poolKey]*loadOrder // the buffers uploads and copies took (see freeBuffer)

	acct      sync.Mutex
	allocated int64                         // bytes obtained from be whose buffers are alive
	owned     map[weak.Pointer[byte]]*entry // the alive buffers obtained from be
}

type poolKey struct {
	dt backend.DType
	n  int
}

// bytes returns the size of a buffer of key.
func (k poolKey) bytes() int64 { return bufferBytes(k.dt, k.n) }

// bufferBytes returns the size of a buffer of n elements of dt.
func bufferBytes(dt backend.DType, n int) int64 {
	size, _ := dt.Bytes(n)
	return int64(size)
}

// loadOrder is the addresses of the buffers of one key that uploads and
// copies took from the pool since its last schedule, at clock, in order,
// and between the two schedules before.
type loadOrder struct {
	clock     int
	cur, last []uintptr
}

type pooled struct {
	buf   backend.Buffer
	clock int // when the buffer was freed
}

// entry is a buffer's account: its size, the buffer, and the cleanup that
// settles the account when the buffer is collected.
type entry struct {
	bytes int64
	ref   weak.Pointer[byte] // the buffer, which a weak pointer does not keep alive
	stop  runtime.Cleanup
}

var pools = map[backend.Backend]*pool{}

// poolOf returns the pool of be.
func poolOf(be backend.Backend) *pool {
	p := pools[be]
	if p == nil {
		p = &pool{be: be, free: map[poolKey][]pooled{}, loads: map[poolKey]*loadOrder{}, owned: map[weak.Pointer[byte]]*entry{}}
		pools[be] = p
	}
	return p
}

// takeWalker returns a walker for a schedule on p's backend, one that an
// earlier schedule used if there is one, so that its slices are reused.
// Each schedule has its own, as several run at once.
func (p *pool) takeWalker() *walker {
	if n := len(p.walkers); n > 0 {
		w := p.walkers[n-1]
		p.walkers = p.walkers[:n-1]
		return w
	}
	return new(walker)
}

// giveWalker returns w, which takeWalker returned, once its schedule has
// ended.
func (p *pool) giveWalker(w *walker) {
	w.forget()
	p.walkers = append(p.walkers, w)
}

// alloc returns a buffer of n elements of dt with unspecified contents,
// the most recently freed one if the pool has one.
func (p *pool) alloc(dt backend.DType, n int) (backend.Buffer, error) {
	buf, _, err := p.take(poolKey{dt, n}, 0, false)
	return buf, err
}

// take returns a buffer of key with unspecified contents: the one at
// address at if the pool holds it; otherwise a new one from the backend
// if obtain is set or the pool has none of key, and otherwise the most
// recently freed one. It reports whether the buffer is new.
//
// A program asks for the buffers it took in its last run, so that a
// replayed schedule runs over the same buffers every time, rather than
// ones the order of the pool shuffled, and a backend can keep what it
// derived from them, such as a CUDA graph. A buffer that a result took
// away leaves the pool, so the program's run must obtain a new one from
// the backend: it does so for the buffer that went missing (obtain)
// rather than take the buffer of another, which would have to take
// another's in turn. The address is a preference only: a newer buffer at
// the address of a collected one serves as well as any other.
func (p *pool) take(key poolKey, at uintptr, obtain bool) (backend.Buffer, bool, error) {
	if bufs := p.free[key]; len(bufs) > 0 {
		i := -1
		if at != 0 {
			for j := len(bufs) - 1; j >= 0; j-- {
				if uintptr(addressOf(bufs[j].buf)) == at {
					i = j
					break
				}
			}
		}
		if i >= 0 || !obtain {
			if i < 0 {
				i = len(bufs) - 1
			}
			buf := bufs[i].buf
			p.free[key] = append(bufs[:i], bufs[i+1:]...)
			return buf, false, nil
		}
	}
	buf, err := p.be.Alloc(key.dt, key.n)
	if err != nil {
		return nil, false, err
	}
	p.obtained += key.bytes()
	p.track(buf)
	return buf, true, nil
}

// unfree takes buf off the free lists, if it is on them, and reports
// whether it was. The most recently freed buffers are at the end.
func (p *pool) unfree(buf backend.Buffer) bool {
	key := poolKey{buf.DType(), buf.Len()}
	bufs := p.free[key]
	for i := len(bufs) - 1; i >= 0; i-- {
		if bufs[i].buf == buf {
			p.free[key] = append(bufs[:i], bufs[i+1:]...)
			return true
		}
	}
	return false
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
	// A list that alloc emptied keeps its key and capacity, so that a
	// buffer released to it in every schedule allocates nothing.
	switch {
	case n == 0:
	case n == len(bufs):
		delete(p.free, key)
	default:
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
// happens to a result whose tensor is dropped. A buffer on the account
// already is left alone, and one that is not a pointer cannot carry a
// cleanup and is left out of the account.
//
// Release tracks the buffers it gives to the pool too: one the pool did
// not obtain, such as an uploaded gradient that ZeroGrad releases, is the
// pool's from then on, to hand out or to free like its own, so the
// account covers every buffer the pool holds and freeing a pooled buffer
// takes its bytes off.
func (p *pool) track(buf backend.Buffer) {
	ptr := addressOf(buf)
	if ptr == nil {
		return
	}
	ref := weak.Make((*byte)(ptr))
	p.acct.Lock()
	defer p.acct.Unlock()
	if p.owned[ref] != nil {
		return
	}
	e := &entry{bytes: bufferBytes(buf.DType(), buf.Len()), ref: ref}
	p.allocated += e.bytes
	p.owned[e.ref] = e
	e.stop = runtime.AddCleanup((*byte)(ptr), p.settle, e)
}

// untrack settles the account of buf as the pool frees it.
func (p *pool) untrack(buf backend.Buffer) {
	ptr := addressOf(buf)
	if ptr == nil {
		return
	}
	ref := weak.Make((*byte)(ptr))
	p.acct.Lock()
	defer p.acct.Unlock()
	e := p.owned[ref]
	if e == nil {
		return
	}
	e.stop.Stop()
	p.drop(e)
}

// settle is the cleanup of a buffer the pool obtained: the buffer has
// been collected, so its bytes are no longer held, unless its account is
// settled already.
func (p *pool) settle(e *entry) {
	p.acct.Lock()
	defer p.acct.Unlock()
	p.drop(e)
}

// accounted returns the bytes on the account for buf, or 0 if buf is not
// on it. The caller holds acct.
func (p *pool) accounted(buf backend.Buffer) int64 {
	ptr := addressOf(buf)
	if ptr == nil {
		return 0
	}
	if e := p.owned[weak.Make((*byte)(ptr))]; e != nil {
		return e.bytes
	}
	return 0
}

// drop takes e off the account, if it is on it. The caller holds acct.
func (p *pool) drop(e *entry) {
	if p.owned[e.ref] == e {
		delete(p.owned, e.ref)
		p.allocated -= e.bytes
	}
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
