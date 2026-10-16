package sched

import (
	"cmp"
	"fmt"
	"maps"
	"runtime"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// A budget caps the memory that schedules on a device hold, so that a
// graph that does not fit is refused before it starts instead of failing
// midway or taking the process down. Before a schedule transfers an input
// or runs a kernel it predicts the memory it would obtain from the backend
// by mirroring its own allocations on a copy of the pool's free lists (see
// forecast), and it is refused with ErrMemoryBudget when that, with the
// memory the pool holds already, exceeds the budget. Pooled buffers the
// schedule would not take are freed first when that makes it fit, the
// oldest first and only as many as it needs: a schedule of new shapes,
// such as a smaller batch after a refusal, cannot reuse them, but the
// next one of the old shapes can reuse those left.
//
// The memory of results that were dropped counts until the garbage
// collector reclaims them (see memory.go). Before it refuses a schedule,
// the pool runs the collector and takes what it reclaimed off the
// account, so that whether a schedule fits does not depend on when the
// collector last ran.

// SetBudget limits the memory that schedules on be may hold to bytes, or
// removes the limit when bytes is 0 or less.
func SetBudget(be backend.Backend, bytes int64) {
	mu.Lock()
	defer mu.Unlock()
	poolOf(be).budget = max(bytes, 0)
}

// Budget returns the memory budget of be, or 0 if it has none.
func Budget(be backend.Backend) int64 {
	mu.Lock()
	defer mu.Unlock()
	return poolOf(be).budget
}

// Memory describes the device memory that the pool of a backend manages.
type Memory struct {
	Allocated int64 // bytes obtained from the backend whose buffers are alive
	Obtained  int64 // bytes obtained from the backend in all
	Predicted int64 // bytes that schedules and downloads predicted they would obtain, in all
}

// MemoryOf returns the memory of be's pool. Predicted equals Obtained as
// long as every schedule predicted its allocations exactly and none
// failed midway, which tests check.
func MemoryOf(be backend.Backend) Memory {
	mu.Lock()
	defer mu.Unlock()
	p := poolOf(be)
	return Memory{Allocated: p.held(), Obtained: p.obtained, Predicted: p.predicted}
}

// A sim predicts what an operation obtains from the backend by mirroring
// the pool's allocations on a count of its free lists. The pool serves
// the most recently freed buffer of a key first, so a buffer the
// operation frees is taken before one pooled earlier.
type sim struct {
	pooled map[poolKey]int // buffers in the pool not taken yet
	freed  map[poolKey]int // buffers the operation freed and has not taken back
	extra  int64           // bytes taken from the backend
}

// simulate returns a sim of p's free lists.
func (p *pool) simulate() *sim {
	s := &sim{pooled: make(map[poolKey]int, len(p.free)), freed: map[poolKey]int{}}
	for key, bufs := range p.free {
		s.pooled[key] = len(bufs)
	}
	return s
}

// take mirrors alloc: a pooled buffer of the key serves if there is one,
// and otherwise the backend does.
func (s *sim) take(dt backend.DType, n int) {
	key := poolKey{dt, n}
	switch {
	case s.freed[key] > 0:
		s.freed[key]--
	case s.pooled[key] > 0:
		s.pooled[key]--
	default:
		s.extra += key.bytes()
	}
}

// give mirrors release.
func (s *sim) give(key poolKey) { s.freed[key]++ }

// reserve admits an operation that s predicts, recording the prediction,
// or refuses it with ErrMemoryBudget when the bytes it obtains from the
// backend and those held already exceed the budget. When freeing pooled
// buffers that it would not take makes it fit, the oldest of them are
// freed until it does. Before refusing, it collects dropped results.
func (p *pool) reserve(s *sim) error {
	if p.budget > 0 && p.held()+s.extra > p.budget {
		spares, n := p.room(s)
		if n < 0 {
			p.collect()
			spares, n = p.room(s)
		}
		if n < 0 {
			return fmt.Errorf("sched: schedule needs %s with %s in use, budget %s: %w", mib(s.extra), mib(p.held()), mib(p.budget), backend.ErrMemoryBudget)
		}
		evict := map[poolKey]int{}
		for _, sp := range spares[:n] {
			evict[sp.key]++
		}
		for key, n := range evict {
			p.evict(key, n)
		}
	}
	p.predicted += s.extra
	return nil
}

// room returns the pooled buffers that the operation s predicts would not
// take, oldest first, and how many of them it frees to fit in the budget,
// none if it fits already, or -1 if freeing all of them is not enough.
func (p *pool) room(s *sim) ([]spare, int) {
	over := p.held() + s.extra - p.budget
	spares := p.spares(s)
	n := 0
	for ; n < len(spares) && over > 0; n++ {
		over -= spares[n].bytes
	}
	if over > 0 {
		return nil, -1
	}
	return spares, n
}

// collect runs the garbage collector and takes the buffers it reclaimed
// off the account at once, rather than when their cleanups run, which
// may be later. The collector has cleared the weak pointers to them by
// the time runtime.GC returns, and a cleanup that runs afterwards finds
// its buffer off the account already.
func (p *pool) collect() {
	runtime.GC()
	p.acct.Lock()
	defer p.acct.Unlock()
	for ref, e := range p.owned {
		if ref.Value() == nil {
			p.drop(e)
		}
	}
}

// spare is a pooled buffer that an operation would not take.
type spare struct {
	key   poolKey
	clock int   // when the buffer was freed
	bytes int64 // on the account, which freeing the buffer takes off
}

// spares returns the pooled buffers that the operation s predicts would
// not take, oldest first: of each key, the oldest, since the pool serves
// the most recently freed first.
func (p *pool) spares(s *sim) []spare {
	var out []spare
	p.acct.Lock()
	defer p.acct.Unlock()
	for key, bufs := range p.free {
		for _, b := range bufs[:s.pooled[key]] {
			out = append(out, spare{key, b.clock, p.accounted(b.buf)})
		}
	}
	// Each list is oldest first, and a stable sort keeps the buffers of a
	// key in that order, so that freeing the first n of a key frees those.
	slices.SortStableFunc(out, func(a, b spare) int {
		return cmp.Or(cmp.Compare(a.clock, b.clock), cmp.Compare(a.key.n, b.key.n), cmp.Compare(a.key.dt, b.key.dt))
	})
	return out
}

// mib formats bytes as mebibytes.
func mib(bytes int64) string { return fmt.Sprintf("%.1f MiB", float64(bytes)/(1<<20)) }

// forecast predicts the buffers that realizing p takes, by mirroring what
// realize does on a sim of m's free lists: the transfers of inputs held on
// other devices, the buffers each kernel writes (see work.kernel and
// runInto), the views that share their inputs' buffers, and the release of
// each buffer after its last reader, which lets the pool serve the next
// buffer of the same dtype and length. Nothing is freed to the backend
// within a schedule, so what the sim takes from the backend in all is
// also the most the schedule holds beyond what the pool held before.
func (p *plan) forecast(m *pool, uses map[*graph.Node]int, into map[*graph.Node]*graph.Node, isTarget map[*graph.Node]bool) (*sim, error) {
	s := m.simulate()
	uses = maps.Clone(uses)
	// Buffers taken in the schedule are numbered from 1, so that 0 is
	// memory the schedule does not own, such as a realized input's or a
	// destination's.
	type result struct {
		buf  int
		view backend.View
	}
	var keys []poolKey // of the buffers taken, by number
	take := func(dt backend.DType, n int) int {
		s.take(dt, n)
		keys = append(keys, poolKey{dt, n})
		return len(keys)
	}
	results := make(map[*graph.Node]result, len(p.order))
	refs := map[int]int{}
	moved := map[*graph.Node]result{}
	for _, n := range p.order {
		for _, in := range p.work[n].deps {
			if _, ok := moved[in]; ok || !in.Realized() || in.Mem.Dev == m.be {
				continue
			}
			// The schedule holds the copy until its end, which gives
			// nothing that a kernel could take.
			buf := take(in.DType, in.Size())
			refs[buf] = 1
			moved[in] = result{buf, backend.Contiguous(in.Shape)}
		}
	}
	lookup := func(n *graph.Node) result {
		if r, ok := moved[n]; ok {
			return r
		}
		if n.Realized() {
			return result{view: n.View}
		}
		return results[n]
	}
	release := func(n *graph.Node) {
		uses[n]--
		if uses[n] > 0 || isTarget[n] || n.Realized() {
			return
		}
		r := results[n]
		delete(results, n)
		if _, owned := refs[r.buf]; owned {
			if refs[r.buf]--; refs[r.buf] == 0 {
				delete(refs, r.buf)
				s.give(keys[r.buf-1])
			}
		}
	}

	for _, n := range p.order {
		w := p.work[n]
		views := make([]backend.View, len(w.deps))
		for i, in := range w.deps {
			views[i] = lookup(in).view
		}
		k, view, err := w.kernel(n, views)
		if err != nil {
			return nil, err
		}
		if k == nil {
			r := result{lookup(w.deps[0]).buf, view}
			if _, owned := refs[r.buf]; owned {
				refs[r.buf]++
			}
			results[n] = r
		} else {
			size, err := elements(k)
			if err != nil {
				return nil, err
			}
			for _, x := range w.nodes(n) {
				r := result{view: resultView(k, x)}
				if into[x] == nil {
					r.buf = take(k.DType, size)
					refs[r.buf] = 1
				}
				results[x] = r
			}
		}
		for _, in := range w.deps {
			release(in)
		}
	}
	return s, nil
}
