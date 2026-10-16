package cpu

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// A team is a set of goroutines that help run the chunks of parallel
// loops. Starting a goroutine per chunk makes every loop wait for the
// runtime to wake idle threads, which costs tens of microseconds, as much
// as a mid-sized kernel takes to run. Team goroutines instead keep polling
// for work for a while after each loop, so that the loops of a training
// step, which follow each other closely, start at once.
//
// One loop runs on the team at a time; the goroutine that starts it runs
// chunks too, so a loop never waits for a helper that has yet to wake.
// The helpers live as long as the process and park when there is no work.
type team struct {
	mu sync.Mutex // held by the goroutine running a loop on the team

	// The running loop. todo is the number of chunks not yet claimed: a
	// goroutine that decrements it to c >= 0 has claimed chunk c. The other
	// fields are set before todo, and helpers read them only after
	// claiming a chunk. left is the number of chunks not yet finished.
	todo atomic.Int64
	left atomic.Int64
	ctx  context.Context
	fn   func(lo, hi int) error
	n    int
	size int
	errs []error

	sleepers atomic.Int32  // helpers parked or about to park
	wake     chan struct{} // receives a token per helper to wake
}

// teamSpin is how long a helper polls for a new loop, once the last has
// finished, before parking. It covers the host work between the schedules
// of a training step, such as building the next graph: a helper that
// parked meanwhile starts the next loop late, after the runtime wakes it.
const teamSpin = 5 * time.Millisecond

// yieldEvery is how many polls a helper makes between yields. Yielding
// lets other goroutines run, but every yield takes the scheduler's lock,
// which the helpers contend for when they yield after every poll.
const yieldEvery = 16

// yieldBusy is how long a yield takes when another goroutine ran during
// it. A yield with nothing else to run returns well within a microsecond.
const yieldBusy = 20 * time.Microsecond

// defaultTeam returns the process-wide team, with a helper for every
// processor beyond the caller's, or nil if there is a single processor.
var defaultTeam = sync.OnceValue(func() *team {
	helpers := runtime.GOMAXPROCS(0) - 1
	if helpers < 1 {
		return nil
	}
	t := &team{wake: make(chan struct{}, helpers)}
	for range helpers {
		go t.help()
	}
	return t
})

// run calls fn over [0, n) in chunks of size elements, using the calling
// goroutine and any helpers that are free, and returns the error of the
// first chunk that failed. The caller must hold t.mu.
func (t *team) run(ctx context.Context, n, size int, fn func(lo, hi int) error) error {
	chunks := (n + size - 1) / size
	t.ctx, t.fn, t.n, t.size = ctx, fn, n, size
	t.errs = append(t.errs[:0], make([]error, chunks)...)
	t.left.Store(int64(chunks))
	t.todo.Store(int64(chunks))
	// A helper that parked before seeing the loop needs a token. One that
	// sees the loop after announcing it parks leaves its token unused, to
	// wake a helper spuriously later.
	if s := t.sleepers.Load(); s > 0 {
		for range min(int(s), chunks-1) {
			select {
			case t.wake <- struct{}{}:
			default:
			}
		}
	}
	t.work()
	for t.left.Load() > 0 {
		runtime.Gosched()
	}
	t.ctx, t.fn = nil, nil
	for _, err := range t.errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// work runs chunks of the current loop until none is left to claim.
func (t *team) work() {
	for {
		c := int(t.todo.Add(-1))
		if c < 0 {
			return
		}
		lo := c * t.size
		t.errs[c] = runChunk(t.ctx, t.fn, lo, min(t.n, lo+t.size))
		t.left.Add(-1)
	}
}

// pollStolen is how long a poll may take, many times what its loads
// need, before the helper concludes that its thread lost the processor
// during it.
const pollStolen = 200 * time.Microsecond

// help runs chunks of every loop it sees, forever.
func (t *team) help() {
	for {
		idle := time.Now()
		last := idle
		for polls := 0; !t.poll(); polls++ {
			now := time.Now()
			switch {
			case now.Sub(last) > pollStolen:
				// The operating system ran another thread on this
				// processor during the poll: the machine has more threads
				// to run than processors, such as those of other
				// processes, and polling takes time from them, including
				// from the goroutines that finish the current loop. The
				// helper parks until the next loop starts.
				t.park()
				now = time.Now()
				idle = now
			case t.left.Load() > 0:
				// Others are finishing a loop, and the next is likely to
				// follow: a helper parked now would start it late.
				idle = now
			case now.Sub(idle) > teamSpin:
				t.park()
				now = time.Now()
				idle = now
			}
			// Let other goroutines have the processor now and then. One
			// that takes it has work of its own, such as reading the next
			// batch, which polling would only slow down: the helper parks
			// instead, and run wakes it when the next loop starts.
			if polls%yieldEvery == 0 {
				runtime.Gosched()
				yielded := time.Now()
				if yielded.Sub(now) > yieldBusy {
					t.park()
					yielded = time.Now()
					idle = yielded
				}
				now = yielded
			}
			last = now
		}
		t.work()
	}
}

// poll reports whether a loop has chunks to claim, checking repeatedly
// for a moment.
func (t *team) poll() bool {
	for range 1000 {
		if t.todo.Load() > 0 {
			return true
		}
	}
	return false
}

// park blocks until a loop starts. Announcing the helper before checking
// for work, while run sets todo before counting sleepers, means that one
// of the two sees the other: either the helper finds the loop or run sends
// it a token.
func (t *team) park() {
	t.sleepers.Add(1)
	if t.todo.Load() <= 0 {
		<-t.wake
	}
	t.sleepers.Add(-1)
}
