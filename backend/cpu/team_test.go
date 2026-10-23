package cpu

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestParallelForCoversEveryIndexOnce(t *testing.T) {
	for _, workers := range []int{1, 2, 3, 4, 7, 16} {
		for _, n := range []int{1, 5, 100, 1001} {
			for _, grain := range []int{1, 3, 64} {
				b := &Backend{workers: workers}
				seen := make([]atomic.Int32, n)
				err := b.parallelFor(t.Context(), n, grain, func(lo, hi int) error {
					for i := lo; i < hi; i++ {
						seen[i].Add(1)
					}
					return nil
				})
				if err != nil {
					t.Fatal(err)
				}
				for i := range seen {
					if got := seen[i].Load(); got != 1 {
						t.Fatalf("%d workers, n %d, grain %d: index %d visited %d times", workers, n, grain, i, got)
					}
				}
			}
		}
	}
}

func TestParallelForReturnsErrorOfFirstFailedChunk(t *testing.T) {
	b := &Backend{workers: 4}
	useHelpers()
	var mu sync.Mutex
	first := -1
	err := b.parallelFor(t.Context(), 100, 1, func(lo, hi int) error {
		if lo < 50 {
			return nil
		}
		mu.Lock()
		defer mu.Unlock()
		if first < 0 || lo < first {
			first = lo
		}
		return fmt.Errorf("chunk at %d", lo)
	})
	if want := fmt.Sprintf("chunk at %d", first); first < 0 || err == nil || err.Error() != want {
		t.Fatalf("got %v, want %q, the error of the first chunk that failed", err, want)
	}
}

func TestParallelForFromManyGoroutines(t *testing.T) {
	// Loops started together share the team or run on goroutines of their
	// own; each must still cover its range exactly once.
	b := &Backend{workers: 4}
	var wg sync.WaitGroup
	errs := make([]error, 8)
	for g := range errs {
		wg.Go(func() {
			for range 50 {
				var sum atomic.Int64
				err := b.parallelFor(t.Context(), 1000, 10, func(lo, hi int) error {
					for i := lo; i < hi; i++ {
						sum.Add(int64(i))
					}
					return nil
				})
				if err == nil && sum.Load() != 999*1000/2 {
					err = fmt.Errorf("sum %d, want %d", sum.Load(), 999*1000/2)
				}
				if err != nil {
					errs[g] = err
					return
				}
			}
		})
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
}

func TestParallelForWakesParkedHelpers(t *testing.T) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 {
		t.Skip("needs two processors")
	}
	b := &Backend{workers: workers}
	useHelpers()
	for range 3 {
		time.Sleep(10 * teamSpin) // long enough for the helpers to park
		// Every chunk waits for all of them to start, so the loop ends only
		// if every helper joins it.
		var started sync.WaitGroup
		started.Add(workers)
		all := make(chan struct{})
		go func() {
			started.Wait()
			close(all)
		}()
		err := b.parallelFor(t.Context(), workers, 1, func(lo, hi int) error {
			started.Done()
			select {
			case <-all:
				return nil
			case <-time.After(10 * time.Second):
				return errors.New("a helper did not join the loop")
			}
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestHelpersPollAboutAsLongAsLoopsRun(t *testing.T) {
	tm := &team{}
	for _, c := range []struct {
		loops time.Duration
		want  time.Duration
	}{
		{10 * time.Microsecond, spinPerBusy * 10 * time.Microsecond},
		{60 * time.Microsecond, minSpin},
		{400 * time.Microsecond, spinPerBusy * 400 * time.Microsecond},
		{10 * time.Millisecond, teamSpin},
		{time.Microsecond, spinPerBusy * time.Microsecond},
	} {
		for range 200 {
			tm.ran(c.loops, 0)
		}
		if got := tm.spin(); got < c.want*9/10 || got > c.want*11/10 {
			t.Errorf("after loops of %v, helpers poll for %v, want %v", c.loops, got, c.want)
		}
	}
}

// useHelpers has the default team's next loop use its helpers, which
// other tests' loops over in no time may have sent solo (see solo).
func useHelpers() {
	defaultTeam().busy.Store(int64(teamSpin))
}

func TestShortLoopsRunOnTheCaller(t *testing.T) {
	tm := newTeam(3)
	run := func(fn func(lo, hi int) error) (calls int) {
		tm.mu.Lock()
		defer tm.mu.Unlock()
		var n atomic.Int32
		if err := tm.run(t.Context(), 64, 1, func(lo, hi int) error {
			n.Add(1)
			return fn(lo, hi)
		}); err != nil {
			t.Fatal(err)
		}
		return int(n.Load())
	}
	// A new team uses its helpers, so a loop that could use them does.
	if calls := run(func(int, int) error { return nil }); calls != 64 {
		t.Fatalf("the first loop ran in %d calls, want 64 chunks", calls)
	}
	// After a run of loops over in no time, the next runs on the caller:
	// one call over the whole range. The loops are recorded rather than
	// run, since how long a loop of nothing takes depends on the machine
	// (and the race detector).
	for range 200 {
		tm.ran(time.Microsecond, time.Millisecond)
	}
	if !tm.solo() {
		t.Fatalf("after 200 loops of a microsecond a millisecond apart, helpers still join: recent loops ran %v", time.Duration(tm.busy.Load()))
	}
	if calls := run(func(int, int) error { return nil }); calls != 1 {
		t.Fatalf("a short loop ran in %d calls, want 1, on the caller", calls)
	}
	// A loop that runs long brings the helpers back for the next.
	run(func(int, int) error { time.Sleep(5 * time.Millisecond); return nil })
	if calls := run(func(int, int) error { return nil }); calls != 64 {
		t.Fatalf("after a long loop, the next ran in %d calls, want 64 chunks", calls)
	}
}
