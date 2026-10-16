package sched

import (
	"errors"
	"runtime"
	"strings"
	"testing"
	"time"
	"weak"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
)

// exact realizes targets on be and returns the bytes the schedule obtained
// from the backend, failing the test unless it predicted them exactly.
func exact(t *testing.T, be backend.Backend, targets ...*graph.Node) int64 {
	t.Helper()
	before := MemoryOf(be)
	if err := Realize(t.Context(), be, targets...); err != nil {
		t.Fatal(err)
	}
	return predictedExactly(t, be, before)
}

// predictedExactly fails the test unless every schedule on be since
// before obtained from the backend exactly what it predicted, and returns
// the bytes obtained.
func predictedExactly(t *testing.T, be backend.Backend, before Memory) int64 {
	t.Helper()
	after := MemoryOf(be)
	got, want := after.Obtained-before.Obtained, after.Predicted-before.Predicted
	if got != want {
		t.Errorf("schedule obtained %d bytes from the backend, predicted %d", got, want)
	}
	return got
}

func TestSchedulesPredictTheirAllocationsExactly(t *testing.T) {
	const f = 4 // bytes per float32
	neg := func(n *graph.Node) *graph.Node { return must(graph.Unary(backend.OpNeg, n)) }
	add := func(a, b *graph.Node) *graph.Node { return must(graph.Binary(backend.OpAdd, a, b)) }
	sum := func(n *graph.Node, axes ...int) *graph.Node { return must(graph.Reduce(backend.OpReduceSum, n, axes)) }
	tests := []struct {
		name string
		// build returns the targets, and the bytes the schedule obtains
		// from the backend.
		build func(t *testing.T, be backend.Backend) ([]*graph.Node, int64)
	}{
		{"fused elementwise chain", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x := realized(t, be, []int{1000}, 1)
			y := x
			for range 20 {
				y = must(graph.Unary(backend.OpSqrt, add(y, x)))
			}
			return []*graph.Node{y}, 1000 * f // one kernel, one buffer
		}},
		{"unfused chain reuses its buffers", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			y := must(graph.Rand([]int{1000, 1}, 1, false))
			for range 50 {
				y = sum(y, 1) // reductions do not fuse; each frees its input
			}
			return []*graph.Node{y}, 2 * 1000 * f
		}},
		{"reduction", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x := must(graph.Rand([]int{64, 32}, 1, false))
			return []*graph.Node{sum(x, 1)}, (64*32 + 64) * f
		}},
		{"matmul with a fused epilogue", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x, w, b := realized(t, be, []int{64, 48}, 1), realized(t, be, []int{48, 30}, 2), realized(t, be, []int{30}, 3)
			bias := must(graph.Expand(must(graph.Reshape(b, []int{1, 30})), []int{64, 30}))
			zero := must(graph.Const(backend.Float32, []int{64, 30}, 0))
			y := must(graph.Binary(backend.OpMax, add(must(graph.MatMul(x, w)), bias), zero))
			return []*graph.Node{y}, 64 * 30 * f // the product never gets a buffer
		}},
		{"transposed product", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			a, b := realized(t, be, []int{5, 3}, 1), realized(t, be, []int{3, 4}, 2)
			return []*graph.Node{must(graph.Permute(must(graph.MatMul(a, b)), []int{1, 0}))}, 5 * 4 * f
		}},
		{"summed product with its gradient-like accumulation", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x, w := realized(t, be, []int{4, 30, 20}, 1), realized(t, be, []int{4, 20, 25}, 2)
			return []*graph.Node{sum(must(graph.MatMul(x, w)), 0)}, 30 * 25 * f
		}},
		{"permute then reshape copies", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x := realized(t, be, []int{4, 6}, 1)
			r := must(graph.Reshape(must(graph.Permute(x, []int{1, 0})), []int{24}))
			return []*graph.Node{neg(r)}, 2 * 24 * f // the copy, then the result
		}},
		{"reshape, shrink, flip and expand are views", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x := realized(t, be, []int{4, 6}, 1)
			v := must(graph.Flip(must(graph.Shrink(must(graph.Reshape(x, []int{2, 12})), [][2]int{{0, 1}, {0, 12}})), []int{1}))
			e := must(graph.Expand(v, []int{3, 12}))
			return []*graph.Node{neg(e)}, 3 * 12 * f
		}},
		{"views shared by several readers", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			// The permuted copy is read by two kernels and freed after the
			// second, in time for the third to reuse it.
			x := realized(t, be, []int{8, 8}, 1)
			c := must(graph.Reshape(must(graph.Permute(x, []int{1, 0})), []int{64}))
			a, b := sum(c, 0), must(graph.Reduce(backend.OpReduceMax, c, []int{0}))
			d := neg(must(graph.Reshape(x, []int{64})))
			return []*graph.Node{a, b, d}, (64 + 1 + 1) * f
		}},
		{"broadcasting", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x, row := realized(t, be, []int{16, 32}, 1), realized(t, be, []int{1, 32}, 2)
			return []*graph.Node{add(x, must(graph.Expand(row, []int{16, 32})))}, 16 * 32 * f
		}},
		{"gather", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x := realized(t, be, []int{5, 3}, 1)
			idx := must(graph.Const(backend.Int32, []int{2, 3}, 1))
			return []*graph.Node{must(graph.Gather(x, idx, 0))}, 1*f + 2*3*f // the index constant, then the result
		}},
		{"constants read by programs cost nothing", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x := realized(t, be, []int{10}, 1)
			two := must(graph.Const(backend.Float32, []int{10}, 2))
			return []*graph.Node{must(graph.Binary(backend.OpMul, x, two))}, 10 * f
		}},
		{"pad and unfold", func(t *testing.T, be backend.Backend) ([]*graph.Node, int64) {
			x := realized(t, be, []int{2, 5, 5}, 1)
			p := must(graph.Pad(x, [][2]int{{0, 0}, {1, 1}, {1, 1}}, 0))
			w := graph.Window{Kernel: []int{3, 3}, Stride: []int{1, 1}, Dilation: []int{1, 1}, Pads: [][2]int{{0, 0}, {0, 0}}}
			u := must(graph.Unfold(p, w, 0))
			return []*graph.Node{u}, (2*7*7 + 2*3*3*5*5) * f
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			be := &counting{Backend: cpu.New()}
			targets, want := tt.build(t, be)
			if got := exact(t, be, targets...); got != want {
				t.Errorf("schedule obtained %d bytes, want %d", got, want)
			}
		})
	}
}

func TestInPlaceUpdatesArePredictedToAllocateNothing(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x, m, g := realized(t, be, []int{3000}, 1), realized(t, be, []int{3000}, 2), realized(t, be, []int{3000}, 3)
	bin := func(op backend.Op, a, b *graph.Node) *graph.Node { return must(graph.Binary(op, a, b)) }
	nextM := bin(backend.OpAdd, m, g)
	nextX := bin(backend.OpAdd, nextM, bin(backend.OpMul, x, m))
	before := MemoryOf(be)
	if err := RealizeInto(t.Context(), be, []*graph.Node{nextX, nextM}, []*graph.Node{x, m}, nil); err != nil {
		t.Fatal(err)
	}
	if got := predictedExactly(t, be, before); got != 0 {
		t.Errorf("an in-place update obtained %d bytes", got)
	}

	// A kept node computed first costs its own schedule, predicted too.
	y := realized(t, be, []int{3000}, 4)
	reader := must(graph.Unary(backend.OpNeg, y))
	before = MemoryOf(be)
	keep := func() []*graph.Node { return []*graph.Node{reader} }
	if err := RealizeInto(t.Context(), be, []*graph.Node{bin(backend.OpAdd, y, g)}, []*graph.Node{y}, keep); err != nil {
		t.Fatal(err)
	}
	if got := predictedExactly(t, be, before); got != 3000*4 {
		t.Errorf("update with a kept reader obtained %d bytes, want the reader's %d", got, 3000*4)
	}
}

func TestTransfersFromOtherDevicesArePredicted(t *testing.T) {
	a, b := &counting{Backend: cpu.New()}, &counting{Backend: cpu.New()}
	x := realized(t, a, []int{1000}, 1)
	y := must(graph.Unary(backend.OpNeg, x))
	if got := exact(t, b, y); got != 2*1000*4 {
		t.Errorf("schedule reading an input from another device obtained %d bytes, want the copy and the result", got)
	}
	// The copy is pooled afterwards and serves the next transfer.
	if got := exact(t, b, must(graph.Unary(backend.OpNeg, x))); got != 1000*4 {
		t.Errorf("schedule reading the same input again obtained %d bytes, want only its result's", got)
	}

	// A view of a moved input shares the copy, which is released at the end
	// of the schedule, and once only.
	c, d := &counting{Backend: cpu.New()}, &counting{Backend: cpu.New()}
	z := realized(t, c, []int{1000}, 2)
	v := must(graph.Reshape(z, []int{10, 100}))
	s := must(graph.Reduce(backend.OpReduceSum, v, []int{1}))
	if got := exact(t, d, s); got != (1000+10)*4 {
		t.Errorf("schedule reading a view of an input from another device obtained %d bytes", got)
	}
	if d.allocs != 2 {
		t.Errorf("%d allocations, want 2", d.allocs)
	}
	if pooled := poolOf(d).pooledElements(); pooled != 1000 {
		t.Errorf("%d elements pooled after the schedule, want the copy's 1000 once", pooled)
	}
}

func TestMovedInputsOutliveTheViewsOfTheirCopies(t *testing.T) {
	a, b := &counting{Backend: cpu.New()}, &counting{Backend: cpu.New()}
	sum := func(n *graph.Node) *graph.Node { return must(graph.Reduce(backend.OpReduceSum, n, []int{0})) }
	flat := func(n *graph.Node) *graph.Node { return must(graph.Reshape(n, []int{16})) }
	x := realized(t, a, []int{4, 4}, 1)
	xs := floats(t, a, x)
	// The sum is the last reader of a view of x's copy; the product reads
	// the copy itself after it, and must not find it handed out as its own
	// output: the copy, the sum and the product each take a buffer.
	p := must(graph.MatMul(x, x))
	if got := exact(t, b, sum(flat(x)), p); got != (16+1+16)*4 {
		t.Errorf("schedule obtained %d bytes, want %d", got, (16+1+16)*4)
	}
	for i, v := range floats(t, b, p) {
		var want float32
		for k := range 4 {
			want += xs[i/4*4+k] * xs[k*4+i%4]
		}
		if d := v - want; d > 1e-5 || d < -1e-5 {
			t.Fatalf("product element %d = %v, want %v", i, v, want)
		}
	}
	if pooled := poolOf(b).pooledElements(); pooled != 16 {
		t.Errorf("%d elements pooled after the schedule, want the copy's 16 once", pooled)
	}

	// Nor may an elementwise kernel, which may run in place, write its
	// result over the copy while a later kernel still reads it.
	y := realized(t, a, []int{4, 4}, 2)
	ys := floats(t, a, y)
	neg := must(graph.Unary(backend.OpNeg, must(graph.Permute(y, []int{1, 0}))))
	if err := Realize(t.Context(), b, sum(flat(y)), must(graph.Unary(backend.OpExp2, y)), neg); err != nil {
		t.Fatal(err)
	}
	for i, v := range floats(t, b, neg) {
		if want := -ys[i%4*4+i/4]; v != want {
			t.Fatalf("negated transpose element %d = %v, want %v", i, v, want)
		}
	}
}

func TestBudgetRefusesSchedulesBeforeAllocating(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	// The random input and the sum take a buffer of 1000 elements each.
	x := must(graph.Rand([]int{1000, 1}, 1, false))
	y := must(graph.Reduce(backend.OpReduceSum, x, []int{1}))
	const need = 2 * 1000 * 4
	SetBudget(be, need-1)
	err := Realize(t.Context(), be, y)
	if !errors.Is(err, backend.ErrMemoryBudget) {
		t.Fatalf("got %v, want ErrMemoryBudget", err)
	}
	if want := "schedule needs 0.0 MiB with 0.0 MiB in use, budget 0.0 MiB"; !strings.Contains(err.Error(), want) {
		t.Errorf("error %q, want it to say %q", err, want)
	}
	if be.allocs != 0 || len(be.ops) != 0 {
		t.Errorf("a refused schedule made %d allocations and ran %d kernels", be.allocs, len(be.ops))
	}
	if y.Realized() || x.Realized() {
		t.Error("a refused schedule realized nodes")
	}
	if m := MemoryOf(be); m.Predicted != 0 || m.Obtained != 0 {
		t.Errorf("memory after a refused schedule: %+v", m)
	}
	SetBudget(be, need)
	if err := Realize(t.Context(), be, y); err != nil {
		t.Fatalf("schedule within a budget equal to its need: %v", err)
	}
	if m := MemoryOf(be); m.Allocated != need {
		t.Errorf("%d bytes allocated, want %d", m.Allocated, need)
	}

	// Memory that results hold counts against the next schedule.
	be = &counting{Backend: cpu.New()}
	held := realized(t, be, []int{1000}, 2)
	z := must(graph.Reduce(backend.OpReduceSum, must(graph.Reshape(held, []int{10, 100})), []int{1}))
	SetBudget(be, 1000*4+10*4-1)
	if err := Realize(t.Context(), be, z); !errors.Is(err, backend.ErrMemoryBudget) {
		t.Fatalf("got %v, want ErrMemoryBudget with %d bytes in use", err, 1000*4)
	}
	SetBudget(be, 1000*4+10*4)
	if err := Realize(t.Context(), be, z); err != nil {
		t.Fatal(err)
	}
	if got := Budget(be); got != 1000*4+10*4 {
		t.Errorf("Budget = %d, want %d", got, 1000*4+10*4)
	}
	SetBudget(be, -5)
	if got := Budget(be); got != 0 {
		t.Errorf("Budget after a negative one = %d, want 0 for no limit", got)
	}
}

func TestBudgetFreesPooledBuffersTheScheduleCannotReuse(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	// Two buffers of 1000 elements: the result holds one and the pool
	// the other.
	y := must(graph.Rand([]int{1000, 1}, 1, false))
	for range 3 {
		y = must(graph.Reduce(backend.OpReduceSum, y, []int{1}))
	}
	if err := Realize(t.Context(), be, y); err != nil {
		t.Fatal(err)
	}
	if held := MemoryOf(be).Allocated; held != 2*1000*4 {
		t.Fatalf("%d bytes held, want %d", held, 2*1000*4)
	}
	// A schedule of another shape cannot use the pooled buffer, so it is
	// freed to make room.
	z := must(graph.Reduce(backend.OpReduceSum, must(graph.Rand([]int{500, 1}, 2, false)), []int{1}))
	SetBudget(be, 1000*4+2*500*4)
	allocs := be.allocs
	if err := Realize(t.Context(), be, z); err != nil {
		t.Fatal(err)
	}
	if n := be.allocs - allocs; n != 2 {
		t.Errorf("%d allocations, want 2", n)
	}
	if m := MemoryOf(be); m.Allocated != 1000*4+2*500*4 || poolOf(be).pooledElements() != 500 {
		t.Errorf("after the schedule %d bytes are held and %d elements pooled, want the pooled buffer of 1000 freed", m.Allocated, poolOf(be).pooledElements())
	}
	// Freeing would not have been enough for a larger schedule, so the
	// pool is left alone and the schedule refused.
	w := must(graph.Reduce(backend.OpReduceSum, must(graph.Rand([]int{2000, 1}, 3, false)), []int{1}))
	if err := Realize(t.Context(), be, w); !errors.Is(err, backend.ErrMemoryBudget) {
		t.Fatalf("got %v, want ErrMemoryBudget", err)
	}
	if pooled := poolOf(be).pooledElements(); pooled != 500 {
		t.Errorf("a refused schedule left %d elements pooled, want 500", pooled)
	}
}

func TestDownloadsCheckTheirCopyAgainstTheBudget(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{4, 6}, 1)
	p := must(graph.Permute(x, []int{1, 0}))
	if err := Realize(t.Context(), be, p); err != nil {
		t.Fatal(err)
	}
	held := MemoryOf(be).Allocated
	out := make([]byte, 24*4)
	SetBudget(be, held+24*4-1)
	if err := Download(t.Context(), be, p, out); !errors.Is(err, backend.ErrMemoryBudget) {
		t.Fatalf("got %v, want ErrMemoryBudget", err)
	}
	SetBudget(be, held+24*4)
	before := MemoryOf(be)
	if err := Download(t.Context(), be, p, out); err != nil {
		t.Fatal(err)
	}
	if got := predictedExactly(t, be, before); got != 24*4 {
		t.Errorf("download obtained %d bytes, want its copy's %d", got, 24*4)
	}
}

func TestCollectedResultsLeaveTheAccount(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	const n = 1 << 20
	if err := Realize(t.Context(), be, must(graph.Rand([]int{n}, 1, false))); err != nil {
		t.Fatal(err)
	}
	if held := MemoryOf(be).Allocated; held != n*4 {
		t.Fatalf("%d bytes held, want the result's %d", held, n*4)
	}
	// The result is dropped: once collected, its bytes are no longer held.
	deadline := time.Now().Add(10 * time.Second)
	for MemoryOf(be).Allocated != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("%d bytes still held after the result was collected", MemoryOf(be).Allocated)
		}
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
	if m := MemoryOf(be); m.Obtained != n*4 || m.Predicted != n*4 {
		t.Errorf("memory %+v, want %d obtained and predicted", m, n*4)
	}
}

// collectedEntry obtains a buffer of n elements of be through its pool,
// drops it and returns its address and its account entry, whose cleanup
// it stops: the entry then outlives the buffer, as the entry of a
// collected buffer does until its cleanup runs, and the test settles it.
func collectedEntry(t *testing.T, be backend.Backend, n int) (uintptr, *entry) {
	t.Helper()
	p := poolOf(be)
	buf, err := p.alloc(backend.Float32, n)
	if err != nil {
		t.Fatal(err)
	}
	addr := uintptr(addressOf(buf))
	p.acct.Lock()
	e := p.owned[weak.Make((*byte)(addressOf(buf)))]
	p.acct.Unlock()
	if e == nil {
		t.Fatal("a buffer the pool obtained is not on its account")
	}
	e.stop.Stop()
	return addr, e
}

func TestStaleAccountEntriesAreSettledOnce(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	p := poolOf(be)
	const n = 100
	addr, e := collectedEntry(t, be, n)
	// Another buffer, allocated outside the pool, takes the collected
	// buffer's address.
	var buf backend.Buffer
	var keep []backend.Buffer
	for round := 0; buf == nil && round < 64; round++ {
		runtime.GC()
		for range 4096 {
			b, err := be.Alloc(backend.Float32, n)
			if err != nil {
				t.Fatal(err)
			}
			if uintptr(addressOf(b)) == addr {
				buf = b
				break
			}
			keep = append(keep, b)
		}
	}
	if buf == nil {
		t.Skip("the allocator did not reuse the collected buffer's address")
	}
	runtime.KeepAlive(keep)
	// It reaches the pool, as a released gradient does, and is freed. The
	// collected buffer's entry is not its account, and its bytes come off
	// once, when its cleanup settles it.
	node, err := graph.Load(be, buf, []int{n})
	if err != nil {
		t.Fatal(err)
	}
	Release(node, nil)
	if held := p.held(); held < 0 {
		t.Fatalf("%d bytes held with a buffer given to the pool", held)
	}
	p.evict(poolKey{backend.Float32, n}, 1)
	if held := p.held(); held != n*4 {
		t.Errorf("%d bytes held after the pool freed the buffer, want the collected buffer's %d, not settled yet", held, n*4)
	}
	p.settle(e)
	if held := p.held(); held != 0 {
		t.Errorf("%d bytes held after the collected buffer's account settled, want 0", held)
	}
	p.acct.Lock()
	left := len(p.owned)
	p.acct.Unlock()
	if left != 0 {
		t.Errorf("%d buffers on the account after both were freed", left)
	}
}

func TestBudgetFreesTheOldestPooledBuffersItNeedsOnly(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	// A pooled buffer of 300 elements, then eight of 1000 freed later.
	small := realized(t, be, []int{300}, 1)
	large := make([]*graph.Node, 8)
	for i := range large {
		large[i] = realized(t, be, []int{1000}, uint64(i+2))
	}
	Release(small, nil)
	kept := realized(t, be, []int{1}, 10)
	defer runtime.KeepAlive(kept)
	for _, n := range large {
		Release(n, nil)
	}
	held := MemoryOf(be).Allocated
	if want := int64(300*4 + 8*1000*4 + 4); held != want {
		t.Fatalf("%d bytes held, want %d", held, want)
	}
	// A schedule needing 2000 bytes is 1996 over the budget: the oldest
	// pooled buffer, of 300, is not enough, and with one of 1000 it fits.
	SetBudget(be, held+4)
	if got := exact(t, be, must(graph.Rand([]int{500}, 11, false))); got != 500*4 {
		t.Errorf("schedule obtained %d bytes, want %d", got, 500*4)
	}
	if pooled := poolOf(be).pooledElements(); pooled != 7*1000 {
		t.Errorf("%d elements pooled after the schedule, want %d: the buffer of 300 and one of 1000 freed", pooled, 7*1000)
	}
	if got, want := MemoryOf(be).Allocated, held-300*4-1000*4+500*4; got != want {
		t.Errorf("%d bytes held after the schedule, want %d", got, want)
	}
}

func TestBudgetCountsMemoryGivenToThePool(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	held := realized(t, be, []int{1000}, 1)
	defer runtime.KeepAlive(held)
	// An uploaded buffer, as a gradient moved with its parameter is, is not
	// counted until its memory is given to the pool, which may serve or
	// free it from then on like any other.
	buf, err := be.Alloc(backend.Float32, 1000)
	if err != nil {
		t.Fatal(err)
	}
	given, err := graph.Load(be, buf, []int{1000})
	if err != nil {
		t.Fatal(err)
	}
	if m := MemoryOf(be); m.Allocated != 1000*4 {
		t.Fatalf("%d bytes held with an uploaded buffer, want %d", m.Allocated, 1000*4)
	}
	Release(given, nil)
	if m := MemoryOf(be); m.Allocated != 2*1000*4 {
		t.Fatalf("%d bytes held once the uploaded buffer is pooled, want %d", m.Allocated, 2*1000*4)
	}
	// Freeing the pooled buffer would not make 2000 more bytes fit beside
	// the 4000 held in a budget of 5000, so the schedule is refused and
	// the pool left alone.
	y := must(graph.Rand([]int{500}, 2, false))
	SetBudget(be, 5000)
	if err := Realize(t.Context(), be, y); !errors.Is(err, backend.ErrMemoryBudget) {
		t.Fatalf("got %v, want ErrMemoryBudget", err)
	}
	if pooled := poolOf(be).pooledElements(); pooled != 1000 {
		t.Errorf("a refused schedule left %d elements pooled, want 1000", pooled)
	}
	// With room for it once the pooled buffer is freed, it runs.
	SetBudget(be, 6000)
	if got := exact(t, be, y); got != 500*4 {
		t.Errorf("schedule obtained %d bytes, want %d", got, 500*4)
	}
	if m := MemoryOf(be); m.Allocated != 6000 || poolOf(be).pooledElements() != 0 {
		t.Errorf("after the schedule %d bytes are held and %d elements pooled, want 6000 and the given buffer freed", m.Allocated, poolOf(be).pooledElements())
	}
}
