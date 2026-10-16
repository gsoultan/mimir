package sched

import (
	"context"
	"errors"
	"reflect"
	"runtime"
	"slices"
	"testing"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
)

// null is a backend whose kernels do nothing, so that a test or benchmark
// sees the scheduler's own work and allocations.
type null struct{ allocs int }

type nullBuffer struct {
	dt backend.DType
	n  int
}

func (b *nullBuffer) DType() backend.DType { return b.dt }
func (b *nullBuffer) Len() int             { return b.n }

func (*null) Name() string { return "null" }
func (b *null) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	b.allocs++
	return &nullBuffer{dt, n}, nil
}
func (*null) Free(backend.Buffer)                                    {}
func (*null) Upload(context.Context, backend.Buffer, []byte) error   { return nil }
func (*null) Download(context.Context, []byte, backend.Buffer) error { return nil }
func (*null) Run(context.Context, *backend.Kernel, []backend.Buffer, []backend.Buffer) error {
	return nil
}
func (*null) Sync(context.Context) error { return nil }

// load returns a realized node of shape on be, without computing it.
func load(tb testing.TB, be backend.Backend, dt backend.DType, shape ...int) *graph.Node {
	tb.Helper()
	n, _ := backend.NumElements(shape)
	buf, err := be.Alloc(dt, n)
	if err != nil {
		tb.Fatal(err)
	}
	return must(graph.Load(be, buf, shape))
}

// mlp is a perceptron of layers linear layers, trained by gradient
// descent with momentum on a squared error. Its step builds a graph the
// way the tensor package does: the forward pass, the loss, the backward
// pass and the update of each parameter and momentum, written over the
// old ones.
type mlp struct {
	x, y       *graph.Node   // a batch and its targets
	ws, bs     []*graph.Node // weights and biases
	mws, mbs   []*graph.Node // their momenta
	batch, dim int
}

func newMLP(tb testing.TB, be backend.Backend, layers, batch, dim int) *mlp {
	tb.Helper()
	m := &mlp{batch: batch, dim: dim}
	m.x, m.y = load(tb, be, backend.Float32, batch, dim), load(tb, be, backend.Float32, batch, dim)
	for range layers {
		m.ws = append(m.ws, load(tb, be, backend.Float32, dim, dim))
		m.bs = append(m.bs, load(tb, be, backend.Float32, dim))
		m.mws = append(m.mws, load(tb, be, backend.Float32, dim, dim))
		m.mbs = append(m.mbs, load(tb, be, backend.Float32, dim))
	}
	return m
}

// randomize fills every input of m with random values.
func (m *mlp) randomize(tb testing.TB, be backend.Backend) {
	tb.Helper()
	nodes := slices.Concat([]*graph.Node{m.x, m.y}, m.ws, m.bs, m.mws, m.mbs)
	for i, n := range nodes {
		r := must(graph.Rand(n.Shape, uint64(i+1), false))
		half := must(graph.Const(backend.Float32, n.Shape, 0.5))
		r = must(graph.Binary(backend.OpAdd, r, must(graph.Unary(backend.OpNeg, half))))
		if err := RealizeInto(context.Background(), be, []*graph.Node{r}, []*graph.Node{n}, nil); err != nil {
			tb.Fatal(err)
		}
		nodes[i] = r
	}
	m.x, m.y = nodes[0], nodes[1]
	l := len(m.ws)
	m.ws, m.bs, m.mws, m.mbs = nodes[2:2+l], nodes[2+l:2+2*l], nodes[2+2*l:2+3*l], nodes[2+3*l:]
}

// step returns the targets of one training step at learning rate lr, the
// loss first, and their destinations.
func (m *mlp) step(lr float64) (targets, dsts []*graph.Node) {
	bin := func(op backend.Op, a, b *graph.Node) *graph.Node { return must(graph.Binary(op, a, b)) }
	full := func(v float64, shape []int) *graph.Node { return must(graph.Const(backend.Float32, shape, v)) }
	tr := func(a *graph.Node) *graph.Node { return must(graph.Permute(a, []int{1, 0})) }
	shape := []int{m.batch, m.dim}
	var acts, pres []*graph.Node
	h := m.x
	for i, w := range m.ws {
		acts = append(acts, h)
		bias := must(graph.Expand(must(graph.Reshape(m.bs[i], []int{1, m.dim})), shape))
		pre := bin(backend.OpAdd, must(graph.MatMul(h, w)), bias)
		pres = append(pres, pre)
		h = bin(backend.OpMax, pre, full(0, shape))
	}
	d := bin(backend.OpAdd, h, must(graph.Unary(backend.OpNeg, m.y)))
	scale := 1 / float64(m.batch*m.dim)
	loss := bin(backend.OpMul, must(graph.Reduce(backend.OpReduceSum, bin(backend.OpMul, d, d), []int{0, 1})), full(scale, []int{1, 1}))
	targets, dsts = []*graph.Node{loss}, []*graph.Node{nil}
	g := bin(backend.OpMul, d, full(2*scale, shape))
	for i := len(m.ws) - 1; i >= 0; i-- {
		g = must(graph.Where(bin(backend.OpCmpLT, full(0, shape), pres[i]), g, full(0, shape)))
		gw := must(graph.MatMul(tr(acts[i]), g))
		gb := must(graph.Reshape(must(graph.Reduce(backend.OpReduceSum, g, []int{0})), []int{m.dim}))
		if i > 0 {
			g = must(graph.MatMul(g, tr(m.ws[i])))
		}
		for _, u := range []struct{ p, mom, grad *graph.Node }{{m.ws[i], m.mws[i], gw}, {m.bs[i], m.mbs[i], gb}} {
			mom := bin(backend.OpAdd, bin(backend.OpMul, u.mom, full(0.9, u.mom.Shape)), u.grad)
			p := bin(backend.OpAdd, u.p, must(graph.Unary(backend.OpNeg, bin(backend.OpMul, mom, full(lr, mom.Shape)))))
			targets, dsts = append(targets, p, mom), append(dsts, u.p, u.mom)
		}
	}
	return targets, dsts
}

// train runs a step and points m at the new parameters.
func (m *mlp) train(tb testing.TB, be backend.Backend, lr float64) *graph.Node {
	tb.Helper()
	targets, dsts := m.step(lr)
	if err := RealizeInto(context.Background(), be, targets, dsts, nil); err != nil {
		tb.Fatal(err)
	}
	m.adopt(targets)
	return targets[0]
}

// adopt points m at the parameters and momenta of a step's targets.
func (m *mlp) adopt(targets []*graph.Node) {
	for i, j := len(m.ws)-1, 1; i >= 0; i, j = i-1, j+4 {
		m.ws[i], m.mws[i], m.bs[i], m.mbs[i] = targets[j], targets[j+1], targets[j+2], targets[j+3]
	}
}

// values downloads the elements of n.
func values(tb testing.TB, be backend.Backend, n *graph.Node) []float32 {
	tb.Helper()
	b := make([]byte, 4*n.Size())
	if err := Download(context.Background(), be, n, b); err != nil {
		tb.Fatal(err)
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n.Size())
}

// withCaching runs f with the cache on or off.
func withCaching(on bool, f func()) {
	was := SetCaching(on)
	defer SetCaching(was)
	f()
}

func TestReplayedStepsMatchPlannedOnes(t *testing.T) {
	var runs [2][][]float32
	for i, on := range []bool{false, true} {
		withCaching(on, func() {
			be := cpu.New()
			m := newMLP(t, be, 3, 16, 8)
			m.randomize(t, be)
			for s := range 6 {
				// The learning rate changes every step, as a schedule's does,
				// which changes the arguments of the update's kernels only.
				loss := m.train(t, be, 0.1/float64(s+1))
				runs[i] = append(runs[i], values(t, be, loss))
			}
			for _, p := range slices.Concat(m.ws, m.bs, m.mws, m.mbs) {
				runs[i] = append(runs[i], values(t, be, p))
			}
			if c := poolOf(be).programs; on && c.hits < 5 {
				t.Errorf("%d of 6 steps replayed a program, want at least 5", c.hits)
			}
		})
	}
	for i := range runs[0] {
		if !slices.Equal(runs[0][i], runs[1][i]) {
			t.Fatalf("value %d differs between planned and replayed steps:\n%v\n%v", i, runs[0][i], runs[1][i])
		}
	}
}

// TestCacheReplaysOnlyTheSameStructure realizes pairs of graphs that a
// variant builds on one backend, and checks that the second replays the
// first's program exactly when it should, and computes what a planned
// schedule computes either way.
func TestCacheReplaysOnlyTheSameStructure(t *testing.T) {
	type variant func(be backend.Backend, second bool) (targets, dsts []*graph.Node)
	bin := func(op backend.Op, a, b *graph.Node) *graph.Node { return must(graph.Binary(op, a, b)) }
	neg := func(a *graph.Node) *graph.Node { return must(graph.Unary(backend.OpNeg, a)) }
	pick := func(second bool, a, b int) int {
		if second {
			return b
		}
		return a
	}
	tests := []struct {
		name   string
		replay bool
		build  variant
	}{
		{"same graph, new buffers", true, func(be backend.Backend, _ bool) ([]*graph.Node, []*graph.Node) {
			x := realizedOn(be, []int{4, 5}, 1)
			return []*graph.Node{neg(must(graph.MatMul(x, realizedOn(be, []int{5, 3}, 2))))}, nil
		}},
		{"another shape", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x := realizedOn(be, []int{4, pick(second, 5, 6)}, 1)
			return []*graph.Node{must(graph.Reduce(backend.OpReduceSum, neg(x), []int{1}))}, nil
		}},
		{"another dtype", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x := realizedOn(be, []int{8}, 1)
			if second {
				x = must(graph.Cast(x, backend.BFloat16))
				if err := Realize(context.Background(), be, x); err != nil {
					panic(err)
				}
			}
			return []*graph.Node{must(graph.Cast(neg(x), backend.Float32))}, nil
		}},
		{"other reduced axes", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x := realizedOn(be, []int{4, 4}, 1)
			return []*graph.Node{must(graph.Reduce(backend.OpReduceSum, neg(x), []int{pick(second, 0, 1)}))}, nil
		}},
		{"another constant", true, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x := realizedOn(be, []int{6}, 1)
			c := must(graph.Const(backend.Float32, []int{6}, float64(pick(second, 2, 3))))
			p := must(graph.Pad(x, [][2]int{{1, 1}}, float64(pick(second, 7, -7))))
			return []*graph.Node{bin(backend.OpMul, x, c), p}, nil
		}},
		{"another seed", true, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			r := must(graph.Rand([]int{100}, uint64(pick(second, 1, 2)), true))
			return []*graph.Node{must(graph.Reduce(backend.OpReduceMax, r, []int{0}))}, nil
		}},
		{"inputs that share memory", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x, y := realizedOn(be, []int{6}, 1), realizedOn(be, []int{6}, 2)
			if second {
				y = x
			}
			return []*graph.Node{bin(backend.OpAdd, x, y)}, nil
		}},
		{"a view of an input", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x := realizedOn(be, []int{4, 4}, 1)
			if second {
				x = must(graph.Permute(x, []int{1, 0}))
				if err := Realize(context.Background(), be, x); err != nil {
					panic(err)
				}
			}
			return []*graph.Node{neg(x)}, nil
		}},
		{"an input on another device", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			on := be
			if second {
				on = cpu.New()
			}
			return []*graph.Node{neg(realizedOn(on, []int{6}, 1))}, nil
		}},
		{"a destination", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x := realizedOn(be, []int{6}, 1)
			var d *graph.Node
			if second {
				d = x
			}
			return []*graph.Node{neg(x)}, []*graph.Node{d}
		}},
		{"a destination that is stale", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x, d := realizedOn(be, []int{6}, 1), realizedOn(be, []int{6}, 2)
			if second {
				d.Mem.Version++
			}
			return []*graph.Node{neg(x)}, []*graph.Node{d}
		}},
		{"another target", false, func(be backend.Backend, second bool) ([]*graph.Node, []*graph.Node) {
			x := realizedOn(be, []int{6}, 1)
			e := must(graph.Unary(backend.OpExp2, x))
			if second {
				return []*graph.Node{neg(e), e}, nil
			}
			return []*graph.Node{neg(e)}, nil
		}},
	}
	realize := func(t *testing.T, be backend.Backend, targets, dsts []*graph.Node) {
		t.Helper()
		if dsts == nil {
			dsts = make([]*graph.Node, len(targets))
		}
		if err := RealizeInto(t.Context(), be, targets, dsts, nil); err != nil {
			t.Fatal(err)
		}
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			be := &counting{Backend: cpu.New()}
			var got, want [][]float32
			for _, second := range []bool{false, true} {
				targets, dsts := tt.build(be, second)
				c := poolOf(be).programs
				realize(t, be, targets, dsts)
				if second {
					if replayed := poolOf(be).programs.hits > c.hits; replayed != tt.replay {
						t.Errorf("replayed %v, want %v", replayed, tt.replay)
					}
					for _, n := range targets {
						got = append(got, values(t, be, n))
					}
				}
			}
			withCaching(false, func() {
				targets, dsts := tt.build(be, true)
				realize(t, be, targets, dsts)
				for _, n := range targets {
					want = append(want, values(t, be, n))
				}
			})
			for i := range want {
				if !slices.Equal(got[i], want[i]) {
					t.Errorf("target %d = %v, want %v", i, got[i], want[i])
				}
			}
		})
	}
}

// realizedOn returns n random values on be, computed.
func realizedOn(be backend.Backend, shape []int, seed uint64) *graph.Node {
	n := must(graph.Rand(shape, seed, false))
	if err := Realize(context.Background(), be, n); err != nil {
		panic(err)
	}
	return n
}

func TestReplaysCheckTheBudget(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	build := func() *graph.Node {
		x := must(graph.Rand([]int{1000, 1}, 1, false))
		return must(graph.Reduce(backend.OpReduceSum, x, []int{1}))
	}
	first := build()
	defer runtime.KeepAlive(first) // its memory counts against the budget
	if err := Realize(t.Context(), be, first); err != nil {
		t.Fatal(err)
	}
	// The pool holds the random values' buffer and the result holds the
	// sum's: a replay needs a new buffer for its sum.
	held := MemoryOf(be).Allocated
	SetBudget(be, held+1000*4-1)
	defer SetBudget(be, 0)
	y := build()
	if err := Realize(t.Context(), be, y); !errors.Is(err, backend.ErrMemoryBudget) {
		t.Fatalf("got %v, want ErrMemoryBudget", err)
	}
	if y.Realized() {
		t.Error("a refused replay realized its target")
	}
	SetBudget(be, held+1000*4)
	before := MemoryOf(be)
	if err := Realize(t.Context(), be, y); err != nil {
		t.Fatal(err)
	}
	predictedExactly(t, be, before)
	if c := poolOf(be).programs; c.hits != 2 {
		t.Errorf("%d replays, want 2", c.hits)
	}
}

func TestReplaysAllocateNothing(t *testing.T) {
	if testing.Short() {
		t.Skip("allocation counts")
	}
	be := &null{}
	// An optimizer's update of a parameter p and its momentum from the
	// gradient g, written over them, as optim.Step schedules it.
	bin := func(op backend.Op, a, b *graph.Node) *graph.Node { return must(graph.Binary(op, a, b)) }
	full := func(v float64) *graph.Node { return must(graph.Const(backend.Float32, []int{64, 32}, v)) }
	p, mom, g := load(t, be, backend.Float32, 64, 32), load(t, be, backend.Float32, 64, 32), load(t, be, backend.Float32, 64, 32)
	// Building the graphs allocates, so they are built first, each step
	// from the results of the one before. Two runs warm the cache and the
	// pool up, and AllocsPerRun runs once more than it counts.
	const runs = 20
	steps := make([][2][]*graph.Node, runs+3)
	for i := range steps {
		lr := 0.1 / float64(i+1)
		nextMom := bin(backend.OpAdd, bin(backend.OpMul, mom, full(0.9)), g)
		nextP := bin(backend.OpAdd, p, must(graph.Unary(backend.OpNeg, bin(backend.OpMul, nextMom, full(lr)))))
		steps[i] = [2][]*graph.Node{{nextP, nextMom}, {p, mom}}
		p, mom = nextP, nextMom
	}
	i := 0
	update := func() {
		if err := RealizeInto(context.Background(), be, steps[i][0], steps[i][1], nil); err != nil {
			t.Fatal(err)
		}
		i++
	}
	update()
	update()
	if allocs := testing.AllocsPerRun(runs, update); allocs != 0 {
		t.Errorf("a replayed in-place update made %v allocations, want none", allocs)
	}

	// A new result needs a storage, and nothing else.
	w := load(t, be, backend.Float32, 32, 16)
	xs := make([]*graph.Node, runs+3)
	for i := range xs {
		xs[i] = must(graph.Unary(backend.OpExp2, must(graph.MatMul(g, w))))
	}
	i = 0
	forward := func() {
		if err := Realize(context.Background(), be, xs[i]); err != nil {
			t.Fatal(err)
		}
		// The caller gives the result back, as ZeroGrad gives back a
		// gradient, so that the next run takes it from the pool.
		Release(xs[i], nil)
		i++
	}
	forward()
	forward()
	allocs := testing.AllocsPerRun(runs, forward)
	if allocs != 1 {
		t.Errorf("a replayed schedule with a new result made %v allocations, want 1", allocs)
	}
}

func TestEveryAttrsFieldButTheArgumentsIsInTheKey(t *testing.T) {
	if keyAttrs == nil {
		t.Fatal("backend.Attrs has a field the schedule key cannot hold")
	}
	type attrs struct {
		Axes  []int
		Scale float64 // a field a new op might add
		Pads  [][2]int
		Value float64
		Seed  uint64
		Name  string
	}
	enc, err := encoderOf(reflect.TypeFor[attrs](), argFields)
	if err != nil {
		t.Fatal(err)
	}
	key := func(a attrs) []uint64 {
		var w walker
		enc(&w, unsafe.Pointer(&a))
		return w.key
	}
	base := attrs{Axes: []int{1, 2}, Scale: 0.5, Pads: [][2]int{{1, 0}}, Name: "q4"}
	for name, a := range map[string]attrs{
		"axes":  {Axes: []int{1}, Scale: 0.5, Pads: [][2]int{{1, 0}}, Name: "q4"},
		"scale": {Axes: []int{1, 2}, Scale: 0.25, Pads: [][2]int{{1, 0}}, Name: "q4"},
		"pads":  {Axes: []int{1, 2}, Scale: 0.5, Pads: [][2]int{{0, 1}}, Name: "q4"},
		"name":  {Axes: []int{1, 2}, Scale: 0.5, Pads: [][2]int{{1, 0}}, Name: "q8"},
		// One empty slice and another holding an empty value differ.
		"no axes": {Axes: []int{}, Scale: 0.5, Pads: [][2]int{{1, 0}, {0, 0}}, Name: "q4"},
	} {
		if slices.Equal(key(a), key(base)) {
			t.Errorf("attrs that differ in %s have the same key", name)
		}
	}
	args := base
	args.Value, args.Seed = 3, 9
	if !slices.Equal(key(args), key(base)) {
		t.Error("attrs that differ in their arguments only have different keys")
	}
	if _, err := encoderOf(reflect.TypeFor[struct{ P *int }](), nil); err == nil {
		t.Error("a pointer field has a key")
	}
}

// BenchmarkSchedule times the scheduler alone, on a backend whose kernels
// do nothing, for the training step of a perceptron of 8 layers: the
// walk that finds a schedule's key, planning and compiling it, and a
// whole schedule, replayed from the cache or planned anew.
func BenchmarkSchedule(b *testing.B) {
	be := &null{}
	m := newMLP(b, be, 8, 32, 32)
	m.train(b, be, 0.1)
	targets, dsts := m.step(0.1)
	b.Run("walk", func(b *testing.B) {
		for b.Loop() {
			if err := walker0.run(be, targets, dsts); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(len(walker0.order)), "nodes")
		walker0.forget()
	})
	b.Run("plan", func(b *testing.B) {
		w := &walker0
		if err := w.run(be, targets, dsts); err != nil {
			b.Fatal(err)
		}
		defer w.forget()
		isTarget := map[*graph.Node]bool{}
		for _, t := range targets {
			isTarget[t] = true
		}
		for b.Loop() {
			p := newPlan(be, w.order, isTarget, candidates(be, targets, dsts))
			if _, err := compile(p, w, targets, dsts, p.destinations(targets, dsts), isTarget); err != nil {
				b.Fatal(err)
			}
		}
	})
	for _, on := range []bool{true, false} {
		name := "planned"
		if on {
			name = "replayed"
		}
		b.Run(name, func(b *testing.B) {
			withCaching(on, func() {
				for b.Loop() {
					b.StopTimer()
					targets, dsts := m.step(0.1)
					b.StartTimer()
					if err := RealizeInto(context.Background(), be, targets, dsts, nil); err != nil {
						b.Fatal(err)
					}
					m.adopt(targets)
				}
			})
		})
	}
}
