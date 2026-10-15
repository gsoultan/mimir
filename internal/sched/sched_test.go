package sched

import (
	"context"
	"errors"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
)

// counting wraps the CPU backend and tracks allocations and kernels.
type counting struct {
	*cpu.Backend
	mu         sync.Mutex
	live, peak int
	allocs     int
	ops        []backend.Op
	failOn     backend.Op
}

func (c *counting) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	b, err := c.Backend.Alloc(dt, n)
	if err == nil {
		c.mu.Lock()
		c.live += n
		c.peak = max(c.peak, c.live)
		c.allocs++
		c.mu.Unlock()
	}
	return b, err
}

func (c *counting) Free(b backend.Buffer) {
	c.mu.Lock()
	c.live -= b.Len()
	c.mu.Unlock()
}

func (c *counting) Run(ctx context.Context, k *backend.Kernel, outs, ins []backend.Buffer) error {
	c.mu.Lock()
	c.ops = append(c.ops, k.Op)
	c.mu.Unlock()
	if k.Op == c.failOn {
		return backend.ErrIndex
	}
	return c.Backend.Run(ctx, k, outs, ins)
}

// held returns the number of elements allocated and not in the pool.
func (c *counting) held() int {
	mu.Lock()
	defer mu.Unlock()
	return c.live - poolOf(c).pooledElements()
}

func must(n *graph.Node, err error) *graph.Node {
	if err != nil {
		panic(err)
	}
	return n
}

func TestIntermediatesAreFreedAfterLastUse(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	const size = 1 << 16
	x := must(graph.Rand([]int{size, 1}, 1, false))
	y := x
	for range 50 {
		// Reductions do not fuse, so each needs a buffer.
		y = must(graph.Reduce(backend.OpReduceSum, y, []int{1}))
	}
	if err := Realize(t.Context(), be, y); err != nil {
		t.Fatal(err)
	}
	if held := be.held(); held != size {
		t.Errorf("%d elements held after realize, want only the result's %d", held, size)
	}
	if be.peak > 2*size {
		t.Errorf("peak %d elements for a chain, want at most two buffers of %d", be.peak, size)
	}
	if !y.Realized() || y.Inputs != nil {
		t.Error("target should hold its result and drop its inputs")
	}
}

func TestViewsShareBuffersAndConstantsTakeOneElement(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	c := must(graph.Const(backend.Float32, []int{1 << 30, 1 << 30}, 3))
	s := must(graph.Shrink(c, [][2]int{{5, 7}, {0, 3}}))
	p := must(graph.Permute(s, []int{1, 0}))
	r := must(graph.Reshape(p, []int{6}))
	if err := Realize(t.Context(), be, r); err != nil {
		t.Fatal(err)
	}
	if be.peak != 1 {
		t.Errorf("peak %d elements, want 1 for a broadcast constant", be.peak)
	}
	out := make([]byte, 6*4)
	if err := Download(t.Context(), be, r, out); err != nil {
		t.Fatal(err)
	}
	for i := range 6 {
		if v := math.Float32frombits(uint32(out[4*i]) | uint32(out[4*i+1])<<8 | uint32(out[4*i+2])<<16 | uint32(out[4*i+3])<<24); v != 3 {
			t.Fatalf("element %d = %v, want 3", i, v)
		}
	}
}

func TestFailedRealizeFreesEverythingAndChangesNothing(t *testing.T) {
	be := &counting{Backend: cpu.New(), failOn: backend.OpFused}
	x := must(graph.Rand([]int{100}, 1, false))
	e := must(graph.Unary(backend.OpExp2, x))
	y := must(graph.Binary(backend.OpMul, e, e))
	if err := Realize(t.Context(), be, y); !errors.Is(err, backend.ErrIndex) {
		t.Fatalf("got %v, want the kernel's error", err)
	}
	if held := be.held(); held != 0 {
		t.Errorf("%d elements still held after a failed realize", held)
	}
	if y.Realized() || x.Realized() || len(y.Inputs) != 2 {
		t.Error("failed realize modified the graph")
	}
}

// deferring reports a kernel's error at the next Sync instead of from Run,
// as a GPU backend does.
type deferring struct {
	*counting
	failed error
}

func (d *deferring) Run(ctx context.Context, k *backend.Kernel, outs, ins []backend.Buffer) error {
	if err := d.counting.Run(ctx, k, outs, ins); err != nil && d.failed == nil {
		d.failed = err
	}
	return nil
}

func (d *deferring) Sync(context.Context) error {
	err := d.failed
	d.failed = nil
	return err
}

// held returns the number of elements allocated and not in d's pool.
func (d *deferring) held() int {
	mu.Lock()
	defer mu.Unlock()
	return d.live - poolOf(d).pooledElements()
}

func TestRealizeWaitsForErrorsReportedAtSync(t *testing.T) {
	be := &deferring{counting: &counting{Backend: cpu.New()}}
	x := realized(t, be, []int{100}, 1)
	y := must(graph.Unary(backend.OpExp2, x))
	be.failOn = backend.OpExp2
	if err := Realize(t.Context(), be, y); !errors.Is(err, backend.ErrIndex) {
		t.Fatalf("got %v, want the kernel's error", err)
	}
	if y.Realized() {
		t.Error("the target of a failed kernel was realized")
	}
	if held := be.held(); held != 100 {
		t.Errorf("%d elements held after a failed realize, want x's 100", held)
	}
	be.failOn = 0
	if err := Realize(t.Context(), be, must(graph.Unary(backend.OpNeg, x))); err != nil {
		t.Errorf("the next realize reported %v", err)
	}
}

func TestElementwiseChainsFuse(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := must(graph.Rand([]int{1000}, 1, false))
	e := must(graph.Unary(backend.OpExp2, x)) // read twice, so it gets a buffer
	two := must(graph.Expand(must(graph.Const(backend.Float32, []int{1}, 2)), []int{1000}))
	sq := must(graph.Binary(backend.OpMul, e, e))
	y := must(graph.Binary(backend.OpAdd, must(graph.Binary(backend.OpMul, sq, two)), e))
	for range 50 {
		y = must(graph.Unary(backend.OpSqrt, y))
	}
	xs := make([]byte, 4000)
	if err := Download(t.Context(), be, x, xs); err != nil {
		t.Fatal(err)
	}
	be.ops = nil
	out := make([]byte, 4000)
	if err := Download(t.Context(), be, y, out); err != nil {
		t.Fatal(err)
	}
	if want := []backend.Op{backend.OpExp2, backend.OpFused}; !slices.Equal(be.ops, want) {
		t.Errorf("kernels %v, want %v", be.ops, want)
	}
	for i := range 1000 {
		xi := float32frombytes(xs[4*i:])
		ei := float32(math.Exp2(float64(xi)))
		want := ei*ei*2 + ei
		for range 50 {
			want = float32(math.Sqrt(float64(want)))
		}
		if got := float32frombytes(out[4*i:]); got != want {
			t.Fatalf("element %d = %v, want %v", i, got, want)
		}
	}
}

func TestRepeatedSchedulesReuseBuffers(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	step := func() {
		x := must(graph.Rand([]int{64, 32}, 1, false))
		w := must(graph.Rand([]int{32, 16}, 2, false))
		h := must(graph.Unary(backend.OpExp2, must(graph.MatMul(x, w))))
		y := must(graph.Reduce(backend.OpReduceSum, h, []int{0}))
		if err := Realize(t.Context(), be, y); err != nil {
			t.Fatal(err)
		}
	}
	step()
	first := be.allocs
	step()
	if got := be.allocs - first; got != 1 {
		t.Errorf("second schedule made %d allocations, want 1 for its result (the first made %d)", got, first)
	}
}

func TestPoolFreesBuffersNoScheduleTakes(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := must(graph.Rand([]int{1000, 1}, 1, false))
	y := must(graph.Reduce(backend.OpReduceSum, x, []int{1}))
	if err := Realize(t.Context(), be, y); err != nil {
		t.Fatal(err)
	}
	if be.live != 2000 {
		t.Fatalf("%d elements live, want the result and the pooled input", be.live)
	}
	for i := range poolAge + 1 {
		if err := Realize(t.Context(), be, must(graph.Rand([]int{1}, uint64(i), false))); err != nil {
			t.Fatal(err)
		}
	}
	if be.live != 1000+poolAge+1 {
		t.Errorf("%d elements live, want the pooled input freed", be.live)
	}
}

func float32frombytes(b []byte) float32 {
	return math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
}

func TestElementwiseConsumersOfAProductRunAsItsEpilogue(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := must(graph.Rand([]int{2, 64, 48}, 1, false))
	w := must(graph.Rand([]int{2, 48, 30}, 2, false))
	b := must(graph.Rand([]int{30}, 3, false))
	if err := Realize(t.Context(), be, x, w, b); err != nil {
		t.Fatal(err)
	}
	// relu(x·w + b), reading the product through a reshape.
	layer := func(p *graph.Node) *graph.Node {
		flat := must(graph.Reshape(p, []int{128, 30}))
		bias := must(graph.Expand(must(graph.Reshape(b, []int{1, 30})), []int{128, 30}))
		zero := must(graph.Const(backend.Float32, []int{128, 30}, 0))
		return must(graph.Binary(backend.OpMax, must(graph.Binary(backend.OpAdd, flat, bias)), zero))
	}
	product := must(graph.MatMul(x, w))
	if err := Realize(t.Context(), be, product); err != nil {
		t.Fatal(err)
	}
	separate := layer(product) // the product is realized, so this fuses alone
	fused := layer(must(graph.MatMul(x, w)))
	be.ops = nil
	want, got := make([]byte, 128*30*4), make([]byte, 128*30*4)
	if err := Download(t.Context(), be, separate, want); err != nil {
		t.Fatal(err)
	}
	if err := Download(t.Context(), be, fused, got); err != nil {
		t.Fatal(err)
	}
	if ops := []backend.Op{backend.OpFused, backend.OpMatMul}; !slices.Equal(be.ops, ops) {
		t.Errorf("kernels %v, want %v", be.ops, ops)
	}
	if !slices.Equal(got, want) {
		t.Error("product with epilogue differs from the separate kernels")
	}
}

func TestSumsOfProductsOverBatchAxesRunAsOneMatMul(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{4, 30, 20}, 1)
	w := realized(t, be, []int{4, 20, 25}, 2)
	b := realized(t, be, []int{25}, 3)
	product := must(graph.MatMul(x, w))
	if err := Realize(t.Context(), be, product); err != nil {
		t.Fatal(err)
	}
	sum := func(p *graph.Node, axis int) *graph.Node {
		return must(graph.Reduce(backend.OpReduceSum, p, []int{axis}))
	}
	biased := func(s *graph.Node) *graph.Node {
		flat := must(graph.Reshape(s, []int{30, 25}))
		bias := must(graph.Expand(must(graph.Reshape(b, []int{1, 25})), []int{30, 25}))
		return must(graph.Binary(backend.OpAdd, flat, bias))
	}
	// The product is realized in separate, so it is summed on its own.
	tests := []struct {
		name               string
		separate, combined *graph.Node
		ops                []backend.Op
	}{
		{"batch sum", sum(product, 0), sum(must(graph.MatMul(x, w)), 0), []backend.Op{backend.OpMatMul}},
		{"with an epilogue", biased(sum(product, 0)), biased(sum(must(graph.MatMul(x, w)), 0)), []backend.Op{backend.OpMatMul}},
		{"column axis", sum(product, 2), sum(must(graph.MatMul(x, w)), 2), []backend.Op{backend.OpMatMul, backend.OpReduceSum}},
		{"row axis", sum(product, 1), sum(must(graph.MatMul(x, w)), 1), []backend.Op{backend.OpMatMul, backend.OpReduceSum}},
	}
	for _, tt := range tests {
		want := floats(t, be, tt.separate)
		be.ops = nil
		got := floats(t, be, tt.combined)
		if !slices.Equal(be.ops, tt.ops) {
			t.Errorf("%s: kernels %v, want %v", tt.name, be.ops, tt.ops)
		}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-5*math.Abs(float64(want[i]))+1e-6 {
				t.Fatalf("%s: element %d = %v, want %v", tt.name, i, got[i], want[i])
			}
		}
	}
}

func TestProductsReadElsewhereAreComputedOnce(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{4, 30, 20}, 1)
	w := realized(t, be, []int{4, 20, 25}, 2)
	p := must(graph.MatMul(x, w))
	s := must(graph.Reduce(backend.OpReduceSum, p, []int{0}))
	be.ops = nil
	if err := Realize(t.Context(), be, p, s); err != nil {
		t.Fatal(err)
	}
	if want := []backend.Op{backend.OpMatMul, backend.OpReduceSum}; !slices.Equal(be.ops, want) {
		t.Errorf("kernels %v, want %v", be.ops, want)
	}
}

func TestAccumulatingASumOfProductsGoesInPlace(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{4, 30, 20}, 1)
	w := realized(t, be, []int{4, 20, 25}, 2)
	old := realized(t, be, []int{1, 30, 25}, 3)
	sum := must(graph.Reduce(backend.OpReduceSum, must(graph.MatMul(x, w)), []int{0}))
	want := floats(t, be, must(graph.Binary(backend.OpAdd, old, sum)))
	sum = must(graph.Reduce(backend.OpReduceSum, must(graph.MatMul(x, w)), []int{0}))
	next := must(graph.Binary(backend.OpAdd, old, sum))
	buf := old.Buf()
	if err := RealizeInto(t.Context(), be, []*graph.Node{next}, []*graph.Node{old}, nil); err != nil {
		t.Fatal(err)
	}
	// A gradient accumulated over a batch, as a convolution's is, is added
	// to its old value in that value's memory.
	if next.Buf() != buf {
		t.Error("the sum was not written over the value it adds to")
	}
	if got := floats(t, be, next); !slices.Equal(got, want) {
		t.Errorf("accumulated sum = %v, want %v", got, want)
	}
}

func TestTargetsReadByOneConsumerAreWrittenByItsKernel(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := must(graph.Rand([]int{1000}, 1, false))
	g := must(graph.Rand([]int{1000}, 2, false))
	if err := Realize(t.Context(), be, x, g); err != nil {
		t.Fatal(err)
	}
	c := func(v float64) *graph.Node { return must(graph.Const(backend.Float32, []int{1000}, v)) }
	bin := func(op backend.Op, a, b *graph.Node) *graph.Node { return must(graph.Binary(op, a, b)) }
	un := func(op backend.Op, a *graph.Node) *graph.Node { return must(graph.Unary(op, a)) }
	// An optimizer-like step: new state m and v, and from them a new x.
	m := bin(backend.OpAdd, bin(backend.OpMul, x, c(0.9)), g)
	v := bin(backend.OpMul, g, g)
	step := bin(backend.OpMul, m, un(backend.OpRecip, bin(backend.OpAdd, un(backend.OpSqrt, v), c(1))))
	next := bin(backend.OpAdd, x, un(backend.OpNeg, step))
	be.ops = nil
	if err := Realize(t.Context(), be, next, m, v); err != nil {
		t.Fatal(err)
	}
	if want := []backend.Op{backend.OpFused}; !slices.Equal(be.ops, want) {
		t.Errorf("kernels %v, want %v", be.ops, want)
	}
	get := func(n *graph.Node) []float32 {
		b := make([]byte, 4000)
		if err := Download(t.Context(), be, n, b); err != nil {
			t.Fatal(err)
		}
		f := make([]float32, 1000)
		for i := range f {
			f[i] = float32frombytes(b[4*i:])
		}
		return f
	}
	xs, gs, ms, vs, nexts := get(x), get(g), get(m), get(v), get(next)
	for i := range 1000 {
		// The conversions round each product, as the kernel does, where
		// the compiler could otherwise fuse it with the addition.
		wm := float32(xs[i]*0.9) + gs[i]
		wv := gs[i] * gs[i]
		wn := xs[i] + -float32(wm*(1/(float32(math.Sqrt(float64(wv)))+1)))
		if ms[i] != wm || vs[i] != wv || nexts[i] != wn {
			t.Fatalf("element %d = %v, %v, %v, want %v, %v, %v", i, nexts[i], ms[i], vs[i], wn, wm, wv)
		}
	}
}

func TestTransposedProductsAreComputedTransposed(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	a := must(graph.Rand([]int{2, 5, 3}, 1, false))
	b := must(graph.Rand([]int{2, 3, 4}, 2, false))
	if err := Realize(t.Context(), be, a, b); err != nil {
		t.Fatal(err)
	}
	product := must(graph.MatMul(a, b))
	if err := Realize(t.Context(), be, product); err != nil {
		t.Fatal(err)
	}
	want := make([]byte, 2*4*5*4)
	if err := Download(t.Context(), be, must(graph.Permute(product, []int{0, 2, 1})), want); err != nil {
		t.Fatal(err)
	}
	transposed := must(graph.Permute(must(graph.MatMul(a, b)), []int{0, 2, 1}))
	be.ops = nil
	if err := Realize(t.Context(), be, transposed); err != nil {
		t.Fatal(err)
	}
	if want := []backend.Op{backend.OpMatMul}; !slices.Equal(be.ops, want) {
		t.Errorf("kernels %v, want %v", be.ops, want)
	}
	if v := transposed.View; !v.IsContiguous() || v.Offset != 0 {
		t.Errorf("result view %+v, want contiguous", v)
	}
	got := make([]byte, len(want))
	if err := Download(t.Context(), be, transposed, got); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Error("transposed product differs from the product transposed")
	}
}

// floats downloads n's elements.
func floats(t *testing.T, be backend.Backend, n *graph.Node) []float32 {
	t.Helper()
	b := make([]byte, 4*n.Size())
	if err := Download(t.Context(), be, n, b); err != nil {
		t.Fatal(err)
	}
	f := make([]float32, n.Size())
	for i := range f {
		f[i] = float32frombytes(b[4*i:])
	}
	return f
}

// realized returns n random values, computed.
func realized(t *testing.T, be backend.Backend, shape []int, seed uint64) *graph.Node {
	t.Helper()
	n := must(graph.Rand(shape, seed, false))
	if err := Realize(t.Context(), be, n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestRealizeIntoWritesOverTheDestinations(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	const n = 3000
	x, m, g := realized(t, be, []int{n}, 1), realized(t, be, []int{n}, 2), realized(t, be, []int{n}, 3)
	xs, ms, gs := floats(t, be, x), floats(t, be, m), floats(t, be, g)
	bin := func(op backend.Op, a, b *graph.Node) *graph.Node { return must(graph.Binary(op, a, b)) }
	// The new x reads the old m after the new m is written over it.
	nextM := bin(backend.OpAdd, m, g)
	nextX := bin(backend.OpAdd, nextM, bin(backend.OpMul, x, m))
	xbuf, mbuf := x.Buf(), m.Buf()
	allocs := be.allocs
	be.ops = nil
	if err := RealizeInto(t.Context(), be, []*graph.Node{nextX, nextM}, []*graph.Node{x, m}, nil); err != nil {
		t.Fatal(err)
	}
	if want := []backend.Op{backend.OpFused}; !slices.Equal(be.ops, want) {
		t.Errorf("kernels %v, want %v", be.ops, want)
	}
	if be.allocs != allocs {
		t.Errorf("%d buffers allocated, want none", be.allocs-allocs)
	}
	if nextX.Buf() != xbuf || nextM.Buf() != mbuf {
		t.Error("results not written over their destinations")
	}
	gotX, gotM := floats(t, be, nextX), floats(t, be, nextM)
	for i := range n {
		wm := ms[i] + gs[i]
		if wx := wm + float32(xs[i]*ms[i]); gotX[i] != wx || gotM[i] != wm {
			t.Fatalf("element %d = %v, %v, want %v, %v", i, gotX[i], gotM[i], wx, wm)
		}
	}
	for _, old := range []*graph.Node{x, m} {
		if err := Download(t.Context(), be, old, make([]byte, 4*n)); !errors.Is(err, graph.ErrOverwritten) {
			t.Errorf("reading an overwritten value: got %v, want ErrOverwritten", err)
		}
		if err := Realize(t.Context(), be, must(graph.Unary(backend.OpNeg, old))); !errors.Is(err, graph.ErrOverwritten) {
			t.Errorf("computing from an overwritten value: got %v, want ErrOverwritten", err)
		}
	}
}

func TestRealizeIntoKeepsValuesReadAfterTheWrite(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{8, 8}, 1)
	xs := floats(t, be, x)
	y := must(graph.Unary(backend.OpNeg, x))
	// The product reads x after y is computed, so y cannot go over x.
	p := must(graph.MatMul(y, x))
	xbuf := x.Buf()
	if err := RealizeInto(t.Context(), be, []*graph.Node{y, p}, []*graph.Node{x, nil}, nil); err != nil {
		t.Fatal(err)
	}
	if y.Buf() == xbuf {
		t.Error("y written over x, which the product reads after")
	}
	ps := floats(t, be, p)
	for i := range 8 {
		for j := range 8 {
			var want float32
			for k := range 8 {
				want += -xs[i*8+k] * xs[k*8+j]
			}
			if d := ps[i*8+j] - want; d > 1e-5 || d < -1e-5 {
				t.Fatalf("product element %d,%d = %v, want %v", i, j, ps[i*8+j], want)
			}
		}
	}
	if !x.Stale() {
		t.Error("destination still readable after its target was computed elsewhere")
	}
}

func TestRealizeIntoLeavesSharedMemoryAlone(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{4, 6}, 1)
	// A transposed view is not the destination's own layout, and a target
	// that is a view of the destination keeps reading its memory.
	view := must(graph.Permute(x, []int{1, 0}))
	if err := Realize(t.Context(), be, view); err != nil {
		t.Fatal(err)
	}
	y := must(graph.Unary(backend.OpNeg, x))
	if err := RealizeInto(t.Context(), be, []*graph.Node{must(graph.Unary(backend.OpNeg, view))}, []*graph.Node{view}, nil); err != nil {
		t.Fatal(err)
	}
	if !x.Stale() {
		t.Error("memory of a consumed view still readable")
	}
	if err := Realize(t.Context(), be, y); !errors.Is(err, graph.ErrOverwritten) {
		t.Errorf("got %v, want ErrOverwritten", err)
	}

	z := realized(t, be, []int{5}, 2)
	same := must(graph.Reshape(z, []int{5}))
	if err := RealizeInto(t.Context(), be, []*graph.Node{same}, []*graph.Node{z}, nil); err != nil {
		t.Fatal(err)
	}
	if z.Stale() || same.Stale() {
		t.Error("a target sharing its destination's memory made it stale")
	}
}

func TestRealizeIntoStopsOnlyBeforeWriting(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{10}, 1)
	y := must(graph.Unary(backend.OpNeg, x))
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := RealizeInto(ctx, be, []*graph.Node{y}, []*graph.Node{x}, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if x.Stale() || y.Realized() {
		t.Error("a cancelled schedule changed nodes")
	}
}

func TestRealizeIntoLeavesStaleDestinationsAlone(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{4}, 1)
	want := floats(t, be, x)
	next := must(graph.Unary(backend.OpNeg, x))
	if err := RealizeInto(t.Context(), be, []*graph.Node{next}, []*graph.Node{x}, nil); err != nil {
		t.Fatal(err)
	}
	if next.Mem != x.Mem {
		t.Fatal("the update was not written over its destination")
	}
	// x is stale and its memory holds next: passing x again changes
	// nothing.
	other := realized(t, be, []int{4}, 2)
	if err := RealizeInto(t.Context(), be, []*graph.Node{must(graph.Unary(backend.OpNeg, other))}, []*graph.Node{x}, nil); err != nil {
		t.Fatal(err)
	}
	if next.Stale() {
		t.Fatal("the value in a stale destination's memory was made stale")
	}
	for i, v := range floats(t, be, next) {
		if v != -want[i] {
			t.Fatalf("value %d = %v, want %v", i, v, -want[i])
		}
	}
}

func TestRealizeIntoKeepsWhatOtherTargetsRead(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	neg := func(n *graph.Node) *graph.Node { return must(graph.Unary(backend.OpNeg, n)) }
	equal := func(what string, got, want []float32) {
		t.Helper()
		if !slices.Equal(got, want) {
			t.Errorf("%s = %v, want %v", what, got, want)
		}
	}
	negated := func(xs []float32) []float32 {
		out := make([]float32, len(xs))
		for i, v := range xs {
			out[i] = -v
		}
		return out
	}

	// A target that is a view of the destination keeps its values.
	x := realized(t, be, []int{3}, 1)
	xs, xbuf := floats(t, be, x), x.Buf()
	view, y := must(graph.Reshape(x, []int{3, 1})), neg(x)
	if err := RealizeInto(t.Context(), be, []*graph.Node{view, y}, []*graph.Node{nil, x}, nil); err != nil {
		t.Fatal(err)
	}
	if y.Buf() == xbuf {
		t.Error("update written over the memory of a view target")
	}
	equal("view of the destination", floats(t, be, view), xs)
	equal("update", floats(t, be, y), negated(xs))

	// Of two targets with one destination, at most one goes over it.
	a, b := realized(t, be, []int{3}, 2), realized(t, be, []int{3}, 3)
	as, bs, abuf := floats(t, be, a), floats(t, be, b), a.Buf()
	na, nb := neg(a), neg(b)
	if err := RealizeInto(t.Context(), be, []*graph.Node{na, nb}, []*graph.Node{a, a}, nil); err != nil {
		t.Fatal(err)
	}
	if na.Buf() == abuf && nb.Buf() == abuf {
		t.Error("two targets written over one destination")
	}
	equal("first target", floats(t, be, na), negated(as))
	equal("second target", floats(t, be, nb), negated(bs))

	// An update that reads its destination in another order cannot go
	// over it element by element.
	views := map[string]func(*graph.Node) *graph.Node{
		"transposed": func(n *graph.Node) *graph.Node { return must(graph.Permute(n, []int{1, 0})) },
		"flipped":    func(n *graph.Node) *graph.Node { return must(graph.Flip(n, []int{0})) },
	}
	for name, view := range views {
		x := realized(t, be, []int{4, 4}, 4)
		want := floats(t, be, neg(view(x)))
		xbuf := x.Buf()
		y := neg(view(x))
		if err := RealizeInto(t.Context(), be, []*graph.Node{y}, []*graph.Node{x}, nil); err != nil {
			t.Fatal(err)
		}
		if y.Buf() == xbuf {
			t.Errorf("update reading its destination %s written over it", name)
		}
		equal(name+" update", floats(t, be, y), want)
	}

	// A destination that views part of a larger buffer is not written.
	big := realized(t, be, []int{8}, 5)
	part := must(graph.Shrink(big, [][2]int{{2, 6}}))
	if err := Realize(t.Context(), be, part); err != nil {
		t.Fatal(err)
	}
	bigs, bigbuf := floats(t, be, big), big.Buf()
	z := neg(realized(t, be, []int{4}, 6))
	if err := RealizeInto(t.Context(), be, []*graph.Node{z}, []*graph.Node{part}, nil); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 4*8)
	if err := be.Download(t.Context(), raw, bigbuf); err != nil {
		t.Fatal(err)
	}
	for i, v := range bigs {
		if got := float32frombytes(raw[4*i:]); got != v {
			t.Fatalf("element %d of the larger buffer = %v, want %v", i, got, v)
		}
	}
}

// cancelling cancels a context, once it has one, when it runs a kernel.
type cancelling struct {
	*counting
	cancel context.CancelFunc
}

func (c *cancelling) Run(ctx context.Context, k *backend.Kernel, outs, ins []backend.Buffer) error {
	if c.cancel != nil {
		c.cancel()
	}
	return c.counting.Run(ctx, k, outs, ins)
}

func TestRealizeIntoFinishesOnceItStartsWriting(t *testing.T) {
	be := &cancelling{counting: &counting{Backend: cpu.New()}}
	x, y := realized(t, be, []int{4, 4}, 1), realized(t, be, []int{4, 4}, 2)
	xs, ys := floats(t, be, x), floats(t, be, y)
	nx := must(graph.Unary(backend.OpNeg, x))
	p := must(graph.MatMul(nx, y))
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	be.ops, be.cancel = nil, cancel
	if err := RealizeInto(ctx, be, []*graph.Node{nx, p}, []*graph.Node{x, nil}, nil); err != nil {
		t.Fatalf("cancelled after the first kernel wrote over its destination: %v", err)
	}
	if len(be.ops) != 2 {
		t.Errorf("kernels %v, want the update and the product", be.ops)
	}
	got := floats(t, be, p)
	for i := range 4 {
		for j := range 4 {
			var want float32
			for k := range 4 {
				want += -xs[i*4+k] * ys[k*4+j]
			}
			if d := got[i*4+j] - want; d > 1e-5 || d < -1e-5 {
				t.Fatalf("product element %d,%d = %v, want %v", i, j, got[i*4+j], want)
			}
		}
	}
}

func TestReleaseLeavesSharedMemoryAlone(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{2, 3}, 1)
	view := must(graph.Reshape(x, []int{6}))
	if err := Realize(t.Context(), be, view); err != nil {
		t.Fatal(err)
	}
	want := floats(t, be, x)
	Release(x, nil)
	if x.Stale() {
		t.Fatal("released memory that a view still reads")
	}
	if got := floats(t, be, view); !slices.Equal(got, want) {
		t.Errorf("view after Release = %v, want %v", got, want)
	}
}

func TestRealizeIntoComputesKeptNodesFirst(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	neg := func(n *graph.Node) *graph.Node { return must(graph.Unary(backend.OpNeg, n)) }
	twice := func(n *graph.Node) *graph.Node { return must(graph.Binary(backend.OpAdd, n, n)) }
	x, y := realized(t, be, []int{6}, 1), realized(t, be, []int{6}, 2)
	xs, ys := floats(t, be, x), floats(t, be, y)
	buf := x.Buf()
	next := twice(x)
	// Of these, only reader loses its input: the update stays readable.
	reader, other, update := neg(x), neg(y), neg(next)
	keep := func() []*graph.Node { return []*graph.Node{reader, other, reader, update} }
	if err := RealizeInto(t.Context(), be, []*graph.Node{next}, []*graph.Node{x}, keep); err != nil {
		t.Fatal(err)
	}
	if next.Buf() != buf {
		t.Error("the update was not written over its destination")
	}
	if !reader.Realized() || other.Realized() || update.Realized() {
		t.Errorf("kept nodes realized: %v reading the destination, %v reading another value, %v reading the update; want true, false, false",
			reader.Realized(), other.Realized(), update.Realized())
	}
	for i, v := range floats(t, be, update) {
		if v != -2*xs[i] {
			t.Fatalf("kept node reading the update: element %d = %v, want %v", i, v, -2*xs[i])
		}
	}
	for i, v := range floats(t, be, reader) {
		if v != -xs[i] {
			t.Fatalf("kept node element %d = %v, want %v", i, v, -xs[i])
		}
	}
	for i, v := range floats(t, be, other) {
		if v != -ys[i] {
			t.Fatalf("other kept node element %d = %v, want %v", i, v, -ys[i])
		}
	}

	// A kept node that fails to compute is left to go stale, without
	// failing the schedule or the other kept nodes.
	z := realized(t, be, []int{6}, 3)
	zs := floats(t, be, z)
	failing, reader := must(graph.Unary(backend.OpExp2, z)), neg(z)
	keep = func() []*graph.Node { return []*graph.Node{failing, reader} }
	be.failOn = backend.OpExp2
	err := RealizeInto(t.Context(), be, []*graph.Node{twice(z)}, []*graph.Node{z}, keep)
	be.failOn = 0
	if err != nil {
		t.Fatalf("a failing kept node failed the schedule: %v", err)
	}
	if err := Realize(t.Context(), be, failing); !errors.Is(err, graph.ErrOverwritten) {
		t.Errorf("failed kept node: got %v, want ErrOverwritten", err)
	}
	for i, v := range floats(t, be, reader) {
		if v != -zs[i] {
			t.Fatalf("kept node beside a failing one: element %d = %v, want %v", i, v, -zs[i])
		}
	}

	// A kept view of the destination keeps its memory, as a target does.
	v := realized(t, be, []int{2, 3}, 5)
	vs := floats(t, be, v)
	view := must(graph.Permute(v, []int{1, 0}))
	keep = func() []*graph.Node { return []*graph.Node{view} }
	if err := RealizeInto(t.Context(), be, []*graph.Node{twice(v)}, []*graph.Node{v}, keep); err != nil {
		t.Fatal(err)
	}
	for i, got := range floats(t, be, view) {
		if want := vs[i%2*3+i/2]; got != want {
			t.Fatalf("kept view of the destination: element %d = %v, want %v", i, got, want)
		}
	}

	// So is one that can no longer be computed.
	broken := must(graph.Binary(backend.OpAdd, failing, next))
	keep = func() []*graph.Node { return []*graph.Node{broken} }
	if err := RealizeInto(t.Context(), be, []*graph.Node{twice(next)}, []*graph.Node{next}, keep); err != nil {
		t.Errorf("a stale kept node failed the schedule: %v", err)
	}
}

func TestReleaseKeepsMemoryThatKeptNodesRead(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x := realized(t, be, []int{4}, 1)
	want := floats(t, be, x)
	reader := must(graph.Unary(backend.OpNeg, must(graph.Reshape(x, []int{2, 2}))))
	Release(x, func() []*graph.Node { return []*graph.Node{reader} })
	if x.Stale() {
		t.Fatal("released memory that a kept node reads")
	}
	for i, v := range floats(t, be, reader) {
		if v != -want[i] {
			t.Fatalf("kept node element %d = %v, want %v", i, v, -want[i])
		}
	}
	Release(x, func() []*graph.Node { return []*graph.Node{reader} })
	if !x.Stale() {
		t.Error("kept memory once nothing kept reads it")
	}

	// Nor does a kept node that can no longer be computed.
	a, b := realized(t, be, []int{4}, 2), realized(t, be, []int{4}, 3)
	broken := must(graph.Binary(backend.OpAdd, a, b))
	if err := RealizeInto(t.Context(), be, []*graph.Node{must(graph.Unary(backend.OpNeg, b))}, []*graph.Node{b}, nil); err != nil {
		t.Fatal(err)
	}
	Release(a, func() []*graph.Node { return []*graph.Node{broken} })
	if !a.Stale() {
		t.Error("kept memory that only a node reading a stale value reads")
	}
}
