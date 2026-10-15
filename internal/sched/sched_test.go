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

func (c *counting) Run(ctx context.Context, k *backend.Kernel, out backend.Buffer, ins []backend.Buffer) error {
	c.mu.Lock()
	c.ops = append(c.ops, k.Op)
	c.mu.Unlock()
	if k.Op == c.failOn {
		return backend.ErrIndex
	}
	return c.Backend.Run(ctx, k, out, ins)
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
		wm := xs[i]*0.9 + gs[i]
		wv := gs[i] * gs[i]
		wn := xs[i] + -(wm * (1 / (float32(math.Sqrt(float64(wv))) + 1)))
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
