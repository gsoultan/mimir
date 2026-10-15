package sched

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
)

// counting wraps the CPU backend and tracks live allocations.
type counting struct {
	*cpu.Backend
	mu         sync.Mutex
	live, peak int
	failOn     backend.Op
}

func (c *counting) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	b, err := c.Backend.Alloc(dt, n)
	if err == nil {
		c.mu.Lock()
		c.live += n
		c.peak = max(c.peak, c.live)
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
	if k.Op == c.failOn {
		return backend.ErrIndex
	}
	return c.Backend.Run(ctx, k, out, ins)
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
	x := must(graph.Rand([]int{size}, 1, false))
	y := x
	for range 50 {
		y = must(graph.Unary(backend.OpSqrt, y))
	}
	if err := Realize(t.Context(), be, y); err != nil {
		t.Fatal(err)
	}
	if be.live != size {
		t.Errorf("%d elements live after realize, want only the result's %d", be.live, size)
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
	be := &counting{Backend: cpu.New(), failOn: backend.OpMul}
	x := must(graph.Rand([]int{100}, 1, false))
	e := must(graph.Unary(backend.OpExp2, x))
	y := must(graph.Binary(backend.OpMul, e, e))
	if err := Realize(t.Context(), be, y); !errors.Is(err, backend.ErrIndex) {
		t.Fatalf("got %v, want the kernel's error", err)
	}
	if be.live != 0 {
		t.Errorf("%d elements still allocated after a failed realize", be.live)
	}
	if y.Realized() || x.Realized() || len(y.Inputs) != 2 {
		t.Error("failed realize modified the graph")
	}
}
