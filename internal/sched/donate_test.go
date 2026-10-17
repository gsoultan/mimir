package sched

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
)

// uploaded returns a donated node of shape holding random values, as an
// upload of a batch is, and its values.
func uploaded(t *testing.T, be backend.Backend, shape []int, seed uint64) (*graph.Node, []float32) {
	t.Helper()
	vals := floats(t, be, realized(t, be, shape, seed))
	n, err := Upload(t.Context(), be, backend.Float32, bytesOf(vals), shape)
	if err != nil {
		t.Fatal(err)
	}
	Donate(n)
	return n, vals
}

func bytesOf(f []float32) []byte {
	b := make([]byte, 0, 4*len(f))
	for _, v := range f {
		b = append(b, float32bytes(v)...)
	}
	return b
}

func float32bytes(v float32) []byte {
	u := math.Float32bits(v)
	return []byte{byte(u), byte(u >> 8), byte(u >> 16), byte(u >> 24)}
}

// near reports whether got and want are equal to float32 rounding, as
// products summed in another order are.
func near(got, want []float32) bool {
	return slices.EqualFunc(got, want, func(g, w float32) bool {
		return math.Abs(float64(g-w)) <= 1e-5*max(1, math.Abs(float64(w)))
	})
}

// matmul returns the product of n×n matrices a and b, by rows.
func matmul(a, b []float32, n int) []float32 {
	out := make([]float32, n*n)
	for i := range n {
		for k := range n {
			for j := range n {
				out[i*n+j] += a[i*n+k] * b[k*n+j]
			}
		}
	}
	return out
}

func TestDonatedLeavesServeLaterKernels(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	const n = 16
	w := realized(t, be, []int{n, n}, 2)
	ws := floats(t, be, w)
	for _, donate := range []bool{false, true} {
		x, xs := uploaded(t, be, []int{n, n}, 1)
		if !donate {
			x.Donated = false
		}
		xbuf := x.Buf()
		// Products do not fuse, so a and b each need a buffer, and the
		// second can take x's, which only the first reads.
		a := must(graph.MatMul(x, w))
		b := must(graph.MatMul(a, w))
		if err := Realize(t.Context(), be, b); err != nil {
			t.Fatal(err)
		}
		if got, want := floats(t, be, b), matmul(matmul(xs, ws, n), ws, n); !near(got, want) {
			t.Fatalf("donate %v: result %v, want %v", donate, got[:4], want[:4])
		}
		taken := b.Buf() == xbuf
		if taken != donate {
			t.Errorf("donate %v: the result took the input's buffer: %v", donate, taken)
		}
		err := Download(t.Context(), be, x, make([]byte, 4*n*n), nil)
		switch {
		case donate && !(errors.Is(err, graph.ErrDonated) && errors.Is(err, graph.ErrOverwritten)):
			t.Errorf("reading a donated input: %v, want ErrDonated and ErrOverwritten", err)
		case !donate && err != nil:
			t.Errorf("reading an input not donated: %v", err)
		}
	}
}

func TestDonatedLeavesOutliveTheirViews(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	const n = 16
	x, xs := uploaded(t, be, []int{n, n}, 1)
	w := realized(t, be, []int{n, n}, 2)
	ws := floats(t, be, w)
	// The transpose views x, and the second product reads it after the
	// first, whose output must not take x's buffer.
	xt := must(graph.Permute(x, []int{1, 0}))
	a := must(graph.MatMul(x, w))
	b := must(graph.MatMul(a, xt))
	if err := Realize(t.Context(), be, b); err != nil {
		t.Fatal(err)
	}
	xts := make([]float32, n*n)
	for i := range n {
		for j := range n {
			xts[i*n+j] = xs[j*n+i]
		}
	}
	if got, want := floats(t, be, b), matmul(matmul(xs, ws, n), xts, n); !near(got, want) {
		t.Fatalf("result %v, want %v", got[:4], want[:4])
	}
}

func TestDonationsKeepWhatTheScheduleReturnsOrOthersRead(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	const n = 16
	cases := map[string]func(x *graph.Node) (targets, dsts []*graph.Node, keep func() []*graph.Node){
		"view target": func(x *graph.Node) ([]*graph.Node, []*graph.Node, func() []*graph.Node) {
			return []*graph.Node{must(graph.Unary(backend.OpExp2, x)), must(graph.Reshape(x, []int{n * n}))}, nil, nil
		},
		"destination": func(x *graph.Node) ([]*graph.Node, []*graph.Node, func() []*graph.Node) {
			return []*graph.Node{must(graph.Unary(backend.OpNeg, x))}, []*graph.Node{x}, nil
		},
		"kept": func(x *graph.Node) ([]*graph.Node, []*graph.Node, func() []*graph.Node) {
			kept := must(graph.Unary(backend.OpExp2, x))
			return []*graph.Node{must(graph.Unary(backend.OpNeg, x))}, nil, func() []*graph.Node { return []*graph.Node{kept} }
		},
		"shared": func(x *graph.Node) ([]*graph.Node, []*graph.Node, func() []*graph.Node) {
			v := must(graph.Reshape(x, []int{n * n}))
			if err := Realize(t.Context(), be, v); err != nil {
				t.Fatal(err)
			}
			return []*graph.Node{must(graph.Unary(backend.OpNeg, x))}, nil, nil
		},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			x, _ := uploaded(t, be, []int{n, n}, 1)
			targets, dsts, keep := c(x)
			if err := RealizeInto(t.Context(), be, targets, dsts, keep); err != nil {
				t.Fatal(err)
			}
			if x.Mem.Donated {
				t.Error("the schedule took the memory of the donated input")
			}
		})
	}
}

func TestMappedMemoryIsNeverTaken(t *testing.T) {
	be := cpu.New()
	raw := make([]byte, 4*64)
	buf, err := cpu.Wrap(backend.Float32, raw)
	if err != nil {
		t.Fatal(err)
	}
	x := must(graph.Load(be, buf, []int{64}))
	Donate(x)
	if err := Realize(t.Context(), be, must(graph.Unary(backend.OpNeg, x))); err != nil {
		t.Fatal(err)
	}
	if x.Stale() {
		t.Error("a schedule took memory the caller lent")
	}
}

func TestRefusedSchedulesLeaveDonationsAlone(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	x, _ := uploaded(t, be, []int{256, 256}, 1)
	defer SetBudget(be, 0)
	SetBudget(be, 1)
	y := must(graph.MatMul(x, x))
	if err := Realize(t.Context(), be, y); !errors.Is(err, backend.ErrMemoryBudget) {
		t.Fatalf("got %v, want ErrMemoryBudget", err)
	}
	if x.Stale() {
		t.Error("a schedule refused before it ran took a donated input")
	}
}

func TestReplaysTakeDonatedLeavesAndPlannedOnesDoNot(t *testing.T) {
	be := &counting{Backend: cpu.New()}
	const n = 16
	w := realized(t, be, []int{n, n}, 2)
	c := &poolOf(be).programs
	replays := 0
	run := func(donate bool) *graph.Node {
		x, _ := uploaded(t, be, []int{n, n}, 1)
		x.Donated = donate
		b := must(graph.MatMul(must(graph.MatMul(x, w)), w))
		hits := c.hits
		if err := Realize(t.Context(), be, b); err != nil {
			t.Fatal(err)
		}
		replays += c.hits - hits
		return x
	}
	run(false)
	run(true)
	replays = 0
	if x := run(true); !x.Stale() {
		t.Error("a replay did not take the donated input")
	}
	if x := run(false); x.Stale() {
		t.Error("a replay took an input not donated")
	}
	if replays != 2 {
		t.Errorf("%d replays, want 2", replays)
	}
}
