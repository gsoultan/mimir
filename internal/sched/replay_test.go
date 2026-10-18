package sched

import (
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
)

// TestReplayEachMatchesRealizeEach traces a step that scales x·w + x by
// a constant and writes it over x, and replays it over new leaves with a
// new constant, which computes what realizing the step's graph does; and
// refuses leaves that would not find the same program.
func TestReplayEachMatchesRealizeEach(t *testing.T) {
	bes := []backend.Backend{cpu.NewStream(1), cpu.NewStream(1)}
	const n = 4
	type step struct {
		targets, dsts, leaves []*graph.Node
		scale                 *graph.Node
	}
	build := func(be backend.Backend, seed uint64, shape []int, c float64) step {
		x := realized(t, be, shape, seed)
		w := realized(t, be, []int{shape[1], shape[1]}, seed+10)
		k := must(graph.Const(backend.Float32, shape, c))
		y := must(graph.Binary(backend.OpMul, must(graph.Binary(backend.OpAdd, must(graph.MatMul(x, w)), x)), k))
		return step{targets: []*graph.Node{y}, dsts: []*graph.Node{x}, leaves: []*graph.Node{x, w}, scale: k}
	}
	steps := func(seed uint64, c float64) []step {
		return []step{build(bes[0], seed, []int{n, n}, c), build(bes[1], seed+1, []int{n, n}, c)}
	}
	each := func(ss []step, f func(s step) []*graph.Node) [][]*graph.Node {
		out := make([][]*graph.Node, len(ss))
		for i, s := range ss {
			out[i] = f(s)
		}
		return out
	}
	// The first step is traced.
	first := steps(1, 2)
	traces := make([]*Trace, len(bes))
	err := RealizeEachTraced(t.Context(), bes, each(first, func(s step) []*graph.Node { return s.targets }), each(first, func(s step) []*graph.Node { return s.dsts }), nil,
		func(r int, tr *Trace, order, leaves []*graph.Node) {
			traces[r] = tr
			if !slices.Equal(leaves[:tr.Leaves()], first[r].leaves) {
				t.Errorf("device %d: traced leaves are not x and w", r)
			}
		})
	if err != nil {
		t.Fatal(err)
	}
	for r, tr := range traces {
		if tr == nil {
			t.Fatalf("device %d was not traced", r)
		}
	}
	// The second, computed by the graph and by replaying the trace.
	want := steps(5, 3)
	if err := RealizeEach(t.Context(), bes, each(want, func(s step) []*graph.Node { return s.targets }), each(want, func(s step) []*graph.Node { return s.dsts }), nil); err != nil {
		t.Fatal(err)
	}
	replay := func(ss []step) ([]bool, [][]*graph.Node) {
		args := make([][]*graph.Node, len(bes))
		for r, tr := range traces {
			args[r] = make([]*graph.Node, 64)
			for _, a := range tr.Args() {
				args[r][a] = ss[r].scale
			}
		}
		targets := make([][]*graph.Node, len(bes))
		missed, err := ReplayEach(t.Context(), traces, each(ss, func(s step) []*graph.Node { return s.leaves }), args, func(r int) []*graph.Node {
			y := ss[r].targets[0]
			targets[r] = []*graph.Node{{Op: y.Op, DType: y.DType, Shape: y.Shape, Dev: y.Dev}}
			return targets[r]
		}, each(ss, func(s step) []*graph.Node { return s.dsts }), nil)
		if err != nil {
			t.Fatal(err)
		}
		return missed, targets
	}
	got := steps(5, 3)
	missed, targets := replay(got)
	for r, be := range bes {
		if missed[r] {
			t.Fatalf("device %d: the trace did not fit the same graph over new leaves", r)
		}
		if g, w := floats(t, be, targets[r][0]), floats(t, be, want[r].targets[0]); !slices.Equal(g, w) {
			t.Fatalf("device %d: replayed %v, want %v", r, g, w)
		}
		if !got[r].dsts[0].Stale() {
			t.Errorf("device %d: the destination is not stale", r)
		}
	}
	// Leaves of another shape, or a destination already written over, do
	// not fit.
	other := []step{build(bes[0], 9, []int{2 * n, n}, 3), build(bes[1], 9, []int{n, n}, 3)}
	if err := RealizeEach(t.Context(), bes[1:], [][]*graph.Node{{must(graph.Unary(backend.OpNeg, other[1].dsts[0]))}}, [][]*graph.Node{{other[1].dsts[0]}}, nil); err != nil {
		t.Fatal(err)
	}
	if missed, targets := replay(other); !missed[0] || !missed[1] || targets[0] != nil || targets[1] != nil {
		t.Errorf("replays of leaves that do not fit: missed %v, targets %v", missed, targets)
	}
}
