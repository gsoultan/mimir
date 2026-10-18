package sched

import (
	"errors"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
)

func TestRealizeEachMatchesRealizeOnEachDevice(t *testing.T) {
	bes := []backend.Backend{cpu.NewStream(1), cpu.NewStream(1), cpu.NewStream(1)}
	const n = 8
	// Each device computes x·w + x into x's memory, the way an optimizer's
	// update writes over a parameter, and a sum that it reads afterwards.
	build := func(be backend.Backend, seed uint64) (targets, dsts []*graph.Node) {
		x := realized(t, be, []int{n, n}, seed)
		w := realized(t, be, []int{n, n}, seed+10)
		y := must(graph.Binary(backend.OpAdd, must(graph.MatMul(x, w)), x))
		s := must(graph.Reduce(backend.OpReduceSum, y, []int{0, 1}))
		return []*graph.Node{y, s}, []*graph.Node{x, nil}
	}
	want := make([][]float32, len(bes))
	for i, be := range bes {
		ts, ds := build(be, uint64(i))
		if err := RealizeInto(t.Context(), be, ts, ds, nil); err != nil {
			t.Fatal(err)
		}
		want[i] = append(floats(t, be, ts[0]), floats(t, be, ts[1])...)
	}
	targets, dsts := make([][]*graph.Node, len(bes)), make([][]*graph.Node, len(bes))
	for i, be := range bes {
		targets[i], dsts[i] = build(be, uint64(i))
	}
	if err := RealizeEach(t.Context(), bes, targets, dsts, nil); err != nil {
		t.Fatal(err)
	}
	for i, be := range bes {
		got := append(floats(t, be, targets[i][0]), floats(t, be, targets[i][1])...)
		for j := range got {
			if got[j] != want[i][j] {
				t.Fatalf("device %d: element %d is %v, want %v", i, j, got[j], want[i][j])
			}
		}
		if !dsts[i][0].Stale() {
			t.Errorf("device %d: the destination is not stale", i)
		}
	}
}

func TestRealizeEachRefusesOneDeviceTwice(t *testing.T) {
	be := cpu.New()
	x := realized(t, be, []int{4}, 1)
	y := must(graph.Unary(backend.OpNeg, x))
	err := RealizeEach(t.Context(), []backend.Backend{be, be}, [][]*graph.Node{{y}, nil}, nil, nil)
	if !errors.Is(err, backend.ErrInternal) {
		t.Fatalf("got %v, want backend.ErrInternal", err)
	}
	if y.Realized() {
		t.Error("a target was computed")
	}
}
