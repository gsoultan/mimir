package tensor_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/sched"
	"github.com/gsoultan/mimir/tensor"
)

// A recording replays its program over new leaves, computing what
// building the graph again would, without compiling anything; leaves of
// another shape do not fit it.
func TestRecordReplaysOverNewLeaves(t *testing.T) {
	if tensor.Eager() {
		t.Skip("eager mode computes every operation as it is created, leaving nothing to record")
	}
	ctx := context.Background()
	const n = 4
	step := func(x, w *tensor.Tensor, c float32) (y, z *tensor.Tensor) {
		y = x.MatMul(w).Add(x).Mul(tensor.Full(tensor.Float32, float64(c), n, n))
		return y, y.Add(x)
	}
	leaves := func(seed uint64) (x, w *tensor.Tensor) {
		x, w = tensor.Randn(seed, n, n), tensor.Randn(seed+1, n, n)
		if err := tensor.Realize(ctx, x, w); err != nil {
			t.Fatal(err)
		}
		return x, w
	}
	values := func(ts ...*tensor.Tensor) [][]float32 {
		out := make([][]float32, len(ts))
		for i, v := range ts {
			d, err := v.Data[float32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = d
		}
		return out
	}

	x, w := leaves(1)
	y, z := step(x, w, 2)
	rec, err := tensor.Record(ctx, []*tensor.Tensor{x, w}, []*tensor.Tensor{y, z, y})
	if err != nil {
		t.Fatal(err)
	}
	if rec == nil {
		t.Fatal("the step was not recorded")
	}
	wy, wz := step(x, w, 2)
	if got, want := values(y, z), values(wy, wz); !slices.Equal(got[0], want[0]) || !slices.Equal(got[1], want[1]) {
		t.Fatalf("Record computed %v, want %v", got, want)
	}

	x2, w2 := leaves(5)
	wy, wz = step(x2, w2, 2)
	want := values(wy, wz)
	compiled := sched.Compiled()
	outs, ok, err := rec.Replay(ctx, []*tensor.Tensor{x2, w2})
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("the recording did not fit leaves of the same shapes")
	}
	if sched.Compiled() != compiled {
		t.Error("a replay compiled a program")
	}
	if len(outs) != 3 || outs[0] != outs[2] {
		t.Fatalf("Replay returned %d targets, want 3 with the same tensor first and last", len(outs))
	}
	if got := values(outs[0], outs[1]); !slices.Equal(got[0], want[0]) || !slices.Equal(got[1], want[1]) {
		t.Fatalf("Replay computed %v, want %v", got, want)
	}

	// A leaf not yet computed is computed first; a replay's output serves
	// as a leaf of the next, as a model's state does.
	outs2, ok, err := rec.Replay(ctx, []*tensor.Tensor{outs[1], tensor.Randn(9, n, n)})
	if err != nil || !ok {
		t.Fatalf("Replay over an output and a lazy leaf: %v, %v", ok, err)
	}
	if got := values(outs2[0]); len(got[0]) != n*n {
		t.Fatalf("Replay gave %d values", len(got[0]))
	}

	// Leaves of other shapes, or on another layout, do not fit.
	x3, w3 := leaves(7)
	if _, ok, err := rec.Replay(ctx, []*tensor.Tensor{x3.Reshape(2, 8), w3}); err != nil || ok {
		t.Errorf("Replay over leaves of another shape: fit %v, %v", ok, err)
	}
	if _, ok, err := rec.Replay(ctx, []*tensor.Tensor{x3.Transpose(0, 1), w3}); err != nil || ok {
		t.Errorf("Replay over a transposed leaf: fit %v, %v", ok, err)
	}
	if _, _, err := rec.Replay(ctx, []*tensor.Tensor{x3}); err == nil {
		t.Error("Replay over too few leaves did not fail")
	}
}

// A computation that cannot be replayed, as one with a leaf among its
// targets, is computed and not recorded. A realized tensor that the
// leaves do not name is read again by every replay.
func TestRecordRefusesWhatItCannotReplay(t *testing.T) {
	if tensor.Eager() {
		t.Skip("eager mode computes every operation as it is created, leaving nothing to record")
	}
	ctx := context.Background()
	x := tensor.Randn(1, 4)
	if err := tensor.Realize(ctx, x); err != nil {
		t.Fatal(err)
	}
	rec, err := tensor.Record(ctx, []*tensor.Tensor{x}, []*tensor.Tensor{x.Add(x), x})
	if err != nil || rec != nil {
		t.Errorf("a leaf among the targets: %v, %v", rec, err)
	}
	table := tensor.FromSlice([]float32{1, 2, 3, 4}, 4)
	y := x.Add(table)
	rec, err = tensor.Record(ctx, []*tensor.Tensor{x}, []*tensor.Tensor{y})
	if err != nil || rec == nil {
		t.Fatalf("a realized tensor the leaves do not name: %v, %v", rec, err)
	}
	x2 := tensor.FromSlice([]float32{10, 10, 10, 10}, 4)
	outs, ok, err := rec.Replay(ctx, []*tensor.Tensor{x2})
	if err != nil || !ok {
		t.Fatalf("Replay: %v, %v", ok, err)
	}
	if got, err := outs[0].Data[float32](ctx); err != nil || !slices.Equal(got, []float32{11, 12, 13, 14}) {
		t.Errorf("Replay read the table as %v, %v", got, err)
	}
}

// Replays of one recording from several goroutines each compute their
// own leaves' result, replaying or, while the program is busy, reporting
// a miss.
func TestReplayConcurrent(t *testing.T) {
	if tensor.Eager() {
		t.Skip("eager mode computes every operation as it is created, leaving nothing to record")
	}
	ctx := context.Background()
	const n = 64
	x := tensor.Randn(1, n, n)
	w := tensor.Randn(2, n, n)
	rec, err := tensor.Record(ctx, []*tensor.Tensor{x, w}, []*tensor.Tensor{x.MatMul(w).Exp()})
	if err != nil || rec == nil {
		t.Fatalf("Record: %v, %v", rec, err)
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	hits := 0
	for g := range 8 {
		wg.Go(func() {
			for i := range 20 {
				x := tensor.Randn(uint64(100*g+i), n, n)
				want, err := x.MatMul(w).Exp().Data[float32](ctx)
				if err != nil {
					t.Error(err)
					return
				}
				outs, ok, err := rec.Replay(ctx, []*tensor.Tensor{x, w})
				if err != nil {
					t.Error(err)
					return
				}
				if !ok {
					continue
				}
				got, err := outs[0].Data[float32](ctx)
				if err != nil {
					t.Error(err)
					return
				}
				if !slices.Equal(got, want) {
					t.Errorf("goroutine %d: replay %d differs", g, i)
					return
				}
				mu.Lock()
				hits++
				mu.Unlock()
			}
		})
	}
	wg.Wait()
	if hits == 0 {
		t.Error("no replay fit")
	}
}

// A recording made into destinations replays into new ones: a decoding
// step that writes a row of a cache writes it over the cache's memory
// every time, allocating nothing the cache's size.
func TestRecordIntoReplaysOverTheDestinations(t *testing.T) {
	if tensor.Eager() {
		t.Skip("eager mode computes every operation as it is created, leaving nothing to record")
	}
	ctx := t.Context()
	be := &allocCounter{Backend: cpu.New()}
	defer tensor.SetDevice(be)()
	const heads, capacity, dim = 2, 64, 8
	want := make([]float32, heads*capacity*dim)
	cache := tensor.FromSlice(make([]float32, len(want)), 1, heads, capacity, dim)
	var rec *tensor.Recording
	for i := range 6 {
		pos := tensor.Full(tensor.Int32, float64(i), 1, heads, 1, dim)
		row := tensor.Randn(uint64(i), 1, heads, 1, dim)
		if err := tensor.Realize(ctx, cache, pos, row); err != nil {
			t.Fatal(err)
		}
		r, _ := row.Data[float32](ctx)
		for h := range heads {
			copy(want[(h*capacity+i)*dim:][:dim], r[h*dim:][:dim])
		}
		leaves, dsts := []*tensor.Tensor{cache, pos, row}, []*tensor.Tensor{cache, nil}
		before, buf := be.elements, tensor.BufferOf(cache)
		var outs []*tensor.Tensor
		if rec == nil {
			next := cache.Scatter(2, pos, row)
			targets := []*tensor.Tensor{next, next.Sum()}
			var err error
			if rec, err = tensor.RecordInto(ctx, leaves, dsts, targets); err != nil || rec == nil {
				t.Fatalf("recording: %v, %v", rec, err)
			}
			outs = targets
		} else {
			var ok bool
			var err error
			if outs, ok, err = rec.ReplayInto(ctx, leaves, dsts); err != nil || !ok {
				t.Fatalf("step %d: replay %v, %v", i, ok, err)
			}
			if n := be.elements - before; n >= heads*capacity*dim {
				t.Errorf("step %d allocated %d elements, as much as the cache", i, n)
			}
			if tensor.BufferOf(outs[0]) != buf {
				t.Errorf("step %d did not write the cache over its memory", i)
			}
		}
		if _, err := cache.Data[float32](ctx); !errors.Is(err, tensor.ErrOverwritten) {
			t.Errorf("step %d: the old cache reads %v, want ErrOverwritten", i, err)
		}
		cache = outs[0]
		got, err := cache.Data[float32](ctx)
		if err != nil || !slices.Equal(got, want) {
			t.Fatalf("step %d: cache %v, %v, want %v", i, got, err, want)
		}
	}
	// A replay given no destinations does not fit a recording made into
	// them.
	pos, row := tensor.Full(tensor.Int32, 9, 1, heads, 1, dim), tensor.Randn(9, 1, heads, 1, dim)
	if _, ok, err := rec.Replay(ctx, []*tensor.Tensor{cache, pos, row}); ok || err != nil {
		t.Errorf("replay without destinations: %v, %v", ok, err)
	}
}
