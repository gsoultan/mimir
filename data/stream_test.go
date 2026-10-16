package data_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/tensor"
)

// shards returns an in-memory source of two-feature examples in shards
// of the given sizes, where example k of the whole set has x = [k, -k]
// and y = k.
func shards(t *testing.T, sizes ...int) *data.Slices {
	t.Helper()
	var out []data.Shard
	k := 0
	for _, n := range sizes {
		sh := data.Shard{X: make([]float32, 2*n), Y: make([]float32, n)}
		for i := range n {
			sh.X[2*i], sh.X[2*i+1], sh.Y[i] = float32(k), -float32(k), float32(k)
			k++
		}
		out = append(out, sh)
	}
	src, err := data.FromSlices(2, out...)
	if err != nil {
		t.Fatal(err)
	}
	return src
}

// targets returns the Y values of each batch of one pass over s, checking
// the shapes and types of the batches and that each x matches its y.
func targets(t *testing.T, s *data.Stream, classes bool) [][]float32 {
	t.Helper()
	var out [][]float32
	for b, err := range s.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		x, err := b.X.Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var y []float32
		if classes {
			yi, err := b.Y.Data[int32](t.Context())
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range yi {
				y = append(y, float32(v))
			}
		} else if y, err = b.Y.Data[float32](t.Context()); err != nil {
			t.Fatal(err)
		}
		if xs, ys := b.X.Shape(), b.Y.Shape(); len(xs) != 2 || xs[0] != len(y) || xs[1] != 2 || len(ys) != 1 || ys[0] != len(y) {
			t.Fatalf("batch shapes X %v Y %v", xs, ys)
		}
		for i, v := range y {
			if x[2*i] != v || x[2*i+1] != -v {
				t.Fatalf("batch row %d: x %v does not match y %v", i, x[2*i:2*i+2], v)
			}
		}
		out = append(out, y)
	}
	return out
}

func TestStreamVisitsEveryExampleOnce(t *testing.T) {
	src := shards(t, 5, 1, 7)
	s, err := data.NewStream(src, data.BatchSize(4))
	if err != nil {
		t.Fatal(err)
	}
	if s.Features() != 2 {
		t.Errorf("Features() = %d", s.Features())
	}
	got := targets(t, s, false)
	want := [][]float32{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12}}
	if !slices.EqualFunc(got, want, slices.Equal) {
		t.Errorf("batches %v, want %v", got, want)
	}
	for b, err := range s.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if b.X.DType() != tensor.Float32 || b.Y.DType() != tensor.Float32 {
			t.Fatalf("batch types %s and %s, want Float32", b.X.DType(), b.Y.DType())
		}
	}

	dl, err := data.NewStream(src, data.BatchSize(4), data.DropLast())
	if err != nil {
		t.Fatal(err)
	}
	if got := targets(t, dl, false); len(got) != 3 || len(got[2]) != 4 {
		t.Errorf("DropLast gave %v", got)
	}

	cls, err := data.NewStream(src, data.BatchSize(6), data.ClassTargets())
	if err != nil {
		t.Fatal(err)
	}
	for b, err := range cls.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if b.Y.DType() != tensor.Int32 {
			t.Fatalf("class targets are %s", b.Y.DType())
		}
	}
	if got := slices.Concat(targets(t, cls, true)...); !slices.Equal(got, want13()) {
		t.Errorf("class targets %v", got)
	}
}

func want13() []float32 {
	out := make([]float32, 13)
	for i := range out {
		out[i] = float32(i)
	}
	return out
}

func TestStreamRejectsBadClassTargets(t *testing.T) {
	for name, y := range map[string]float32{"fraction": 1.5, "negative": -1} {
		src, err := data.FromSlices(1, data.Shard{X: []float32{1}, Y: []float32{1}}, data.Shard{X: []float32{2}, Y: []float32{y}})
		if err != nil {
			t.Fatal(err)
		}
		s, err := data.NewStream(src, data.ClassTargets())
		if err != nil {
			t.Fatal(err)
		}
		var got error
		n := 0
		for _, err := range s.All(t.Context()) {
			if err != nil {
				got = err
				break
			}
			n++
		}
		if !errors.Is(got, data.ErrTarget) || !strings.Contains(got.Error(), "shard 1") || n != 1 {
			t.Errorf("%s target: %d batches then %v", name, n, got)
		}
	}
	// A target within rounding noise of an integer is that integer.
	src, _ := data.FromSlices(1, data.Shard{X: []float32{1}, Y: []float32{2.0000002}})
	s, _ := data.NewStream(src, data.ClassTargets())
	for b, err := range s.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if y, _ := b.Y.Data[int32](t.Context()); !slices.Equal(y, []int32{2}) {
			t.Errorf("rounded target %v", y)
		}
	}
}

func TestStreamShuffleIsDeterministic(t *testing.T) {
	src := shards(t, 50, 30, 20)
	stream := func(opts ...data.Option) *data.Stream {
		s, err := data.NewStream(src, append([]data.Option{data.BatchSize(16), data.Shuffle(7), data.ShuffleBuffer(16)}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	all := func(s *data.Stream) []float32 { return slices.Concat(targets(t, s, false)...) }

	a, b := stream(), stream()
	first, second := all(a), all(a)
	if slices.Equal(first, second) {
		t.Error("two epochs had the same order")
	}
	if sorted := slices.Sorted(slices.Values(first)); !slices.Equal(sorted, all(stream(data.InOrder()))) {
		t.Errorf("epoch visited %v, want each example once", first)
	}
	if got := all(b); !slices.Equal(got, first) {
		t.Error("same seed and epoch gave a different order")
	}
	b.SetEpoch(1)
	if got := all(b); !slices.Equal(got, second) {
		t.Error("SetEpoch(1) did not reproduce the second epoch")
	}
	for _, n := range []int{1, 8} {
		if got := all(stream(data.Prefetch(n))); !slices.Equal(got, first) {
			t.Errorf("Prefetch(%d) changed the order", n)
		}
	}

	// Without the buffer, whole shards follow one another in order.
	only := all(stream(data.ShuffleBuffer(0)))
	if slices.Equal(only, first) {
		t.Error("ShuffleBuffer(0) gave the buffered order")
	}
	for i := 0; i < len(only); {
		lo := only[i]
		n := map[float32]int{0: 50, 50: 30, 80: 20}[lo]
		if n == 0 {
			t.Fatalf("position %d starts with %v, not a shard", i, lo)
		}
		for j := range n {
			if only[i+j] != lo+float32(j) {
				t.Fatalf("shard at %v broken at %d: %v", lo, j, only[i:i+n])
			}
		}
		i += n
	}
	if sorted := slices.Sorted(slices.Values(all(stream(data.ShuffleBuffer(1000))))); !slices.Equal(sorted, all(stream(data.InOrder()))) {
		t.Error("a buffer larger than the data lost examples")
	}
}

// countingSource wraps a Source, counting opens and closes and failing
// shard failShard with failErr: at Open, or at its failAt'th Read when
// failAt is positive.
type countingSource struct {
	data.Source
	opened, closed atomic.Int32
	failShard      int
	failAt         int
	failErr        error
}

func (s *countingSource) Open(ctx context.Context, shard int) (data.Rows, error) {
	if shard == s.failShard && s.failErr != nil && s.failAt == 0 {
		return nil, s.failErr
	}
	rows, err := s.Source.Open(ctx, shard)
	if err != nil {
		return nil, err
	}
	s.opened.Add(1)
	r := &countingRows{Rows: rows, src: s}
	if shard == s.failShard && s.failErr != nil {
		r.failAt = s.failAt
	}
	return r, nil
}

type countingRows struct {
	data.Rows
	src    *countingSource
	reads  int
	failAt int
}

func (r *countingRows) Read(x, y []float32) (int, error) {
	r.reads++
	if r.failAt > 0 && r.reads == r.failAt {
		return 0, r.src.failErr
	}
	return r.Rows.Read(x, y)
}

func (r *countingRows) Close() error {
	r.src.closed.Add(1)
	return r.Rows.Close()
}

func TestStreamShardErrorNamesTheShard(t *testing.T) {
	boom := errors.New("boom")
	for name, src := range map[string]*countingSource{
		"open": {Source: shards(t, 5, 5, 5), failShard: 1, failErr: boom},
		"read": {Source: shards(t, 5, 5, 5), failShard: 1, failAt: 1, failErr: boom},
	} {
		s, err := data.NewStream(src, data.BatchSize(5), data.Prefetch(3))
		if err != nil {
			t.Fatal(err)
		}
		var got error
		var before [][]float32
		for b, err := range s.All(t.Context()) {
			if err != nil {
				got = err
				break
			}
			y, _ := b.Y.Data[float32](t.Context())
			before = append(before, y)
		}
		if !errors.Is(got, boom) || !strings.Contains(got.Error(), "data: shard 1:") {
			t.Errorf("%s failure: %v", name, got)
		}
		if want := [][]float32{{0, 1, 2, 3, 4}}; !slices.EqualFunc(before, want, slices.Equal) {
			t.Errorf("%s failure: batches before it %v, want %v", name, before, want)
		}
		if o, c := src.opened.Load(), src.closed.Load(); o != c {
			t.Errorf("%s failure: %d opened, %d closed", name, o, c)
		}
	}
}

// settled reports whether the goroutine count returns to at most n
// within a short wait.
func settled(n int) bool {
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > n {
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(time.Millisecond)
	}
	return true
}

func TestStreamReleasesReadersOnEarlyExit(t *testing.T) {
	sizes := make([]int, 20)
	for i := range sizes {
		sizes[i] = 1000
	}
	src := &countingSource{Source: shards(t, sizes...)}
	s, err := data.NewStream(src, data.BatchSize(16), data.Shuffle(1), data.Prefetch(4))
	if err != nil {
		t.Fatal(err)
	}
	// A full epoch first, so whatever the tensor runtime starts lazily
	// is counted in the baseline.
	for _, err := range s.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if o, c := src.opened.Load(), src.closed.Load(); o != 20 || c != 20 {
		t.Fatalf("full epoch opened %d and closed %d shards", o, c)
	}

	t.Run("break", func(t *testing.T) {
		base := runtime.NumGoroutine()
		src.opened.Store(0)
		src.closed.Store(0)
		n := 0
		for _, err := range s.All(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
			if n++; n == 1 {
				break
			}
		}
		if !settled(base) {
			t.Errorf("%d goroutines after breaking out, %d before", runtime.NumGoroutine(), base)
		}
		if o, c := src.opened.Load(), src.closed.Load(); o == 0 || o != c {
			t.Errorf("%d opened, %d closed", o, c)
		}
	})

	t.Run("cancel", func(t *testing.T) {
		base := runtime.NumGoroutine()
		src.opened.Store(0)
		src.closed.Store(0)
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		var got error
		n := 0
		start := time.Now()
		for _, err := range s.All(ctx) {
			if err != nil {
				got = err
				break
			}
			if n++; n == 2 {
				cancel()
			}
		}
		if !errors.Is(got, context.Canceled) || n != 2 {
			t.Errorf("%d batches then %v, want context.Canceled after 2", n, got)
		}
		if since := time.Since(start); since > 2*time.Second {
			t.Errorf("cancelled epoch took %v", since)
		}
		if !settled(base) {
			t.Errorf("%d goroutines after cancelling, %d before", runtime.NumGoroutine(), base)
		}
		if o, c := src.opened.Load(), src.closed.Load(); o == 0 || o != c {
			t.Errorf("%d opened, %d closed", o, c)
		}
	})
}

func TestStreamCancelledBeforeStart(t *testing.T) {
	s, err := data.NewStream(shards(t, 3), data.BatchSize(2))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	n := 0
	for _, err := range s.All(ctx) {
		if !errors.Is(err, context.Canceled) {
			t.Errorf("cancelled epoch: %v", err)
		}
		n++
	}
	if n != 1 {
		t.Errorf("cancelled epoch yielded %d times", n)
	}
}

// emptySource is a Source with no shards or features.
type emptySource struct{ shards, features int }

func (s emptySource) Shards() int   { return s.shards }
func (s emptySource) Features() int { return s.features }
func (s emptySource) Open(context.Context, int) (data.Rows, error) {
	return nil, io.EOF
}

func TestNewStreamErrors(t *testing.T) {
	src := shards(t, 2)
	stats, err := data.Statistics(t.Context(), shards(t, 3))
	if err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct {
		src  data.Source
		opts []data.Option
		want error
	}{
		"nil source":       {nil, nil, data.ErrSource},
		"no shards":        {emptySource{0, 2}, nil, data.ErrSource},
		"no features":      {emptySource{1, 0}, nil, data.ErrSource},
		"batch 0":          {src, []data.Option{data.BatchSize(0)}, tensor.ErrShape},
		"prefetch 0":       {src, []data.Option{data.Prefetch(0)}, data.ErrOption},
		"negative buffer":  {src, []data.Option{data.ShuffleBuffer(-1)}, data.ErrOption},
		"nil stats":        {src, []data.Option{data.Normalize(nil)}, data.ErrOption},
		"stats of 3":       {src, []data.Option{data.Normalize(&data.Stats{Mean: make([]float32, 3), Std: make([]float32, 3)})}, tensor.ErrShape},
		"stats without sd": {src, []data.Option{data.Normalize(&data.Stats{Mean: make([]float32, 2)})}, tensor.ErrShape},
	} {
		if _, err := data.NewStream(tc.src, tc.opts...); !errors.Is(err, tc.want) {
			t.Errorf("%s: %v, want %v", name, err, tc.want)
		}
	}
	if _, err := data.NewStream(src, data.Normalize(stats)); err != nil {
		t.Errorf("matching stats: %v", err)
	}
}

func TestFromSlicesErrors(t *testing.T) {
	for name, tc := range map[string]struct {
		features int
		shards   []data.Shard
	}{
		"no features": {0, []data.Shard{{X: nil, Y: nil}}},
		"no shards":   {2, nil},
		"mismatch":    {2, []data.Shard{{X: []float32{1, 2, 3}, Y: []float32{1}}}},
	} {
		if _, err := data.FromSlices(tc.features, tc.shards...); !errors.Is(err, data.ErrSource) {
			t.Errorf("%s: %v", name, err)
		}
	}
	src := shards(t, 1)
	if _, err := src.Open(t.Context(), 1); !errors.Is(err, data.ErrSource) {
		t.Errorf("shard out of range: %v", err)
	}
	rows, err := src.Open(t.Context(), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	if _, err := rows.Read(make([]float32, 1), make([]float32, 1)); !errors.Is(err, io.ErrShortBuffer) {
		t.Errorf("read into less than one example: %v", err)
	}
	if n, err := rows.Read(make([]float32, 4), make([]float32, 2)); n != 1 || err != nil {
		t.Errorf("read %d, %v", n, err)
	}
	if n, err := rows.Read(make([]float32, 4), make([]float32, 2)); n != 0 || !errors.Is(err, io.EOF) {
		t.Errorf("read past the end: %d, %v", n, err)
	}
}

func ExampleNewStream() {
	src, err := data.FromSlices(2,
		data.Shard{X: []float32{1, 2, 3, 4}, Y: []float32{0, 1}},
		data.Shard{X: []float32{5, 6}, Y: []float32{1}},
	)
	if err != nil {
		fmt.Println(err)
		return
	}
	stream, err := data.NewStream(src, data.BatchSize(2), data.ClassTargets())
	if err != nil {
		fmt.Println(err)
		return
	}
	for batch, err := range stream.All(context.Background()) {
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println(batch.X, batch.Y)
	}
	// Output:
	// Tensor(float32[2 2]) Tensor(int32[2])
	// Tensor(float32[1 2]) Tensor(int32[1])
}
