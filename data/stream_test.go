package data_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"slices"
	"strings"
	"sync"
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
	return shardsOf(t, 2, sizes...)
}

// shardsOf is shards with features values per example: x = [k, -k, 0...].
func shardsOf(t *testing.T, features int, sizes ...int) *data.Slices {
	t.Helper()
	var out []data.Shard
	k := 0
	for _, n := range sizes {
		sh := data.Shard{X: make([]float32, features*n), Y: make([]float32, n)}
		for i := range n {
			sh.X[i*features], sh.Y[i] = float32(k), float32(k)
			if features > 1 {
				sh.X[i*features+1] = -float32(k)
			}
			k++
		}
		out = append(out, sh)
	}
	src, err := data.FromSlices(features, out...)
	if err != nil {
		t.Fatal(err)
	}
	return src
}

// seq returns lo, lo+1, ..., hi-1.
func seq(lo, hi int) []float32 {
	out := make([]float32, 0, hi-lo)
	for k := lo; k < hi; k++ {
		out = append(out, float32(k))
	}
	return out
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
		f := s.Features()
		if xs, ys := b.X.Shape(), b.Y.Shape(); len(xs) != 2 || xs[0] != len(y) || xs[1] != f || len(ys) != 1 || ys[0] != len(y) {
			t.Fatalf("batch shapes X %v Y %v", xs, ys)
		}
		for i, v := range y {
			if x[i*f] != v || f > 1 && x[i*f+1] != -v {
				t.Fatalf("batch row %d: x %v does not match y %v", i, x[i*f:i*f+min(f, 2)], v)
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

func want13() []float32 { return seq(0, 13) }

func TestStreamRejectsBadClassTargets(t *testing.T) {
	// A NaN fails every comparison, so a check written as what fails
	// lets it through as a garbage index.
	for name, y := range map[string]float32{"fraction": 1.5, "negative": -1, "nan": float32(math.NaN()), "inf": float32(math.Inf(1))} {
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

func TestStreamBatchLargerThanChunk(t *testing.T) {
	// Examples of 4096 features fill a chunk of about 4 MiB at 255 of
	// them (TestChunkRows pins that), so batches of 256 span chunks and
	// the last batch takes the tail of one.
	const features, batch = 4096, 256
	src := shardsOf(t, features, 300, 300)
	stream := func(opts ...data.Option) *data.Stream {
		s, err := data.NewStream(src, append([]data.Option{data.BatchSize(batch), data.Prefetch(1), data.ShuffleBuffer(100)}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	got := targets(t, stream(), false)
	want := [][]float32{seq(0, 256), seq(256, 512), seq(512, 600)}
	if !slices.EqualFunc(got, want, slices.Equal) {
		t.Errorf("%d batches, want 256, 256 and 88 examples in order", len(got))
	}
	if got := targets(t, stream(data.DropLast()), false); !slices.EqualFunc(got, want[:2], slices.Equal) {
		t.Errorf("DropLast gave %d batches", len(got))
	}
	all := func(s *data.Stream) []float32 { return slices.Concat(targets(t, s, false)...) }
	first := all(stream(data.Shuffle(7)))
	if !slices.Equal(slices.Sorted(slices.Values(first)), seq(0, 600)) {
		t.Error("shuffled epoch did not visit each example once")
	}
	if got := all(stream(data.Shuffle(7), data.Prefetch(8))); !slices.Equal(got, first) {
		t.Error("Prefetch(8) changed the shuffled order")
	}
}

// allocated returns the bytes the heap allocated while f ran.
func allocated(f func()) uint64 {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// Examples of wideFeatures values make chunks of about chunkBytes, far
// above whatever else an epoch over a few rows allocates, so a chunk
// allocated shows in the bytes an epoch allocates.
const wideFeatures, chunkBytes = 1024, 4 << 20

func TestStreamKeepsChunksAcrossEpochs(t *testing.T) {
	src := shardsOf(t, wideFeatures, 10, 10, 10, 10)
	s, err := data.NewStream(src, data.BatchSize(10), data.Prefetch(1))
	if err != nil {
		t.Fatal(err)
	}
	run := func() {
		for _, err := range s.All(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	if got := allocated(run); got < chunkBytes/2 {
		t.Fatalf("first epoch allocated %d bytes, want at least a chunk of about %d", got, chunkBytes)
	}
	if got := allocated(run); got > chunkBytes/4 {
		t.Errorf("second epoch allocated %d bytes, want the chunks of the first kept", got)
	}
}

// funcSource is a Source whose shards open makes.
type funcSource struct {
	features, shards int
	open             func(ctx context.Context, shard int) (data.Rows, error)
}

func (s *funcSource) Shards() int   { return s.shards }
func (s *funcSource) Features() int { return s.features }
func (s *funcSource) Open(ctx context.Context, shard int) (data.Rows, error) {
	return s.open(ctx, shard)
}

// readFunc is a Rows that reads with itself and closes with nothing.
type readFunc func(x, y []float32) (int, error)

func (f readFunc) Read(x, y []float32) (int, error) { return f(x, y) }
func (readFunc) Close() error                       { return nil }

func TestStreamReaderReturnsChunkOnFailure(t *testing.T) {
	// Shard 1 fails at its first read, after shard 0 has been read. The
	// reader must hand the chunk it holds back, or every epoch after
	// the first allocates another in its place.
	good := shardsOf(t, wideFeatures, 10)
	badTarget, err := data.FromSlices(wideFeatures,
		data.Shard{X: make([]float32, 10*wideFeatures), Y: make([]float32, 10)},
		data.Shard{X: make([]float32, wideFeatures), Y: []float32{1.5}},
	)
	if err != nil {
		t.Fatal(err)
	}
	overCount := &funcSource{features: wideFeatures, shards: 2, open: func(ctx context.Context, shard int) (data.Rows, error) {
		if shard == 0 {
			return good.Open(ctx, 0)
		}
		return readFunc(func(x, y []float32) (int, error) { return len(y) + 1, nil }), nil
	}}
	for name, tc := range map[string]struct {
		src  data.Source
		opts []data.Option
		want error
	}{
		"target": {badTarget, []data.Option{data.ClassTargets()}, data.ErrTarget},
		"count":  {overCount, nil, data.ErrSource},
	} {
		s, err := data.NewStream(tc.src, append([]data.Option{data.BatchSize(10), data.Prefetch(1)}, tc.opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		run := func() {
			var got error
			for _, err := range s.All(t.Context()) {
				if err != nil {
					got = err
				}
			}
			if !errors.Is(got, tc.want) || !strings.Contains(got.Error(), "shard 1") {
				t.Fatalf("%s: epoch ended with %v", name, got)
			}
		}
		run() // allocates the chunks of an epoch
		if got := allocated(func() {
			for range 3 {
				run()
			}
		}); got > chunkBytes/4 {
			t.Errorf("%s: three more failing epochs allocated %d bytes, want the chunks of the first kept", name, got)
		}
	}
}

// aheadSource is a two-shard source of one example per read, for
// TestStreamReaderReturnsChunkOnCancel: shard 1 reads freely and shard 0
// yields nothing until shard 1 has been read twice in the same epoch.
type aheadSource struct {
	mu    sync.Mutex
	ahead chan struct{}
}

func (s *aheadSource) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ahead = make(chan struct{})
}

func (s *aheadSource) Shards() int   { return 2 }
func (s *aheadSource) Features() int { return wideFeatures }

func (s *aheadSource) Open(_ context.Context, shard int) (data.Rows, error) {
	s.mu.Lock()
	ahead := s.ahead
	s.mu.Unlock()
	reads := 0
	return readFunc(func(x, y []float32) (int, error) {
		reads++
		switch {
		case shard == 0:
			<-ahead
		case reads == 2:
			close(ahead)
		}
		clear(x[:wideFeatures])
		y[0] = float32(shard)
		return 1, nil
	}), nil
}

func TestStreamReaderReturnsChunkOnCancel(t *testing.T) {
	// The consumer leaves the epoch after one batch, from shard 0, while
	// shard 1's reader, ahead of it, has filled its channel with one
	// chunk and holds a second one to send. Cancelled there, the reader
	// must hand that chunk back, or the epochs keep allocating.
	src := &aheadSource{}
	s, err := data.NewStream(src, data.BatchSize(1), data.Prefetch(2))
	if err != nil {
		t.Fatal(err)
	}
	run := func() {
		src.reset()
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		defer cancel()
		for b, err := range s.All(ctx) {
			if err != nil {
				t.Fatal(err)
			}
			if y, _ := b.Y.Data[float32](ctx); !slices.Equal(y, []float32{0}) {
				t.Fatalf("first batch has targets %v, want shard 0's", y)
			}
			break
		}
	}
	for range 5 {
		run() // by now every chunk an epoch can take is allocated
	}
	if got := allocated(func() {
		for range 3 {
			run()
		}
	}); got > chunkBytes/4 {
		t.Errorf("three more cut-short epochs allocated %d bytes, want the chunks of the readers cancelled kept", got)
	}
}

func TestStreamFailsAShardThatMakesNoProgress(t *testing.T) {
	// A Rows that returns no examples and no error for ever would spin
	// the reader until the context ends; one that does so a few times
	// before its examples is fine.
	stuck := &funcSource{features: 1, shards: 1, open: func(context.Context, int) (data.Rows, error) {
		return readFunc(func([]float32, []float32) (int, error) { return 0, nil }), nil
	}}
	slow := &funcSource{features: 1, shards: 1, open: func(context.Context, int) (data.Rows, error) {
		reads := 0
		return readFunc(func(x, y []float32) (int, error) {
			switch reads++; {
			case reads < 50:
				return 0, nil
			case reads == 50:
				x[0], y[0] = 1, 1
				return 1, nil
			}
			return 0, io.EOF
		}), nil
	}}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	s, err := data.NewStream(stuck)
	if err != nil {
		t.Fatal(err)
	}
	var got error
	n := 0
	for _, err := range s.All(ctx) {
		got = err
		n++
	}
	if !errors.Is(got, io.ErrNoProgress) || !errors.Is(got, data.ErrSource) || !strings.Contains(got.Error(), "shard 0") || n != 1 {
		t.Errorf("stuck shard: %d yields, the last %v", n, got)
	}
	if _, err := data.Statistics(ctx, stuck); !errors.Is(err, io.ErrNoProgress) || !errors.Is(err, data.ErrSource) {
		t.Errorf("statistics of a stuck shard: %v", err)
	}
	if s, err = data.NewStream(slow); err != nil {
		t.Fatal(err)
	}
	if got := slices.Concat(targets(t, s, false)...); !slices.Equal(got, []float32{1}) {
		t.Errorf("slow shard gave %v", got)
	}
	if st, err := data.Statistics(ctx, slow); err != nil || st.Rows != 1 {
		t.Errorf("statistics of a slow shard: %v, %v", st, err)
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
