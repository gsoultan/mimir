package data_test

import (
	"context"
	"errors"
	"io"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
	"time"

	"github.com/gsoultan/mimir/data"
)

// varied returns a source of three features, the third constant, in
// shards of the given sizes, with the examples as one row-major slice.
func varied(t *testing.T, sizes ...int) (*data.Slices, []float32) {
	t.Helper()
	r := rand.New(rand.NewPCG(3, 4))
	var all []float32
	var out []data.Shard
	for _, n := range sizes {
		sh := data.Shard{X: make([]float32, 3*n), Y: make([]float32, n)}
		for i := range n {
			sh.X[3*i] = float32(r.NormFloat64()*10 + 100)
			sh.X[3*i+1] = float32(r.Float64() - 0.5)
			sh.X[3*i+2] = 7
			sh.Y[i] = float32(i)
		}
		all = append(all, sh.X...)
		out = append(out, sh)
	}
	src, err := data.FromSlices(3, out...)
	if err != nil {
		t.Fatal(err)
	}
	return src, all
}

// reference computes the statistics of the rows of x, with f features,
// directly in float64.
func reference(x []float32, f int) data.Stats {
	n := len(x) / f
	st := data.Stats{Rows: int64(n), Mean: make([]float32, f), Std: make([]float32, f), Min: make([]float32, f), Max: make([]float32, f)}
	for j := range f {
		sum, lo, hi := 0.0, math.Inf(1), math.Inf(-1)
		for i := range n {
			v := float64(x[i*f+j])
			sum += v
			lo, hi = min(lo, v), max(hi, v)
		}
		mean := sum / float64(n)
		sq := 0.0
		for i := range n {
			d := float64(x[i*f+j]) - mean
			sq += d * d
		}
		st.Mean[j], st.Std[j], st.Min[j], st.Max[j] = float32(mean), float32(math.Sqrt(sq/float64(n))), float32(lo), float32(hi)
	}
	return st
}

func TestStatisticsMatchesReference(t *testing.T) {
	src, all := varied(t, 10000, 1, 9000, 3000)
	want := reference(all, 3)
	for _, prefetch := range []int{1, 3} {
		got, err := data.Statistics(t.Context(), src, data.Prefetch(prefetch))
		if err != nil {
			t.Fatal(err)
		}
		if got.Rows != want.Rows {
			t.Errorf("Rows %d, want %d", got.Rows, want.Rows)
		}
		for j := range 3 {
			close := func(name string, g, w float32, tol float64) {
				if math.Abs(float64(g-w)) > tol*math.Max(1, math.Abs(float64(w))) {
					t.Errorf("Prefetch(%d) feature %d %s %v, want %v", prefetch, j, name, g, w)
				}
			}
			close("Mean", got.Mean[j], want.Mean[j], 1e-6)
			close("Std", got.Std[j], want.Std[j], 1e-5)
			close("Min", got.Min[j], want.Min[j], 0)
			close("Max", got.Max[j], want.Max[j], 0)
		}
		if got.Std[2] != 0 || got.Mean[2] != 7 {
			t.Errorf("constant feature: mean %v std %v", got.Mean[2], got.Std[2])
		}
	}
}

func TestNormalizeStandardizes(t *testing.T) {
	src, _ := varied(t, 5000, 2000)
	stats, err := data.Statistics(t.Context(), src)
	if err != nil {
		t.Fatal(err)
	}
	s, err := data.NewStream(src, data.BatchSize(512), data.Shuffle(1), data.Normalize(stats))
	if err != nil {
		t.Fatal(err)
	}
	sum, sq := make([]float64, 3), make([]float64, 3)
	n := 0
	for b, err := range s.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		x, err := b.X.Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		for i := range len(x) / 3 {
			for j := range 3 {
				v := float64(x[3*i+j])
				sum[j] += v
				sq[j] += v * v
			}
			n++
		}
	}
	if n != 7000 {
		t.Fatalf("%d examples", n)
	}
	for j := range 3 {
		mean := sum[j] / float64(n)
		std := math.Sqrt(sq[j]/float64(n) - mean*mean)
		wantStd := 1.0
		if j == 2 {
			wantStd = 0 // constant, so only centered
		}
		if math.Abs(mean) > 1e-5 || math.Abs(std-wantStd) > 1e-5 {
			t.Errorf("feature %d: mean %v, std %v", j, mean, std)
		}
	}
}

func TestStatisticsErrors(t *testing.T) {
	src, _ := varied(t, 10, 10)
	empty, err := data.FromSlices(3, data.Shard{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := data.Statistics(t.Context(), empty); !errors.Is(err, data.ErrSource) {
		t.Errorf("empty source: %v", err)
	}
	if _, err := data.Statistics(t.Context(), src, data.Prefetch(0)); !errors.Is(err, data.ErrOption) {
		t.Errorf("prefetch 0: %v", err)
	}
	if _, err := data.Statistics(t.Context(), nil); !errors.Is(err, data.ErrSource) {
		t.Errorf("nil source: %v", err)
	}
	if _, err := data.Statistics(t.Context(), emptySource{0, 1}); !errors.Is(err, data.ErrSource) {
		t.Errorf("no shards: %v", err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := data.Statistics(ctx, src); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: %v", err)
	}

	boom := errors.New("boom")
	failing := &countingSource{Source: src, failShard: 1, failAt: 1, failErr: boom}
	_, err = data.Statistics(t.Context(), failing)
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "data: shard 1:") {
		t.Errorf("failing shard: %v", err)
	}
	if o, c := failing.opened.Load(), failing.closed.Load(); o != c {
		t.Errorf("%d opened, %d closed", o, c)
	}
}

func TestStatisticsReportsTheLowestFailedShard(t *testing.T) {
	// Shard 2 fails first, and shard 1 once it has. Shard 0 reads one
	// example once the failure has cancelled the context, so it is cut
	// short by the cancellation, which the failures outrank.
	boom := errors.New("boom")
	failed := make(chan struct{})
	src := &funcSource{features: 1, shards: 4, open: func(ctx context.Context, shard int) (data.Rows, error) {
		return readFunc(func(x, y []float32) (int, error) {
			switch shard {
			case 0:
				<-ctx.Done()
				x[0], y[0] = 1, 1
				return 1, nil
			case 1:
				<-failed
				return 0, boom
			case 2:
				close(failed)
				return 0, boom
			}
			return 0, io.EOF
		}), nil
	}}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	_, err := data.Statistics(ctx, src, data.Prefetch(3))
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "data: shard 1:") {
		t.Errorf("two failed shards: %v, want shard 1's", err)
	}
}
