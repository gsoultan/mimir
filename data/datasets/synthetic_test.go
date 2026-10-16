package datasets

import (
	"errors"
	"io"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/data"
)

// synthRows reads every row of src shard by shard.
func synthRows(t *testing.T, src data.Source) (x, y []float32) {
	t.Helper()
	f := src.Features()
	bx, by := make([]float32, 7*f), make([]float32, 7)
	for i := range src.Shards() {
		rows, err := src.Open(t.Context(), i)
		if err != nil {
			t.Fatal(err)
		}
		for {
			n, err := rows.Read(bx, by)
			x = append(x, bx[:n*f]...)
			y = append(y, by[:n]...)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		if err := rows.Close(); err != nil {
			t.Fatal(err)
		}
	}
	return x, y
}

func TestSynthetic(t *testing.T) {
	src := Synthetic(1000, 5, 3, 42)
	if src.Shards() != 3 || src.Features() != 5 {
		t.Fatalf("%d shards of %d features", src.Shards(), src.Features())
	}
	x, y := synthRows(t, src)
	if len(y) != 1000 || len(x) != 5000 {
		t.Fatalf("%d rows, %d values", len(y), len(x))
	}
	ones := 0
	for i, v := range y {
		if v != 0 && v != 1 {
			t.Fatalf("row %d target %v", i, v)
		}
		ones += int(v)
	}
	if ones < 300 || ones > 700 {
		t.Errorf("%d of 1000 targets are 1", ones)
	}
	for i, v := range x {
		if v < -1 || v >= 1 {
			t.Fatalf("value %d is %v", i, v)
		}
	}
	// The same seed gives the same rows, through any buffer size, and
	// another seed does not.
	x2, y2 := synthRows(t, Synthetic(1000, 5, 3, 42))
	if !slices.Equal(x, x2) || !slices.Equal(y, y2) {
		t.Error("the same seed gave different rows")
	}
	if x3, _ := synthRows(t, Synthetic(1000, 5, 3, 43)); slices.Equal(x, x3) {
		t.Error("another seed gave the same rows")
	}
	if _, y4 := synthRows(t, Synthetic(1000, 5, 7, 42)); len(y4) != 1000 {
		t.Errorf("7 shards hold %d rows", len(y4))
	}
	rows, err := src.Open(t.Context(), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	if _, err := rows.Read(make([]float32, 4), make([]float32, 1)); !errors.Is(err, io.ErrShortBuffer) {
		t.Errorf("short buffer: %v", err)
	}
	if _, err := src.Open(t.Context(), 3); !errors.Is(err, data.ErrSource) {
		t.Errorf("shard out of range: %v", err)
	}
	if s := Synthetic(-1, 0, 0, 1); s.Shards() != 1 || s.Features() != 1 {
		t.Errorf("clamped: %d shards of %d features", s.Shards(), s.Features())
	}
}

func TestSyntheticThroughStream(t *testing.T) {
	src := Synthetic(100, 3, 4, 1)
	s, err := data.NewStream(src, data.BatchSize(32), data.ClassTargets(), data.Shuffle(2))
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for b, err := range s.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		n += b.Y.Shape()[0]
	}
	if n != 100 {
		t.Errorf("streamed %d rows", n)
	}
}

func BenchmarkSynthetic(b *testing.B) {
	const features = 32
	src := Synthetic(1<<40, features, 1, 1)
	x, y := make([]float32, 8192*features), make([]float32, 8192)
	rows, err := src.Open(b.Context(), 0)
	if err != nil {
		b.Fatal(err)
	}
	defer rows.Close()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := rows.Read(x, y); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(b.N)*8192*features/b.Elapsed().Seconds()/1e9, "Gvalues/s")
}
