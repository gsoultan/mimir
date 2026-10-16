package datasets

import (
	"context"
	"errors"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/io/parquet"
)

// parquetFixture returns the path of a fixture that tools/parquet/gen.py
// wrote.
func parquetFixture(name string) string { return filepath.Join("testdata", name+".parquet") }

// regressionRow returns row i of the regression table of the fixtures.
func regressionRow(i int) (x []float32, y float32) {
	even := float32(0)
	if i%2 == 0 {
		even = 1
	}
	return []float32{float32(i) / 4, float32(i%7 - 3), even, float32(int64(i) * 1_000_000_007)}, float32(i)/2 + 0.5
}

func TestParquetShardsStreamsRowGroups(t *testing.T) {
	src, features, err := ParquetShards([]string{parquetFixture("reg_0"), parquetFixture("reg_1")}, "y")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"x1", "x2", "x3", "x4"}; !slices.Equal(features, want) {
		t.Errorf("features %v, want %v", features, want)
	}
	if src.Shards() != 3 || src.Features() != 4 {
		t.Errorf("%d shards of %d features, want 3 of 4", src.Shards(), src.Features())
	}
	ys, xs := readAll(t, src)
	if len(ys) != 130 {
		t.Fatalf("%d rows, want 130", len(ys))
	}
	for i := range ys {
		x, y := regressionRow(i)
		if ys[i] != y || !slices.Equal(xs[i], x) {
			t.Errorf("row %d: %v -> %v, want %v -> %v", i, xs[i], ys[i], x, y)
		}
	}
	// Shuffled, the same rows come in another order.
	sys, _ := readAll(t, src, data.Shuffle(7), data.ShuffleBuffer(16))
	if slices.Equal(sys, ys) {
		t.Error("shuffled epoch is in order")
	}
	slices.Sort(sys)
	if !slices.Equal(sys, ys) {
		t.Errorf("shuffled rows %v, want a permutation of %v", sys, ys)
	}
	// Statistics read the source as a stream does.
	stats, err := data.Statistics(t.Context(), src)
	if err != nil {
		t.Fatal(err)
	}
	if want := float32(129) / 8; math.Abs(float64(stats.Mean[0]-want)) > 1e-5 {
		t.Errorf("mean of x1 %v, want %v", stats.Mean[0], want)
	}
}

func TestParquetShardsClasses(t *testing.T) {
	src, features, err := ParquetShards([]string{parquetFixture("cls_0"), parquetFixture("cls_1")}, "label", Classes("a", "b", "c"))
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"x1", "x2"}; !slices.Equal(features, want) {
		t.Errorf("features %v, want %v", features, want)
	}
	ys, xs := readAll(t, src, data.ClassTargets())
	if len(ys) != 130 {
		t.Fatalf("%d rows, want 130", len(ys))
	}
	for i := range ys {
		x, _ := regressionRow(i)
		if ys[i] != float32(i%3) || !slices.Equal(xs[i], x[:2]) {
			t.Errorf("row %d: %v -> %v, want %v -> %v", i, xs[i], ys[i], x[:2], i%3)
		}
	}

	// A shard reads in pieces, like an io.Reader, and ends with io.EOF.
	rows, err := src.Open(t.Context(), 1) // rows 50 to 99 of cls_0
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	if _, err := rows.Read(make([]float32, 1), make([]float32, 1)); !errors.Is(err, io.ErrShortBuffer) {
		t.Errorf("Read into less than a row: %v", err)
	}
	x, y := make([]float32, 2*7), make([]float32, 7)
	var got []float32
	for {
		n, err := rows.Read(x, y)
		got = append(got, y[:n]...)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if n == 0 {
			t.Fatal("Read returned nothing without io.EOF")
		}
	}
	if len(got) != 50 {
		t.Fatalf("%d rows from shard 1, want 50", len(got))
	}
	for i, c := range got {
		if c != float32((50+i)%3) {
			t.Errorf("row %d of shard 1: class %v, want %v", i, c, (50+i)%3)
		}
	}
	if n, err := rows.Read(x, y); n != 0 || !errors.Is(err, io.EOF) {
		t.Errorf("Read after the end: %d, %v", n, err)
	}
}

func TestParquetShardsErrors(t *testing.T) {
	reg0, reg1, cls0 := parquetFixture("reg_0"), parquetFixture("reg_1"), parquetFixture("cls_0")
	garbage := filepath.Join(t.TempDir(), "garbage.parquet")
	if err := os.WriteFile(garbage, []byte("PAR1 this is not a parquet file PAR1"), 0o644); err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct {
		paths  []string
		target string
		opts   []CSVOption
		is     error
		want   []string // substrings of the error
	}{
		"no files":          {nil, "y", nil, data.ErrSource, nil},
		"missing file":      {[]string{filepath.Join(t.TempDir(), "none.parquet")}, "y", nil, fs.ErrNotExist, nil},
		"garbage":           {[]string{garbage}, "y", nil, ErrFormat, []string{"garbage.parquet"}},
		"garbage (parquet)": {[]string{garbage}, "y", nil, parquet.ErrFormat, nil},
		"With":              {[]string{reg0}, "y", []CSVOption{With(data.BatchSize(2))}, data.ErrOption, []string{"NewStream"}},
		"Classes unnamed":   {[]string{cls0}, "label", []CSVOption{Classes()}, data.ErrOption, []string{"Classes"}},
		"no target":         {[]string{reg0}, "z", nil, ErrFormat, []string{"reg_0.parquet", `"z"`, "x1"}},
		"no features":       {[]string{parquetFixture("single")}, "y", nil, ErrFormat, []string{"single.parquet", "no feature columns"}},
		"string feature":    {[]string{cls0}, "x1", nil, ErrFormat, []string{`"label"`, "BYTE_ARRAY", "not a number"}},
		"string target":     {[]string{cls0}, "label", nil, ErrFormat, []string{`"label"`, "Classes"}},
		"numeric classes":   {[]string{reg0}, "y", []CSVOption{Classes("a")}, data.ErrOption, []string{`"y"`, "DOUBLE", "Classes"}},
		"type mismatch":     {[]string{reg0, parquetFixture("mismatch")}, "y", nil, ErrFormat, []string{"mismatch.parquet", "column 2", `"x2" INT64`, `"x2" INT32`, "reg_0.parquet"}},
		"column mismatch":   {[]string{reg1, parquetFixture("single")}, "y", nil, ErrFormat, []string{"single.parquet", "1 columns, want 5", "reg_1.parquet"}},
	} {
		_, _, err := ParquetShards(tc.paths, tc.target, tc.opts...)
		if !errors.Is(err, tc.is) {
			t.Errorf("%s: %v, want %v", name, err, tc.is)
			continue
		}
		for _, w := range tc.want {
			if !strings.Contains(err.Error(), w) {
				t.Errorf("%s: %v does not mention %q", name, err, w)
			}
		}
	}

	// Reading fails with the file, row and column.
	for name, tc := range map[string]struct {
		paths  []string
		target string
		opts   []CSVOption
		want   []string
	}{
		"null":          {[]string{parquetFixture("nulls")}, "y", nil, []string{"nulls.parquet", "row 3", `column "x2"`, "null"}},
		"unknown class": {[]string{cls0}, "label", []CSVOption{Classes("a", "b")}, []string{"cls_0.parquet", "row 2", `column "label"`, `"c"`}},
	} {
		src, _, err := ParquetShards(tc.paths, tc.target, tc.opts...)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		s, err := data.NewStream(src)
		if err != nil {
			t.Fatal(err)
		}
		var got error
		for _, err := range s.All(t.Context()) {
			if err != nil {
				got = err
			}
		}
		if !errors.Is(got, ErrFormat) {
			t.Errorf("%s: %v", name, got)
			continue
		}
		for _, w := range tc.want {
			if !strings.Contains(got.Error(), w) {
				t.Errorf("%s: %v does not mention %q", name, got, w)
			}
		}
	}

	// Open rejects a shard that does not exist and a cancelled context.
	src, _, err := ParquetShards([]string{reg0}, "y")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := src.Open(t.Context(), 2); !errors.Is(err, data.ErrSource) {
		t.Errorf("shard 2 of 2: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := src.Open(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Errorf("Open with a cancelled context: %v", err)
	}

	// A file that changed since ParquetShards read it fails its shards.
	changed := filepath.Join(t.TempDir(), "changed.parquet")
	copyFile(t, reg0, changed)
	src, _, err = ParquetShards([]string{changed}, "y")
	if err != nil {
		t.Fatal(err)
	}
	copyFile(t, reg1, changed) // one row group instead of two
	if _, err := src.Open(t.Context(), 1); !errors.Is(err, data.ErrSource) || !strings.Contains(err.Error(), "1 row groups, had 2") {
		t.Errorf("fewer row groups: %v", err)
	}
	copyFile(t, cls0, changed)
	if _, err := src.Open(t.Context(), 0); !errors.Is(err, ErrFormat) || !strings.Contains(err.Error(), "3 columns, want 5") {
		t.Errorf("other columns: %v", err)
	}
}

func copyFile(t *testing.T, from, to string) {
	t.Helper()
	b, err := os.ReadFile(from)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(to, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
