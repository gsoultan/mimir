package datasets

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
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
		"nested":            {[]string{parquetFixture("nested")}, "y", nil, ErrFormat, []string{"nested.parquet", `"vals.list.element"`, "nested"}},
		"nested elsewhere":  {[]string{reg0, parquetFixture("nested")}, "y", nil, ErrFormat, []string{"nested.parquet", "6 columns, want 5"}},
		"no row groups":     {[]string{parquetFixture("no_rows")}, "y", nil, data.ErrSource, []string{"no_rows.parquet", "no row groups"}},
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
		"null": {[]string{parquetFixture("nulls")}, "y", nil, []string{"nulls.parquet", "row 3", `column "x2"`, "null"}},
		// The first file declares x1 required, the second does not and
		// holds a null in it, which is not read as NaN.
		"null in a column required in the first file": {[]string{parquetFixture("required"), parquetFixture("x1_null")}, "y", nil, []string{"x1_null.parquet", "row 4", `column "x1"`, "null"}},
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
	closeSource(t, src)
	copyFile(t, reg1, changed) // one row group instead of two
	if _, err := src.Open(t.Context(), 1); !errors.Is(err, data.ErrSource) || !strings.Contains(err.Error(), "1 row groups, had 2") {
		t.Errorf("fewer row groups: %v", err)
	}
	copyFile(t, cls0, changed)
	if _, err := src.Open(t.Context(), 0); !errors.Is(err, ErrFormat) || !strings.Contains(err.Error(), "3 columns, want 5") {
		t.Errorf("other columns: %v", err)
	}
}

func TestParquetShardsLogicalTypes(t *testing.T) {
	src, features, err := ParquetShards([]string{parquetFixture("logical")}, "y")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"d64", "d32", "day"}; !slices.Equal(features, want) {
		t.Errorf("features %v, want %v", features, want)
	}
	ys, xs := readAll(t, src)
	// DECIMALs are their value, DATEs their days since the epoch.
	want := [][]float32{{12.34, -1.234, 10}, {0.01, 0.5, 10957}}
	if !slices.Equal(ys, []float32{1, 2}) || len(xs) != 2 {
		t.Fatalf("targets %v, rows %v", ys, xs)
	}
	for i := range want {
		for j, w := range want[i] {
			if got := xs[i][j]; math.Abs(float64(got-w)) > 1e-6*math.Abs(float64(w)) {
				t.Errorf("row %d %s = %v, want %v", i, features[j], got, w)
			}
		}
	}
}

func TestParquetShardsReadsAnySizes(t *testing.T) {
	src, _, err := ParquetShards([]string{parquetFixture("reg_0")}, "y")
	if err != nil {
		t.Fatal(err)
	}
	// Blocks of 3 rows, so that most Reads span several.
	src.(*parquetSource).block = 3
	for _, sizes := range [][][2]int{
		{{40, 10}, {8, 1}},          // a smaller y after a larger one
		{{4, 7}, {40, 2}, {13, 13}}, // x and y the limit in turn, x not a whole number of rows
		{{4 * 50, 50}},
		{{4 * 64, 64}},
	} {
		rows, err := src.Open(t.Context(), 1) // rows 50 to 99
		if err != nil {
			t.Fatal(err)
		}
		row := 50
		for k := 0; ; k++ {
			size := sizes[k%len(sizes)]
			x, y := make([]float32, size[0]), make([]float32, size[1])
			n, err := rows.Read(x, y)
			if want := min(size[0]/4, size[1], 100-row); n != want && !(want == 0 && n == 0) {
				t.Fatalf("%v: read %d at row %d, want %d", sizes, n, row, want)
			}
			for i := range n {
				wx, wy := regressionRow(row + i)
				if !slices.Equal(x[4*i:4*i+4], wx) || y[i] != wy {
					t.Fatalf("%v: row %d: %v -> %v, want %v -> %v", sizes, row+i, x[4*i:4*i+4], y[i], wx, wy)
				}
			}
			row += n
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		if row != 100 {
			t.Errorf("%v: read to row %d, want 100", sizes, row)
		}
		rows.Close()
	}
}

func TestParquetShardsShareOpenFiles(t *testing.T) {
	// The shards of a file read one after another, by Stream and
	// Statistics, parse its footer once.
	src, _, err := ParquetShards([]string{parquetFixture("reg_0")}, "y")
	if err != nil {
		t.Fatal(err)
	}
	ps := src.(*parquetSource)
	for range 2 {
		if ys, _ := readAll(t, src); len(ys) != 100 {
			t.Fatalf("%d rows", len(ys))
		}
	}
	if _, err := data.Statistics(t.Context(), src); err != nil {
		t.Fatal(err)
	}
	if n := ps.opened.Load(); n != 1 {
		t.Errorf("reading the 2 shards of a file 3 times opened it %d times", n)
	}

	// Of six files read in turn, four stay open, the last read.
	dir := t.TempDir()
	var paths []string
	for i := range 6 {
		paths = append(paths, filepath.Join(dir, fmt.Sprintf("reg_%d.parquet", i)))
		copyFile(t, parquetFixture("reg_1"), paths[i])
	}
	src, _, err = ParquetShards(paths, "y")
	if err != nil {
		t.Fatal(err)
	}
	closeSource(t, src)
	ps = src.(*parquetSource)
	if ys, _ := readAll(t, src, data.Prefetch(1)); len(ys) != 180 {
		t.Fatalf("%d rows", len(ys))
	}
	checkIdle(t, ps, paths[2:])
	// Reading them again in order finds the first two closed and the
	// rest evicted by the time it gets to them.
	readAll(t, src, data.Prefetch(1))
	if n := ps.opened.Load(); n != 12 {
		t.Errorf("reading 6 files twice opened them %d times", n)
	}
	// Read in the order the cache keeps them, none is opened again.
	for _, shard := range []int{2, 3, 4, 5} {
		rows, err := src.Open(t.Context(), shard)
		if err != nil {
			t.Fatal(err)
		}
		if err := rows.Close(); err != nil {
			t.Fatal(err)
		}
		if err := rows.Close(); err != nil { // a second Close does nothing
			t.Fatal(err)
		}
	}
	if n := ps.opened.Load(); n != 12 {
		t.Errorf("reading the open files opened %d", n-12)
	}
	checkIdle(t, ps, paths[2:])

	// Shards opened, read and closed at random from several goroutines
	// share files and leave at most 4 open.
	if err := src.(io.Closer).Close(); err != nil {
		t.Fatal(err)
	}
	checkIdle(t, ps, []string{})
	paths = append(paths, parquetFixture("reg_0"), parquetFixture("reg_0")) // a path twice
	src, _, err = ParquetShards(paths, "y")
	if err != nil {
		t.Fatal(err)
	}
	closeSource(t, src)
	ps = src.(*parquetSource)
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Go(func() {
			rng := rand.New(rand.NewPCG(uint64(g), 0))
			x, y := make([]float32, 4*16), make([]float32, 16)
			for range 50 {
				shard := rng.IntN(src.Shards())
				rows, err := src.Open(t.Context(), shard)
				if err != nil {
					t.Error(err)
					return
				}
				n := 0
				for {
					m, err := rows.Read(x, y)
					n += m
					if errors.Is(err, io.EOF) {
						break
					}
					if err != nil {
						t.Error(err)
						return
					}
				}
				if want := []int{30, 30, 30, 30, 30, 30, 50, 50, 50, 50}[shard]; n != want {
					t.Errorf("shard %d: %d rows, want %d", shard, n, want)
				}
				if err := rows.Close(); err != nil {
					t.Error(err)
				}
			}
		})
	}
	wg.Wait()
	checkIdle(t, ps, nil)

	// Close closes the idle files, and a shard closed after it closes its
	// file at once, so that the files can be deleted.
	rows, err := src.Open(t.Context(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.(io.Closer).Close(); err != nil {
		t.Fatal(err)
	}
	ps.mu.Lock()
	if len(ps.idle) != 0 || len(ps.open) != 1 {
		t.Errorf("after Close, %d files open, %d of them idle", len(ps.open), len(ps.idle))
	}
	ps.mu.Unlock()
	if err := rows.Close(); err != nil {
		t.Fatal(err)
	}
	ps.mu.Lock()
	if len(ps.open) != 0 || len(ps.idle) != 0 {
		t.Errorf("%d files open after the last shard closed", len(ps.open))
	}
	ps.mu.Unlock()
	for _, path := range paths[:6] {
		if err := os.Remove(path); err != nil {
			t.Error(err)
		}
	}
}

// closeSource closes src, a source of ParquetShards, when the test ends,
// before its temporary directory is removed.
func closeSource(t *testing.T, src data.Source) {
	t.Helper()
	t.Cleanup(func() {
		if err := src.(io.Closer).Close(); err != nil {
			t.Error(err)
		}
	})
}

// checkIdle checks that every file ps keeps open is idle, at most 4 of
// them, and, if want is not nil, that they are those of want in order of
// last use.
func checkIdle(t *testing.T, ps *parquetSource, want []string) {
	t.Helper()
	ps.mu.Lock()
	defer ps.mu.Unlock()
	var idle []string
	for _, sf := range ps.idle {
		idle = append(idle, sf.path)
		if sf.refs != 0 || ps.open[sf.path] != sf {
			t.Errorf("idle file %s has %d references", sf.path, sf.refs)
		}
	}
	if len(ps.open) != len(ps.idle) || len(ps.idle) > maxIdleFiles {
		t.Errorf("%d files open, %d of them idle", len(ps.open), len(ps.idle))
	}
	if want != nil && !slices.Equal(idle, want) {
		t.Errorf("idle files %v, want %v", idle, want)
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

// BenchmarkParquetShards reads the file named by MIMIR_PARQUET_BENCH, such
// as one of many row groups and columns, through the shard reader with the
// last column as the target: "open" times Open and Close of the first
// shard, "read" the Reads of every shard in order with the timer stopped
// around Open and Close, and "epoch" all of it, as an in-order epoch does.
// MB/s is the file's bytes.
func BenchmarkParquetShards(b *testing.B) {
	path := os.Getenv("MIMIR_PARQUET_BENCH")
	if path == "" {
		b.Skip("MIMIR_PARQUET_BENCH names no file")
	}
	f, err := parquet.OpenFile(path)
	if err != nil {
		b.Fatal(err)
	}
	cols := f.Columns()
	f.Close()
	src, features, err := ParquetShards([]string{path}, cols[len(cols)-1].Name)
	if err != nil {
		b.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		b.Fatal(err)
	}
	const chunk = 4096
	x, y := make([]float32, chunk*len(features)), make([]float32, chunk)
	b.Run("open", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			rows, err := src.Open(b.Context(), 0)
			if err != nil {
				b.Fatal(err)
			}
			if err := rows.Close(); err != nil {
				b.Fatal(err)
			}
		}
	})
	epoch := func(b *testing.B, pause bool) {
		b.SetBytes(info.Size())
		b.ReportAllocs()
		var n int
		for b.Loop() {
			n = 0
			for shard := range src.Shards() {
				if pause {
					b.StopTimer()
				}
				rows, err := src.Open(b.Context(), shard)
				if err != nil {
					b.Fatal(err)
				}
				if pause {
					b.StartTimer()
				}
				for {
					m, err := rows.Read(x, y)
					n += m
					if errors.Is(err, io.EOF) {
						break
					}
					if err != nil {
						b.Fatal(err)
					}
				}
				if pause {
					b.StopTimer()
				}
				if err := rows.Close(); err != nil {
					b.Fatal(err)
				}
				if pause {
					b.StartTimer()
				}
			}
		}
		b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "rows/s")
	}
	b.Run("read", func(b *testing.B) { epoch(b, true) })
	b.Run("epoch", func(b *testing.B) { epoch(b, false) })
}
