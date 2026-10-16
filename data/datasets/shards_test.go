package datasets

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/tensor"
)

// writeShards writes each body to a file in a new temp dir, gzipping
// those whose name ends in .gz, and returns the paths in order.
func writeShards(t *testing.T, files ...[2]string) []string {
	t.Helper()
	dir := t.TempDir()
	var paths []string
	for _, f := range files {
		b := []byte(f[1])
		if strings.HasSuffix(f[0], ".gz") {
			b = gz(t, b)
		}
		path := filepath.Join(dir, f[0])
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	return paths
}

// readAll reads every example of src through a Stream in order and
// returns the targets and the rows of X.
func readAll(t *testing.T, src data.Source, opts ...data.Option) ([]float32, [][]float32) {
	t.Helper()
	s, err := data.NewStream(src, append([]data.Option{data.BatchSize(3)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	var ys []float32
	var xs [][]float32
	for b, err := range s.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		x, err := b.X.Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		f := b.X.Shape()[1]
		for i := range len(x) / f {
			xs = append(xs, x[i*f:(i+1)*f])
		}
		if b.Y.DType() == tensor.Int32 {
			y, _ := b.Y.Data[int32](t.Context())
			for _, v := range y {
				ys = append(ys, float32(v))
			}
		} else {
			y, _ := b.Y.Data[float32](t.Context())
			ys = append(ys, y...)
		}
	}
	return ys, xs
}

func TestCSVShardsStreamsFiles(t *testing.T) {
	paths := writeShards(t,
		[2]string{"a.csv", "x1,price,x2\n1,10.5,2\n3,20,4\n\n"},
		[2]string{"b.csv", "x1,price,x2\r\n5,30,6\r\n"},
		[2]string{"c.csv.gz", "x1,price,x2\n7,40,8\n9,50,10\n11,60,12\n13,70,14"},
	)
	src, features, err := CSVShards(paths, "price")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(features, []string{"x1", "x2"}) || src.Shards() != 3 || src.Features() != 2 {
		t.Errorf("features %v, %d shards of %d features", features, src.Shards(), src.Features())
	}
	ys, xs := readAll(t, src)
	if !slices.Equal(ys, []float32{10.5, 20, 30, 40, 50, 60, 70}) {
		t.Errorf("targets %v", ys)
	}
	wantX := [][]float32{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}}
	if !slices.EqualFunc(xs, wantX, slices.Equal) {
		t.Errorf("rows %v", xs)
	}
	// The same examples shuffled are the same multiset.
	shuffled, _ := readAll(t, src, data.Shuffle(3), data.ShuffleBuffer(4))
	if slices.Equal(shuffled, ys) || !slices.Equal(slices.Sorted(slices.Values(shuffled)), ys) {
		t.Errorf("shuffled targets %v", shuffled)
	}
}

func TestCSVShardsClasses(t *testing.T) {
	paths := writeShards(t,
		[2]string{"a.csv", "sepal,species,petal\n5.1,setosa,1.4\n7.0,versicolor,4.7\n"},
		[2]string{"b.csv", "sepal,species,petal\n6.3,\"virginica\",6.0\n4.9,setosa,1.4\n"},
	)
	src, features, err := CSVShards(paths, "species", Classes("setosa", "versicolor", "virginica"))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(features, []string{"sepal", "petal"}) {
		t.Errorf("features %v", features)
	}
	ys, xs := readAll(t, src, data.ClassTargets())
	if !slices.Equal(ys, []float32{0, 1, 2, 0}) || xs[2][1] != 6 {
		t.Errorf("targets %v, rows %v", ys, xs)
	}

	if _, _, err := CSVShards(paths, "species", Classes()); !errors.Is(err, data.ErrOption) {
		t.Errorf("Classes without names: %v", err)
	}
	if _, _, err := CSVShards(paths, "species", Classes("a", "a")); !errors.Is(err, data.ErrOption) {
		t.Errorf("duplicate class: %v", err)
	}
}

func TestCSVShardsQuotedFields(t *testing.T) {
	// Quoted fields are unquoted in place, a comma inside kept; a quote
	// inside, escaped as "", goes through encoding/csv, line after line.
	paths := writeShards(t, [2]string{"q.csv", "x1,x2,label\n1,2,\"cat\"\n\"3\",\"4\",\"a,b\"\n5,6,\"say \"\"hi\"\"\"\n7,8,\"say \"\"hi\"\"\"\n\"9\",10,cat\n"})
	src, features, err := CSVShards(paths, "label", Classes("cat", "a,b", `say "hi"`))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(features, []string{"x1", "x2"}) {
		t.Errorf("features %v", features)
	}
	ys, xs := readAll(t, src)
	if !slices.Equal(ys, []float32{0, 1, 2, 2, 0}) {
		t.Errorf("targets %v", ys)
	}
	if want := [][]float32{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}}; !slices.EqualFunc(xs, want, slices.Equal) {
		t.Errorf("rows %v", xs)
	}

	// A quote where encoding/csv rejects one is this reader's line and
	// csv's column, not csv's count of the lines it has seen.
	for name, tc := range map[string]struct {
		body string
		want []string // substrings of the error
	}{
		"bare":         {"x1,x2,label\n1,a\"b,cat\n", []string{"line 2 column 4", `bare "`}},
		"unterminated": {"x1,x2,label\n1,2,\"cat\n", []string{"line 2 column"}},
		"after quote":  {"x1,x2,label\n1,2,\"cat\"dog\n", []string{"line 2 column"}},
	} {
		paths := writeShards(t, [2]string{name + ".csv", tc.body})
		src, _, err := CSVShards(paths, "label", Classes("cat"))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		rows, err := src.Open(t.Context(), 0)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		defer rows.Close()
		_, err = rows.Read(make([]float32, 2), make([]float32, 1))
		if !errors.Is(err, ErrFormat) || strings.Contains(err.Error(), "on line") {
			t.Errorf("%s: %v", name, err)
			continue
		}
		for _, w := range tc.want {
			if !strings.Contains(err.Error(), w) {
				t.Errorf("%s: %v does not mention %q", name, err, w)
			}
		}
	}
}

func TestCSVShardsQuotedLabelsAllocateNothing(t *testing.T) {
	var b strings.Builder
	b.WriteString("x1,label,x2\n")
	for i := range 2000 {
		fmt.Fprintf(&b, "%d,%q,%d\n", i, []string{"cat", "dog"}[i%2], -i)
	}
	paths := writeShards(t, [2]string{"quoted.csv", b.String()})
	src, _, err := CSVShards(paths, "label", Classes("cat", "dog"))
	if err != nil {
		t.Fatal(err)
	}
	rows, err := src.Open(t.Context(), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	x, y := make([]float32, 100*2), make([]float32, 100)
	allocs := testing.AllocsPerRun(10, func() {
		if n, err := rows.Read(x, y); n != 100 || err != nil {
			t.Fatalf("read %d, %v", n, err)
		}
	})
	if allocs > 0 {
		t.Errorf("reading 100 rows of quoted labels allocated %v times", allocs)
	}
	// The eleventh read holds rows 1000 to 1099.
	if y[0] != 0 || y[1] != 1 || x[0] != 1000 || x[3] != -1001 {
		t.Errorf("read y %v x %v", y[:2], x[:4])
	}
}

func TestCSVShardsSkipsBOMAndDetectsGzip(t *testing.T) {
	paths := writeShards(t,
		[2]string{"bom.csv", "\xEF\xBB\xBFx1,price,x2\n1,10,2\n"},
		[2]string{"zipped.csv.gz", "x1,price,x2\n3,20,4\n"},
	)
	// Gzipped, but not named so.
	unnamed := filepath.Join(t.TempDir(), "unnamed.csv")
	if err := os.WriteFile(unnamed, gz(t, []byte("x1,price,x2\n5,30,6\n")), 0o644); err != nil {
		t.Fatal(err)
	}
	src, features, err := CSVShards(append(paths, unnamed), "price")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(features, []string{"x1", "x2"}) {
		t.Errorf("features %v", features)
	}
	ys, xs := readAll(t, src)
	if !slices.Equal(ys, []float32{10, 20, 30}) || !slices.EqualFunc(xs, [][]float32{{1, 2}, {3, 4}, {5, 6}}, slices.Equal) {
		t.Errorf("targets %v, rows %v", ys, xs)
	}
}

func TestCSVShardsErrors(t *testing.T) {
	good := "x1,label,x2\n1,cat,2\n"
	paths := writeShards(t,
		[2]string{"good.csv", good},
		[2]string{"header.csv", "x1,x2,label\n1,2,cat\n"},
		[2]string{"columns.csv", "x1,label\n1,cat\n"},
		[2]string{"class.csv", "x1,label,x2\n1,cat,2\n3,dog,4\n"},
		[2]string{"number.csv", "x1,label,x2\n1,cat,2\n3,cat,four\n"},
		[2]string{"fields.csv", "x1,label,x2\n1,cat\n"},
		[2]string{"target.csv", "x1,label,x2\n1,2.5,2\n"},
	)
	notGz := filepath.Join(t.TempDir(), "plain.csv.gz")
	if err := os.WriteFile(notGz, []byte(good), 0o644); err != nil {
		t.Fatal(err)
	}
	good0 := paths[0]
	for name, tc := range map[string]struct {
		path string
		want []string // substrings of the error
	}{
		"mismatched header": {paths[1], []string{filepath.Base(paths[1]), "line 1", "column 2", `"x2"`, `"label"`}},
		"fewer columns":     {paths[2], []string{filepath.Base(paths[2]), "line 1", "2 columns"}},
		"unknown class":     {paths[3], []string{filepath.Base(paths[3]), "line 3", `column "label"`, `"dog"`}},
		"non-numeric":       {paths[4], []string{filepath.Base(paths[4]), "line 3", `column "x2"`, `"four"`}},
		"short line":        {paths[5], []string{filepath.Base(paths[5]), "line 2", "2 fields, want 3"}},
	} {
		src, _, err := CSVShards([]string{good0, tc.path}, "label", Classes("cat"))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		s, err := data.NewStream(src, data.ClassTargets())
		if err != nil {
			t.Fatal(err)
		}
		var got error
		for _, err := range s.All(t.Context()) {
			if err != nil {
				got = err
			}
		}
		if !errors.Is(got, ErrFormat) || !strings.Contains(got.Error(), "data: shard 1:") {
			t.Errorf("%s: %v", name, got)
			continue
		}
		for _, w := range tc.want {
			if !strings.Contains(got.Error(), w) {
				t.Errorf("%s: %v does not mention %q", name, got, w)
			}
		}
	}

	// A non-numeric target without Classes, named with its file.
	src, _, err := CSVShards([]string{good0}, "label")
	if err != nil {
		t.Fatal(err)
	}
	s, _ := data.NewStream(src)
	for _, err := range s.All(t.Context()) {
		if !errors.Is(err, ErrFormat) || !strings.Contains(err.Error(), "line 2") || !strings.Contains(err.Error(), "Classes") {
			t.Errorf("string target: %v", err)
		}
	}
	// So is a target that is not a class index under ClassTargets.
	src, _, err = CSVShards([]string{paths[6]}, "label")
	if err != nil {
		t.Fatal(err)
	}
	s, _ = data.NewStream(src, data.ClassTargets())
	for _, err := range s.All(t.Context()) {
		if !errors.Is(err, data.ErrTarget) {
			t.Errorf("fractional class: %v", err)
		}
	}

	if _, _, err := CSVShards([]string{good0, filepath.Join(t.TempDir(), "missing.csv")}, "label"); !errors.Is(err, fs.ErrNotExist) || !strings.Contains(err.Error(), "missing.csv") {
		t.Errorf("missing file: %v", err)
	}
	if _, _, err := CSVShards([]string{good0}, "nope"); !errors.Is(err, ErrFormat) {
		t.Errorf("missing target: %v", err)
	}
	if _, _, err := CSVShards(nil, "label"); !errors.Is(err, data.ErrSource) {
		t.Errorf("no paths: %v", err)
	}
	if _, _, err := CSVShards([]string{notGz}, "label"); !errors.Is(err, ErrFormat) {
		t.Errorf("not gzipped: %v", err)
	}
	empty := writeShards(t, [2]string{"empty.csv", ""})
	if _, _, err := CSVShards(empty, "label"); !errors.Is(err, ErrFormat) || !strings.Contains(err.Error(), "line 1") {
		t.Errorf("empty file: %v", err)
	}
	only := writeShards(t, [2]string{"only.csv", "label\ncat\n"})
	if _, _, err := CSVShards(only, "label"); !errors.Is(err, ErrFormat) || !strings.Contains(err.Error(), "only.csv") || !strings.Contains(err.Error(), `"label"`) {
		t.Errorf("target as the only column: %v", err)
	}
	if _, _, err := CSVShards([]string{good0}, "label", With(data.BatchSize(2))); !errors.Is(err, data.ErrOption) {
		t.Errorf("With: %v", err)
	}
}

func TestCSVShardsLongLines(t *testing.T) {
	// A line longer than the read buffer is reassembled.
	var b strings.Builder
	b.WriteString("y")
	for i := range 300000 {
		fmt.Fprintf(&b, ",f%d", i)
	}
	b.WriteString("\n1")
	for range 300000 {
		b.WriteString(",0.5")
	}
	b.WriteString("\n")
	paths := writeShards(t, [2]string{"wide.csv", b.String()})
	src, features, err := CSVShards(paths, "y")
	if err != nil {
		t.Fatal(err)
	}
	if len(features) != 300000 {
		t.Fatalf("%d features", len(features))
	}
	rows, err := src.Open(t.Context(), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	x, y := make([]float32, 300000), make([]float32, 1)
	if n, err := rows.Read(x, y); n != 1 || err != nil || y[0] != 1 || x[299999] != 0.5 {
		t.Errorf("read %d, %v: y %v x[last] %v", n, err, y[0], x[299999])
	}
	if n, err := rows.Read(x, y); n != 0 || !errors.Is(err, io.EOF) {
		t.Errorf("read past the end: %d, %v", n, err)
	}
}

// writeBenchCSV writes a CSV of rows rows and features features to a new
// temp file and returns its path and size. With quoted, the target is a
// quoted class label, "no" or "yes", as R and pandas write strings; else
// it is 0 or 1.
func writeBenchCSV(b *testing.B, rows, features int, quoted bool) (string, int64) {
	b.Helper()
	path := filepath.Join(b.TempDir(), "bench.csv")
	f, err := os.Create(path)
	if err != nil {
		b.Fatal(err)
	}
	w := bytes.NewBuffer(make([]byte, 0, 1<<20))
	for j := range features {
		if j > 0 {
			w.WriteByte(',')
		}
		fmt.Fprintf(w, "f%d", j)
	}
	w.WriteString(",y\n")
	r := rand.New(rand.NewPCG(1, 2))
	labels := []string{"0", "1"}
	if quoted {
		labels = []string{`"no"`, `"yes"`}
	}
	for range rows {
		for j := range features {
			if j > 0 {
				w.WriteByte(',')
			}
			fmt.Fprintf(w, "%.4f", r.Float64()*100-50)
		}
		fmt.Fprintf(w, ",%s\n", labels[r.IntN(2)])
		if w.Len() > 1<<19 {
			if _, err := f.Write(w.Bytes()); err != nil {
				b.Fatal(err)
			}
			w.Reset()
		}
	}
	if _, err := f.Write(w.Bytes()); err != nil {
		b.Fatal(err)
	}
	if err := f.Close(); err != nil {
		b.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		b.Fatal(err)
	}
	return path, info.Size()
}

// BenchmarkCSVShards reads a 50,000-row CSV of 20 features through the
// shard reader, with a numeric target and with a quoted class label; MB/s
// is the file's bytes.
func BenchmarkCSVShards(b *testing.B) {
	const rows, features = 50000, 20
	for _, bm := range []struct {
		name   string
		quoted bool
		opts   []CSVOption
	}{
		{"numeric", false, nil},
		{"quoted", true, []CSVOption{Classes("no", "yes")}},
	} {
		b.Run(bm.name, func(b *testing.B) {
			path, size := writeBenchCSV(b, rows, features, bm.quoted)
			src, _, err := CSVShards([]string{path}, "y", bm.opts...)
			if err != nil {
				b.Fatal(err)
			}
			x, y := make([]float32, 8192*features), make([]float32, 8192)
			b.SetBytes(size)
			b.ReportAllocs()
			for b.Loop() {
				rows, err := src.Open(b.Context(), 0)
				if err != nil {
					b.Fatal(err)
				}
				for {
					_, err := rows.Read(x, y)
					if errors.Is(err, io.EOF) {
						break
					}
					if err != nil {
						b.Fatal(err)
					}
				}
				if err := rows.Close(); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(rows)*float64(b.N)/b.Elapsed().Seconds(), "rows/s")
		})
	}
}
