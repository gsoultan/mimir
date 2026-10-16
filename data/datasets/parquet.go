package datasets

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/io/parquet"
)

// ParquetShards streams the Parquet files at paths as one data.Source, a
// shard per row group, so that a large file is read a few row groups at
// a time too. Every file has the columns of the first, in the same order
// and of the same types: the column named target is Y and every other
// column, of which there must be at least one, is a feature, read as a
// float32 from a BOOLEAN (as 0 or 1), INT32, INT64, FLOAT or DOUBLE
// column. It returns the source and the feature names.
//
// A numeric target is read the same way. Class labels in a string
// (BYTE_ARRAY) column need Classes with the names, which map each label
// to its index in the list: Y then holds the indices as Float32, and
// data.ClassTargets on data.NewStream makes them Int32. A label not among
// them and a null in any column fail the shard with the file, the row's
// index in it and the column. Every path must exist when ParquetShards is
// called, which reads the metadata of each; a shard reopens its file. The
// options are those of CSV, though With is rejected: the data options of
// a stream go to data.NewStream.
func ParquetShards(paths []string, target string, opts ...CSVOption) (data.Source, []string, error) {
	var cfg csvConfig
	for _, o := range opts {
		o(&cfg)
	}
	if len(paths) == 0 {
		return nil, nil, fmt.Errorf("datasets: parquet shards of no files: %w", data.ErrSource)
	}
	if cfg.with {
		return nil, nil, fmt.Errorf("datasets: parquet shards take no With options; data.NewStream takes the data options: %w", data.ErrOption)
	}
	if cfg.classes && cfg.names == nil {
		return nil, nil, fmt.Errorf("datasets: parquet shards need the class names in Classes: %w", data.ErrOption)
	}
	src := &parquetSource{paths: slices.Clone(paths)}
	if cfg.names != nil {
		var err error
		if src.classes, err = classIndex(cfg.names); err != nil {
			return nil, nil, err
		}
	}
	for i, path := range src.paths {
		f, err := openParquet(path)
		if err != nil {
			return nil, nil, err
		}
		if i == 0 {
			err = src.describe(f, target)
		} else {
			err = src.check(f, path)
		}
		if cerr := f.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return nil, nil, err
		}
		var first int64
		for g := range f.RowGroups() {
			src.shards = append(src.shards, parquetShard{file: i, group: g, first: first})
			first += f.RowGroupRows(g)
		}
		src.groups = append(src.groups, f.RowGroups())
	}
	features := make([]string, 0, len(src.columns)-1)
	for i, c := range src.columns {
		if i != src.target {
			features = append(features, c.Name)
		}
	}
	return src, features, nil
}

// parquetSource is the data.Source of ParquetShards.
type parquetSource struct {
	paths   []string
	groups  []int            // row groups of each file
	shards  []parquetShard   // the row groups of every file, in order
	columns []parquet.Column // of the first file, which every file must match
	target  int              // column of the target
	classes map[string]int32 // index of each class name; nil for a numeric target
}

// parquetShard is one row group of one file.
type parquetShard struct {
	file, group int
	first       int64 // the index of the row group's first row in the file
}

func (s *parquetSource) Shards() int   { return len(s.shards) }
func (s *parquetSource) Features() int { return len(s.columns) - 1 }

// openParquet opens the file at path, wrapping a malformed file's error
// so that it is ErrFormat as well as parquet.ErrFormat.
func openParquet(path string) (*parquet.File, error) {
	f, err := parquet.OpenFile(path)
	if errors.Is(err, parquet.ErrFormat) || errors.Is(err, parquet.ErrUnsupported) {
		return nil, fmt.Errorf("datasets: %w (%w)", err, ErrFormat)
	}
	if err != nil {
		return nil, fmt.Errorf("datasets: %w", err)
	}
	return f, nil
}

// describe takes the columns of the first file, f, and checks that
// ParquetShards can read them.
func (s *parquetSource) describe(f *parquet.File, target string) error {
	path := s.paths[0]
	s.columns = f.Columns()
	s.target = -1
	names := make([]string, len(s.columns))
	for i, c := range s.columns {
		names[i] = c.Name
		if c.Name == target {
			s.target = i
		}
	}
	if s.target < 0 {
		return fmt.Errorf("datasets: %s has no column %q in %v: %w", path, target, names, ErrFormat)
	}
	if len(s.columns) < 2 {
		return fmt.Errorf("datasets: %s has no feature columns besides the target %q: %w", path, target, ErrFormat)
	}
	for i, c := range s.columns {
		strings := c.Type == parquet.ByteArray || c.Type == parquet.FixedLenByteArray
		switch {
		case i != s.target && !numeric(c.Type):
			return fmt.Errorf("datasets: %s column %q is %v, not a number: %w", path, c.Name, c.Type, ErrFormat)
		case i == s.target && s.classes == nil && strings:
			return fmt.Errorf("datasets: %s column %q holds strings (Classes names the labels of a class target): %w", path, c.Name, ErrFormat)
		case i == s.target && s.classes != nil && !strings:
			return fmt.Errorf("datasets: %s column %q is %v; Classes names the labels of a string target: %w", path, c.Name, c.Type, data.ErrOption)
		case i == s.target && s.classes == nil && !numeric(c.Type):
			return fmt.Errorf("datasets: %s column %q is %v, not a number: %w", path, c.Name, c.Type, ErrFormat)
		}
	}
	return nil
}

// numeric reports whether ParquetShards reads a column of type t as a
// float32.
func numeric(t parquet.Type) bool {
	switch t {
	case parquet.Boolean, parquet.Int32, parquet.Int64, parquet.Float, parquet.Double:
		return true
	}
	return false
}

// check reports a file whose columns are unlike the first file's.
func (s *parquetSource) check(f *parquet.File, path string) error {
	cols := f.Columns()
	if len(cols) != len(s.columns) {
		return fmt.Errorf("datasets: %s has %d columns, want %d as in %s: %w", path, len(cols), len(s.columns), s.paths[0], ErrFormat)
	}
	for i, c := range cols {
		want := s.columns[i]
		if c.Name != want.Name || c.Type != want.Type {
			return fmt.Errorf("datasets: %s column %d is %q %v, want %q %v as in %s: %w", path, i+1, c.Name, c.Type, want.Name, want.Type, s.paths[0], ErrFormat)
		}
	}
	return nil
}

func (s *parquetSource) Open(ctx context.Context, shard int) (data.Rows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if shard < 0 || shard >= len(s.shards) {
		return nil, fmt.Errorf("datasets: parquet shard %d of %d: %w", shard, len(s.shards), data.ErrSource)
	}
	sh := s.shards[shard]
	path := s.paths[sh.file]
	f, err := openParquet(path)
	if err != nil {
		return nil, err
	}
	err = s.check(f, path)
	if err == nil && f.RowGroups() != s.groups[sh.file] {
		err = fmt.Errorf("datasets: %s has %d row groups, had %d: %w", path, f.RowGroups(), s.groups[sh.file], data.ErrSource)
	}
	if err != nil {
		f.Close()
		return nil, err
	}
	r := &parquetRows{src: s, path: path, file: f, first: sh.first, cols: make([]*parquet.Reader, len(s.columns))}
	for i := range s.columns {
		if r.cols[i], err = f.Reader(sh.group, i); err != nil {
			f.Close()
			return nil, fmt.Errorf("datasets: %s row group %d: %w", path, sh.group, err)
		}
	}
	return r, nil
}

// parquetRows reads the rows of one row group, a column at a time.
type parquetRows struct {
	src   *parquetSource
	path  string
	file  *parquet.File
	first int64 // the index in the file of the row group's first row
	row   int64 // rows read so far
	cols  []*parquet.Reader
	// Buffers for one column of a read.
	vals   []float32
	labels [][]byte
	nulls  []bool
}

func (r *parquetRows) Close() error { return r.file.Close() }

// Read fills x and y with up to len(y) examples.
func (r *parquetRows) Read(x, y []float32) (int, error) {
	f := r.src.Features()
	room := min(len(x)/f, len(y))
	if room == 0 {
		return 0, io.ErrShortBuffer
	}
	if cap(r.nulls) < room {
		r.nulls = make([]bool, room)
		r.vals = make([]float32, room)
	}
	n, xi := -1, 0
	for i, c := range r.src.columns {
		var m int
		var err error
		switch {
		case i == r.src.target && r.src.classes != nil:
			m, err = r.readLabels(i, y[:room])
		case i == r.src.target:
			m, err = r.readNumbers(i, y[:room], 1)
		default:
			m, err = r.readNumbers(i, x[xi:], f)
			xi++
		}
		if err != nil {
			return 0, err
		}
		if n >= 0 && m != n {
			return 0, fmt.Errorf("datasets: %s column %q has %d rows after row %d, column %q %d: %w", r.path, c.Name, m, r.first+r.row, r.src.columns[0].Name, n, ErrFormat)
		}
		n = m
	}
	if n == 0 {
		return 0, io.EOF
	}
	r.row += int64(n)
	return n, nil
}

// readNumbers reads the next values of column col into dst at every
// stride-th position, as many as fit, and returns how many.
func (r *parquetRows) readNumbers(col int, dst []float32, stride int) (int, error) {
	c := r.src.columns[col]
	room := min(len(r.vals), (len(dst)+stride-1)/stride)
	n, err := r.cols[col].ReadFloat32(r.vals[:room], r.nullsFor(c, room))
	if err != nil && !errors.Is(err, io.EOF) {
		return 0, fmt.Errorf("datasets: %s: %w", r.path, err)
	}
	if err := r.checkNulls(c, n); err != nil {
		return 0, err
	}
	if stride == 1 {
		copy(dst, r.vals[:n])
	} else {
		for i, v := range r.vals[:n] {
			dst[i*stride] = v
		}
	}
	return n, nil
}

// readLabels reads the next labels of the target column col into y as
// class indices, as many as fit, and returns how many.
func (r *parquetRows) readLabels(col int, y []float32) (int, error) {
	c := r.src.columns[col]
	if cap(r.labels) < len(y) {
		r.labels = make([][]byte, len(y))
	}
	n, err := r.cols[col].ReadBytes(r.labels[:len(y)], r.nullsFor(c, len(y)))
	if err != nil && !errors.Is(err, io.EOF) {
		return 0, fmt.Errorf("datasets: %s: %w", r.path, err)
	}
	if err := r.checkNulls(c, n); err != nil {
		return 0, err
	}
	for i, l := range r.labels[:n] {
		k, ok := r.src.classes[string(l)]
		if !ok {
			return 0, fmt.Errorf("datasets: %s row %d column %q: unknown class %q: %w", r.path, r.first+r.row+int64(i), c.Name, l, ErrFormat)
		}
		y[i] = float32(k)
	}
	return n, nil
}

// nullsFor returns the nulls slice to read n values of c with: none for a
// column that cannot hold nulls.
func (r *parquetRows) nullsFor(c parquet.Column, n int) []bool {
	if !c.Nullable {
		return nil
	}
	return r.nulls[:n]
}

// checkNulls reports the first null among the n values just read of c.
func (r *parquetRows) checkNulls(c parquet.Column, n int) error {
	if !c.Nullable {
		return nil
	}
	if i := slices.Index(r.nulls[:n], true); i >= 0 {
		return fmt.Errorf("datasets: %s row %d column %q: null: %w", r.path, r.first+r.row+int64(i), c.Name, ErrFormat)
	}
	return nil
}
