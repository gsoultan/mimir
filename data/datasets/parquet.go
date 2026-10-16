package datasets

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/io/parquet"
)

// ParquetShards streams the Parquet files at paths as one data.Source, a
// shard per row group, so that a large file is read a few row groups at
// a time too. Every file has the columns of the first, in the same order
// and of the same types: the column named target is Y and every other
// column, of which there must be at least one, is a feature, read as a
// float32 from a BOOLEAN (as 0 or 1), INT32, INT64, FLOAT or DOUBLE
// column. A DECIMAL stored as an INT32 or INT64 is read as its value, the
// integer divided by 10^scale in float64 and then rounded to float32; a
// DATE, TIME or TIMESTAMP as its integer, a number of days or of time
// units. A nested column, such as a list, is rejected. It returns the
// source and the feature names.
//
// A numeric target is read the same way. Class labels in a string
// (BYTE_ARRAY) column need Classes with the names, which map each label
// to its index in the list: Y then holds the indices as Float32, and
// data.ClassTargets on data.NewStream makes them Int32. A label not among
// them and a null in any column, whatever the file's schema says of it,
// fail the shard with the file, the row's index in it and the column.
// Every path must exist when ParquetShards is called, which reads the
// metadata of each, and the files must hold at least one row group
// between them. The options are those of CSV, though With is rejected:
// the data options of a stream go to data.NewStream.
//
// Shards of one file share one open file, whose footer is parsed when a
// shard opens it and no other shard has it open; that open checks that
// the file still has the columns and row groups it had. A file that no
// shard is reading stays open, up to 4 such files, the least recently
// used closed first, so that reading the shards of a file one after
// another, as an epoch in order or Statistics does, parses its footer
// once. The source thus keeps open, with their metadata, the files of
// the shards being read, at most Prefetch of them for one epoch, and 4
// more, however many files there are; the 4 stay open between epochs. A
// file that changes while it is open is not noticed. The source is an
// io.Closer: Close closes the files that no shard is reading, and from
// then on a file closes as soon as its last shard does, which a program
// that deletes or replaces the files needs on Windows.
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
	if len(src.shards) == 0 {
		return nil, nil, fmt.Errorf("datasets: parquet files %v hold no row groups: %w", paths, data.ErrSource)
	}
	features := make([]string, 0, len(src.columns)-1)
	for i, c := range src.columns {
		if i != src.target {
			features = append(features, c.Name)
		}
	}
	src.block = blockRows(len(features))
	return src, features, nil
}

// blockBytes is the size of the X block that a Parquet Read fills a column
// at a time: small enough to stay in a core's L2 cache while each column
// is scattered into it.
const blockBytes = 256 << 10

// blockRows returns the rows of a Read block of features values each.
func blockRows(features int) int { return max(1, blockBytes/(4*features)) }

// maxIdleFiles is how many files a Parquet source keeps open while no
// shard reads them.
const maxIdleFiles = 4

// parquetSource is the data.Source of ParquetShards.
type parquetSource struct {
	paths    []string
	groups   []int            // row groups of each file
	shards   []parquetShard   // the row groups of every file, in order
	columns  []parquet.Column // of the first file, which every file must match
	divisors []float64        // 10^scale for a DECIMAL column, 0 for the others
	target   int              // column of the target
	classes  map[string]int32 // index of each class name; nil for a numeric target
	block    int              // rows of a Read block

	// The open files, by path: those shards read and, in idle, up to
	// maxIdleFiles that none reads, the least recently used first.
	mu     sync.Mutex
	open   map[string]*sharedFile
	idle   []*sharedFile
	closed bool         // Close ran: files no shard reads are closed, not kept
	opened atomic.Int64 // footers parsed by Open, for the tests
}

// sharedFile is a file open for the shards that read it.
type sharedFile struct {
	path  string
	refs  int           // shards reading it, under parquetSource.mu
	ready chan struct{} // closed once f or err is set
	f     *parquet.File
	err   error
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

// kind describes the type of a column for errors: its physical type and
// its logical type, if any.
func kind(c parquet.Column) string {
	switch c.Logical {
	case "":
		return c.Type.String()
	case "DECIMAL":
		return fmt.Sprintf("%v DECIMAL(%d, %d)", c.Type, c.Precision, c.Scale)
	}
	return fmt.Sprintf("%v %s", c.Type, c.Logical)
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
	s.divisors = make([]float64, len(s.columns))
	for i, c := range s.columns {
		strings := c.Type == parquet.ByteArray || c.Type == parquet.FixedLenByteArray
		switch {
		case c.Nested:
			return fmt.Errorf("datasets: %s column %q is nested, as a list or a struct is, which ParquetShards cannot read: %w", path, c.Name, ErrFormat)
		case i != s.target && !numeric(c.Type):
			return fmt.Errorf("datasets: %s column %q is %s, not a number: %w", path, c.Name, kind(c), ErrFormat)
		case i == s.target && s.classes == nil && strings:
			return fmt.Errorf("datasets: %s column %q holds strings (Classes names the labels of a class target): %w", path, c.Name, ErrFormat)
		case i == s.target && s.classes != nil && !strings:
			return fmt.Errorf("datasets: %s column %q is %s; Classes names the labels of a string target: %w", path, c.Name, kind(c), data.ErrOption)
		case i == s.target && s.classes == nil && !numeric(c.Type):
			return fmt.Errorf("datasets: %s column %q is %s, not a number: %w", path, c.Name, kind(c), ErrFormat)
		}
		s.divisors[i] = divisor(c)
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

// divisor returns what the integers of c are divided by to give its
// values: 10^Scale for a DECIMAL with a scale, and 0, for none, otherwise.
func divisor(c parquet.Column) float64 {
	if c.Logical == "DECIMAL" && c.Scale > 0 && (c.Type == parquet.Int32 || c.Type == parquet.Int64) {
		return math.Pow10(c.Scale)
	}
	return 0
}

// check reports a file whose columns are unlike the first file's in a way
// that changes how they are read.
func (s *parquetSource) check(f *parquet.File, path string) error {
	cols := f.Columns()
	if len(cols) != len(s.columns) {
		return fmt.Errorf("datasets: %s has %d columns, want %d as in %s: %w", path, len(cols), len(s.columns), s.paths[0], ErrFormat)
	}
	for i, c := range cols {
		want := s.columns[i]
		if c.Name != want.Name || c.Type != want.Type || c.Nested != want.Nested || divisor(c) != s.divisors[i] {
			return fmt.Errorf("datasets: %s column %d is %q %s, want %q %s as in %s: %w", path, i+1, c.Name, kind(c), want.Name, kind(want), s.paths[0], ErrFormat)
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
	file, err := s.acquire(sh.file)
	if err != nil {
		return nil, err
	}
	r := &parquetRows{src: s, path: file.path, file: file, first: sh.first, cols: make([]*parquet.Reader, len(s.columns))}
	for i := range s.columns {
		if r.cols[i], err = file.f.Reader(sh.group, i); err != nil {
			r.Close()
			return nil, fmt.Errorf("datasets: %s row group %d: %w", file.path, sh.group, err)
		}
	}
	return r, nil
}

// acquire returns the open file of the given index for a shard to read,
// opening it if no other shard has it open and none keeps it idle. Shards
// that ask for a file being opened wait for it, so its footer is parsed
// once. The shard releases the file when it is done.
func (s *parquetSource) acquire(file int) (*sharedFile, error) {
	path := s.paths[file]
	s.mu.Lock()
	if sf := s.open[path]; sf != nil {
		if sf.refs == 0 {
			s.idle = slices.DeleteFunc(s.idle, func(o *sharedFile) bool { return o == sf })
		}
		sf.refs++
		s.mu.Unlock()
		<-sf.ready
		// A file that failed to open is no longer in s.open, so the
		// reference taken on it needs no release.
		return sf, sf.err
	}
	sf := &sharedFile{path: path, refs: 1, ready: make(chan struct{})}
	if s.open == nil {
		s.open = make(map[string]*sharedFile)
	}
	s.open[path] = sf
	s.mu.Unlock()

	f, err := s.openChecked(file)
	s.mu.Lock()
	sf.f, sf.err = f, err
	if err != nil {
		delete(s.open, path)
	}
	s.mu.Unlock()
	close(sf.ready)
	return sf, err
}

// openChecked opens the file of the given index and checks that it still
// has the columns and the row groups ParquetShards found in it.
func (s *parquetSource) openChecked(file int) (*parquet.File, error) {
	path := s.paths[file]
	f, err := openParquet(path)
	if err != nil {
		return nil, err
	}
	s.opened.Add(1)
	err = s.check(f, path)
	if err == nil && f.RowGroups() != s.groups[file] {
		err = fmt.Errorf("datasets: %s has %d row groups, had %d: %w", path, f.RowGroups(), s.groups[file], data.ErrSource)
	}
	if err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}

// release ends a shard's use of sf. A file no shard reads joins the idle
// files, and the least recently used of those is closed when there are
// more than maxIdleFiles.
func (s *parquetSource) release(sf *sharedFile) error {
	s.mu.Lock()
	if sf.refs--; sf.refs > 0 {
		s.mu.Unlock()
		return nil
	}
	var evicted *sharedFile
	if s.closed {
		evicted = sf
		delete(s.open, sf.path)
	} else if s.idle = append(s.idle, sf); len(s.idle) > maxIdleFiles {
		evicted = s.idle[0]
		s.idle = slices.Delete(s.idle, 0, 1)
		delete(s.open, evicted.path)
	}
	s.mu.Unlock()
	if evicted != nil {
		return evicted.f.Close()
	}
	return nil
}

// Close closes the files that no shard is reading, and makes the source
// close every other file once its last shard is closed instead of keeping
// it open. Shards may still be opened afterwards; each opens its file
// again. Close returns the first error closing a file.
func (s *parquetSource) Close() error {
	s.mu.Lock()
	s.closed = true
	idle := s.idle
	s.idle = nil
	for _, sf := range idle {
		delete(s.open, sf.path)
	}
	s.mu.Unlock()
	var err error
	for _, sf := range idle {
		if cerr := sf.f.Close(); err == nil {
			err = cerr
		}
	}
	return err
}

// parquetRows reads the rows of one row group.
type parquetRows struct {
	src    *parquetSource
	path   string
	file   *sharedFile
	first  int64 // the index in the file of the row group's first row
	row    int64 // rows read so far
	cols   []*parquet.Reader
	closed bool
	// Buffers for one column of a block.
	vals   []float32
	wide   []float64 // the values of a DECIMAL column, before scaling
	labels [][]byte
	nulls  []bool
}

func (r *parquetRows) Close() error {
	if r.closed {
		return nil
	}
	r.closed = true
	return r.src.release(r.file)
}

// Read fills x and y with up to min(len(x)/features, len(y)) examples,
// the lesser of the room in x and in y, as data.Rows says. It reads them
// in blocks of rows, a column at a time, so that the X block a column is
// scattered into stays in cache.
func (r *parquetRows) Read(x, y []float32) (int, error) {
	f := r.src.Features()
	room := min(len(x)/f, len(y))
	if room == 0 {
		return 0, io.ErrShortBuffer
	}
	n := 0
	for n < room {
		want := min(r.src.block, room-n)
		m, err := r.readBlock(x[n*f:(n+want)*f], y[n:n+want])
		if err != nil {
			return 0, err
		}
		n += m
		r.row += int64(m)
		if m < want {
			break
		}
	}
	if n == 0 {
		return 0, io.EOF
	}
	return n, nil
}

// readBlock reads the next len(y) rows, or as many as are left, into x
// and y, and returns how many. Every column must hold as many.
func (r *parquetRows) readBlock(x, y []float32) (int, error) {
	f, m := r.src.Features(), len(y)
	if cap(r.vals) < m {
		r.vals = make([]float32, m)
		r.nulls = make([]bool, m)
	}
	n, xi := -1, 0
	for i, c := range r.src.columns {
		var got int
		var err error
		switch {
		case i == r.src.target && r.src.classes != nil:
			got, err = r.readLabels(i, y)
		case i == r.src.target:
			got, err = r.readNumbers(i, y)
		default:
			vals := r.vals[:m]
			got, err = r.readNumbers(i, vals)
			dst := x[xi:]
			for k, v := range vals[:got] {
				dst[k*f] = v
			}
			xi++
		}
		if err != nil {
			return 0, err
		}
		if n >= 0 && got != n {
			return 0, fmt.Errorf("datasets: %s column %q has %d rows after row %d, column %q %d: %w", r.path, c.Name, got, r.first+r.row, r.src.columns[0].Name, n, ErrFormat)
		}
		n = got
	}
	return n, nil
}

// readNumbers reads the next len(dst) values of column col, or as many as
// are left, into dst and returns how many.
func (r *parquetRows) readNumbers(col int, dst []float32) (int, error) {
	nulls := r.nulls[:len(dst)]
	var n int
	var err error
	if div := r.src.divisors[col]; div != 0 {
		if cap(r.wide) < len(dst) {
			r.wide = make([]float64, len(dst))
		}
		wide := r.wide[:len(dst)]
		n, err = r.cols[col].ReadFloat64(wide, nulls)
		for i, v := range wide[:n] {
			dst[i] = float32(v / div)
		}
	} else {
		n, err = r.cols[col].ReadFloat32(dst, nulls)
	}
	if err != nil && !errors.Is(err, io.EOF) {
		return 0, fmt.Errorf("datasets: %s: %w", r.path, err)
	}
	return n, r.checkNulls(col, n)
}

// readLabels reads the next labels of the target column col into y as
// class indices, as many as fit, and returns how many.
func (r *parquetRows) readLabels(col int, y []float32) (int, error) {
	if cap(r.labels) < len(y) {
		r.labels = make([][]byte, len(y))
	}
	n, err := r.cols[col].ReadBytes(r.labels[:len(y)], r.nulls[:len(y)])
	if err != nil && !errors.Is(err, io.EOF) {
		return 0, fmt.Errorf("datasets: %s: %w", r.path, err)
	}
	if err := r.checkNulls(col, n); err != nil {
		return 0, err
	}
	for i, l := range r.labels[:n] {
		k, ok := r.src.classes[string(l)]
		if !ok {
			return 0, fmt.Errorf("datasets: %s row %d column %q: unknown class %q: %w", r.path, r.first+r.row+int64(i), r.src.columns[col].Name, l, ErrFormat)
		}
		y[i] = float32(k)
	}
	return n, nil
}

// firstNull returns the index of the first true in nulls, or -1. It
// searches the bools as bytes, which bytes.IndexByte does a vector at a
// time, since every value of every column is checked.
func firstNull(nulls []bool) int {
	if len(nulls) == 0 {
		return -1
	}
	return bytes.IndexByte(unsafe.Slice((*byte)(unsafe.Pointer(&nulls[0])), len(nulls)), 1)
}

// checkNulls reports the first null among the n values just read of
// column col. Every column is read with a nulls slice, since a file may
// hold nulls in a column that the first file declares required.
func (r *parquetRows) checkNulls(col, n int) error {
	if i := firstNull(r.nulls[:n]); i >= 0 {
		return fmt.Errorf("datasets: %s row %d column %q: null: %w", r.path, r.first+r.row+int64(i), r.src.columns[col].Name, ErrFormat)
	}
	return nil
}
