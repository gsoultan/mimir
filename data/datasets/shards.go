package datasets

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/gsoultan/mimir/data"
)

// csvBuffer is the read buffer of a CSV shard.
const csvBuffer = 1 << 20

// gzipMagic begins every gzip file.
var gzipMagic = []byte{0x1f, 0x8b}

// utf8BOM is the byte order mark some exporters put before the first
// line.
var utf8BOM = []byte("\xEF\xBB\xBF")

// CSVShards streams the CSV files at paths as one data.Source, a shard per
// file, as CSV reads one file: each file has the same header row, the
// column named target is Y and every other column, of which there must
// be at least one, is a Float32 feature. A file that begins with the
// gzip magic bytes, or whose path ends in .gz, is decompressed as it is
// read, and a UTF-8 byte order mark before the header is skipped. It
// returns the source and the feature names.
//
// A numeric target is parsed as a float32. Class labels need Classes with
// the names, which map each label to its index in the list: Y then holds
// the indices as Float32, and data.ClassTargets on data.NewStream makes
// them Int32. A label not among them, a feature that is not a number and
// a header unlike the first file's fail the shard with the file, line and
// column. A field may be quoted, with "" for a quote inside, but a quoted
// field may not contain a line break. Every path must exist when
// CSVShards is called. With is for CSV only: the data options of a
// stream go to data.NewStream.
func CSVShards(paths []string, target string, opts ...CSVOption) (data.Source, []string, error) {
	var cfg csvConfig
	for _, o := range opts {
		o(&cfg)
	}
	if len(paths) == 0 {
		return nil, nil, fmt.Errorf("datasets: csv shards of no files: %w", data.ErrSource)
	}
	if cfg.with {
		return nil, nil, fmt.Errorf("datasets: csv shards take no With options; data.NewStream takes the data options: %w", data.ErrOption)
	}
	if cfg.classes && cfg.names == nil {
		return nil, nil, fmt.Errorf("datasets: csv shards need the class names in Classes: %w", data.ErrOption)
	}
	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			return nil, nil, fmt.Errorf("datasets: %w", err)
		}
	}
	src := &csvSource{paths: slices.Clone(paths)}
	if cfg.names != nil {
		var err error
		if src.classes, err = classIndex(cfg.names); err != nil {
			return nil, nil, err
		}
	}
	r, err := openCSV(src, 0)
	if err != nil {
		return nil, nil, err
	}
	header, err := r.readHeader()
	if cerr := r.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, nil, err
	}
	src.header = header
	src.target = slices.Index(header, target)
	if src.target < 0 {
		return nil, nil, fmt.Errorf("datasets: %s has no column %q in %v: %w", paths[0], target, header, ErrFormat)
	}
	if len(header) < 2 {
		return nil, nil, fmt.Errorf("datasets: %s has no feature columns besides the target %q: %w", paths[0], target, ErrFormat)
	}
	features := slices.Delete(slices.Clone(header), src.target, src.target+1)
	return src, features, nil
}

// csvSource is the data.Source of CSVShards.
type csvSource struct {
	paths   []string
	header  []string         // of the first file, which every file must match
	target  int              // column of the target
	classes map[string]int32 // index of each class name; nil for a numeric target
}

func (s *csvSource) Shards() int   { return len(s.paths) }
func (s *csvSource) Features() int { return len(s.header) - 1 }

func (s *csvSource) Open(ctx context.Context, shard int) (data.Rows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if shard < 0 || shard >= len(s.paths) {
		return nil, fmt.Errorf("datasets: csv shard %d of %d: %w", shard, len(s.paths), data.ErrSource)
	}
	r, err := openCSV(s, shard)
	if err != nil {
		return nil, err
	}
	header, err := r.readHeader()
	if err == nil {
		err = r.checkHeader(header)
	}
	if err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

// csvRows reads the lines of one CSV file.
type csvRows struct {
	src  *csvSource
	path string
	file *os.File
	gz   *gzip.Reader
	br   *bufio.Reader
	long []byte // a line longer than the buffer
	line int    // 1-based number of the last line read
	eof  bool
	// fields holds the fields of the line being parsed, as slices of it
	// or, for a line that went through encoding/csv, of unquoted.
	fields   [][]byte
	unquoted []byte
	// quoted parses a line with a quote inside a quoted field, reading
	// it from quotedLine; both are made on first use.
	quoted     *csv.Reader
	quotedLine *bytes.Reader
}

func openCSV(src *csvSource, shard int) (*csvRows, error) {
	path := src.paths[shard]
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("datasets: %w", err)
	}
	r := &csvRows{src: src, path: path, file: f}
	br := bufio.NewReaderSize(f, csvBuffer)
	magic, _ := br.Peek(len(gzipMagic))
	if bytes.Equal(magic, gzipMagic) || strings.HasSuffix(path, ".gz") {
		if r.gz, err = gzip.NewReader(br); err != nil {
			f.Close()
			return nil, fmt.Errorf("datasets: %s: %v: %w", path, err, ErrFormat)
		}
		r.br = bufio.NewReaderSize(r.gz, csvBuffer)
	} else {
		r.br = br
	}
	return r, nil
}

func (r *csvRows) Close() error {
	var gzErr error
	if r.gz != nil {
		gzErr = r.gz.Close()
	}
	return errors.Join(gzErr, r.file.Close())
}

// readHeader reads the header row, skipping a byte order mark.
func (r *csvRows) readHeader() ([]string, error) {
	line, err := r.nextLine()
	if errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("datasets: %s line 1: no header: %w", r.path, ErrFormat)
	}
	if err != nil {
		return nil, err
	}
	if err := r.split(bytes.TrimPrefix(line, utf8BOM)); err != nil {
		return nil, err
	}
	header := make([]string, len(r.fields))
	for i, f := range r.fields {
		header[i] = string(f)
	}
	return header, nil
}

// checkHeader reports a header unlike the first file's.
func (r *csvRows) checkHeader(header []string) error {
	want := r.src.header
	if len(header) != len(want) {
		return fmt.Errorf("datasets: %s line 1: %d columns %v, want %d %v: %w", r.path, len(header), header, len(want), want, ErrFormat)
	}
	for i := range header {
		if header[i] != want[i] {
			return fmt.Errorf("datasets: %s line 1 column %d: %q, want %q: %w", r.path, i+1, header[i], want[i], ErrFormat)
		}
	}
	return nil
}

// Read fills x and y with up to len(y) examples.
func (r *csvRows) Read(x, y []float32) (int, error) {
	f := r.src.Features()
	room := min(len(x)/f, len(y))
	if room == 0 {
		return 0, io.ErrShortBuffer
	}
	n := 0
	for n < room {
		line, err := r.nextLine()
		if err != nil {
			if errors.Is(err, io.EOF) && n > 0 {
				return n, nil
			}
			return n, err
		}
		if len(line) == 0 {
			continue
		}
		if err := r.parseLine(line, x[n*f:(n+1)*f], &y[n]); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

// nextLine returns the next line without its line ending, valid until the
// next call, or io.EOF after the last. Errors name the file and line.
func (r *csvRows) nextLine() ([]byte, error) {
	if r.eof {
		return nil, io.EOF
	}
	line, err := r.br.ReadSlice('\n')
	if errors.Is(err, bufio.ErrBufferFull) {
		r.long = append(r.long[:0], line...)
		for errors.Is(err, bufio.ErrBufferFull) {
			line, err = r.br.ReadSlice('\n')
			r.long = append(r.long, line...)
		}
		line = r.long
	}
	if err != nil {
		if !errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("datasets: %s line %d: %w", r.path, r.line+1, err)
		}
		r.eof = true
		if len(line) == 0 {
			return nil, io.EOF
		}
	}
	r.line++
	line = bytes.TrimSuffix(line, []byte("\n"))
	return bytes.TrimSuffix(line, []byte("\r")), nil
}

// split sets r.fields to the comma-separated fields of line, as slices
// of it. A field that begins and ends with a quote loses the quotes and
// keeps a comma inside; one with a quote inside, escaped as "", goes
// through encoding/csv, as does a line with a bare quote, which it
// rejects.
func (r *csvRows) split(line []byte) error {
	r.fields = r.fields[:0]
	if bytes.IndexByte(line, '"') < 0 {
		for {
			i := bytes.IndexByte(line, ',')
			if i < 0 {
				r.fields = append(r.fields, line)
				return nil
			}
			r.fields = append(r.fields, line[:i])
			line = line[i+1:]
		}
	}
	rest := line
	for {
		var field []byte
		if len(rest) > 0 && rest[0] == '"' {
			end := bytes.IndexByte(rest[1:], '"') + 1 // the closing quote
			if end < 1 || (end+1 < len(rest) && rest[end+1] != ',') {
				return r.splitCSV(line)
			}
			field, rest = rest[1:end], rest[end+1:]
		} else {
			i := bytes.IndexByte(rest, ',')
			if i < 0 {
				i = len(rest)
			}
			field, rest = rest[:i], rest[i:]
			if bytes.IndexByte(field, '"') >= 0 {
				return r.splitCSV(line)
			}
		}
		r.fields = append(r.fields, field)
		if len(rest) == 0 {
			return nil
		}
		rest = rest[1:] // the comma
	}
}

// splitCSV sets r.fields to the fields of line as encoding/csv reads it,
// through one reader kept for the file.
func (r *csvRows) splitCSV(line []byte) error {
	if r.quoted == nil {
		r.quotedLine = bytes.NewReader(nil)
		r.quoted = csv.NewReader(r.quotedLine)
		r.quoted.FieldsPerRecord = -1
		r.quoted.ReuseRecord = true
	}
	r.quotedLine.Reset(line)
	rec, err := r.quoted.Read()
	if err != nil {
		// The reader counts the lines it has seen, not the file's.
		var pe *csv.ParseError
		if errors.As(err, &pe) {
			return fmt.Errorf("datasets: %s line %d column %d: %v: %w", r.path, r.line, pe.Column, pe.Err, ErrFormat)
		}
		return fmt.Errorf("datasets: %s line %d: %v: %w", r.path, r.line, err, ErrFormat)
	}
	buf := r.unquoted[:0]
	for _, f := range rec {
		buf = append(buf, f...)
	}
	r.unquoted = buf
	r.fields = r.fields[:0]
	for _, f := range rec {
		r.fields = append(r.fields, buf[:len(f):len(f)])
		buf = buf[len(f):]
	}
	return nil
}

// parseLine parses one example from line into xrow and y.
func (r *csvRows) parseLine(line []byte, xrow []float32, y *float32) error {
	if err := r.split(line); err != nil {
		return err
	}
	header := r.src.header
	if len(r.fields) != len(header) {
		return fmt.Errorf("datasets: %s line %d: %d fields, want %d: %w", r.path, r.line, len(r.fields), len(header), ErrFormat)
	}
	xi := 0
	for col, field := range r.fields {
		if col == r.src.target {
			if r.src.classes != nil {
				c, ok := r.src.classes[string(field)]
				if !ok {
					return fmt.Errorf("datasets: %s line %d column %q: unknown class %q: %w", r.path, r.line, header[col], field, ErrFormat)
				}
				*y = float32(c)
				continue
			}
			v, err := strconv.ParseFloat(string(field), 32)
			if err != nil {
				return fmt.Errorf("datasets: %s line %d column %q: %q is not a number (Classes names the labels of a class target): %w", r.path, r.line, header[col], field, ErrFormat)
			}
			*y = float32(v)
			continue
		}
		v, err := strconv.ParseFloat(string(field), 32)
		if err != nil {
			return fmt.Errorf("datasets: %s line %d column %q: %q is not a number: %w", r.path, r.line, header[col], field, ErrFormat)
		}
		xrow[xi] = float32(v)
		xi++
	}
	return nil
}
