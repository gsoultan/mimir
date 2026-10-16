// Package parquet reads the subset of Apache Parquet that data science
// tools write by default, streaming one column chunk at a time so that
// training sets of any size can be read shard by shard.
//
//	f, err := parquet.OpenFile("train.parquet")
//	...
//	col, _ := f.Column("feature")
//	for g := range f.RowGroups() {
//		r, err := f.Reader(g, col)
//		...
//		for {
//			n, err := r.ReadFloat32(buf, nil)
//			if err == io.EOF {
//				break
//			}
//			...
//		}
//	}
//
// Supported are flat schemas (required or optional top-level columns) of the
// physical types BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY and
// FIXED_LEN_BYTE_ARRAY; data pages V1 and V2; the PLAIN, RLE, dictionary,
// DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY and
// BYTE_STREAM_SPLIT encodings; and the UNCOMPRESSED, SNAPPY, GZIP and ZSTD
// codecs. A column is read by its physical type: its logical type, such as
// DECIMAL or TIMESTAMP, is reported on Column but not applied.
//
// Nested columns (repeated fields and fields inside groups) and INT96
// columns are listed but cannot be read, and neither can encrypted files or
// the LZ4, LZ4_RAW, BROTLI and LZO codecs; these fail with ErrUnsupported.
// A malformed file fails with ErrFormat, never a panic, whatever its bytes.
// There is no writer.
//
// A File holds only the file's metadata and its io.ReaderAt, so readers of
// different columns may run in separate goroutines.
package parquet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/klauspost/compress/zstd"
)

// ErrFormat reports a malformed file.
var ErrFormat = errors.New("parquet: malformed file")

// ErrUnsupported reports a feature of the format that the package does not
// implement.
var ErrUnsupported = errors.New("parquet: unsupported feature")

// Type is the physical type of a column.
type Type uint8

// The physical types of parquet.thrift.
const (
	Boolean Type = iota
	Int32
	Int64
	Int96
	Float
	Double
	ByteArray
	FixedLenByteArray
)

var typeNames = [...]string{"BOOLEAN", "INT32", "INT64", "INT96", "FLOAT", "DOUBLE", "BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY"}

func (t Type) String() string {
	if int(t) < len(typeNames) {
		return typeNames[t]
	}
	return fmt.Sprintf("TYPE(%d)", uint8(t))
}

// width returns the number of bytes of a PLAIN value of a fixed-width type,
// or 0 for the other types.
func (t Type) width() int {
	switch t {
	case Int32, Float:
		return 4
	case Int64, Double:
		return 8
	case Int96:
		return 12
	}
	return 0
}

// Column describes a leaf column of the schema.
type Column struct {
	Name     string // the path in the schema, "a.b" for a nested column
	Type     Type
	Nullable bool
	// Nested is set for a column below a group or repeated, such as the
	// element of a list, which Reader cannot read.
	Nested bool
	Length int // the size of a FixedLenByteArray value
	// Logical is the name of the column's logical type, from the
	// LogicalType of the schema or, in files of older writers, its
	// ConvertedType: "DECIMAL", "DATE", "TIME", "TIMESTAMP", "STRING",
	// "ENUM", "UUID", "JSON", "BSON", "INTERVAL", "INT", "UINT", "LIST",
	// "MAP", "FLOAT16", or "" for none. The values are those of the
	// physical type: a DECIMAL is its unscaled integer, a DATE a number
	// of days and a TIMESTAMP one of units since the epoch.
	Logical string
	// Precision and Scale are those of a DECIMAL, whose value is the
	// unscaled integer divided by 10^Scale.
	Precision, Scale int
}

// leaf is a leaf of the schema with what the reader needs beyond Column.
type leaf struct {
	Column
	maxDef int
	maxRep int
}

// File is an open Parquet file. Its methods are safe for concurrent use.
type File struct {
	r       io.ReaderAt
	size    int64
	closer  io.Closer
	meta    fileMetaData
	leaves  []leaf
	columns []Column
	index   map[string]int
	footer  int64 // the offset of the footer, where the pages end

	zstdOnce sync.Once
	zstdDec  *zstd.Decoder
	zstdErr  error
}

const magic = "PAR1"

// Open parses the footer of the file that r holds, which is size bytes
// long. r must stay valid while the File is read and is not closed by Close.
func Open(r io.ReaderAt, size int64) (*File, error) {
	if size < 2*int64(len(magic))+4 {
		return nil, fmt.Errorf("%w: %d bytes", ErrFormat, size)
	}
	f := &File{r: r, size: size}
	var head [4]byte
	if err := f.readAt(head[:], 0); err != nil {
		return nil, err
	}
	if string(head[:]) != magic {
		return nil, fmt.Errorf("%w: magic %q", ErrFormat, head[:])
	}
	var tail [8]byte
	if err := f.readAt(tail[:], size-8); err != nil {
		return nil, err
	}
	switch string(tail[4:]) {
	case magic:
	case "PARE":
		return nil, fmt.Errorf("%w: encrypted footer", ErrUnsupported)
	default:
		return nil, fmt.Errorf("%w: footer magic %q", ErrFormat, tail[4:])
	}
	n := int64(binary.LittleEndian.Uint32(tail[:4]))
	if n > size-12 {
		return nil, fmt.Errorf("%w: footer of %d bytes in %d-byte file", ErrFormat, n, size)
	}
	f.footer = size - 8 - n
	footer := make([]byte, n)
	if err := f.readAt(footer, f.footer); err != nil {
		return nil, err
	}
	meta, err := parseFileMetaData(footer)
	if err != nil {
		return nil, formatErr("footer", err)
	}
	if meta.encrypted {
		return nil, fmt.Errorf("%w: encrypted columns", ErrUnsupported)
	}
	if meta.numRows < 0 {
		return nil, fmt.Errorf("%w: %d rows", ErrFormat, meta.numRows)
	}
	f.meta = meta
	if f.leaves, err = buildSchema(meta.schema); err != nil {
		return nil, err
	}
	f.columns = make([]Column, len(f.leaves))
	f.index = make(map[string]int, len(f.leaves))
	for i, l := range f.leaves {
		f.columns[i] = l.Column
		f.index[l.Name] = i
	}
	for g := range meta.rowGroups {
		rg := &meta.rowGroups[g]
		if rg.numRows < 0 {
			return nil, fmt.Errorf("%w: row group %d has %d rows", ErrFormat, g, rg.numRows)
		}
		if len(rg.columns) != len(f.leaves) {
			return nil, fmt.Errorf("%w: row group %d has %d column chunks for %d columns", ErrFormat, g, len(rg.columns), len(f.leaves))
		}
		for c := range rg.columns {
			ch := &rg.columns[c]
			if !ch.hasMeta {
				continue
			}
			if ch.meta.typ != f.leaves[c].Type {
				return nil, fmt.Errorf("%w: column chunk %d of row group %d is %v, column %q is %v", ErrFormat, c, g, ch.meta.typ, f.leaves[c].Name, f.leaves[c].Type)
			}
			if len(ch.meta.path) > 0 && strings.Join(ch.meta.path, ".") != f.leaves[c].Name {
				return nil, fmt.Errorf("%w: column chunk %d of row group %d holds %q, want %q", ErrFormat, c, g, strings.Join(ch.meta.path, "."), f.leaves[c].Name)
			}
			if ch.meta.numValues < 0 {
				return nil, fmt.Errorf("%w: column %q of row group %d has %d values", ErrFormat, f.leaves[c].Name, g, ch.meta.numValues)
			}
		}
	}
	return f, nil
}

// OpenFile opens the named file. Close closes it.
func OpenFile(path string) (*File, error) {
	of, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := of.Stat()
	if err != nil {
		of.Close()
		return nil, err
	}
	f, err := Open(of, info.Size())
	if err != nil {
		of.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	f.closer = of
	return f, nil
}

// Close releases the decoders and closes the file when OpenFile opened it.
// Readers of the file must not be used afterwards.
func (f *File) Close() error {
	if f.zstdDec != nil {
		f.zstdDec.Close()
	}
	if f.closer != nil {
		return f.closer.Close()
	}
	return nil
}

// Rows returns the number of rows in the file.
func (f *File) Rows() int64 { return f.meta.numRows }

// RowGroups returns the number of row groups.
func (f *File) RowGroups() int { return len(f.meta.rowGroups) }

// RowGroupRows returns the number of rows in a row group.
func (f *File) RowGroupRows(group int) int64 { return f.meta.rowGroups[group].numRows }

// Columns returns the leaf columns of the schema, in schema order.
func (f *File) Columns() []Column { return slices.Clone(f.columns) }

// Column returns the index of the named column.
func (f *File) Column(name string) (int, bool) {
	i, ok := f.index[name]
	return i, ok
}

// Metadata returns the key-value metadata of the footer, which tools use
// for their own schemas, like the "pandas" key. It must not be modified.
func (f *File) Metadata() map[string]string { return f.meta.keyValue }

// CreatedBy returns the writer's description of itself.
func (f *File) CreatedBy() string { return f.meta.createdBy }

// readAt fills b from offset off, failing with ErrFormat when the file ends
// first.
func (f *File) readAt(b []byte, off int64) error {
	n, err := f.r.ReadAt(b, off)
	if n == len(b) {
		return nil
	}
	if err == nil || err == io.EOF {
		return fmt.Errorf("%w: %d of %d bytes at offset %d", ErrFormat, n, len(b), off)
	}
	return fmt.Errorf("parquet: read %d bytes at offset %d: %w", len(b), off, err)
}

// zstd returns the file's shared Zstandard decoder.
func (f *File) zstd() (*zstd.Decoder, error) {
	f.zstdOnce.Do(func() {
		f.zstdDec, f.zstdErr = zstd.NewReader(nil, zstd.WithDecoderMaxMemory(maxPageSize))
	})
	return f.zstdDec, f.zstdErr
}

// formatErr wraps an error from parsing a thrift message as ErrFormat.
func formatErr(what string, err error) error {
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return fmt.Errorf("%w: %s is truncated", ErrFormat, what)
	}
	return fmt.Errorf("%w: %s: %v", ErrFormat, what, err)
}

// buildSchema walks the flat, depth-first schema list and returns its
// leaves, named by their dotted paths.
func buildSchema(elems []schemaElement) ([]leaf, error) {
	if len(elems) == 0 {
		return nil, fmt.Errorf("%w: empty schema", ErrFormat)
	}
	var leaves []leaf
	pos := 1
	var visit func(prefix string, depth, def, rep int) error
	visit = func(prefix string, depth, def, rep int) error {
		if pos >= len(elems) {
			return fmt.Errorf("%w: schema element %d is missing", ErrFormat, pos)
		}
		e := &elems[pos]
		pos++
		name := e.name
		if prefix != "" {
			name = prefix + "." + name
		}
		switch e.repetition {
		case repOptional:
			def++
		case repRepeated:
			def++
			rep++
		}
		if e.numChildren > 0 {
			for range e.numChildren {
				if err := visit(name, depth+1, def, rep); err != nil {
					return err
				}
			}
			return nil
		}
		if !e.hasType {
			return fmt.Errorf("%w: column %q has no type", ErrFormat, name)
		}
		l := leaf{Column: Column{Name: name, Type: e.typ, Nullable: def > 0, Nested: depth > 0 || rep > 0}, maxDef: def, maxRep: rep}
		logical, precision, scale := e.logicalType()
		if logical == "DECIMAL" && (precision < 1 || scale < 0 || scale > precision) {
			return fmt.Errorf("%w: column %q is a DECIMAL of precision %d and scale %d", ErrFormat, name, precision, scale)
		}
		l.Logical, l.Precision, l.Scale = logical, int(precision), int(scale)
		if e.typ == FixedLenByteArray {
			if e.typeLength <= 0 {
				return fmt.Errorf("%w: column %q has length %d", ErrFormat, name, e.typeLength)
			}
			l.Length = int(e.typeLength)
		}
		leaves = append(leaves, l)
		return nil
	}
	for range elems[0].numChildren {
		if err := visit("", 0, 0, 0); err != nil {
			return nil, err
		}
	}
	if elems[0].numChildren == 0 && len(elems) > 1 {
		return nil, fmt.Errorf("%w: root with no children in a schema of %d elements", ErrFormat, len(elems))
	}
	if pos != len(elems) {
		return nil, fmt.Errorf("%w: %d schema elements outside the root", ErrFormat, len(elems)-pos)
	}
	return leaves, nil
}
