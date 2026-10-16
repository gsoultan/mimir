package parquet

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"unsafe"

	"github.com/klauspost/compress/gzip"
)

// kind says how the values of the current page are decoded.
type kind uint8

const (
	kindNone        kind = iota
	kindPlain            // fixed-width PLAIN values, decoded from values as they are read
	kindDict             // dictionary indices, decoded from rle as they are read
	kindDecoded          // decoded when the page was loaded, into u8s, u32s, u64s or bytes and spans
	kindBools            // RLE booleans, decoded from bools as they are read
	kindDelta            // DELTA_BINARY_PACKED integers, decoded from d32 or d64 as they are read
	kindDeltaLength      // DELTA_LENGTH_BYTE_ARRAY values, their lengths decoded from lenStream as they are read
	kindDeltaBytes       // DELTA_BYTE_ARRAY values, materialized from prefixStream, lenStream and suffixes as they are read
)

// dictionary holds the decoded dictionary page of a column chunk, in the
// same forms as decoded page values.
type dictionary struct {
	n         int
	u32s      []uint32
	u64s      []uint64
	bytes     []byte
	spans     []uint32 // the start and end in bytes of each entry
	fitsInt32 bool     // every Int64 entry fits in an int32
}

// Reader streams the values of one column chunk, page by page, decoding
// straight into the caller's slice. Each Read method returns the number of
// values read, and io.EOF once the chunk is exhausted; n may be less than
// len(dst) with a nil error until then. The nulls slice, when not nil,
// must be at least len(dst) long and receives true for each null value,
// where dst gets the zero value; when it is nil, a null becomes NaN for the
// float methods and an error wrapping ErrFormat for the others. The
// methods convert between types as documented on each; a column of a type
// a method does not accept fails with ErrUnsupported.
//
// Reading allocates per page at most, never per value, except that
// ReadStrings and ReadBytes allocate one buffer per call for the values
// they return; Reset moves a Reader to another chunk with its buffers.
// After an error, every call returns that error again. A Reader is not
// safe for concurrent use, but readers of the same File are independent.
type Reader struct {
	f    *File
	col  *leaf
	meta *columnMetaData
	name string

	chunk chunk
	left  int64 // values of the chunk not yet loaded into a page
	row   int64 // values read so far
	err   error

	hdr  pageHeader
	page []byte // the decompressed page
	gz   *gzip.Reader
	src  bytes.Reader

	dict    dictionary
	hasDict bool

	// The current page.
	kind         kind
	remaining    int               // values, nulls included, not yet read
	levels       rleDecoder[uint8] // the definition levels not yet read, when the page has nulls
	defs         []uint8           // the definition levels of the values being read
	pageNulls    int
	valPos       int    // non-null values read so far
	values       []byte // kindPlain
	rle          rleDecoder[uint32]
	idx          []uint32 // kindDict
	bools        rleDecoder[uint8]
	d32          deltaStream[uint32]
	d64          deltaStream[uint64]
	u8s          []uint8
	u32s         []uint32
	u64s         []uint64
	bytes        []byte
	bytesOff     int      // kindDeltaLength: the offset in bytes of the next value
	spans        []uint32 // the start and end in bytes of each value
	bytesBuf     []byte
	lenStream    deltaStream[uint32]
	prefixStream deltaStream[uint32]
	lens         []uint32
	prefixes     []uint32
	suffixes     []byte // kindDeltaBytes: the suffix bytes, read from suffixOff on
	suffixOff    int
	prev         []byte   // kindDeltaBytes: the last value materialized
	sel          []uint32 // the spans of the dictionary entries read
}

// Reader returns a reader of one column chunk. Readers of different
// columns of a group return the same number of values in the same order,
// so callers assemble rows by reading equal counts from each. Reading a
// nested or INT96 column, an encrypted one, or one stored with a codec the
// package lacks fails with ErrUnsupported.
func (f *File) Reader(group, column int) (*Reader, error) {
	r := &Reader{f: f}
	if err := r.Reset(group, column); err != nil {
		return nil, err
	}
	return r, nil
}

// Reset makes r read another column chunk of the same file, keeping its
// buffers, which saves allocating them again for every chunk of a large
// file. It fails as Reader does.
func (r *Reader) Reset(group, column int) error {
	f := r.f
	if group < 0 || group >= len(f.meta.rowGroups) {
		return fmt.Errorf("parquet: row group %d of %d", group, len(f.meta.rowGroups))
	}
	if column < 0 || column >= len(f.leaves) {
		return fmt.Errorf("parquet: column %d of %d", column, len(f.leaves))
	}
	l := &f.leaves[column]
	ch := &f.meta.rowGroups[group].columns[column]
	switch {
	case l.Nested:
		return fmt.Errorf("%w: nested column %q", ErrUnsupported, l.Name)
	case l.Type == Int96 || l.Type > FixedLenByteArray:
		return fmt.Errorf("%w: %v column %q", ErrUnsupported, l.Type, l.Name)
	case ch.encrypted:
		return fmt.Errorf("%w: encrypted column %q", ErrUnsupported, l.Name)
	case ch.filePath != "":
		return fmt.Errorf("%w: column %q stored in file %q", ErrUnsupported, l.Name, ch.filePath)
	case !ch.hasMeta:
		return fmt.Errorf("%w: column %q of row group %d has no metadata", ErrFormat, l.Name, group)
	}
	m := &ch.meta
	switch m.codec {
	case codecUncompressed, codecSnappy, codecGzip, codecZstd:
	default:
		return fmt.Errorf("%w: %v codec", ErrUnsupported, m.codec)
	}
	// The dictionary page precedes the data pages. Some writers leave its
	// offset unset or 0 even though there is one, which the first page
	// being a dictionary page covers.
	start := m.dataPageOffset
	if m.hasDictionaryOffset && m.dictionaryPageOffset > 0 && (m.dictionaryPageOffset < start || start == 0) {
		start = m.dictionaryPageOffset
	}
	if m.numValues > 0 && (start < int64(len(magic)) || start > f.footer || m.totalCompressedSize < 0 || m.totalCompressedSize > f.footer-start) {
		return fmt.Errorf("%w: column %q of row group %d spans %d bytes at offset %d in a file of %d bytes", ErrFormat, l.Name, group, m.totalCompressedSize, start, f.size)
	}
	r.col, r.meta, r.name = l, m, l.Name
	r.left, r.row, r.err = m.numValues, 0, nil
	r.chunk = chunk{f: f, pos: start, end: start + m.totalCompressedSize, buf: r.chunk.buf[:0]}
	r.hasDict = false
	r.kind, r.remaining = kindNone, 0
	return nil
}

// bufferBytes returns the bytes of the buffers r holds, for the tests.
func (r *Reader) bufferBytes() int {
	d := &r.dict
	return cap(r.chunk.buf) + cap(r.page) + cap(r.defs) + cap(r.u8s) + cap(r.bytesBuf) + cap(r.prev) + cap(d.bytes) +
		4*(cap(r.idx)+cap(r.u32s)+cap(r.spans)+cap(r.lens)+cap(r.prefixes)+cap(r.sel)+cap(d.u32s)+cap(d.spans)) +
		8*(cap(r.u64s)+cap(d.u64s))
}

// number is the set of types the numeric Read methods fill.
type number interface {
	int32 | int64 | float32 | float64
}

func isInt32[T number]() bool {
	var zero T
	_, ok := any(zero).(int32)
	return ok
}

// ReadFloat32 reads Boolean (as 0 or 1), Int32, Int64, Float and Double
// columns, rounding values that float32 cannot hold.
func (r *Reader) ReadFloat32(dst []float32, nulls []bool) (int, error) {
	return readNumeric(r, dst, nulls, float32(math.NaN()), true)
}

// ReadFloat64 reads Boolean (as 0 or 1), Int32, Int64, Float and Double
// columns, rounding Int64 values that float64 cannot hold.
func (r *Reader) ReadFloat64(dst []float64, nulls []bool) (int, error) {
	return readNumeric(r, dst, nulls, math.NaN(), true)
}

// ReadInt32 reads Boolean (as 0 or 1), Int32 and Int64 columns. An Int64
// value that does not fit in an int32 fails with ErrUnsupported.
func (r *Reader) ReadInt32(dst []int32, nulls []bool) (int, error) {
	return readNumeric(r, dst, nulls, 0, false)
}

// ReadInt64 reads Boolean (as 0 or 1), Int32 and Int64 columns.
func (r *Reader) ReadInt64(dst []int64, nulls []bool) (int, error) {
	return readNumeric(r, dst, nulls, 0, false)
}

// readNumeric reads into dst, where nilNull is what a null becomes when
// nulls is nil if the method allows that.
func readNumeric[T number](r *Reader, dst []T, nulls []bool, nilNull T, floats bool) (int, error) {
	if err := r.begin(len(dst), nulls); err != nil {
		return 0, err
	}
	switch r.col.Type {
	case Boolean, Int32, Int64:
	case Float, Double:
		if !floats {
			return 0, r.reject("integers")
		}
	default:
		return 0, r.reject("numbers")
	}
	null := T(0)
	if nulls == nil {
		null = nilNull
	}
	narrow := isInt32[T]() && r.col.Type == Int64
	return read(r, dst, nulls, null, floats, func(dst []T) error { return decodeNumeric(r, dst, narrow) })
}

// ReadStrings reads ByteArray and FixedLenByteArray columns. The strings of
// one call share one allocation.
func (r *Reader) ReadStrings(dst []string, nulls []bool) (int, error) {
	if err := r.beginBytes(len(dst), nulls); err != nil {
		return 0, err
	}
	return read(r, dst, nulls, "", false, func(dst []string) error { return decodeStrings(r, dst) })
}

// ReadBytes reads ByteArray and FixedLenByteArray columns. The slices of one
// call are consecutive parts of one allocation, each capped at its own
// length so that appending to one copies it; a null is nil.
func (r *Reader) ReadBytes(dst [][]byte, nulls []bool) (int, error) {
	if err := r.beginBytes(len(dst), nulls); err != nil {
		return 0, err
	}
	return read(r, dst, nulls, nil, false, func(dst [][]byte) error { return decodeBytes(r, dst) })
}

func (r *Reader) begin(n int, nulls []bool) error {
	if r.err != nil {
		return r.err
	}
	if nulls != nil && len(nulls) < n {
		return fmt.Errorf("parquet: nulls slice of %d for %d values", len(nulls), n)
	}
	return nil
}

func (r *Reader) beginBytes(n int, nulls []bool) error {
	if err := r.begin(n, nulls); err != nil {
		return err
	}
	if r.col.Type != ByteArray && r.col.Type != FixedLenByteArray {
		return r.reject("bytes")
	}
	return nil
}

func (r *Reader) reject(as string) error {
	return fmt.Errorf("%w: reading %v column %q as %s", ErrUnsupported, r.col.Type, r.name, as)
}

// read fills dst page by page. decode decodes the next len(dst) non-null
// values of the page; read then spreads them over the null positions,
// which get null, or fails when nulls is nil and nilNull is false.
func read[T any](r *Reader, dst []T, nulls []bool, null T, nilNull bool, decode func([]T) error) (int, error) {
	total := 0
	for len(dst) > 0 {
		if r.remaining == 0 {
			if err := r.nextPage(); err != nil {
				if err == io.EOF {
					if total > 0 {
						return total, nil
					}
					return 0, io.EOF
				}
				r.err = err
				return total, err
			}
			continue
		}
		n := min(len(dst), r.remaining)
		valid := n
		var defs []uint8
		maxDef := uint8(r.col.maxDef)
		if r.pageNulls > 0 {
			r.defs = grow(r.defs, n)
			defs = r.defs
			// decodeLevels counted them, so they are all there.
			if m, err := r.levels.next(defs); err != nil || m < n {
				r.err = fmt.Errorf("%w: column %q: %d of %d definition levels", ErrFormat, r.name, m, n)
				return total, r.err
			}
			valid = 0
			for _, d := range defs {
				if d == maxDef {
					valid++
				}
			}
			if valid < n && nulls == nil && !nilNull {
				first := 0
				for defs[first] == maxDef {
					first++
				}
				r.err = fmt.Errorf("%w: null at row %d of column %q read without a nulls slice", ErrFormat, r.row+int64(first), r.name)
				return total, r.err
			}
		}
		if err := decode(dst[:valid]); err != nil {
			r.err = err
			return total, err
		}
		if valid < n {
			spread(dst[:n], valid, defs, maxDef, nulls, null)
		} else if nulls != nil {
			clear(nulls[:n])
		}
		r.remaining -= n
		r.row += int64(n)
		total += n
		dst = dst[n:]
		if nulls != nil {
			nulls = nulls[n:]
		}
	}
	return total, nil
}

// spread moves the valid values at the start of dst to the positions whose
// definition level is maxDef, filling the others with null, and records
// which positions are null in nulls when it is not nil. It walks backwards,
// so no value is overwritten before it is moved.
func spread[T any](dst []T, valid int, defs []uint8, maxDef uint8, nulls []bool, null T) {
	j := valid
	for i := len(dst) - 1; i >= 0; i-- {
		if defs[i] == maxDef {
			j--
			dst[i] = dst[j]
			if nulls != nil {
				nulls[i] = false
			}
		} else {
			dst[i] = null
			if nulls != nil {
				nulls[i] = true
			}
		}
	}
}

// decodeNumeric decodes the next len(dst) non-null values of the page.
func decodeNumeric[T number](r *Reader, dst []T, narrow bool) error {
	n := len(dst)
	if n == 0 {
		return nil
	}
	t := r.col.Type
	switch r.kind {
	case kindPlain:
		w := t.width()
		if err := plainTo(dst, r.values[r.valPos*w:(r.valPos+n)*w], t, narrow, r.name); err != nil {
			return err
		}
	case kindDict:
		if err := r.readIndices(n); err != nil {
			return err
		}
		if err := gatherTo(dst, r.idx, &r.dict, t); err != nil {
			return fmt.Errorf("%w: column %q: %v", ErrFormat, r.name, err)
		}
		if narrow && !r.dict.fitsInt32 {
			for _, k := range r.idx {
				if v := int64(r.dict.u64s[k]); int64(int32(v)) != v {
					return narrowErr(v, r.name)
				}
			}
		}
	case kindBools:
		r.u8s = grow(r.u8s, n)
		if m, err := r.bools.next(r.u8s); err != nil {
			return fmt.Errorf("column %q: %w", r.name, err)
		} else if m < n {
			return r.short("page", n)
		}
		for i, b := range r.u8s {
			dst[i] = T(b)
		}
	case kindDelta:
		var m int
		var err error
		if t == Int32 {
			r.u32s = grow(r.u32s, n)
			if m, err = r.d32.next(r.u32s); err == nil && m == n {
				convU32(dst, r.u32s, t)
			}
		} else {
			r.u64s = grow(r.u64s, n)
			if m, err = r.d64.next(r.u64s); err == nil && m == n {
				err = convU64(dst, r.u64s, t, narrow, r.name)
			}
		}
		if err != nil {
			return fmt.Errorf("column %q: %w", r.name, err)
		}
		if m < n {
			return r.short("page", n)
		}
	case kindDecoded:
		switch t {
		case Boolean:
			src := r.u8s[r.valPos : r.valPos+n]
			for i, b := range src {
				dst[i] = T(b)
			}
		case Int32, Float:
			convU32(dst, r.u32s[r.valPos:r.valPos+n], t)
		case Int64, Double:
			if err := convU64(dst, r.u64s[r.valPos:r.valPos+n], t, narrow, r.name); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("%w: column %q: no page loaded", ErrFormat, r.name)
	}
	r.valPos += n
	return nil
}

// readIndices decodes the next n dictionary indices into r.idx. The
// lookups check them against the dictionary.
func (r *Reader) readIndices(n int) error {
	r.idx = grow(r.idx, n)
	m, err := r.rle.next(r.idx)
	if err != nil {
		return fmt.Errorf("column %q: dictionary indices: %w", r.name, err)
	}
	if m < n {
		return fmt.Errorf("%w: column %q: %d of %d dictionary indices", ErrFormat, r.name, m, n)
	}
	return nil
}

func indexErr(k uint32, n int) error {
	return fmt.Errorf("dictionary index %d of %d entries", k, n)
}

func narrowErr(v int64, name string) error {
	return fmt.Errorf("%w: value %d of column %q does not fit in an int32", ErrUnsupported, v, name)
}

// plainTo decodes PLAIN values of type t from src.
func plainTo[T number](dst []T, src []byte, t Type, narrow bool, name string) error {
	switch t {
	case Int32:
		for i := range dst {
			dst[i] = T(int32(binary.LittleEndian.Uint32(src[4*i:])))
		}
	case Float:
		for i := range dst {
			dst[i] = T(math.Float32frombits(binary.LittleEndian.Uint32(src[4*i:])))
		}
	case Double:
		for i := range dst {
			dst[i] = T(math.Float64frombits(binary.LittleEndian.Uint64(src[8*i:])))
		}
	case Int64:
		if narrow {
			for i := range dst {
				v := int64(binary.LittleEndian.Uint64(src[8*i:]))
				if int64(int32(v)) != v {
					return narrowErr(v, name)
				}
				dst[i] = T(v)
			}
			return nil
		}
		for i := range dst {
			dst[i] = T(int64(binary.LittleEndian.Uint64(src[8*i:])))
		}
	}
	return nil
}

// gatherTo looks up dictionary entries of type t.
func gatherTo[T number](dst []T, idx []uint32, d *dictionary, t Type) error {
	n := uint32(d.n)
	switch t {
	case Int32:
		dict := d.u32s[:n]
		for i, k := range idx {
			if k >= n {
				return indexErr(k, d.n)
			}
			dst[i] = T(int32(dict[k]))
		}
	case Float:
		dict := d.u32s[:n]
		for i, k := range idx {
			if k >= n {
				return indexErr(k, d.n)
			}
			dst[i] = T(math.Float32frombits(dict[k]))
		}
	case Int64:
		dict := d.u64s[:n]
		for i, k := range idx {
			if k >= n {
				return indexErr(k, d.n)
			}
			dst[i] = T(int64(dict[k]))
		}
	case Double:
		dict := d.u64s[:n]
		for i, k := range idx {
			if k >= n {
				return indexErr(k, d.n)
			}
			dst[i] = T(math.Float64frombits(dict[k]))
		}
	}
	return nil
}

// convU32 converts the bits of Int32 or Float values.
func convU32[T number](dst []T, src []uint32, t Type) {
	if t == Float {
		for i, b := range src {
			dst[i] = T(math.Float32frombits(b))
		}
		return
	}
	for i, b := range src {
		dst[i] = T(int32(b))
	}
}

// convU64 converts the bits of Int64 or Double values.
func convU64[T number](dst []T, src []uint64, t Type, narrow bool, name string) error {
	switch {
	case t == Double:
		for i, b := range src {
			dst[i] = T(math.Float64frombits(b))
		}
	case narrow:
		for i, b := range src {
			v := int64(b)
			if int64(int32(v)) != v {
				return narrowErr(v, name)
			}
			dst[i] = T(v)
		}
	default:
		for i, b := range src {
			dst[i] = T(int64(b))
		}
	}
	return nil
}

// byteValues returns the bytes of the page and the spans, two per value, of
// its next n non-null values.
func (r *Reader) byteValues(n int) ([]byte, []uint32, error) {
	switch r.kind {
	case kindDecoded:
		return r.bytes, r.spans[2*r.valPos : 2*(r.valPos+n)], nil
	case kindDict:
		if err := r.readIndices(n); err != nil {
			return nil, nil, err
		}
		r.sel = grow(r.sel, 2*n)
		spans := r.dict.spans[:2*r.dict.n]
		for i, k := range r.idx {
			if int(k) >= r.dict.n {
				return nil, nil, fmt.Errorf("%w: column %q: %v", ErrFormat, r.name, indexErr(k, r.dict.n))
			}
			r.sel[2*i], r.sel[2*i+1] = spans[2*k], spans[2*k+1]
		}
		return r.dict.bytes, r.sel, nil
	case kindDeltaLength:
		return r.deltaLengthValues(n)
	case kindDeltaBytes:
		return r.deltaBytes(n)
	}
	return nil, nil, fmt.Errorf("%w: column %q: no page loaded", ErrFormat, r.name)
}

// decodeStrings copies the next len(dst) non-null values into one buffer
// and returns strings into it.
func decodeStrings(r *Reader, dst []string) error {
	if len(dst) == 0 {
		return nil
	}
	src, spans, err := r.byteValues(len(dst))
	if err != nil {
		return err
	}
	total := 0
	for i := range dst {
		total += int(spans[2*i+1] - spans[2*i])
	}
	buf := make([]byte, total)
	off := 0
	for i := range dst {
		m := copy(buf[off:], src[spans[2*i]:spans[2*i+1]])
		if m > 0 {
			dst[i] = unsafe.String(&buf[off], m)
		} else {
			dst[i] = ""
		}
		off += m
	}
	r.valPos += len(dst)
	return nil
}

// decodeBytes copies the next len(dst) non-null values into one buffer and
// returns slices of it.
func decodeBytes(r *Reader, dst [][]byte) error {
	if len(dst) == 0 {
		return nil
	}
	src, spans, err := r.byteValues(len(dst))
	if err != nil {
		return err
	}
	total := 0
	for i := range dst {
		total += int(spans[2*i+1] - spans[2*i])
	}
	buf := make([]byte, total)
	off := 0
	for i := range dst {
		m := copy(buf[off:], src[spans[2*i]:spans[2*i+1]])
		dst[i] = buf[off : off+m : off+m]
		off += m
	}
	r.valPos += len(dst)
	return nil
}
