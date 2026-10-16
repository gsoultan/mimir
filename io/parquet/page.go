package parquet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/bits"
	"slices"

	"github.com/klauspost/compress/gzip"
	"github.com/klauspost/compress/snappy"
)

// maxPageSize bounds the decoded size of a page. Writers produce pages of
// about a megabyte.
const maxPageSize = 1 << 30

// chunkWindow is how much of a column chunk one read fetches at least, so
// that small pages and the page headers share reads. A page larger than
// the window is read at its own size, so a Reader holds about one page's
// compressed bytes, plus its decompressed bytes, whatever the size of the
// chunk.
const chunkWindow = 64 << 10

// chunk reads the bytes of one column chunk through a window.
type chunk struct {
	f        *File
	pos, end int64
	buf      []byte
	bufStart int64 // the offset in the file of buf[0]
}

// next returns the next n bytes of the chunk and consumes them. The slice
// is valid until the next call.
func (c *chunk) next(n int) ([]byte, error) {
	b, err := c.peek(n)
	if err != nil {
		return nil, err
	}
	if len(b) < n {
		return nil, fmt.Errorf("%w: page of %d bytes at offset %d runs past the column chunk", ErrFormat, n, c.pos)
	}
	c.pos += int64(n)
	return b, nil
}

// peek returns the next n bytes of the chunk without consuming them, or
// fewer at the end of the chunk. When the window holds only part of them,
// it keeps that part, moved to the front, and reads the rest after it, to
// fill a window of max(n, chunkWindow) bytes, so no byte of the chunk is
// read twice.
func (c *chunk) peek(n int) ([]byte, error) {
	if left := c.end - c.pos; int64(n) > left {
		n = int(left)
	}
	kept := 0
	if off := c.pos - c.bufStart; off >= 0 && off <= int64(len(c.buf)) {
		if int64(n) <= int64(len(c.buf))-off {
			return c.buf[off : off+int64(n)], nil
		}
		kept = copy(c.buf, c.buf[off:])
	}
	size := int(min(int64(max(n, chunkWindow)), c.end-c.pos))
	if refit(cap(c.buf), size) {
		buf := make([]byte, size)
		copy(buf, c.buf[:kept])
		c.buf = buf
	} else {
		c.buf = c.buf[:size]
	}
	if err := c.f.readAt(c.buf[kept:], c.pos+int64(kept)); err != nil {
		c.buf = c.buf[:0]
		return nil, err
	}
	c.bufStart = c.pos
	return c.buf[:n], nil
}

// refit reports whether a buffer of capacity c is reallocated to hold n
// bytes: when it is too small, or more than four times larger than n and
// the window, so that a Reader does not keep the buffers of one large
// page, such as a dictionary, for the smaller pages after it.
func refit(c, n int) bool { return c < n || c > 4*max(n, chunkWindow) }

// fit returns b resized to n bytes, reallocated as refit says.
func fit(b []byte, n int) []byte {
	if refit(cap(b), n) {
		return make([]byte, n)
	}
	return b[:n]
}

// grow returns b resized to n elements, reallocating only when it must.
func grow[T any](b []T, n int) []T {
	return slices.Grow(b[:0], n)[:n]
}

// nextPage loads the next data page of the chunk, reading the dictionary
// page on the way. It returns io.EOF once all the chunk's values are read.
func (r *Reader) nextPage() error {
	for {
		if r.left == 0 {
			return io.EOF
		}
		if r.chunk.pos >= r.chunk.end {
			return fmt.Errorf("%w: column %q ends with %d values to read", ErrFormat, r.name, r.left)
		}
		if err := r.readHeader(); err != nil {
			return err
		}
		h := &r.hdr
		if h.compressedSize < 0 || h.uncompressedSize < 0 || h.uncompressedSize > maxPageSize {
			return fmt.Errorf("%w: column %q: page of %d bytes, %d compressed", ErrFormat, r.name, h.uncompressedSize, h.compressedSize)
		}
		data, err := r.chunk.next(int(h.compressedSize))
		if err != nil {
			return err
		}
		switch h.typ {
		case dictionaryPage:
			if !h.hasDict {
				return fmt.Errorf("%w: column %q: dictionary page without its header", ErrFormat, r.name)
			}
			if r.hasDict {
				return fmt.Errorf("%w: column %q has two dictionary pages", ErrFormat, r.name)
			}
			if err := r.loadDictionary(data); err != nil {
				return err
			}
		case dataPage:
			if !h.hasData {
				return fmt.Errorf("%w: column %q: data page without its header", ErrFormat, r.name)
			}
			return r.loadV1(data)
		case dataPageV2:
			if !h.hasV2 {
				return fmt.Errorf("%w: column %q: data page without its header", ErrFormat, r.name)
			}
			return r.loadV2(data)
		}
		// Index pages and page types from newer versions are skipped.
	}
}

// readHeader parses the page header at the chunk's position, fetching more
// of the chunk until the whole header is in the window.
func (r *Reader) readHeader() error {
	want := 1 << 10
	for {
		b, err := r.chunk.peek(want)
		if err != nil {
			return err
		}
		n, err := parsePageHeader(b, &r.hdr)
		if err == nil {
			r.chunk.pos += int64(n)
			return nil
		}
		if errors.Is(err, io.ErrUnexpectedEOF) && len(b) == want {
			want *= 4
			continue
		}
		return formatErr(fmt.Sprintf("column %q: page header at offset %d", r.name, r.chunk.pos), err)
	}
}

// The most each codec expands its input, which bounds the size a page of
// compressed bytes may declare before its buffer is allocated: Snappy's
// longest copy, 64 bytes, takes 3 bytes, DEFLATE's best ratio is 1032 to
// 1, and a Zstandard block of 128 KiB takes 4 bytes as a run of one byte.
const (
	snappyExpansion  = 22
	deflateExpansion = 1032
	zstdExpansion    = 32 << 10
)

// expansion returns the most the chunk's codec expands its input.
func (r *Reader) expansion() int {
	switch r.meta.codec {
	case codecSnappy:
		return snappyExpansion
	case codecGzip:
		return deflateExpansion
	case codecZstd:
		return zstdExpansion
	}
	return 1
}

// decompress returns the size bytes that src holds, decoded with the
// chunk's codec into a buffer that the next page reuses. A size larger
// than the codec can expand src to fails before anything is allocated.
func (r *Reader) decompress(src []byte, size int) ([]byte, error) {
	if limit := int64(r.expansion())*int64(len(src)) + 64; r.meta.codec != codecUncompressed && int64(size) > limit {
		return nil, fmt.Errorf("%w: column %q: %v page of %d bytes declares %d", ErrFormat, r.name, r.meta.codec, len(src), size)
	}
	switch r.meta.codec {
	case codecUncompressed:
		if len(src) != size {
			return nil, fmt.Errorf("%w: column %q: uncompressed page of %d bytes declares %d", ErrFormat, r.name, len(src), size)
		}
		return src, nil
	case codecSnappy:
		if n, err := snappy.DecodedLen(src); err != nil || n != size {
			return nil, fmt.Errorf("%w: column %q: snappy page of %d bytes declares %d", ErrFormat, r.name, n, size)
		}
		r.page = fit(r.page, size)
		out, err := snappy.Decode(r.page, src)
		if err != nil {
			return nil, fmt.Errorf("%w: column %q: snappy: %v", ErrFormat, r.name, err)
		}
		if len(out) != size {
			return nil, fmt.Errorf("%w: column %q: snappy page of %d bytes declares %d", ErrFormat, r.name, len(out), size)
		}
		return out, nil
	case codecGzip:
		r.src.Reset(src)
		var err error
		if r.gz == nil {
			r.gz, err = gzip.NewReader(&r.src)
		} else {
			err = r.gz.Reset(&r.src)
		}
		if err != nil {
			return nil, fmt.Errorf("%w: column %q: gzip: %v", ErrFormat, r.name, err)
		}
		r.gz.Multistream(false)
		// Into a buffer grown as the stream yields bytes, so a page that
		// declares more than it holds costs no more than it holds.
		page := r.page[:0]
		if cap(page) > 4*max(size, chunkWindow) { // as refit drops a buffer
			page = nil
		}
		for len(page) < size {
			if len(page) == cap(page) {
				page = slices.Grow(page, min(size-len(page), max(len(page), chunkWindow)))
			}
			m, err := r.gz.Read(page[len(page):min(cap(page), size)])
			page = page[:len(page)+m]
			if err == io.EOF && len(page) < size {
				err = io.ErrUnexpectedEOF
			}
			if err != nil && err != io.EOF {
				r.page = page
				return nil, fmt.Errorf("%w: column %q: gzip: %v", ErrFormat, r.name, err)
			}
		}
		r.page = page
		var extra [1]byte
		if n, err := r.gz.Read(extra[:]); n != 0 || err != io.EOF {
			return nil, fmt.Errorf("%w: column %q: gzip page longer than %d bytes", ErrFormat, r.name, size)
		}
		return r.page, nil
	case codecZstd:
		dec, err := r.f.zstd()
		if err != nil {
			return nil, err
		}
		r.page = fit(r.page, size)
		out, err := dec.DecodeAll(src, r.page[:0])
		if err != nil {
			return nil, fmt.Errorf("%w: column %q: zstd: %v", ErrFormat, r.name, err)
		}
		r.page = out
		if len(out) != size {
			return nil, fmt.Errorf("%w: column %q: zstd page of %d bytes declares %d", ErrFormat, r.name, len(out), size)
		}
		return out, nil
	}
	return nil, fmt.Errorf("%w: %v codec", ErrUnsupported, r.meta.codec)
}

// loadV1 loads a data page V1: the definition levels with a length prefix
// when the column is optional, then the values, all compressed together.
func (r *Reader) loadV1(data []byte) error {
	h := &r.hdr.data
	n := int(h.numValues)
	if n < 0 || int64(n) > r.left {
		return fmt.Errorf("%w: column %q: page of %d values with %d to read", ErrFormat, r.name, n, r.left)
	}
	if n == 0 {
		r.remaining = 0
		return nil
	}
	page, err := r.decompress(data, int(r.hdr.uncompressedSize))
	if err != nil {
		return err
	}
	if r.col.maxDef > 0 {
		if h.defEncoding != encRLE {
			return fmt.Errorf("%w: %v definition levels", ErrUnsupported, h.defEncoding)
		}
		if len(page) < 4 {
			return fmt.Errorf("%w: column %q: page without definition levels", ErrFormat, r.name)
		}
		size := binary.LittleEndian.Uint32(page)
		if uint64(size) > uint64(len(page)-4) {
			return fmt.Errorf("%w: column %q: %d bytes of definition levels in a %d-byte page", ErrFormat, r.name, size, len(page))
		}
		if err := r.decodeLevels(page[4:4+size], n); err != nil {
			return err
		}
		page = page[4+size:]
	} else {
		r.pageNulls = 0
	}
	return r.setValues(h.encoding, page, n)
}

// loadV2 loads a data page V2: the levels, uncompressed and with their
// lengths in the header, then the values, compressed on their own.
func (r *Reader) loadV2(data []byte) error {
	h := &r.hdr.v2
	n := int(h.numValues)
	if n < 0 || int64(n) > r.left {
		return fmt.Errorf("%w: column %q: page of %d values with %d to read", ErrFormat, r.name, n, r.left)
	}
	if n == 0 {
		r.remaining = 0
		return nil
	}
	repLen, defLen := int(h.repLength), int(h.defLength)
	if repLen < 0 || defLen < 0 || repLen+defLen > len(data) {
		return fmt.Errorf("%w: column %q: %d+%d bytes of levels in a %d-byte page", ErrFormat, r.name, repLen, defLen, len(data))
	}
	levels := data[repLen : repLen+defLen]
	rest := data[repLen+defLen:]
	size := int(r.hdr.uncompressedSize) - repLen - defLen
	if size < 0 {
		return fmt.Errorf("%w: column %q: page of %d bytes with %d bytes of levels", ErrFormat, r.name, r.hdr.uncompressedSize, repLen+defLen)
	}
	var values []byte
	if h.isCompressed {
		var err error
		if values, err = r.decompress(rest, size); err != nil {
			return err
		}
	} else {
		if len(rest) != size {
			return fmt.Errorf("%w: column %q: page values of %d bytes declare %d", ErrFormat, r.name, len(rest), size)
		}
		values = rest
	}
	if r.col.maxDef > 0 {
		if err := r.decodeLevels(levels, n); err != nil {
			return err
		}
	} else {
		r.pageNulls = 0
	}
	return r.setValues(h.encoding, values, n)
}

// decodeLevels starts decoding the n definition levels of a page, which
// are read with its values, and counts its nulls. A page without nulls,
// whose levels are one run of maxDef, is recognized without decoding them;
// otherwise counting them takes a pass over their bytes that expands no
// run, so a page cannot make the reader hold more levels than are read.
func (r *Reader) decodeLevels(levels []byte, n int) error {
	width := uint(bits.Len(uint(r.col.maxDef)))
	if h, m := binary.Uvarint(levels); m > 0 && h&1 == 0 && h>>1 >= uint64(n) {
		nbytes := int(width+7) / 8
		if m+nbytes <= len(levels) {
			var v uint64
			for i := range nbytes {
				v |= uint64(levels[m+i]) << (8 * i)
			}
			if v == uint64(r.col.maxDef) {
				r.pageNulls = 0
				return nil
			}
		}
	}
	r.levels = rleDecoder[uint8]{data: levels, width: width}
	count := r.levels
	m, defined, err := count.count(n, uint8(r.col.maxDef))
	if err != nil {
		return fmt.Errorf("column %q: definition levels: %w", r.name, err)
	}
	if m < n {
		return fmt.Errorf("%w: column %q: %d of %d definition levels", ErrFormat, r.name, m, n)
	}
	r.pageNulls = n - defined
	return nil
}

// loadDictionary decodes a dictionary page, whose values are PLAIN.
func (r *Reader) loadDictionary(data []byte) error {
	h := &r.hdr.dict
	n := int(h.numValues)
	if n < 0 {
		return fmt.Errorf("%w: column %q: dictionary of %d values", ErrFormat, r.name, n)
	}
	if h.encoding != encPlain && h.encoding != encPlainDictionary {
		return fmt.Errorf("%w: %v dictionary page", ErrUnsupported, h.encoding)
	}
	page, err := r.decompress(data, int(r.hdr.uncompressedSize))
	if err != nil {
		return err
	}
	d := &r.dict
	switch t := r.col.Type; t {
	case Int32, Float:
		if n > len(page)/4 {
			return r.short("dictionary", n)
		}
		d.u32s = grow(d.u32s, n)
		for i := range d.u32s {
			d.u32s[i] = binary.LittleEndian.Uint32(page[4*i:])
		}
	case Int64, Double:
		if n > len(page)/8 {
			return r.short("dictionary", n)
		}
		d.u64s = grow(d.u64s, n)
		d.fitsInt32 = true
		for i := range d.u64s {
			v := binary.LittleEndian.Uint64(page[8*i:])
			d.u64s[i] = v
			if int64(int32(v)) != int64(v) {
				d.fitsInt32 = false
			}
		}
	case ByteArray:
		if n > len(page)/4 { // each entry has a 4-byte length
			return r.short("dictionary", n)
		}
		d.bytes = d.bytes[:0]
		d.spans = grow(d.spans, 2*n)
		off := 0
		for i := range n {
			if len(page)-off < 4 {
				return r.short("dictionary", n)
			}
			l := binary.LittleEndian.Uint32(page[off:])
			off += 4
			if uint64(l) > uint64(len(page)-off) {
				return r.short("dictionary", n)
			}
			d.spans[2*i] = uint32(len(d.bytes))
			d.bytes = append(d.bytes, page[off:off+int(l)]...)
			d.spans[2*i+1] = uint32(len(d.bytes))
			off += int(l)
		}
	case FixedLenByteArray:
		l := r.col.Length
		if n > len(page)/l {
			return r.short("dictionary", n)
		}
		d.bytes = append(d.bytes[:0], page[:n*l]...)
		d.spans = grow(d.spans, 2*n)
		for i := range n {
			d.spans[2*i], d.spans[2*i+1] = uint32(i*l), uint32((i+1)*l)
		}
	default:
		return fmt.Errorf("%w: dictionary of %v values", ErrUnsupported, t)
	}
	d.n = n
	r.hasDict = true
	return nil
}

// short reports a page too small for the values it declares.
func (r *Reader) short(what string, n int) error {
	return fmt.Errorf("%w: column %q: %s too short for %d values", ErrFormat, r.name, what, n)
}

// setValues prepares the n values of a page, of which pageNulls are null,
// for the Read methods: fixed-width PLAIN values and dictionary indices are
// decoded as they are read, everything else is decoded now.
func (r *Reader) setValues(enc encoding, values []byte, n int) error {
	valid := n - r.pageNulls
	r.remaining, r.valPos = n, 0
	r.left -= int64(n)
	t := r.col.Type
	switch enc {
	case encPlain:
		switch t {
		case Boolean:
			if (valid+7)/8 > len(values) {
				return r.short("page", valid)
			}
			r.u8s = grow(r.u8s, valid)
			for i := range r.u8s {
				r.u8s[i] = values[i>>3] >> (i & 7) & 1
			}
			r.kind = kindDecoded
		case Int32, Int64, Float, Double:
			if valid > len(values)/t.width() {
				return r.short("page", valid)
			}
			r.values = values
			r.kind = kindPlain
		case ByteArray:
			if valid > len(values)/4 { // each value has a 4-byte length
				return r.short("page", valid)
			}
			r.spans = grow(r.spans, 2*valid)
			off := 0
			for i := range valid {
				if len(values)-off < 4 {
					return r.short("page", valid)
				}
				l := binary.LittleEndian.Uint32(values[off:])
				off += 4
				if uint64(l) > uint64(len(values)-off) {
					return r.short("page", valid)
				}
				r.spans[2*i], r.spans[2*i+1] = uint32(off), uint32(off+int(l))
				off += int(l)
			}
			r.bytes = values
			r.kind = kindDecoded
		case FixedLenByteArray:
			l := r.col.Length
			if valid > len(values)/l {
				return r.short("page", valid)
			}
			r.spans = grow(r.spans, 2*valid)
			for i := range valid {
				r.spans[2*i], r.spans[2*i+1] = uint32(i*l), uint32((i+1)*l)
			}
			r.bytes = values
			r.kind = kindDecoded
		default:
			return fmt.Errorf("%w: PLAIN %v values", ErrUnsupported, t)
		}
	case encRLE:
		if t != Boolean {
			return fmt.Errorf("%w: column %q: RLE values in a %v column", ErrFormat, r.name, t)
		}
		if len(values) < 4 {
			return r.short("page", valid)
		}
		size := binary.LittleEndian.Uint32(values)
		if uint64(size) > uint64(len(values)-4) {
			return r.short("page", valid)
		}
		r.bools = rleDecoder[uint8]{data: values[4 : 4+size], width: 1}
		r.kind = kindBools
	case encPlainDictionary, encRLEDictionary:
		if !r.hasDict {
			return fmt.Errorf("%w: column %q: dictionary-encoded page before the dictionary page", ErrFormat, r.name)
		}
		r.rle = rleDecoder[uint32]{}
		if valid > 0 {
			if len(values) == 0 {
				return r.short("page", valid)
			}
			w := uint(values[0])
			if w > 32 {
				return fmt.Errorf("%w: column %q: %d-bit dictionary indices", ErrFormat, r.name, w)
			}
			r.rle = rleDecoder[uint32]{data: values[1:], width: w}
		}
		r.kind = kindDict
	case encDeltaBinaryPacked:
		var err error
		switch t {
		case Int32:
			err = r.d32.init(values, valid)
		case Int64:
			err = r.d64.init(values, valid)
		default:
			return fmt.Errorf("%w: column %q: DELTA_BINARY_PACKED values in a %v column", ErrFormat, r.name, t)
		}
		if err != nil {
			return fmt.Errorf("column %q: %w", r.name, err)
		}
		r.kind = kindDelta
	case encDeltaLengthByteArray:
		if t != ByteArray {
			return fmt.Errorf("%w: column %q: DELTA_LENGTH_BYTE_ARRAY values in a %v column", ErrFormat, r.name, t)
		}
		if err := r.deltaLengthByteArray(values, valid); err != nil {
			return fmt.Errorf("column %q: %w", r.name, err)
		}
	case encDeltaByteArray:
		if t != ByteArray && t != FixedLenByteArray {
			return fmt.Errorf("%w: column %q: DELTA_BYTE_ARRAY values in a %v column", ErrFormat, r.name, t)
		}
		if err := r.deltaByteArray(values, valid); err != nil {
			return fmt.Errorf("column %q: %w", r.name, err)
		}
	case encByteStreamSplit:
		switch t {
		case Int32, Float:
			if len(values) != 4*valid {
				return r.short("page", valid)
			}
			r.u32s = grow(r.u32s, valid)
			byteStreamSplit32(r.u32s, values)
		case Int64, Double:
			if len(values) != 8*valid {
				return r.short("page", valid)
			}
			r.u64s = grow(r.u64s, valid)
			byteStreamSplit64(r.u64s, values)
		case FixedLenByteArray:
			l := r.col.Length
			if len(values) != l*valid {
				return r.short("page", valid)
			}
			r.bytesBuf = grow(r.bytesBuf, len(values))
			byteStreamSplitBytes(r.bytesBuf, values, l)
			r.spans = grow(r.spans, 2*valid)
			for i := range valid {
				r.spans[2*i], r.spans[2*i+1] = uint32(i*l), uint32((i+1)*l)
			}
			r.bytes = r.bytesBuf
		default:
			return fmt.Errorf("%w: column %q: BYTE_STREAM_SPLIT values in a %v column", ErrFormat, r.name, t)
		}
		r.kind = kindDecoded
	default:
		return fmt.Errorf("%w: %v encoding", ErrUnsupported, enc)
	}
	return nil
}
