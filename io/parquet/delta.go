package parquet

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// maxDeltaBlock bounds the block size of DELTA_BINARY_PACKED, which writers
// set to 128 or 256 values.
const maxDeltaBlock = 1 << 20

// word is the set of types that hold the bits of int32 and int64 values.
type word interface{ uint32 | uint64 }

// deltaStream decodes DELTA_BINARY_PACKED values as they are read, so
// that what it holds does not grow with the number of values a page
// declares, which a page of width-0 miniblocks can make far larger than
// its bytes. The block headers and miniblocks are checked against the
// bytes as they are reached.
type deltaStream[T word] struct {
	data    []byte
	pos     int // the next block header or miniblock
	per, nb int // values per miniblock and miniblocks per block
	left    int // values not yet decoded
	prev    T   // the last value decoded
	first   bool

	minDelta T
	widths   []byte // the bit widths of the current block's miniblocks
	mini     int    // the next miniblock of the current block
	width    uint   // the bit width of the current miniblock
	bitPos   int    // the bytes of the current miniblock's next group of 8
	miniLeft int    // values of the current miniblock not yet decoded

	group     [8]T // a group decoded but not all returned
	groupPos  int
	groupSize int
}

// init starts decoding the n values of the DELTA_BINARY_PACKED data.
func (d *deltaStream[T]) init(data []byte, n int) error {
	*d = deltaStream[T]{data: data}
	next := func() (uint64, error) {
		v, m := binary.Uvarint(data[d.pos:])
		if m <= 0 {
			return 0, fmt.Errorf("%w: delta header", ErrFormat)
		}
		d.pos += m
		return v, nil
	}
	blockSize, err := next()
	if err != nil {
		return err
	}
	miniblocks, err := next()
	if err != nil {
		return err
	}
	total, err := next()
	if err != nil {
		return err
	}
	first, err := next()
	if err != nil {
		return err
	}
	if blockSize == 0 || blockSize > maxDeltaBlock || miniblocks == 0 || miniblocks > blockSize || blockSize%miniblocks != 0 || (blockSize/miniblocks)%8 != 0 {
		return fmt.Errorf("%w: delta blocks of %d values in %d miniblocks", ErrFormat, blockSize, miniblocks)
	}
	if total != uint64(n) {
		return fmt.Errorf("%w: delta block of %d values, want %d", ErrFormat, total, n)
	}
	d.per, d.nb = int(blockSize/miniblocks), int(miniblocks)
	d.mini = d.nb // the first miniblock starts a block
	d.left, d.first = n, n > 0
	d.prev = T(uint64(zigzag(first)))
	return nil
}

// bits is the width of T.
func (d *deltaStream[T]) bits() uint { return uint(unsafe.Sizeof(T(0))) * 8 }

// nextMiniblock moves to the next miniblock that holds values, reading
// the next block header when the block is done.
func (d *deltaStream[T]) nextMiniblock() error {
	if d.mini == d.nb {
		v, m := binary.Uvarint(d.data[d.pos:])
		if m <= 0 {
			return fmt.Errorf("%w: delta header", ErrFormat)
		}
		d.pos += m
		d.minDelta = T(uint64(zigzag(v)))
		if d.nb > len(d.data)-d.pos {
			return fmt.Errorf("%w: delta block is truncated", ErrFormat)
		}
		d.widths = d.data[d.pos : d.pos+d.nb]
		d.pos += d.nb
		d.mini = 0
	}
	w := uint(d.widths[d.mini])
	if w > d.bits() {
		return fmt.Errorf("%w: %d-bit deltas", ErrFormat, w)
	}
	count := min(d.per, d.left)
	if need := (count*int(w) + 7) / 8; need > len(d.data)-d.pos {
		return fmt.Errorf("%w: delta block is truncated", ErrFormat)
	}
	d.width, d.bitPos, d.miniLeft = w, d.pos, count
	// The last miniblock is padded to its full size, or not.
	d.pos += min(d.per*int(w)/8, len(d.data)-d.pos)
	d.mini++
	return nil
}

// next decodes up to len(dst) values and returns how many it decoded:
// fewer only once all are.
func (d *deltaStream[T]) next(dst []T) (int, error) {
	n := 0
	for n < len(dst) {
		switch {
		case d.groupPos < d.groupSize:
			m := copy(dst[n:], d.group[d.groupPos:d.groupSize])
			d.groupPos += m
			n += m
		case d.left == 0:
			return n, nil
		case d.first:
			dst[n] = d.prev
			n++
			d.left--
			d.first = false
		case d.miniLeft == 0:
			if err := d.nextMiniblock(); err != nil {
				return n, err
			}
		default:
			// Whole groups of 8 straight into dst, or one group, of up
			// to 8, into d.group. unpack gets the rest of the data, not
			// just the miniblock's bytes, so that its wide loads stay in
			// bounds up to the end.
			out := dst[n : n+min(len(dst)-n, d.miniLeft)/8*8]
			if len(out) == 0 {
				d.groupSize, d.groupPos = min(8, d.miniLeft), 0
				out = d.group[:d.groupSize]
			} else {
				n += len(out)
			}
			unpack(out, d.data[d.bitPos:], d.width)
			prev, minDelta := d.prev, d.minDelta
			for k := range out {
				prev += minDelta + out[k]
				out[k] = prev
			}
			d.prev = prev
			d.bitPos += len(out) / 8 * int(d.width)
			d.miniLeft -= len(out)
			d.left -= len(out)
		}
	}
	return n, nil
}

// end returns the offset in the data of the first byte after the values,
// walking the block headers and miniblocks without decoding them. It is
// called on a stream that has decoded nothing yet.
func (d *deltaStream[T]) end() (int, error) {
	c := *d
	if c.first { // the first value is in the header
		c.left--
	}
	for c.left > 0 {
		if err := c.nextMiniblock(); err != nil {
			return 0, err
		}
		c.left -= c.miniLeft
	}
	return c.pos, nil
}

// deltaLengths reads n lengths from s into lens, which it grows, checking
// that each is a valid length.
func deltaLengths(s *deltaStream[uint32], lens []uint32, n int) ([]uint32, error) {
	lens = grow(lens, n)
	m, err := s.next(lens)
	if err != nil {
		return lens, err
	}
	if m < n {
		return lens, fmt.Errorf("%w: %d of %d delta lengths", ErrFormat, m, n)
	}
	for _, l := range lens {
		if l > math.MaxInt32 {
			return lens, fmt.Errorf("%w: length %d", ErrFormat, int32(l))
		}
	}
	return lens, nil
}

// deltaLengthByteArray loads a DELTA_LENGTH_BYTE_ARRAY page of n values: the
// lengths, then the values back to back. The lengths are decoded as the
// values are read.
func (r *Reader) deltaLengthByteArray(values []byte, n int) error {
	if err := r.lenStream.init(values, n); err != nil {
		return err
	}
	end, err := r.lenStream.end()
	if err != nil {
		return err
	}
	r.bytes, r.bytesOff = values[end:], 0
	r.kind = kindDeltaLength
	return nil
}

// deltaLengthValues returns the bytes of a DELTA_LENGTH_BYTE_ARRAY page and
// the spans of its next n values.
func (r *Reader) deltaLengthValues(n int) ([]byte, []uint32, error) {
	var err error
	if r.lens, err = deltaLengths(&r.lenStream, r.lens, n); err != nil {
		return nil, nil, fmt.Errorf("column %q: %w", r.name, err)
	}
	r.spans = grow(r.spans, 2*n)
	off := r.bytesOff
	for i, l := range r.lens {
		if int(l) > len(r.bytes)-off {
			return nil, nil, fmt.Errorf("%w: column %q: values of %d bytes or more in a %d-byte page", ErrFormat, r.name, off+int(l), len(r.bytes))
		}
		r.spans[2*i], r.spans[2*i+1] = uint32(off), uint32(off+int(l))
		off += int(l)
	}
	r.bytesOff = off
	return r.bytes, r.spans, nil
}

// deltaByteArray loads a DELTA_BYTE_ARRAY page of n values: the length of
// the prefix each value shares with the previous one, then the suffixes as
// DELTA_LENGTH_BYTE_ARRAY. The lengths are decoded, and the values
// materialized, as they are read, since shared prefixes can make the
// values far larger than the page.
func (r *Reader) deltaByteArray(values []byte, n int) error {
	if err := r.prefixStream.init(values, n); err != nil {
		return err
	}
	end, err := r.prefixStream.end()
	if err != nil {
		return err
	}
	rest := values[end:]
	if err := r.lenStream.init(rest, n); err != nil {
		return err
	}
	if end, err = r.lenStream.end(); err != nil {
		return err
	}
	r.suffixes, r.suffixOff = rest[end:], 0
	r.prev = r.prev[:0]
	r.kind = kindDeltaBytes
	return nil
}

// deltaBytes materializes the next n values of a DELTA_BYTE_ARRAY page into
// r.bytesBuf and returns it with their spans.
func (r *Reader) deltaBytes(n int) ([]byte, []uint32, error) {
	var err error
	if r.prefixes, err = deltaLengths(&r.prefixStream, r.prefixes, n); err != nil {
		return nil, nil, fmt.Errorf("column %q: %w", r.name, err)
	}
	if r.lens, err = deltaLengths(&r.lenStream, r.lens, n); err != nil {
		return nil, nil, fmt.Errorf("column %q: %w", r.name, err)
	}
	buf := append(r.bytesBuf[:0], r.prev...)
	prevStart, prevEnd := 0, len(buf)
	r.spans = grow(r.spans, 2*n)
	for i := range n {
		p, l := int(r.prefixes[i]), int(r.lens[i])
		if p > prevEnd-prevStart {
			return nil, nil, fmt.Errorf("%w: column %q: prefix of %d bytes after a %d-byte value", ErrFormat, r.name, p, prevEnd-prevStart)
		}
		if l > len(r.suffixes)-r.suffixOff {
			return nil, nil, fmt.Errorf("%w: column %q: suffixes of %d bytes or more in %d", ErrFormat, r.name, r.suffixOff+l, len(r.suffixes))
		}
		if len(buf)+p+l > maxPageSize {
			return nil, nil, fmt.Errorf("%w: column %q: values of more than %d bytes", ErrFormat, r.name, maxPageSize)
		}
		start := len(buf)
		buf = append(buf, buf[prevStart:prevStart+p]...)
		buf = append(buf, r.suffixes[r.suffixOff:r.suffixOff+l]...)
		r.suffixOff += l
		r.spans[2*i], r.spans[2*i+1] = uint32(start), uint32(len(buf))
		prevStart, prevEnd = start, len(buf)
	}
	r.prev = append(r.prev[:0], buf[prevStart:prevEnd]...)
	r.bytesBuf = buf
	return buf, r.spans[:2*n], nil
}

// byteStreamSplit32 gathers 4-byte values from the 4 byte streams in src,
// each len(dst) bytes long.
func byteStreamSplit32(dst []uint32, src []byte) {
	n := len(dst)
	s0, s1, s2, s3 := src[:n], src[n:2*n], src[2*n:3*n], src[3*n:4*n]
	for i := range dst {
		dst[i] = uint32(s0[i]) | uint32(s1[i])<<8 | uint32(s2[i])<<16 | uint32(s3[i])<<24
	}
}

// byteStreamSplit64 gathers 8-byte values from the 8 byte streams in src.
func byteStreamSplit64(dst []uint64, src []byte) {
	n := len(dst)
	for i := range dst {
		var v uint64
		for j := range 8 {
			v |= uint64(src[j*n+i]) << (8 * j)
		}
		dst[i] = v
	}
}

// byteStreamSplitBytes gathers values of width bytes from the width byte
// streams in src into dst, which is as long as src.
func byteStreamSplitBytes(dst, src []byte, width int) {
	n := len(src) / width
	for j := range width {
		s := src[j*n : (j+1)*n]
		for i, b := range s {
			dst[i*width+j] = b
		}
	}
}
