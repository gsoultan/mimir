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

// deltaDecode decodes a DELTA_BINARY_PACKED block of exactly len(dst)
// values into dst, as the bits of int32 or int64 values, and returns the
// bytes that follow it.
func deltaDecode[T word](data []byte, dst []T) ([]byte, error) {
	bits := uint(unsafe.Sizeof(T(0))) * 8
	pos := 0
	next := func() (uint64, error) {
		v, n := binary.Uvarint(data[pos:])
		if n <= 0 {
			return 0, fmt.Errorf("%w: delta header", ErrFormat)
		}
		pos += n
		return v, nil
	}
	blockSize, err := next()
	if err != nil {
		return nil, err
	}
	miniblocks, err := next()
	if err != nil {
		return nil, err
	}
	total, err := next()
	if err != nil {
		return nil, err
	}
	first, err := next()
	if err != nil {
		return nil, err
	}
	if blockSize == 0 || blockSize > maxDeltaBlock || miniblocks == 0 || miniblocks > blockSize || blockSize%miniblocks != 0 || (blockSize/miniblocks)%8 != 0 {
		return nil, fmt.Errorf("%w: delta blocks of %d values in %d miniblocks", ErrFormat, blockSize, miniblocks)
	}
	if total != uint64(len(dst)) {
		return nil, fmt.Errorf("%w: delta block of %d values, want %d", ErrFormat, total, len(dst))
	}
	if len(dst) == 0 {
		return data[pos:], nil
	}
	per, nb := int(blockSize/miniblocks), int(miniblocks)
	prev := T(uint64(zigzag(first)))
	dst[0] = prev
	i := 1
	for i < len(dst) {
		md, err := next()
		if err != nil {
			return nil, err
		}
		minDelta := T(uint64(zigzag(md)))
		if nb > len(data)-pos {
			return nil, fmt.Errorf("%w: delta block is truncated", ErrFormat)
		}
		widths := data[pos : pos+nb]
		pos += nb
		for j := 0; j < nb && i < len(dst); j++ {
			w := uint(widths[j])
			if w > bits {
				return nil, fmt.Errorf("%w: %d-bit deltas", ErrFormat, w)
			}
			count := min(per, len(dst)-i)
			need := (count*int(w) + 7) / 8
			if need > len(data)-pos {
				return nil, fmt.Errorf("%w: delta block is truncated", ErrFormat)
			}
			out := dst[i : i+count]
			unpack(out, data[pos:], w) // the rest of the page, so wide loads stay in bounds
			for k := range out {
				prev += minDelta + out[k]
				out[k] = prev
			}
			// The last miniblock is padded to its full size.
			pos += min(per*int(w)/8, len(data)-pos)
			i += count
		}
	}
	return data[pos:], nil
}

// deltaLengths decodes the DELTA_BINARY_PACKED lengths at the start of
// data into lens and returns the bytes after them.
func deltaLengths(data []byte, lens []uint32) ([]byte, error) {
	rest, err := deltaDecode(data, lens)
	if err != nil {
		return nil, err
	}
	for _, l := range lens {
		if l > math.MaxInt32 {
			return nil, fmt.Errorf("%w: length %d", ErrFormat, int32(l))
		}
	}
	return rest, nil
}

// spansOf fills spans, two entries per value, for values of the given
// lengths stored back to back in limit bytes.
func spansOf(spans []uint32, lens []uint32, limit int) error {
	off := 0
	for i, l := range lens {
		if int(l) > limit-off {
			return fmt.Errorf("%w: values of %d bytes or more in a %d-byte page", ErrFormat, off+int(l), limit)
		}
		spans[2*i], spans[2*i+1] = uint32(off), uint32(off+int(l))
		off += int(l)
	}
	return nil
}

// deltaLengthByteArray loads a DELTA_LENGTH_BYTE_ARRAY page of n values: the
// lengths, then the values back to back.
func (r *Reader) deltaLengthByteArray(values []byte, n int) error {
	r.lens = grow(r.lens, n)
	rest, err := deltaLengths(values, r.lens)
	if err != nil {
		return err
	}
	r.spans = grow(r.spans, 2*n)
	if err := spansOf(r.spans, r.lens, len(rest)); err != nil {
		return err
	}
	r.bytes = rest
	r.kind = kindDecoded
	return nil
}

// deltaByteArray loads a DELTA_BYTE_ARRAY page of n values: the length of
// the prefix each value shares with the previous one, then the suffixes as
// DELTA_LENGTH_BYTE_ARRAY. The values themselves are materialized as they
// are read, since shared prefixes can make them far larger than the page.
func (r *Reader) deltaByteArray(values []byte, n int) error {
	r.prefixes = grow(r.prefixes, n)
	rest, err := deltaLengths(values, r.prefixes)
	if err != nil {
		return err
	}
	r.lens = grow(r.lens, n)
	if rest, err = deltaLengths(rest, r.lens); err != nil {
		return err
	}
	r.suffixes, r.suffixOff = rest, 0
	r.prev = r.prev[:0]
	r.kind = kindDeltaBytes
	return nil
}

// deltaBytes materializes the next n values of a DELTA_BYTE_ARRAY page into
// r.bytesBuf and returns it with their spans.
func (r *Reader) deltaBytes(n int) ([]byte, []uint32, error) {
	buf := append(r.bytesBuf[:0], r.prev...)
	prevStart, prevEnd := 0, len(buf)
	r.spans = grow(r.spans, 2*n)
	for i := range n {
		k := r.valPos + i
		p, l := int(r.prefixes[k]), int(r.lens[k])
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
