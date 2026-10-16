package parquet

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
)

// unsigned is the set of types bit-packed values are unpacked into.
type unsigned interface{ uint8 | uint32 | uint64 }

// zigzag undoes the zigzag mapping of signed integers to unsigned ones.
func zigzag(u uint64) int64 { return int64(u>>1) ^ -int64(u&1) }

// unpack decodes len(dst) values of width bits each from src, where values
// are packed from the least significant bit of each byte on. src must hold
// ceil(len(dst)*width/8) bytes, which callers check; it may hold more,
// which only makes the decoding faster.
func unpack[T unsigned](dst []T, src []byte, width uint) {
	switch {
	case width == 0:
		clear(dst)
	case width == 8:
		for i := range dst {
			dst[i] = T(src[i])
		}
	case width == 16:
		for i := range dst {
			dst[i] = T(binary.LittleEndian.Uint16(src[2*i:]))
		}
	case width == 32:
		for i := range dst {
			dst[i] = T(binary.LittleEndian.Uint32(src[4*i:]))
		}
	case width == 64:
		for i := range dst {
			dst[i] = T(binary.LittleEndian.Uint64(src[8*i:]))
		}
	case width <= 57:
		// A value lies within the 8 bytes from the byte its first bit is
		// in, so while those bytes are within src it is one load, shift
		// and mask. Groups of 8 values start at a byte, w bytes apart.
		mask := uint64(1)<<width - 1
		w := int(width)
		i, off := 0, 0
		for i+8 <= len(dst) && off+w+8 <= len(src) {
			g := src[off : off+w+8]
			d := dst[i : i+8 : i+8]
			d[0] = T(binary.LittleEndian.Uint64(g) & mask)
			d[1] = T(binary.LittleEndian.Uint64(g[w>>3:]) >> (width & 7) & mask)
			d[2] = T(binary.LittleEndian.Uint64(g[(2*w)>>3:]) >> ((2 * width) & 7) & mask)
			d[3] = T(binary.LittleEndian.Uint64(g[(3*w)>>3:]) >> ((3 * width) & 7) & mask)
			d[4] = T(binary.LittleEndian.Uint64(g[(4*w)>>3:]) >> ((4 * width) & 7) & mask)
			d[5] = T(binary.LittleEndian.Uint64(g[(5*w)>>3:]) >> ((5 * width) & 7) & mask)
			d[6] = T(binary.LittleEndian.Uint64(g[(6*w)>>3:]) >> ((6 * width) & 7) & mask)
			d[7] = T(binary.LittleEndian.Uint64(g[(7*w)>>3:]) >> ((7 * width) & 7) & mask)
			i += 8
			off += w
		}
		bit := uint(off) * 8
		for ; i < len(dst) && int(bit>>3)+8 <= len(src); i++ {
			dst[i] = T(binary.LittleEndian.Uint64(src[bit>>3:]) >> (bit & 7) & mask)
			bit += width
		}
		// The rest a byte at a time, through an accumulator that always
		// has room for one more byte.
		s := int(bit >> 3)
		var acc uint64
		var nbits uint
		if i < len(dst) && bit&7 != 0 {
			acc = uint64(src[s]) >> (bit & 7)
			nbits = 8 - bit&7
			s++
		}
		for ; i < len(dst); i++ {
			for nbits < width {
				acc |= uint64(src[s]) << nbits
				s++
				nbits += 8
			}
			dst[i] = T(acc & mask)
			acc >>= width
			nbits -= width
		}
	default:
		// Up to 64 bits, gathered a byte at a time.
		bit := 0
		for i := range dst {
			var v uint64
			for got := uint(0); got < width; {
				b := uint64(src[bit>>3]) >> (bit & 7)
				take := min(8-uint(bit&7), width-got)
				v |= (b & (1<<take - 1)) << got
				got += take
				bit += int(take)
			}
			dst[i] = T(v)
		}
	}
}

// rleDecoder decodes the RLE/bit-packing hybrid: a sequence of runs, each
// either one value repeated or groups of 8 bit-packed values. A zero value
// decodes nothing.
type rleDecoder[T unsigned] struct {
	data  []byte
	pos   int
	width uint

	rleLeft   int // values left in the current repeated run
	rleVal    T
	bpLeft    int  // values left in the current bit-packed run
	group     [8]T // a partly consumed group of bit-packed values
	groupLeft int
}

// next decodes up to len(dst) values and returns how many it decoded.
// Fewer than len(dst) means the data ended.
func (d *rleDecoder[T]) next(dst []T) (int, error) {
	n := 0
	for n < len(dst) {
		switch {
		case d.rleLeft > 0:
			m := min(d.rleLeft, len(dst)-n)
			fill := dst[n : n+m]
			for i := range fill {
				fill[i] = d.rleVal
			}
			n += m
			d.rleLeft -= m
		case d.groupLeft > 0:
			m := copy(dst[n:], d.group[8-d.groupLeft:])
			n += m
			d.groupLeft -= m
		case d.bpLeft > 0:
			w := int(d.width)
			// unpack gets the rest of the data, not just the run's bytes,
			// so that its wide loads stay in bounds up to the end.
			if groups := min(d.bpLeft, len(dst)-n) / 8; groups > 0 {
				unpack(dst[n:n+8*groups], d.data[d.pos:], d.width)
				d.pos += groups * w
				n += 8 * groups
				d.bpLeft -= 8 * groups
				continue
			}
			unpack(d.group[:], d.data[d.pos:], d.width)
			d.pos += w
			d.bpLeft -= 8
			d.groupLeft = 8
		default:
			if d.pos >= len(d.data) {
				return n, nil
			}
			if err := d.run(); err != nil {
				return n, err
			}
		}
	}
	return n, nil
}

// count decodes up to n values without keeping them and returns how many
// it decoded, fewer only when the data ends, and how many of those equal
// v. A repeated run is counted without being expanded, and bit-packed
// groups of width 1, the levels of an optional column, by their bits.
func (d *rleDecoder[T]) count(n int, v T) (decoded, equal int, err error) {
	var buf [256]T
	for decoded < n {
		switch groups := min(d.bpLeft, n-decoded) / 8; {
		case d.rleLeft > 0:
			k := min(d.rleLeft, n-decoded)
			if d.rleVal == v {
				equal += k
			}
			d.rleLeft -= k
			decoded += k
		case d.groupLeft == 0 && groups > 0 && d.width == 1 && v <= 1:
			ones := 0
			b := d.data[d.pos : d.pos+groups]
			for len(b) >= 8 {
				ones += bits.OnesCount64(binary.LittleEndian.Uint64(b))
				b = b[8:]
			}
			for _, x := range b {
				ones += bits.OnesCount8(x)
			}
			if v == 1 {
				equal += ones
			} else {
				equal += 8*groups - ones
			}
			d.pos += groups
			d.bpLeft -= 8 * groups
			decoded += 8 * groups
		case d.groupLeft > 0 || d.bpLeft > 0:
			// The values of the current run, through buf.
			k, err := d.next(buf[:min(len(buf), n-decoded, d.groupLeft+d.bpLeft)])
			for _, x := range buf[:k] {
				if x == v {
					equal++
				}
			}
			decoded += k
			if err != nil {
				return decoded, equal, err
			}
		case d.pos >= len(d.data):
			return decoded, equal, nil
		default:
			if err := d.run(); err != nil {
				return decoded, equal, err
			}
		}
	}
	return decoded, equal, nil
}

// run reads the header of the next run.
func (d *rleDecoder[T]) run() error {
	h, n := binary.Uvarint(d.data[d.pos:])
	if n <= 0 {
		return fmt.Errorf("%w: run header", ErrFormat)
	}
	d.pos += n
	count := h >> 1
	if count > math.MaxInt32 {
		return fmt.Errorf("%w: run of %d values", ErrFormat, count)
	}
	w := int(d.width)
	if h&1 == 0 {
		// A repeated value, stored in the width rounded up to whole bytes.
		nbytes := (w + 7) / 8
		if nbytes > len(d.data)-d.pos {
			return fmt.Errorf("%w: truncated run", ErrFormat)
		}
		var v uint64
		for i := range nbytes {
			v |= uint64(d.data[d.pos+i]) << (8 * i)
		}
		d.pos += nbytes
		if d.width < 64 && v>>d.width != 0 {
			return fmt.Errorf("%w: value %d in a %d-bit run", ErrFormat, v, d.width)
		}
		d.rleLeft, d.rleVal = int(count), T(v)
		return nil
	}
	// count groups of 8 bit-packed values, w bytes each.
	if w > 0 && int(count) > (len(d.data)-d.pos)/w {
		return fmt.Errorf("%w: truncated run", ErrFormat)
	}
	d.bpLeft = int(count) * 8
	return nil
}
