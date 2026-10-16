package parquet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Thrift compact protocol type ids.
const (
	ttStop   = 0
	ttTrue   = 1
	ttFalse  = 2
	ttByte   = 3
	ttI16    = 4
	ttI32    = 5
	ttI64    = 6
	ttDouble = 7
	ttBinary = 8
	ttList   = 9
	ttSet    = 10
	ttMap    = 11
	ttStruct = 12
	ttUUID   = 13
)

// maxDepth bounds the nesting of structs and containers so that a hostile
// footer cannot exhaust the stack.
const maxDepth = 64

// decoder reads the thrift compact protocol from a byte slice. Running out
// of bytes yields io.ErrUnexpectedEOF so that a caller holding only part of
// a message can fetch more; other malformed input yields a plain error.
// Callers wrap both as ErrFormat.
type decoder struct {
	buf   []byte
	pos   int
	depth int
}

func (d *decoder) byte() (byte, error) {
	if d.pos >= len(d.buf) {
		return 0, io.ErrUnexpectedEOF
	}
	b := d.buf[d.pos]
	d.pos++
	return b, nil
}

func (d *decoder) uvarint() (uint64, error) {
	v, n := binary.Uvarint(d.buf[d.pos:])
	if n <= 0 {
		if n == 0 {
			return 0, io.ErrUnexpectedEOF
		}
		return 0, errors.New("varint overflows 64 bits")
	}
	d.pos += n
	return v, nil
}

// varint reads a zigzag-encoded signed integer.
func (d *decoder) varint() (int64, error) {
	u, err := d.uvarint()
	return zigzag(u), err
}

func (d *decoder) i32() (int32, error) {
	v, err := d.varint()
	if err != nil {
		return 0, err
	}
	if v < math.MinInt32 || v > math.MaxInt32 {
		return 0, fmt.Errorf("i32 field holds %d", v)
	}
	return int32(v), nil
}

func (d *decoder) binary() ([]byte, error) {
	n, err := d.uvarint()
	if err != nil {
		return nil, err
	}
	if n > uint64(len(d.buf)-d.pos) {
		return nil, io.ErrUnexpectedEOF
	}
	b := d.buf[d.pos : d.pos+int(n)]
	d.pos += int(n)
	return b, nil
}

// field reads a field header and returns the field's type and id. The type
// is ttStop at the end of the struct. last is the id of the previous field
// of the struct, which short headers encode a delta from.
func (d *decoder) field(last int16) (typ byte, id int16, err error) {
	b, err := d.byte()
	if err != nil {
		return 0, 0, err
	}
	typ = b & 0x0f
	if typ == ttStop {
		return ttStop, 0, nil
	}
	if delta := b >> 4; delta != 0 {
		return typ, last + int16(delta), nil
	}
	v, err := d.varint()
	if err != nil {
		return 0, 0, err
	}
	if v < math.MinInt16 || v > math.MaxInt16 {
		return 0, 0, fmt.Errorf("field id %d", v)
	}
	return typ, int16(v), nil
}

// strct reads a struct, calling fn for each field until the stop field. fn
// reads the field's value, or skips it with skip.
func (d *decoder) strct(fn func(id int16, typ byte) error) error {
	if d.depth >= maxDepth {
		return fmt.Errorf("nesting deeper than %d", maxDepth)
	}
	d.depth++
	var last int16
	for {
		typ, id, err := d.field(last)
		if err != nil {
			return err
		}
		if typ == ttStop {
			d.depth--
			return nil
		}
		if err := fn(id, typ); err != nil {
			return err
		}
		last = id
	}
}

// list reads a list or set header and returns the element type and count.
func (d *decoder) list() (elem byte, n int, err error) {
	b, err := d.byte()
	if err != nil {
		return 0, 0, err
	}
	elem, n = b&0x0f, int(b>>4)
	if n == 15 {
		u, err := d.uvarint()
		if err != nil {
			return 0, 0, err
		}
		if u > math.MaxInt32 {
			return 0, 0, fmt.Errorf("list of %d elements", u)
		}
		n = int(u)
	}
	// Every element takes at least one byte, which bounds allocations.
	if n > len(d.buf)-d.pos {
		return 0, 0, io.ErrUnexpectedEOF
	}
	return elem, n, nil
}

// The typed readers check the field's wire type before reading its value.

func (d *decoder) readBool(typ byte) (bool, error) {
	switch typ {
	case ttTrue:
		return true, nil
	case ttFalse:
		return false, nil
	}
	return false, fmt.Errorf("bool field of type %d", typ)
}

func (d *decoder) readI32(typ byte) (int32, error) {
	if typ != ttI32 {
		return 0, fmt.Errorf("i32 field of type %d", typ)
	}
	return d.i32()
}

func (d *decoder) readI64(typ byte) (int64, error) {
	if typ != ttI64 {
		return 0, fmt.Errorf("i64 field of type %d", typ)
	}
	return d.varint()
}

func (d *decoder) readBinary(typ byte) ([]byte, error) {
	if typ != ttBinary {
		return nil, fmt.Errorf("binary field of type %d", typ)
	}
	return d.binary()
}

func (d *decoder) readString(typ byte) (string, error) {
	b, err := d.readBinary(typ)
	return string(b), err
}

// readList reads the header of a list field whose elements are of type elem.
func (d *decoder) readList(typ, elem byte) (int, error) {
	if typ != ttList && typ != ttSet {
		return 0, fmt.Errorf("list field of type %d", typ)
	}
	got, n, err := d.list()
	if err != nil {
		return 0, err
	}
	if got != elem && n > 0 {
		return 0, fmt.Errorf("list of type %d, want %d", got, elem)
	}
	return n, nil
}

// readStruct reads a struct field.
func (d *decoder) readStruct(typ byte, fn func(id int16, typ byte) error) error {
	if typ != ttStruct {
		return fmt.Errorf("struct field of type %d", typ)
	}
	return d.strct(fn)
}

// skip skips a field value of type typ.
func (d *decoder) skip(typ byte) error {
	switch typ {
	case ttTrue, ttFalse:
		return nil
	case ttByte:
		_, err := d.byte()
		return err
	case ttI16, ttI32, ttI64:
		_, err := d.uvarint()
		return err
	case ttDouble:
		return d.skipBytes(8)
	case ttBinary:
		_, err := d.binary()
		return err
	case ttUUID:
		return d.skipBytes(16)
	case ttList, ttSet:
		elem, n, err := d.list()
		if err != nil {
			return err
		}
		return d.skipElems(elem, n)
	case ttMap:
		n, err := d.uvarint()
		if err != nil {
			return err
		}
		if n == 0 {
			return nil
		}
		if n > uint64(len(d.buf)-d.pos) {
			return io.ErrUnexpectedEOF
		}
		kv, err := d.byte()
		if err != nil {
			return err
		}
		if err := d.nest(); err != nil {
			return err
		}
		for range n {
			if err := d.skipElem(kv >> 4); err != nil {
				return err
			}
			if err := d.skipElem(kv & 0x0f); err != nil {
				return err
			}
		}
		d.depth--
		return nil
	case ttStruct:
		return d.strct(func(_ int16, typ byte) error { return d.skip(typ) })
	}
	return fmt.Errorf("field of type %d", typ)
}

func (d *decoder) skipBytes(n int) error {
	if n > len(d.buf)-d.pos {
		return io.ErrUnexpectedEOF
	}
	d.pos += n
	return nil
}

func (d *decoder) nest() error {
	if d.depth >= maxDepth {
		return fmt.Errorf("nesting deeper than %d", maxDepth)
	}
	d.depth++
	return nil
}

func (d *decoder) skipElems(elem byte, n int) error {
	if err := d.nest(); err != nil {
		return err
	}
	for range n {
		if err := d.skipElem(elem); err != nil {
			return err
		}
	}
	d.depth--
	return nil
}

// skipElem skips a container element, where a bool takes a byte of its own.
func (d *decoder) skipElem(typ byte) error {
	if typ == ttTrue || typ == ttFalse {
		_, err := d.byte()
		return err
	}
	return d.skip(typ)
}
