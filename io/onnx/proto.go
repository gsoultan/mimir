package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Protocol buffer wire types.
const (
	wireVarint  = 0
	wireFixed64 = 1
	wireBytes   = 2
	wireFixed32 = 5
)

// decoder reads protocol buffer messages from a byte slice, bounded by
// it: every length it reads must fit in the bytes left, so a message can
// claim no more than its own bytes, and repeated fields hold at most one
// element per byte. Its first error sticks: later reads return zero
// values, and more reports false.
type decoder struct {
	b    []byte
	off  int
	base int // offset of b in the file, for errors
	err  *error
}

// sub returns a decoder of the n bytes of a length-delimited field, which
// shares d's error.
func (d *decoder) sub(b []byte, at int) *decoder {
	return &decoder{b: b, base: d.base + at, err: d.err}
}

func (d *decoder) fail(format string, args ...any) {
	if *d.err == nil {
		*d.err = fmt.Errorf("%w: at byte %d: %s", ErrFormat, d.base+d.off, fmt.Sprintf(format, args...))
	}
}

// more reports whether d has fields left to read.
func (d *decoder) more() bool { return *d.err == nil && d.off < len(d.b) }

func (d *decoder) uvarint() uint64 {
	if *d.err != nil {
		return 0
	}
	v, n := binary.Uvarint(d.b[d.off:])
	if n <= 0 {
		d.fail("truncated or overlong varint")
		return 0
	}
	d.off += n
	return v
}

// key reads a field's number and wire type.
func (d *decoder) key() (field int, wire int) {
	k := d.uvarint()
	if *d.err == nil && (k>>3 == 0 || k>>3 > math.MaxInt32) {
		d.fail("field number %d", k>>3)
		return 0, 0
	}
	return int(k >> 3), int(k & 7)
}

// next returns the next n bytes, or nil if the message ends first.
func (d *decoder) next(n uint64) []byte {
	if *d.err != nil {
		return nil
	}
	if n > uint64(len(d.b)-d.off) {
		d.fail("%d bytes past the end of the message", n)
		return nil
	}
	b := d.b[d.off : d.off+int(n) : d.off+int(n)]
	d.off += int(n)
	return b
}

// bytes reads the payload of a length-delimited field.
func (d *decoder) bytes() []byte { return d.next(d.uvarint()) }

// message returns a decoder of the embedded message of a length-delimited
// field.
func (d *decoder) message() *decoder {
	n := d.uvarint()
	at := d.off
	return d.sub(d.next(n), at)
}

func (d *decoder) str() string { return string(d.bytes()) }

func (d *decoder) fixed32() uint32 {
	if b := d.next(4); b != nil {
		return binary.LittleEndian.Uint32(b)
	}
	return 0
}

func (d *decoder) fixed64() uint64 {
	if b := d.next(8); b != nil {
		return binary.LittleEndian.Uint64(b)
	}
	return 0
}

// skip passes over a field of an unknown number.
func (d *decoder) skip(wire int) {
	switch wire {
	case wireVarint:
		d.uvarint()
	case wireFixed64:
		d.next(8)
	case wireBytes:
		d.bytes()
	case wireFixed32:
		d.next(4)
	default:
		// Groups are deprecated and ONNX never used them.
		d.fail("wire type %d", wire)
	}
}

// want checks that a field has the wire type its number declares.
func (d *decoder) want(field, wire, declared int) bool {
	if wire != declared {
		d.fail("field %d of wire type %d, want %d", field, wire, declared)
		return false
	}
	return true
}

// int64s appends a repeated varint field, packed or not, to s.
func (d *decoder) int64s(s []int64, wire int) []int64 {
	switch wire {
	case wireVarint:
		return append(s, int64(d.uvarint()))
	case wireBytes:
		p := d.message()
		for p.more() {
			s = append(s, int64(p.uvarint()))
		}
		return s
	}
	d.fail("repeated varint of wire type %d", wire)
	return s
}

// float32s appends a repeated float field, packed or not, to s.
func (d *decoder) float32s(s []float32, wire int) []float32 {
	switch wire {
	case wireFixed32:
		return append(s, math.Float32frombits(d.fixed32()))
	case wireBytes:
		b := d.bytes()
		if len(b)%4 != 0 {
			d.fail("packed floats of %d bytes", len(b))
			return s
		}
		for i := 0; i < len(b); i += 4 {
			s = append(s, math.Float32frombits(binary.LittleEndian.Uint32(b[i:])))
		}
		return s
	}
	d.fail("repeated float of wire type %d", wire)
	return s
}

// float64s appends a repeated double field, packed or not, to s.
func (d *decoder) float64s(s []float64, wire int) []float64 {
	switch wire {
	case wireFixed64:
		return append(s, math.Float64frombits(d.fixed64()))
	case wireBytes:
		b := d.bytes()
		if len(b)%8 != 0 {
			d.fail("packed doubles of %d bytes", len(b))
			return s
		}
		for i := 0; i < len(b); i += 8 {
			s = append(s, math.Float64frombits(binary.LittleEndian.Uint64(b[i:])))
		}
		return s
	}
	d.fail("repeated double of wire type %d", wire)
	return s
}

// The messages of onnx.proto that a model is read from. Fields mimir does
// not use are skipped.

type modelProto struct {
	irVersion int64
	opsets    []opsetID
	producer  string
	graph     *graphProto
}

type opsetID struct {
	domain  string
	version int64
}

type graphProto struct {
	name        string
	nodes       []*nodeProto
	inits       []*tensorProto
	sparseInits int
	inputs      []ValueInfo
	outputs     []ValueInfo
}

type nodeProto struct {
	name    string
	op      string
	domain  string
	inputs  []string
	outputs []string
	attrs   []*attribute
}

// attrType is the type of an attribute, AttributeProto.AttributeType.
type attrType int64

const (
	attrFloat   attrType = 1
	attrInt     attrType = 2
	attrString  attrType = 3
	attrTensor  attrType = 4
	attrGraph   attrType = 5
	attrFloats  attrType = 6
	attrInts    attrType = 7
	attrStrings attrType = 8
)

type attribute struct {
	name    string
	typ     attrType
	f       float32
	i       int64
	s       []byte
	t       *tensorProto
	floats  []float32
	ints    []int64
	strings [][]byte
	ref     string // ref_attr_name, which only functions use
}

type tensorProto struct {
	name     string
	dims     []int64
	dataType DataType
	segment  bool
	raw      []byte
	hasRaw   bool
	floats   []float32
	int32s   []int64 // int32_data, which also holds the bits of narrower types
	int64s   []int64
	doubles  []float64
	uint64s  []int64
	strings  int // string_data elements, which mimir does not read
	external map[string]string
	location int64 // 1 for data in an external file
}

// parseModel decodes a ModelProto.
func parseModel(b []byte) (*modelProto, error) {
	var err error
	d := &decoder{b: b, err: &err}
	m := &modelProto{}
	for d.more() {
		field, wire := d.key()
		switch field {
		case 1:
			if d.want(field, wire, wireVarint) {
				m.irVersion = int64(d.uvarint())
			}
		case 2:
			if d.want(field, wire, wireBytes) {
				m.producer = d.str()
			}
		case 7:
			if d.want(field, wire, wireBytes) {
				if m.graph != nil {
					d.fail("two graphs")
				}
				m.graph = parseGraph(d.message())
			}
		case 8:
			if d.want(field, wire, wireBytes) {
				m.opsets = append(m.opsets, parseOpset(d.message()))
			}
		default:
			d.skip(wire)
		}
	}
	if err != nil {
		return nil, err
	}
	if m.graph == nil {
		return nil, fmt.Errorf("%w: no graph", ErrFormat)
	}
	return m, nil
}

func parseOpset(d *decoder) opsetID {
	var o opsetID
	for d.more() {
		field, wire := d.key()
		switch field {
		case 1:
			if d.want(field, wire, wireBytes) {
				o.domain = d.str()
			}
		case 2:
			if d.want(field, wire, wireVarint) {
				o.version = int64(d.uvarint())
			}
		default:
			d.skip(wire)
		}
	}
	return o
}

func parseGraph(d *decoder) *graphProto {
	g := &graphProto{}
	for d.more() {
		field, wire := d.key()
		switch field {
		case 1, 2, 5, 11, 12, 15:
			if !d.want(field, wire, wireBytes) {
				return g
			}
		}
		switch field {
		case 1:
			g.nodes = append(g.nodes, parseNode(d.message()))
		case 2:
			g.name = d.str()
		case 5:
			g.inits = append(g.inits, parseTensor(d.message()))
		case 11:
			g.inputs = append(g.inputs, parseValueInfo(d.message()))
		case 12:
			g.outputs = append(g.outputs, parseValueInfo(d.message()))
		case 15:
			d.bytes()
			g.sparseInits++
		default:
			d.skip(wire)
		}
	}
	return g
}

func parseNode(d *decoder) *nodeProto {
	n := &nodeProto{}
	for d.more() {
		field, wire := d.key()
		if field >= 1 && field <= 7 && !d.want(field, wire, wireBytes) {
			return n
		}
		switch field {
		case 1:
			n.inputs = append(n.inputs, d.str())
		case 2:
			n.outputs = append(n.outputs, d.str())
		case 3:
			n.name = d.str()
		case 4:
			n.op = d.str()
		case 5:
			n.attrs = append(n.attrs, parseAttribute(d.message()))
		case 7:
			n.domain = d.str()
		default:
			d.skip(wire)
		}
	}
	return n
}

func parseAttribute(d *decoder) *attribute {
	a := &attribute{}
	for d.more() {
		field, wire := d.key()
		switch field {
		case 1:
			if d.want(field, wire, wireBytes) {
				a.name = d.str()
			}
		case 2:
			if d.want(field, wire, wireFixed32) {
				a.f = math.Float32frombits(d.fixed32())
			}
		case 3:
			if d.want(field, wire, wireVarint) {
				a.i = int64(d.uvarint())
			}
		case 4:
			if d.want(field, wire, wireBytes) {
				a.s = d.bytes()
			}
		case 5:
			if d.want(field, wire, wireBytes) {
				a.t = parseTensor(d.message())
			}
		case 7:
			a.floats = d.float32s(a.floats, wire)
		case 8:
			a.ints = d.int64s(a.ints, wire)
		case 9:
			if d.want(field, wire, wireBytes) {
				a.strings = append(a.strings, d.bytes())
			}
		case 20:
			if d.want(field, wire, wireVarint) {
				a.typ = attrType(d.uvarint())
			}
		case 21:
			if d.want(field, wire, wireBytes) {
				a.ref = d.str()
			}
		default:
			// Graphs (6, 11), sparse tensors, type protos and the lists
			// of tensors are skipped: no operator mimir runs takes them.
			d.skip(wire)
		}
	}
	return a
}

func parseTensor(d *decoder) *tensorProto {
	t := &tensorProto{}
	for d.more() {
		field, wire := d.key()
		switch field {
		case 1:
			t.dims = d.int64s(t.dims, wire)
		case 2:
			if d.want(field, wire, wireVarint) {
				t.dataType = DataType(int32(d.uvarint()))
			}
		case 3:
			d.skip(wire)
			t.segment = true
		case 4:
			t.floats = d.float32s(t.floats, wire)
		case 5:
			t.int32s = d.int64s(t.int32s, wire)
		case 6:
			if d.want(field, wire, wireBytes) {
				d.bytes()
				t.strings++
			}
		case 7:
			t.int64s = d.int64s(t.int64s, wire)
		case 8:
			if d.want(field, wire, wireBytes) {
				t.name = d.str()
			}
		case 9:
			if d.want(field, wire, wireBytes) {
				t.raw, t.hasRaw = d.bytes(), true
			}
		case 10:
			t.doubles = d.float64s(t.doubles, wire)
		case 11:
			t.uint64s = d.int64s(t.uint64s, wire)
		case 13:
			if d.want(field, wire, wireBytes) {
				k, v := parseEntry(d.message())
				if t.external == nil {
					t.external = map[string]string{}
				}
				t.external[k] = v
			}
		case 14:
			if d.want(field, wire, wireVarint) {
				t.location = int64(d.uvarint())
			}
		default:
			d.skip(wire)
		}
	}
	return t
}

// parseEntry decodes a StringStringEntryProto.
func parseEntry(d *decoder) (key, value string) {
	for d.more() {
		field, wire := d.key()
		switch field {
		case 1:
			if d.want(field, wire, wireBytes) {
				key = d.str()
			}
		case 2:
			if d.want(field, wire, wireBytes) {
				value = d.str()
			}
		default:
			d.skip(wire)
		}
	}
	return key, value
}

func parseValueInfo(d *decoder) ValueInfo {
	var v ValueInfo
	for d.more() {
		field, wire := d.key()
		switch field {
		case 1:
			if d.want(field, wire, wireBytes) {
				v.Name = d.str()
			}
		case 2:
			if d.want(field, wire, wireBytes) {
				parseType(d.message(), &v)
			}
		default:
			d.skip(wire)
		}
	}
	return v
}

// parseType decodes a TypeProto into v. Only tensor types are read;
// sequences, maps and optional values leave v's Type Undefined.
func parseType(d *decoder, v *ValueInfo) {
	for d.more() {
		field, wire := d.key()
		if field != 1 {
			d.skip(wire)
			continue
		}
		if !d.want(field, wire, wireBytes) {
			return
		}
		t := d.message()
		for t.more() {
			field, wire := t.key()
			switch field {
			case 1:
				if t.want(field, wire, wireVarint) {
					v.Type = DataType(int32(t.uvarint()))
				}
			case 2:
				if t.want(field, wire, wireBytes) {
					v.Shape = parseShape(t.message())
					v.Ranked = true
				}
			default:
				t.skip(wire)
			}
		}
	}
}

// parseShape decodes a TensorShapeProto.
func parseShape(d *decoder) []Dim {
	dims := []Dim{}
	for d.more() {
		field, wire := d.key()
		if field != 1 {
			d.skip(wire)
			continue
		}
		if !d.want(field, wire, wireBytes) {
			break
		}
		dim := Dim{Size: -1}
		e := d.message()
		for e.more() {
			field, wire := e.key()
			switch field {
			case 1:
				if e.want(field, wire, wireVarint) {
					s := int64(e.uvarint())
					if s < 0 || s > math.MaxInt32 {
						e.fail("dimension %d", s)
					}
					dim.Size = int(s)
				}
			case 2:
				if e.want(field, wire, wireBytes) {
					dim.Name = e.str()
				}
			default:
				e.skip(wire)
			}
		}
		dims = append(dims, dim)
	}
	return dims
}
