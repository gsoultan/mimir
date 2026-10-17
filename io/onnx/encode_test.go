package onnx

import (
	"encoding/binary"
	"math"
)

// pb encodes a protocol buffer message, for writing models in tests.
type pb struct{ b []byte }

func (p *pb) key(field, wire int) { p.b = binary.AppendUvarint(p.b, uint64(field<<3|wire)) }

func (p *pb) varint(field int, v int64) *pb {
	p.key(field, wireVarint)
	p.b = binary.AppendUvarint(p.b, uint64(v))
	return p
}

func (p *pb) bytes(field int, b []byte) *pb {
	p.key(field, wireBytes)
	p.b = binary.AppendUvarint(p.b, uint64(len(b)))
	p.b = append(p.b, b...)
	return p
}

func (p *pb) str(field int, s string) *pb { return p.bytes(field, []byte(s)) }

func (p *pb) msg(field int, m *pb) *pb { return p.bytes(field, m.b) }

func (p *pb) float(field int, f float32) *pb {
	p.key(field, wireFixed32)
	p.b = binary.LittleEndian.AppendUint32(p.b, math.Float32bits(f))
	return p
}

// packed encodes a packed repeated varint field.
func (p *pb) packed(field int, vs []int64) *pb {
	var q []byte
	for _, v := range vs {
		q = binary.AppendUvarint(q, uint64(v))
	}
	return p.bytes(field, q)
}

// graph builds a model in tests.
type graph struct {
	opset   int64
	nodes   []*pb
	inits   []*pb
	inputs  []*pb
	outputs []*pb
}

func newGraph() *graph { return &graph{opset: 17} }

// valueInfo encodes a ValueInfoProto; dims are ints or, for symbolic
// dimensions, strings. A nil dims leaves the shape out.
func valueInfo(name string, typ DataType, dims []any) *pb {
	tt := (&pb{}).varint(1, int64(typ))
	if dims != nil {
		shape := &pb{}
		for _, d := range dims {
			dim := &pb{}
			switch d := d.(type) {
			case int:
				dim.varint(1, int64(d))
			case string:
				dim.str(2, d)
			}
			shape.msg(1, dim)
		}
		tt.msg(2, shape)
	}
	return (&pb{}).str(1, name).msg(2, (&pb{}).msg(1, tt))
}

func (g *graph) input(name string, typ DataType, dims ...any) *graph {
	if dims == nil {
		dims = []any{}
	}
	g.inputs = append(g.inputs, valueInfo(name, typ, dims))
	return g
}

func (g *graph) output(name string) *graph {
	g.outputs = append(g.outputs, valueInfo(name, Undefined, nil))
	return g
}

// tensorFloats encodes a float TensorProto in raw_data.
func tensorFloats(name string, shape []int, data []float32) *pb {
	t := &pb{}
	for _, d := range shape {
		t.varint(1, int64(d))
	}
	t.varint(2, int64(Float))
	raw := make([]byte, 4*len(data))
	for i, f := range data {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(f))
	}
	return t.str(8, name).bytes(9, raw)
}

// tensorInts encodes an int64 TensorProto in int64_data.
func tensorInts(name string, shape []int, data []int64) *pb {
	t := &pb{}
	for _, d := range shape {
		t.varint(1, int64(d))
	}
	t.varint(2, int64(Int64))
	return t.packed(7, data).str(8, name)
}

func (g *graph) init(name string, shape []int, data []float32) *graph {
	g.inits = append(g.inits, tensorFloats(name, shape, data))
	return g
}

func (g *graph) initInts(name string, shape []int, data []int64) *graph {
	g.inits = append(g.inits, tensorInts(name, shape, data))
	return g
}

// node adds a node; ins and outs are separated by "->" in io, such as
// []string{"x", "w", "->", "y"}.
func (g *graph) node(op string, io []string, attrs ...*pb) *graph {
	n := (&pb{}).str(4, op).str(3, op+"_"+string(rune('a'+len(g.nodes)%26)))
	out := false
	for _, s := range io {
		switch {
		case s == "->":
			out = true
		case out:
			n.str(2, s)
		default:
			n.str(1, s)
		}
	}
	for _, a := range attrs {
		n.msg(5, a)
	}
	g.nodes = append(g.nodes, n)
	return g
}

func aInt(name string, v int64) *pb {
	return (&pb{}).str(1, name).varint(20, int64(attrInt)).varint(3, v)
}

func aInts(name string, vs ...int64) *pb {
	return (&pb{}).str(1, name).varint(20, int64(attrInts)).packed(8, vs)
}

func aFloat(name string, f float32) *pb {
	return (&pb{}).str(1, name).varint(20, int64(attrFloat)).float(2, f)
}

func aString(name, s string) *pb {
	return (&pb{}).str(1, name).varint(20, int64(attrString)).str(4, s)
}

func aTensor(name string, t *pb) *pb {
	return (&pb{}).str(1, name).varint(20, int64(attrTensor)).msg(5, t)
}

// model encodes the graph as a ModelProto.
func (g *graph) model() []byte {
	gp := &pb{}
	for _, n := range g.nodes {
		gp.msg(1, n)
	}
	gp.str(2, "test")
	for _, t := range g.inits {
		gp.msg(5, t)
	}
	for _, v := range g.inputs {
		gp.msg(11, v)
	}
	for _, v := range g.outputs {
		gp.msg(12, v)
	}
	m := (&pb{}).varint(1, 8).str(2, "mimir-test")
	m.msg(7, gp)
	m.msg(8, (&pb{}).str(1, "").varint(2, g.opset))
	return m.b
}
