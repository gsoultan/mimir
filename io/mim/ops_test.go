package mim_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"

	mimfile "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/interp"
	"github.com/gsoultan/mimir-model/ref"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

// builder writes small graphs for the tests, whose inputs are random and
// whose outputs the format's reference interpreter computes.
type builder struct {
	g mimfile.Graph
}

func newBuilder() *builder {
	return &builder{g: mimfile.Graph{
		PrimitiveVersion: mimfile.CurrentPrimitive,
		StdlibVersion:    mimfile.CurrentStdlib,
		Subgraphs:        []mimfile.Subgraph{{}},
	}}
}

func dims(shape []int) []mimfile.Dim {
	d := make([]mimfile.Dim, len(shape))
	for i, s := range shape {
		d[i] = mimfile.Dim(s)
	}
	return d
}

func (b *builder) value(name string, dt mimfile.DType, kind mimfile.ValueKind, shape ...int) uint32 {
	b.g.Values = append(b.g.Values, mimfile.Value{Name: name, DType: dt, Shape: dims(shape), Kind: kind})
	return uint32(len(b.g.Values) - 1)
}

// input adds an input of the main graph.
func (b *builder) input(name string, dt mimfile.DType, shape ...int) uint32 {
	v := b.value(name, dt, mimfile.KindInput, shape...)
	b.g.Subgraphs[0].Inputs = append(b.g.Subgraphs[0].Inputs, v)
	return v
}

// op adds an op to subgraph sg with one output of dtype dt and shape.
func (b *builder) op(sg int, code mimfile.Opcode, dt mimfile.DType, shape []int, in []uint32, attrs ...mimfile.Attr) uint32 {
	out := b.value("", dt, mimfile.KindIntermediate, shape...)
	b.g.Subgraphs[sg].Ops = append(b.g.Subgraphs[sg].Ops, mimfile.Op{Opcode: code, Inputs: in, Outputs: []uint32{out}, Attrs: attrs})
	return out
}

// output makes v an output of the main graph, named name.
func (b *builder) output(name string, v uint32) {
	b.g.Values[v].Name = name
	b.g.Values[v].Kind = mimfile.KindOutput
	b.g.Subgraphs[0].Outputs = append(b.g.Subgraphs[0].Outputs, v)
}

func (b *builder) file(t *testing.T) []byte {
	t.Helper()
	if err := b.g.Validate(); err != nil {
		t.Fatal(err)
	}
	enc, err := b.g.Encode()
	if err != nil {
		t.Fatal(err)
	}
	w := mimfile.NewWriter()
	if err := w.AddSection(mimfile.SectionGraph, true, enc); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func aInt(key string, v int64) mimfile.Attr {
	return mimfile.Attr{Key: key, Type: mimfile.AttrInt, Int: v}
}
func aInts(key string, v ...int64) mimfile.Attr {
	return mimfile.Attr{Key: key, Type: mimfile.AttrInts, Ints: v}
}
func aFloat(key string, v float64) mimfile.Attr {
	return mimfile.Attr{Key: key, Type: mimfile.AttrFloat, Float: v}
}
func aStr(key, v string) mimfile.Attr {
	return mimfile.Attr{Key: key, Type: mimfile.AttrString, String: v}
}
func aDType(key string, v mimfile.DType) mimfile.Attr {
	return mimfile.Attr{Key: key, Type: mimfile.AttrDType, DType: v}
}
func aSub(key string, v int) mimfile.Attr {
	return mimfile.Attr{Key: key, Type: mimfile.AttrSubgraph, Subgraph: uint32(v)}
}
func calls(name string, version int64, attrs ...mimfile.Attr) []mimfile.Attr {
	return append([]mimfile.Attr{aStr("name", name), aInt("version", version)}, attrs...)
}

// random returns n values for an input of dtype dt: floats from a
// normal distribution, scaled by scale, or integers in [lo, hi).
type gen struct {
	scale  float64
	lo, hi int
}

// inputs returns random inputs for the graph's main inputs, for the
// reference interpreter and for mimir.
func inputs(t *testing.T, g *mimfile.Graph, seed uint64, gens map[string]gen) (map[string]*ref.Tensor, map[string]*tensor.Tensor) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, seed+1))
	refs, ts := map[string]*ref.Tensor{}, map[string]*tensor.Tensor{}
	for _, vi := range g.Subgraphs[0].Inputs {
		v := g.Values[vi]
		shape := make([]int, len(v.Shape))
		n := 1
		for i, d := range v.Shape {
			shape[i] = int(d)
			n *= int(d)
		}
		r, err := ref.New(v.DType, shape)
		if err != nil {
			t.Fatal(err)
		}
		gn, ok := gens[v.Name]
		if !ok {
			gn = gen{scale: 1, lo: -5, hi: 5}
		}
		switch v.DType {
		case mimfile.F32:
			f := make([]float32, n)
			for i := range f {
				f[i] = float32(rng.NormFloat64() * gn.scale)
				binary.LittleEndian.PutUint32(r.Data[4*i:], math.Float32bits(f[i]))
			}
			ts[v.Name] = tensor.FromSlice(f, shape...)
		case mimfile.I32, mimfile.I64, mimfile.Bool:
			if v.DType == mimfile.Bool {
				gn.lo, gn.hi = 0, 2
			}
			x := make([]int32, n)
			for i := range x {
				x[i] = int32(gn.lo + rng.IntN(gn.hi-gn.lo))
				r.SetBits(i, uint64(int64(x[i])))
			}
			ts[v.Name] = tensor.FromSlice(x, shape...)
		default:
			t.Fatalf("input of dtype %s", v.DType)
		}
		if len(shape) == 0 {
			ts[v.Name] = ts[v.Name].Reshape() // FromSlice makes a vector
		}
		refs[v.Name] = r
	}
	return refs, ts
}

// float64s returns the elements of r.
func float64s(t *testing.T, r *ref.Tensor) []float64 {
	t.Helper()
	n := r.NumElems()
	out := make([]float64, n)
	for i := range out {
		b := r.Bits(i)
		switch r.DType {
		case mimfile.F32:
			out[i] = float64(math.Float32frombits(uint32(b)))
		case mimfile.I32:
			out[i] = float64(int32(b))
		case mimfile.I64:
			out[i] = float64(int64(b))
		case mimfile.Bool, mimfile.U8:
			out[i] = float64(b)
		case mimfile.I8:
			out[i] = float64(int8(b))
		case mimfile.I16:
			out[i] = float64(int16(b))
		case mimfile.U16:
			out[i] = float64(uint16(b))
		default:
			t.Fatalf("output of dtype %s", r.DType)
		}
	}
	return out
}

// compare runs the graph in the reference interpreter and in mimir, on
// the same random inputs, and compares the outputs within tol times the
// largest reference output: exactly for integers.
func compare(t *testing.T, b *builder, tol float64, gens map[string]gen) {
	t.Helper()
	data := b.file(t)
	f, err := mimfile.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	p, err := interp.Load(f)
	if err != nil {
		t.Fatal(err)
	}
	m, err := mim.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	s, rs := m.NewSession(), p.NewSession()
	for seed := range uint64(2) {
		refIn, in := inputs(t, &b.g, seed, gens)
		want, err := rs.Run(refIn)
		if err != nil {
			t.Fatalf("reference: %v", err)
		}
		got, err := s.Run(ctx, in)
		if err != nil {
			t.Fatal(err)
		}
		for name, w := range want {
			g := got[name]
			if g == nil {
				t.Fatalf("no output %s", name)
			}
			if !equalShapes(g.Shape(), w.Shape) {
				t.Fatalf("output %s of shape %v, want %v", name, g.Shape(), w.Shape)
			}
			var gv []float64
			if g.DType() == tensor.Int32 {
				x, err := g.Data[int32](ctx)
				if err != nil {
					t.Fatal(err)
				}
				for _, v := range x {
					gv = append(gv, float64(v))
				}
			} else {
				x, err := g.Cast(tensor.Float32).Data[float32](ctx)
				if err != nil {
					t.Fatal(err)
				}
				for _, v := range x {
					gv = append(gv, float64(v))
				}
			}
			wv := float64s(t, w)
			scale := 1e-30
			for _, v := range wv {
				if !math.IsInf(v, 0) && !math.IsNaN(v) {
					scale = max(scale, math.Abs(v))
				}
			}
			for i := range wv {
				same := gv[i] == wv[i] || (math.IsNaN(gv[i]) && math.IsNaN(wv[i]))
				if !same && !(math.Abs(gv[i]-wv[i]) <= tol*scale) {
					t.Fatalf("seed %d: output %s element %d = %g, want %g", seed, name, i, gv[i], wv[i])
				}
			}
		}
	}
}

func equalShapes(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// unaryGraph is a graph of one op on one input x.
func unaryGraph(code mimfile.Opcode, in, out mimfile.DType, shape []int, attrs ...mimfile.Attr) *builder {
	b := newBuilder()
	x := b.input("x", in, shape...)
	b.output("y", b.op(0, code, out, shape, []uint32{x}, attrs...))
	return b
}

// binaryGraph is a graph of one op on inputs a and b.
func binaryGraph(code mimfile.Opcode, in, out mimfile.DType, sa, sb, so []int) *builder {
	b := newBuilder()
	x := b.input("a", in, sa...)
	y := b.input("b", in, sb...)
	b.output("y", b.op(0, code, out, so, []uint32{x, y}))
	return b
}

func TestPrimitivesMatchTheReference(t *testing.T) {
	f32, i32, i64, boolean := mimfile.F32, mimfile.I32, mimfile.I64, mimfile.Bool
	s := []int{3, 4}
	for _, tc := range []struct {
		name string
		b    *builder
		tol  float64
		gens map[string]gen
	}{
		{"neg", unaryGraph(mimfile.OpNeg, f32, f32, s), 0, nil},
		{"neg/int", unaryGraph(mimfile.OpNeg, i32, i32, s), 0, nil},
		{"recip", unaryGraph(mimfile.OpRecip, f32, f32, s), 1e-6, nil},
		{"sqrt", unaryGraph(mimfile.OpSqrt, f32, f32, s), 1e-6, nil},
		{"exp2", unaryGraph(mimfile.OpExp2, f32, f32, s), 1e-6, nil},
		{"log2", unaryGraph(mimfile.OpLog2, f32, f32, s), 1e-6, nil},
		{"sin", unaryGraph(mimfile.OpSin, f32, f32, s), 1e-6, map[string]gen{"x": {scale: 10}}},
		{"trunc", unaryGraph(mimfile.OpTrunc, f32, f32, s), 0, map[string]gen{"x": {scale: 1e3}}},
		{"cast/f32-i32", unaryGraph(mimfile.OpCast, f32, i32, s, aDType("to", i32)), 0, map[string]gen{"x": {scale: 1e3}}},
		{"cast/f32-i8", unaryGraph(mimfile.OpCast, f32, mimfile.I8, s, aDType("to", mimfile.I8)), 0, map[string]gen{"x": {scale: 300}}},
		{"cast/i32-u8", unaryGraph(mimfile.OpCast, i32, mimfile.U8, s, aDType("to", mimfile.U8)), 0, map[string]gen{"x": {lo: -600, hi: 600}}},
		{"cast/i32-i16", unaryGraph(mimfile.OpCast, i32, mimfile.I16, s, aDType("to", mimfile.I16)), 0, map[string]gen{"x": {lo: -1 << 20, hi: 1 << 20}}},
		{"cast/f32-bool", unaryGraph(mimfile.OpCast, f32, boolean, s, aDType("to", boolean)), 0, map[string]gen{"x": {scale: 0}}},
		{"cast/i64-f32", unaryGraph(mimfile.OpCast, i64, f32, s, aDType("to", f32)), 0, nil},
		{"cast/bool-f32", unaryGraph(mimfile.OpCast, boolean, f32, s, aDType("to", f32)), 0, nil},
		{"add", binaryGraph(mimfile.OpAdd, f32, f32, s, []int{1, 4}, s), 0, nil},
		{"mul", binaryGraph(mimfile.OpMul, f32, f32, []int{3, 1}, s, s), 0, nil},
		{"max/int", binaryGraph(mimfile.OpMax, i32, i32, s, s, s), 0, nil},
		{"pow", binaryGraph(mimfile.OpPow, f32, f32, s, s, s), 2e-6, nil},
		{"idiv", binaryGraph(mimfile.OpIdiv, i32, i32, s, s, s), 0, map[string]gen{"a": {lo: -1000, hi: 1000}, "b": {lo: 1, hi: 9}}},
		{"imod", binaryGraph(mimfile.OpImod, i32, i32, s, s, s), 0, map[string]gen{"a": {lo: -1000, hi: 1000}, "b": {lo: -9, hi: -1}}},
		{"and", binaryGraph(mimfile.OpAnd, boolean, boolean, s, s, s), 0, nil},
		{"or", binaryGraph(mimfile.OpOr, boolean, boolean, s, s, s), 0, nil},
		{"xor", binaryGraph(mimfile.OpXor, boolean, boolean, s, s, s), 0, nil},
		{"cmplt", binaryGraph(mimfile.OpCmplt, f32, boolean, s, s, s), 0, nil},
		{"cmpeq", binaryGraph(mimfile.OpCmpeq, i64, boolean, s, s, s), 0, map[string]gen{"a": {lo: 0, hi: 3}, "b": {lo: 0, hi: 3}}},
		{"reduce/add", unaryGraph(mimfile.OpReduce, f32, f32, []int{3, 4, 5}, aStr("mode", "add"), aInts("axes", 0, 2)), 1e-6, nil},
		{"reduce/max", unaryGraph(mimfile.OpReduce, i32, i32, []int{3, 4, 5}, aStr("mode", "max"), aInts("axes", 1)), 0, nil},
		{"reduce/mul", unaryGraph(mimfile.OpReduce, f32, f32, []int{3, 7, 5}, aStr("mode", "mul"), aInts("axes", 1, 2)), 1e-6, nil},
		{"permute", unaryGraph(mimfile.OpPermute, f32, f32, []int{2, 3, 4}, aInts("perm", 2, 0, 1)), 0, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Graphs whose output shape differs from the input's fix it up.
			fixShapes(tc.b)
			compare(t, tc.b, tc.tol, tc.gens)
		})
	}
}

// fixShapes sets the shape of each op output that changes shape: reduced
// axes become 1, permuted axes move.
func fixShapes(b *builder) {
	for _, op := range b.g.Subgraphs[0].Ops {
		in, out := b.g.Values[op.Inputs[0]].Shape, &b.g.Values[op.Outputs[0]]
		switch op.Opcode {
		case mimfile.OpReduce:
			out.Shape = append([]mimfile.Dim(nil), in...)
			for _, a := range op.Attrs {
				if a.Key == "axes" {
					for _, x := range a.Ints {
						out.Shape[x] = 1
					}
				}
			}
		case mimfile.OpPermute:
			out.Shape = make([]mimfile.Dim, len(in))
			for _, a := range op.Attrs {
				if a.Key == "perm" {
					for i, x := range a.Ints {
						out.Shape[i] = in[x]
					}
				}
			}
		}
	}
}

func TestMovementMatchesTheReference(t *testing.T) {
	f32, i32 := mimfile.F32, mimfile.I32
	t.Run("reshape-expand-pad-shrink", func(t *testing.T) {
		b := newBuilder()
		x := b.input("x", f32, 2, 6)
		r := b.op(0, mimfile.OpReshape, f32, []int{2, 1, 6}, []uint32{x}, aInts("shape", 2, 1, 6))
		e := b.op(0, mimfile.OpExpand, f32, []int{2, 3, 6}, []uint32{r})
		p := b.op(0, mimfile.OpPad, f32, []int{3, 5, 7}, []uint32{e}, aInts("pads", 1, 0, 0, 2, 1, 0), aFloat("value", -2))
		b.output("y", b.op(0, mimfile.OpShrink, f32, []int{2, 3, 4}, []uint32{p}, aInts("begin", 1, 1, 2), aInts("end", 3, 4, 6)))
		compare(t, b, 0, nil)
	})
	t.Run("gather", func(t *testing.T) {
		// Out-of-range positions are unspecified; the generator keeps
		// these in range.
		b := newBuilder()
		x := b.input("x", f32, 3, 5, 2)
		idx := b.input("idx", mimfile.I64, 2, 4)
		b.output("y", b.op(0, mimfile.OpGather, f32, []int{3, 2, 4, 2}, []uint32{x, idx}, aInt("axis", 1)))
		compare(t, b, 0, map[string]gen{"idx": {lo: 0, hi: 5}})
	})
	t.Run("scatter", func(t *testing.T) {
		// Five positions among four rows repeat at least one: the last
		// update wins.
		b := newBuilder()
		x := b.input("x", f32, 3, 4, 2)
		idx := b.input("idx", i32, 5)
		u := b.input("u", f32, 3, 5, 2)
		b.output("y", b.op(0, mimfile.OpScatter, f32, []int{3, 4, 2}, []uint32{x, idx, u}, aInt("axis", 1)))
		compare(t, b, 0, map[string]gen{"idx": {lo: 0, hi: 4}})
	})
	t.Run("where-const", func(t *testing.T) {
		b := newBuilder()
		cond := b.input("cond", mimfile.Bool, 3, 4)
		x := b.input("x", f32, 3, 4)
		k := b.op(0, mimfile.OpConst, f32, []int{1, 1}, nil, aFloat("value", 7.5))
		b.output("y", b.op(0, mimfile.OpWhere, f32, []int{3, 4}, []uint32{cond, x, k}))
		compare(t, b, 0, nil)
	})
}

func TestControlFlowMatchesTheReference(t *testing.T) {
	f32, i32, boolean := mimfile.F32, mimfile.I32, mimfile.Bool
	t.Run("while", func(t *testing.T) {
		// x doubles n times.
		b := newBuilder()
		x := b.input("x", f32, 3)
		n := b.input("n", i32)
		zero := b.op(0, mimfile.OpConst, i32, nil, nil, aInt("value", 0))
		b.g.Subgraphs = append(b.g.Subgraphs, mimfile.Subgraph{}, mimfile.Subgraph{})
		ci := b.value("", i32, mimfile.KindIntermediate)
		cx := b.value("", f32, mimfile.KindIntermediate, 3)
		b.g.Subgraphs[1].Inputs = []uint32{ci, cx}
		b.g.Subgraphs[1].Outputs = []uint32{b.op(1, mimfile.OpCmplt, boolean, nil, []uint32{ci, n})}
		bi := b.value("", i32, mimfile.KindIntermediate)
		bx := b.value("", f32, mimfile.KindIntermediate, 3)
		b.g.Subgraphs[2].Inputs = []uint32{bi, bx}
		one := b.op(2, mimfile.OpConst, i32, nil, nil, aInt("value", 1))
		b.g.Subgraphs[2].Outputs = []uint32{
			b.op(2, mimfile.OpAdd, i32, nil, []uint32{bi, one}),
			b.op(2, mimfile.OpAdd, f32, []int{3}, []uint32{bx, bx}),
		}
		outI := b.value("", i32, mimfile.KindIntermediate)
		outX := b.value("", f32, mimfile.KindIntermediate, 3)
		b.g.Subgraphs[0].Ops = append(b.g.Subgraphs[0].Ops, mimfile.Op{
			Opcode: mimfile.OpWhile, Inputs: []uint32{zero, x}, Outputs: []uint32{outI, outX},
			Attrs: []mimfile.Attr{aSub("cond", 1), aSub("body", 2)},
		})
		b.output("y", outX)
		compare(t, b, 0, map[string]gen{"n": {lo: 0, hi: 5}})
	})
	t.Run("if", func(t *testing.T) {
		b := newBuilder()
		p := b.input("p", boolean)
		x := b.input("x", f32, 3)
		b.g.Subgraphs = append(b.g.Subgraphs, mimfile.Subgraph{}, mimfile.Subgraph{})
		for sg, code := range map[int]mimfile.Opcode{1: mimfile.OpNeg, 2: mimfile.OpSqrt} {
			in := b.value("", f32, mimfile.KindIntermediate, 3)
			b.g.Subgraphs[sg].Inputs = []uint32{in}
			b.g.Subgraphs[sg].Outputs = []uint32{b.op(sg, code, f32, []int{3}, []uint32{in})}
		}
		out := b.value("", f32, mimfile.KindIntermediate, 3)
		b.g.Subgraphs[0].Ops = append(b.g.Subgraphs[0].Ops, mimfile.Op{
			Opcode: mimfile.OpIf, Inputs: []uint32{p, x}, Outputs: []uint32{out},
			Attrs: []mimfile.Attr{aSub("then", 1), aSub("else", 2)},
		})
		b.output("y", out)
		compare(t, b, 0, nil)
	})
	t.Run("model-local composite", func(t *testing.T) {
		b := newBuilder()
		x := b.input("x", f32, 2, 3)
		b.g.Subgraphs = append(b.g.Subgraphs, mimfile.Subgraph{})
		in := b.value("", f32, mimfile.KindIntermediate, 2, 3)
		b.g.Subgraphs[1].Inputs = []uint32{in}
		sq := b.op(1, mimfile.OpMul, f32, []int{2, 3}, []uint32{in, in})
		b.g.Subgraphs[1].Outputs = []uint32{b.op(1, mimfile.OpNeg, f32, []int{2, 3}, []uint32{sq})}
		b.g.Composites = []mimfile.Composite{{Name: "test/neg_square", Version: 1, Subgraph: 1}}
		b.output("y", b.op(0, mimfile.OpCall, f32, []int{2, 3}, []uint32{x}, calls("test/neg_square", 1)...))
		compare(t, b, 0, nil)
	})
}

func TestVariablesKeepTheirValuesBetweenRuns(t *testing.T) {
	// acc starts as zeros; each run adds x to it and returns the sum.
	b := newBuilder()
	x := b.input("x", mimfile.F32, 2)
	acc := b.value("acc", mimfile.F32, mimfile.KindVariable, 2)
	b.g.Values[acc].Zero = true
	sum := b.op(0, mimfile.OpAdd, mimfile.F32, []int{2}, []uint32{acc, x})
	b.output("sum", sum)
	b.g.Updates = []mimfile.Update{{Variable: acc, Value: sum}}
	m, err := mim.Parse(b.file(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	s := m.NewSession()
	run := func(s *mim.Session, v float32) []float32 {
		t.Helper()
		out, err := s.Run(ctx, map[string]*tensor.Tensor{"x": tensor.FromSlice([]float32{v, -v}, 2)})
		if err != nil {
			t.Fatal(err)
		}
		got, err := out["sum"].Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	for i, want := range []float32{1, 3, 6} {
		if got := run(s, float32(i+1)); got[0] != want || got[1] != -want {
			t.Fatalf("run %d: %v, want [%v %v]", i, got, want, -want)
		}
	}
	if got := run(m.NewSession(), 5); got[0] != 5 {
		t.Fatalf("a new session starts from %v, want zeros", got[0]-5)
	}
	s.Reset()
	if got := run(s, 2); got[0] != 2 {
		t.Fatalf("after Reset: %v, want 2", got[0])
	}
	// A failed run changes no variable.
	if _, err := s.Run(ctx, map[string]*tensor.Tensor{"x": tensor.FromSlice([]float32{1, 2, 3}, 3)}); !errors.Is(err, tensor.ErrShape) {
		t.Fatalf("run with the wrong shape: %v, want ErrShape", err)
	}
	if got := run(s, 1); got[0] != 3 {
		t.Fatalf("after a failed run: %v, want 3", got[0])
	}
}

func TestSessionsRunConcurrently(t *testing.T) {
	// Each session keeps its own sum of what it was given; sessions of one
	// model run on separate goroutines at once.
	b := newBuilder()
	x := b.input("x", mimfile.F32, 2)
	acc := b.value("acc", mimfile.F32, mimfile.KindVariable, 2)
	b.g.Values[acc].Zero = true
	sum := b.op(0, mimfile.OpAdd, mimfile.F32, []int{2}, []uint32{acc, x})
	b.output("sum", sum)
	b.g.Updates = []mimfile.Update{{Variable: acc, Value: sum}}
	m, err := mim.Parse(b.file(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	const sessions, runs = 4, 20
	got := make([][]float32, sessions)
	errs := make([]error, sessions)
	var wg sync.WaitGroup
	for i := range sessions {
		wg.Go(func() {
			s, v := m.NewSession(), float32(i+1)
			for range runs {
				out, err := s.Run(ctx, map[string]*tensor.Tensor{"x": tensor.FromSlice([]float32{v, -v}, 2)})
				if err != nil {
					errs[i] = err
					return
				}
				if got[i], err = out["sum"].Data[float32](ctx); err != nil {
					errs[i] = err
					return
				}
			}
		})
	}
	wg.Wait()
	for i := range sessions {
		if errs[i] != nil {
			t.Fatalf("session %d: %v", i, errs[i])
		}
		if want := float32(runs * (i + 1)); got[i][0] != want || got[i][1] != -want {
			t.Errorf("session %d: %v, want [%v %v]", i, got[i], want, -want)
		}
	}
}

func TestOpenRejectsWhatMimirDoesNotRun(t *testing.T) {
	for _, tc := range []struct {
		name string
		b    *builder
	}{
		{"shl", binaryGraph(mimfile.OpShl, mimfile.I32, mimfile.I32, []int{2}, []int{2}, []int{2})},
		{"bitwise and", binaryGraph(mimfile.OpAnd, mimfile.I32, mimfile.I32, []int{2}, []int{2}, []int{2})},
		{"f64", unaryGraph(mimfile.OpNeg, mimfile.F64, mimfile.F64, []int{2})},
		{"unknown composite", func() *builder {
			b := newBuilder()
			x := b.input("x", mimfile.F32, 2)
			b.output("y", b.op(0, mimfile.OpCall, mimfile.F32, []int{2}, []uint32{x}, calls("softplus", 7)...))
			return b
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := mim.Parse(tc.b.file(t))
			var u *mim.UnsupportedError
			if !errors.As(err, &u) || !errors.Is(err, mim.ErrUnsupported) {
				t.Fatalf("Parse: %v, want an *UnsupportedError", err)
			}
		})
	}
}

func TestRunChecksInputs(t *testing.T) {
	b := newBuilder()
	b.g.SymDims = []mimfile.SymDim{{Name: "batch", Bound: 4}}
	x := b.value("x", mimfile.F32, mimfile.KindInput, -1, 2)
	b.g.Values[x].Shape[0] = -1 // symbolic dimension 0
	b.g.Subgraphs[0].Inputs = []uint32{x}
	y := b.value("", mimfile.F32, mimfile.KindIntermediate)
	b.g.Values[y].Shape = []mimfile.Dim{-1, 2}
	b.g.Subgraphs[0].Ops = []mimfile.Op{{Opcode: mimfile.OpNeg, Inputs: []uint32{x}, Outputs: []uint32{y}}}
	b.output("y", y)
	m, err := mim.Parse(b.file(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if got := m.Inputs()[0].String(); got != "x: f32[batch 2]" {
		t.Errorf("input %s", got)
	}
	ctx := context.Background()
	if _, err := m.Run(ctx, map[string]*tensor.Tensor{"x": tensor.Zeros(tensor.Float32, 3, 2)}); err != nil {
		t.Fatalf("batch 3: %v", err)
	}
	for name, in := range map[string]map[string]*tensor.Tensor{
		"missing":     {},
		"above bound": {"x": tensor.Zeros(tensor.Float32, 5, 2)},
		"wrong size":  {"x": tensor.Zeros(tensor.Float32, 3, 3)},
		"wrong dtype": {"x": tensor.Zeros(tensor.Int32, 3, 2)},
		"extra":       {"x": tensor.Zeros(tensor.Float32, 3, 2), "z": tensor.Zeros(tensor.Float32, 1)},
	} {
		if _, err := m.Run(ctx, in); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

// TestHugeValuesFailBeforeAllocating runs graphs that declare values far
// larger than memory, as a hostile file may: the run fails with
// tensor.ErrMemoryBudget instead of allocating them.
func TestHugeValuesFailBeforeAllocating(t *testing.T) {
	ctx := context.Background()
	expand := func(n int) []byte {
		b := newBuilder()
		x := b.input("x", mimfile.F32, 1, 4)
		big := b.op(0, mimfile.OpExpand, mimfile.F32, []int{n, 4}, []uint32{x})
		b.output("y", b.op(0, mimfile.OpReduce, mimfile.F32, []int{1, 4}, []uint32{big}, aStr("mode", "add"), aInts("axes", 0)))
		return b.file(t)
	}
	zero := func(n int) []byte {
		b := newBuilder()
		x := b.input("x", mimfile.F32, 1, 4)
		v := b.value("cache", mimfile.F32, mimfile.KindVariable, n, 4)
		b.g.Values[v].Zero = true
		b.output("y", b.op(0, mimfile.OpAdd, mimfile.F32, []int{1, 4}, []uint32{x, x}))
		b.g.Updates = []mimfile.Update{{Variable: v, Value: v}}
		return b.file(t)
	}
	x := map[string]*tensor.Tensor{"x": tensor.Ones(tensor.Float32, 1, 4)}
	for _, tc := range []struct {
		name string
		file []byte
	}{
		{"expanded to 32 GiB", expand(1 << 31)},
		{"expanded to 2^62 elements", expand(1 << 60)},
		{"variable of 16 GiB", zero(1 << 30)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := mim.Parse(tc.file)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if _, err := m.Run(ctx, x); !errors.Is(err, tensor.ErrMemoryBudget) {
				t.Fatalf("Run: %v, want ErrMemoryBudget", err)
			}
		})
	}
	m, err := mim.Parse(expand(3))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	m.SetMaxValueBytes(47) // [3 4] f32 is 48 bytes
	if _, err := m.Run(ctx, x); !errors.Is(err, tensor.ErrMemoryBudget) {
		t.Errorf("value of 48 bytes over a bound of 47: %v", err)
	}
	m.SetMaxValueBytes(48)
	out, err := m.Run(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := out["y"].Data[float32](ctx); !slices.Equal(got, []float32{3, 3, 3, 3}) {
		t.Errorf("y = %v", got)
	}
}

// callGraph is a graph of one call on inputs of the given shapes.
func callGraph(name string, version int64, out []int, ins [][]int, attrs ...mimfile.Attr) *builder {
	b := newBuilder()
	var vs []uint32
	for i, s := range ins {
		vs = append(vs, b.input(string(rune('a'+i)), mimfile.F32, s...))
	}
	b.output("y", b.op(0, mimfile.OpCall, mimfile.F32, out, vs, calls(name, version, attrs...)...))
	return b
}

// positions makes the second input of a rope call positions, of i64.
func positions(b *builder) *builder {
	b.g.Values[b.g.Subgraphs[0].Inputs[1]].DType = mimfile.I64
	return b
}

func TestCompositesMatchTheReference(t *testing.T) {
	for _, tc := range []struct {
		name string
		b    *builder
		tol  float64
	}{
		{"conv2d/square", callGraph("conv2d", 1, []int{2, 4, 3, 3}, [][]int{{2, 3, 7, 7}, {4, 3, 3, 3}, {4}},
			aInts("strides", 2, 2), aInts("pads", 1, 0, 1, 0)), 1e-5},
		{"conv2d/groups-dilation", callGraph("conv2d", 1, []int{1, 4, 5, 5}, [][]int{{1, 4, 9, 9}, {4, 2, 3, 3}},
			aInts("dilations", 2, 2), aInt("groups", 2)), 1e-5},
		{"conv2d/rectangular", callGraph("conv2d", 1, []int{2, 3, 6, 3}, [][]int{{2, 2, 8, 9}, {3, 2, 3, 5}, {3}},
			aInts("strides", 1, 2), aInts("pads", 0, 0, 1, 0)), 1e-5},
		{"cumsum", callGraph("cumsum", 1, []int{3, 5}, [][]int{{3, 5}}, aInt("axis", 1)), 1e-6},
		{"cumsum/exclusive-reverse", callGraph("cumsum", 1, []int{3, 5}, [][]int{{3, 5}}, aInt("axis", -1), aInt("exclusive", 1), aInt("reverse", 1)), 1e-6},
		{"layer_norm", callGraph("layer_norm", 1, []int{2, 3, 8}, [][]int{{2, 3, 8}, {3, 8}, {3, 8}}, aInt("axis", 1), aFloat("epsilon", 1e-3)), 1e-5},
		{"rms_norm", callGraph("rms_norm", 1, []int{2, 8}, [][]int{{2, 8}, {8}}), 1e-5},
		{"mean", callGraph("mean", 1, []int{2, 1, 1}, [][]int{{2, 3, 4}}, aInts("axes", 1, -1)), 1e-6},
		{"softmax", callGraph("softmax", 1, []int{2, 3, 4}, [][]int{{2, 3, 4}}, aInt("axis", 1)), 1e-6},
		{"gelu_erf", callGraph("gelu_erf", 1, []int{4, 4}, [][]int{{4, 4}}), 1e-6},
		{"erf", callGraph("erf", 1, []int{4, 4}, [][]int{{4, 4}}), 1e-6},
		{"swiglu", callGraph("swiglu", 1, []int{3, 4}, [][]int{{3, 4}, {3, 4}}), 1e-6},
		{"matmul", callGraph("matmul", 1, []int{2, 3, 5}, [][]int{{2, 3, 4}, {1, 4, 5}}), 1e-6},
		{"linear", callGraph("linear", 1, []int{2, 3, 5}, [][]int{{2, 3, 4}, {5, 4}, {5}}), 1e-6},
		{"rope/half-partial", positions(callGraph("rope", 1, []int{1, 2, 3, 8}, [][]int{{1, 2, 3, 8}, {1, 3}}, aInt("rotary_dims", 4), aFloat("theta", 500))), 1e-5},
		{"rope/interleaved-yarn", positions(callGraph("rope", 1, []int{2, 2, 3, 8}, [][]int{{2, 2, 3, 8}, {2, 3}}, aInt("interleaved", 1),
			aFloat("yarn_factor", 4), aInt("yarn_original_context", 64))), 1e-5},
		{"attention/causal-gqa", callGraph("attention", 1, []int{1, 4, 2, 8}, [][]int{{1, 4, 2, 8}, {1, 2, 5, 8}, {1, 2, 5, 8}},
			aInt("causal", 1), aInt("num_kv_heads", 2)), 1e-5},
		{"attention/float-mask", callGraph("attention", 1, []int{2, 2, 3, 4}, [][]int{{2, 2, 3, 4}, {2, 2, 3, 4}, {2, 2, 3, 4}, {2, 1, 3, 3}},
			aFloat("scale", 0.3)), 1e-5},
		{"attention/softcap", callGraph("attention", 1, []int{1, 2, 3, 4}, [][]int{{1, 2, 3, 4}, {1, 2, 5, 4}, {1, 2, 5, 4}, {1, 1, 3, 5}},
			aFloat("softcap", 2), aInt("causal", 1)), 1e-5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			compare(t, tc.b, tc.tol, map[string]gen{"b": {scale: 1, lo: 0, hi: 9}})
		})
	}
}

func TestAttentionSinksJoinTheDenominator(t *testing.T) {
	// Scores 0 and ln 2 and a sink of 0: weights 1/4 and 2/4 of the
	// values 4 and 8.
	b := callGraph("attention", 1, []int{1, 1, 1, 1}, [][]int{{1, 1, 1, 1}, {1, 1, 2, 1}, {1, 1, 2, 1}, {1, 1, 1, 2}, {1}}, aFloat("scale", 1))
	m, err := mim.Parse(b.file(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	out, err := m.Run(ctx, map[string]*tensor.Tensor{
		"a": tensor.FromSlice([]float32{1}, 1, 1, 1, 1),
		"b": tensor.FromSlice([]float32{0, float32(math.Ln2)}, 1, 1, 2, 1),
		"c": tensor.FromSlice([]float32{4, 8}, 1, 1, 2, 1),
		"d": tensor.FromSlice([]float32{0, 0}, 1, 1, 1, 2),
		"e": tensor.FromSlice([]float32{0}, 1),
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := out["y"].Item[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(got)-5) > 1e-5 {
		t.Fatalf("attention with a sink = %v, want 5", got)
	}
}

func TestCompositesWithoutAReferenceBody(t *testing.T) {
	x := []float32{0.5, 2, 1, 3, 0.25, 4}
	gelu := func(v float64) float64 { return v * (1 + math.Erf(v/math.Sqrt2)) / 2 }
	geluTanh := func(v float64) float64 {
		return v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v))) / 2
	}
	logSoftmax := func(row []float64) []float64 {
		m, s := math.Inf(-1), 0.0
		for _, v := range row {
			m = max(m, v)
		}
		for _, v := range row {
			s += math.Exp(v - m)
		}
		out := make([]float64, len(row))
		for i, v := range row {
			out[i] = v - m - math.Log(s)
		}
		return out
	}
	rows := func(f func([]float64) []float64) []float64 {
		var out []float64
		for r := range 2 {
			row := make([]float64, 3)
			for i := range row {
				row[i] = float64(x[3*r+i])
			}
			out = append(out, f(row)...)
		}
		return out
	}
	each := func(f func(float64) float64) []float64 {
		out := make([]float64, len(x))
		for i, v := range x {
			out[i] = f(float64(v))
		}
		return out
	}
	for _, tc := range []struct {
		name   string
		b      *builder
		inputs int
		want   []float64
	}{
		{"log", callGraph("log", 1, []int{2, 3}, [][]int{{2, 3}}), 1, each(math.Log)},
		{"rsqrt", callGraph("rsqrt", 1, []int{2, 3}, [][]int{{2, 3}}), 1, each(func(v float64) float64 { return 1 / math.Sqrt(v) })},
		{"log_softmax", callGraph("log_softmax", 1, []int{2, 3}, [][]int{{2, 3}}), 1, rows(logSoftmax)},
		{"variance", callGraph("variance", 1, []int{2, 1}, [][]int{{2, 3}}, aInts("axes", 1), aInt("correction", 1)), 1, []float64{0.5833333, 3.7708333}},
		{"geglu", callGraph("geglu", 1, []int{2, 3}, [][]int{{2, 3}, {2, 3}}), 2, each(func(v float64) float64 { return gelu(v) * v })},
		{"geglu/tanh", callGraph("geglu", 1, []int{2, 3}, [][]int{{2, 3}, {2, 3}}, aStr("approximate", "tanh")), 2, each(func(v float64) float64 { return geluTanh(v) * v })},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := runOne(t, tc.b, x, tc.inputs, "y")
			for i, w := range tc.want {
				if math.Abs(got[i]-w) > 1e-5*max(1, math.Abs(w)) {
					t.Fatalf("element %d = %g, want %g", i, got[i], w)
				}
			}
		})
	}
	t.Run("top_k", func(t *testing.T) {
		b := newBuilder()
		in := b.input("a", mimfile.F32, 2, 3)
		v := b.value("values", mimfile.F32, mimfile.KindOutput, 2, 2)
		i := b.value("indices", mimfile.I64, mimfile.KindOutput, 2, 2)
		b.g.Subgraphs[0].Ops = []mimfile.Op{{Opcode: mimfile.OpCall, Inputs: []uint32{in}, Outputs: []uint32{v, i},
			Attrs: calls("top_k", 1, aInt("k", 2), aInt("largest", 0))}}
		b.g.Subgraphs[0].Outputs = []uint32{v, i}
		if got := runOne(t, b, x, 1, "values"); !slicesEqual(got, []float64{0.5, 1, 0.25, 3}) {
			t.Errorf("values %v", got)
		}
		if got := runOne(t, b, x, 1, "indices"); !slicesEqual(got, []float64{0, 2, 1, 0}) {
			t.Errorf("indices %v", got)
		}
	})
	t.Run("argsort", func(t *testing.T) {
		b := newBuilder()
		in := b.input("a", mimfile.F32, 2, 3)
		b.output("y", b.op(0, mimfile.OpCall, mimfile.I64, []int{2, 3}, []uint32{in}, calls("argsort", 1, aInt("descending", 1))...))
		if got := runOne(t, b, x, 1, "y"); !slicesEqual(got, []float64{1, 2, 0, 2, 0, 1}) {
			t.Errorf("indices %v", got)
		}
	})
}

// runOne runs the graph with x, of shape [2, 3], as each of its first n
// inputs and returns the output name.
func runOne(t *testing.T, b *builder, x []float32, n int, name string) []float64 {
	t.Helper()
	m, err := mim.Parse(b.file(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	in := map[string]*tensor.Tensor{}
	for i := range n {
		in[string(rune('a'+i))] = tensor.FromSlice(x, 2, 3)
	}
	out, err := m.Run(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	y := out[name]
	var got []float64
	if y.DType() == tensor.Int32 {
		v, err := y.Data[int32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range v {
			got = append(got, float64(e))
		}
		return got
	}
	v, err := y.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range v {
		got = append(got, float64(e))
	}
	return got
}

func slicesEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
