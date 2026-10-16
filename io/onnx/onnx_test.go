package onnx

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// arr is a float64 array for computing expected results.
type arr struct {
	shape []int
	data  []float64
}

func newArr(shape ...int) *arr {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return &arr{shape: shape, data: make([]float64, n)}
}

func (a *arr) off(pos ...int) int {
	o := 0
	for i, p := range pos {
		o = o*a.shape[i] + p
	}
	return o
}

func (a *arr) at(pos ...int) float64 { return a.data[a.off(pos...)] }

func (a *arr) f32() []float32 {
	out := make([]float32, len(a.data))
	for i, x := range a.data {
		out[i] = float32(x)
	}
	return out
}

func (a *arr) tensor() *tensor.Tensor { return tensor.FromSlice(a.f32(), a.shape...) }

// random returns an array of values in [-1, 1) from seed.
func random(seed uint64, shape ...int) *arr {
	a := newArr(shape...)
	r := rand.New(rand.NewPCG(seed, 7))
	for i := range a.data {
		a.data[i] = float64(float32(r.Float64()*2 - 1))
	}
	return a
}

// mapArr returns f applied to each element of a.
func mapArr(a *arr, f func(float64) float64) *arr {
	b := newArr(a.shape...)
	for i, x := range a.data {
		b.data[i] = f(x)
	}
	return b
}

// open parses the model g encodes.
func open(t testing.TB, g *graph) *Model {
	t.Helper()
	m, err := Parse(g.model())
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// runOne runs g on inputs and returns its output out as float32s.
func runOne(t testing.TB, g *graph, inputs map[string]*tensor.Tensor, out string) ([]float32, []int) {
	t.Helper()
	m := open(t, g)
	res, err := m.Run(context.Background(), inputs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	y, ok := res[out]
	if !ok {
		t.Fatalf("no output %q in %v", out, res)
	}
	if y.DType() != tensor.Float32 {
		y = y.Cast(tensor.Float32)
	}
	data, err := y.Data[float32](context.Background())
	if err != nil {
		t.Fatalf("Data: %v", err)
	}
	return data, y.Shape()
}

// expect checks got against want to within tol, relative to 1 or to
// want's magnitude.
func expect(t testing.TB, got []float32, gotShape []int, want *arr, tol float64) {
	t.Helper()
	if !slices.Equal(gotShape, want.shape) {
		t.Fatalf("shape %v, want %v", gotShape, want.shape)
	}
	for i, w := range want.data {
		g := float64(got[i])
		if math.IsNaN(w) && math.IsNaN(g) || g == w {
			continue
		}
		if !(math.Abs(g-w) <= tol*max(1, math.Abs(w))) {
			t.Fatalf("element %d is %v, want %v (of %v, want %v)", i, g, w, got[:min(len(got), 12)], want.data[:min(len(want.data), 12)])
		}
	}
}

func TestUnary(t *testing.T) {
	x := random(1, 3, 4)
	pos := mapArr(x, func(v float64) float64 { return math.Abs(v) + 0.1 })
	sig := func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }
	cases := []struct {
		op    string
		in    *arr
		attrs []*pb
		f     func(float64) float64
	}{
		{"Relu", x, nil, func(v float64) float64 { return max(v, 0) }},
		{"Sigmoid", x, nil, sig},
		{"Tanh", x, nil, math.Tanh},
		{"Exp", x, nil, math.Exp},
		{"Log", pos, nil, math.Log},
		{"Sqrt", pos, nil, math.Sqrt},
		{"Neg", x, nil, func(v float64) float64 { return -v }},
		{"Abs", x, nil, math.Abs},
		{"Erf", x, nil, math.Erf},
		{"Reciprocal", pos, nil, func(v float64) float64 { return 1 / v }},
		{"Sin", x, nil, math.Sin},
		{"Cos", x, nil, math.Cos},
		{"Floor", mapArr(x, func(v float64) float64 { return 3 * v }), nil, math.Floor},
		{"Ceil", mapArr(x, func(v float64) float64 { return 3 * v }), nil, math.Ceil},
		{"Sign", x, nil, func(v float64) float64 { return math.Copysign(1, v) }},
		{"Softplus", x, nil, func(v float64) float64 { return math.Log1p(math.Exp(v)) }},
		{"Softsign", x, nil, func(v float64) float64 { return v / (1 + math.Abs(v)) }},
		{"LeakyRelu", x, []*pb{aFloat("alpha", 0.2)}, func(v float64) float64 {
			if v < 0 {
				return float64(float32(0.2)) * v
			}
			return v
		}},
		{"Elu", x, []*pb{aFloat("alpha", 0.5)}, func(v float64) float64 {
			if v < 0 {
				return 0.5 * (math.Exp(v) - 1)
			}
			return v
		}},
		{"HardSigmoid", mapArr(x, func(v float64) float64 { return 5 * v }), nil, func(v float64) float64 {
			return min(max(float64(float32(0.2))*v+0.5, 0), 1)
		}},
		{"HardSwish", mapArr(x, func(v float64) float64 { return 5 * v }), nil, func(v float64) float64 {
			return v * min(max(v/6+0.5, 0), 1)
		}},
		{"Gelu", x, nil, func(v float64) float64 { return v / 2 * (1 + math.Erf(v/math.Sqrt2)) }},
		{"Gelu", x, []*pb{aString("approximate", "tanh")}, func(v float64) float64 {
			return v / 2 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v)))
		}},
		{"Identity", x, nil, func(v float64) float64 { return v }},
		{"Dropout", x, nil, func(v float64) float64 { return v }},
	}
	for _, tc := range cases {
		t.Run(tc.op, func(t *testing.T) {
			g := newGraph().input("x", Float, 3, 4).node(tc.op, []string{"x", "->", "y"}, tc.attrs...).output("y")
			got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": tc.in.tensor()}, "y")
			expect(t, got, shape, mapArr(tc.in, tc.f), 1e-5)
		})
	}
}

func TestBinary(t *testing.T) {
	a, b := random(2, 2, 3), random(3, 3)
	posA := mapArr(a, func(v float64) float64 { return math.Abs(v) + 0.5 })
	cases := []struct {
		op   string
		a, b *arr
		f    func(x, y float64) float64
	}{
		{"Add", a, b, func(x, y float64) float64 { return x + y }},
		{"Sub", a, b, func(x, y float64) float64 { return x - y }},
		{"Mul", a, b, func(x, y float64) float64 { return x * y }},
		{"Div", a, mapArr(b, func(v float64) float64 { return v + 2 }), func(x, y float64) float64 { return x / y }},
		{"Pow", posA, b, math.Pow},
		{"Max", a, b, math.Max},
		{"Min", a, b, math.Min},
		{"Sum", a, b, func(x, y float64) float64 { return x + y }},
		{"Mean", a, b, func(x, y float64) float64 { return (x + y) / 2 }},
		{"Equal", a, a, func(x, y float64) float64 { return b2f(x == y) }},
		{"Less", a, b, func(x, y float64) float64 { return b2f(x < y) }},
		{"Greater", a, b, func(x, y float64) float64 { return b2f(x > y) }},
		{"LessOrEqual", a, b, func(x, y float64) float64 { return b2f(x <= y) }},
		{"GreaterOrEqual", a, b, func(x, y float64) float64 { return b2f(x >= y) }},
		{"Mod", mapArr(a, func(v float64) float64 { return 7 * v }), mapArr(b, func(v float64) float64 { return v + 2 }), func(x, y float64) float64 {
			r := math.Mod(x, y)
			if r != 0 && (r < 0) != (y < 0) {
				r += y
			}
			return r
		}},
	}
	for _, tc := range cases {
		t.Run(tc.op, func(t *testing.T) {
			g := newGraph().input("a", Float, 2, 3).input("b", Float, 3).node(tc.op, []string{"a", "b", "->", "y"}).output("y")
			if tc.op == "Equal" {
				g = newGraph().input("a", Float, 2, 3).input("b", Float, 2, 3).node(tc.op, []string{"a", "b", "->", "y"}).output("y")
			}
			got, shape := runOne(t, g, map[string]*tensor.Tensor{"a": tc.a.tensor(), "b": tc.b.tensor()}, "y")
			want := newArr(2, 3)
			for i := range 2 {
				for j := range 3 {
					y := tc.b.data[j]
					if len(tc.b.shape) == 2 {
						y = tc.b.at(i, j)
					}
					want.data[i*3+j] = tc.f(tc.a.at(i, j), y)
				}
			}
			expect(t, got, shape, want, 1e-5)
		})
	}
}

func TestPowSpecialExponents(t *testing.T) {
	x := random(4, 2, 3)
	for _, e := range []float32{2, 3, 0.5, -1, 1, 1.5} {
		in := x
		if e == 0.5 || e == -1 || e == 1.5 {
			in = mapArr(x, func(v float64) float64 { return math.Abs(v) + 0.25 })
		}
		g := newGraph().input("x", Float, 2, 3).init("e", []int{}, []float32{e}).node("Pow", []string{"x", "e", "->", "y"}).output("y")
		got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": in.tensor()}, "y")
		expect(t, got, shape, mapArr(in, func(v float64) float64 { return math.Pow(v, float64(e)) }), 1e-5)
	}
	// Negative bases with integer exponents keep their sign.
	g := newGraph().input("x", Float, 2, 3).init("e", []int{}, []float32{5}).node("Pow", []string{"x", "e", "->", "y"}).output("y")
	got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
	expect(t, got, shape, mapArr(x, func(v float64) float64 { return math.Pow(v, 5) }), 1e-4)
}

func TestGemm(t *testing.T) {
	a, b, c := random(5, 4, 3), random(6, 5, 4), random(7, 5)
	g := newGraph().input("a", Float, 4, 3).init("b", []int{5, 4}, b.f32()).init("c", []int{5}, c.f32()).
		node("Gemm", []string{"a", "b", "c", "->", "y"}, aInt("transA", 1), aInt("transB", 1), aFloat("alpha", 0.5), aFloat("beta", 2)).output("y")
	got, shape := runOne(t, g, map[string]*tensor.Tensor{"a": a.tensor()}, "y")
	want := newArr(3, 5)
	for i := range 3 {
		for j := range 5 {
			s := 0.0
			for k := range 4 {
				s += a.at(k, i) * b.at(j, k)
			}
			want.data[i*5+j] = 0.5*s + 2*c.data[j]
		}
	}
	expect(t, got, shape, want, 1e-5)

	// MatMul broadcasts batches.
	x, w := random(8, 2, 3, 4), random(9, 4, 2)
	g = newGraph().input("x", Float, 2, 3, 4).init("w", []int{4, 2}, w.f32()).node("MatMul", []string{"x", "w", "->", "y"}).output("y")
	got, shape = runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
	want = newArr(2, 3, 2)
	for b := range 2 {
		for i := range 3 {
			for j := range 2 {
				s := 0.0
				for k := range 4 {
					s += x.at(b, i, k) * w.at(k, j)
				}
				want.data[want.off(b, i, j)] = s
			}
		}
	}
	expect(t, got, shape, want, 1e-5)
}

// TestShapeOnHost checks that the chain Shape, Gather, Unsqueeze, Concat
// that computes a Reshape's target runs on the host.
func TestShapeOnHost(t *testing.T) {
	g := newGraph().input("x", Float, "batch", 3, 4).
		initInts("zero", []int{}, []int64{0}).initInts("minus1", []int{1}, []int64{-1}).
		node("Shape", []string{"x", "->", "s"}).
		node("Gather", []string{"s", "zero", "->", "n"}, aInt("axis", 0)).
		node("Unsqueeze", []string{"n", "axes", "->", "n1"}).
		node("Concat", []string{"n1", "minus1", "->", "shape"}, aInt("axis", 0)).
		node("Reshape", []string{"x", "shape", "->", "y"}).output("y")
	g.initInts("axes", []int{1}, []int64{0})
	x := random(10, 2, 3, 4)
	m := open(t, g)
	r := &run{ctx: context.Background(), m: m, dev: tensor.CPU(), env: map[string]*value{"x": devValue(Float, x.tensor())}}
	for _, n := range m.nodes {
		outs, err := r.eval(n)
		if err != nil {
			t.Fatal(err)
		}
		r.env[n.outputs[0]] = outs[0]
		if n.op != "Reshape" && (!outs[0].host || outs[0].t != nil) {
			t.Errorf("%s computed on a device", n.op)
		}
	}
	if got := r.env["shape"]; !slices.Equal(got.ints, []int64{2, -1}) {
		t.Errorf("shape %v, want [2 -1]", got.ints)
	}
	got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
	want := &arr{shape: []int{2, 12}, data: x.data}
	expect(t, got, shape, want, 0)
}

// TestFoldConstants checks that nodes of initializers alone run once,
// when the model opens.
func TestFoldConstants(t *testing.T) {
	w := random(11, 3, 2)
	g := newGraph().input("x", Float, 1, 2).init("w", []int{3, 2}, w.f32()).
		node("Transpose", []string{"w", "->", "wt"}).
		node("MatMul", []string{"x", "wt", "->", "y"}).output("y")
	m := open(t, g)
	if len(m.nodes) != 1 || m.nodes[0].op != "MatMul" {
		t.Fatalf("nodes after folding: %d", len(m.nodes))
	}
	x := random(12, 1, 2)
	got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
	want := newArr(1, 3)
	for j := range 3 {
		want.data[j] = x.data[0]*w.at(j, 0) + x.data[1]*w.at(j, 1)
	}
	expect(t, got, shape, want, 1e-6)
}

func TestErrors(t *testing.T) {
	ctx := context.Background()
	x := tensor.FromSlice([]float32{1, 2}, 2)
	_, err := Parse(newGraph().input("x", Float, 2).node("Loop", []string{"x", "->", "y"}).output("y").model())
	var u *UnsupportedError
	if !errors.Is(err, ErrUnsupported) || !errors.As(err, &u) || u.Op != "Loop" {
		t.Errorf("Loop: %v", err)
	}
	g := newGraph().input("x", Float, 2).node("Relu", []string{"x", "->", "y"}).output("y")
	g.nodes[0].str(7, "com.microsoft")
	if _, err := Parse(g.model()); !errors.As(err, &u) || u.Domain != "com.microsoft" {
		t.Errorf("custom domain: %v", err)
	}
	if _, err := Parse(newGraph().input("x", Float, 2).node("Relu", []string{"z", "->", "y"}).output("y").model()); !errors.Is(err, ErrFormat) {
		t.Errorf("undefined input: %v", err)
	}
	cyc := newGraph().input("x", Float, 2).node("Add", []string{"x", "b", "->", "a"}).node("Relu", []string{"a", "->", "b"}).output("b")
	if _, err := Parse(cyc.model()); !errors.Is(err, ErrFormat) {
		t.Errorf("cycle: %v", err)
	}
	old := newGraph().input("x", Float, 2).node("Relu", []string{"x", "->", "y"}).output("y")
	old.opset = 9
	if _, err := Parse(old.model()); !errors.Is(err, ErrUnsupported) {
		t.Errorf("opset 9: %v", err)
	}
	m := open(t, newGraph().input("x", Float, 2).node("Relu", []string{"x", "->", "y"}).output("y"))
	if _, err := m.Run(ctx, nil); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("missing input: %v", err)
	}
	if _, err := m.Run(ctx, map[string]*tensor.Tensor{"x": tensor.FromSlice([]int32{1, 2}, 2)}); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("int32 for float: %v", err)
	}
	if _, err := m.Run(ctx, map[string]*tensor.Tensor{"x": tensor.FromSlice([]float32{1, 2, 3}, 3)}); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("wrong shape: %v", err)
	}
	if _, err := m.Run(ctx, map[string]*tensor.Tensor{"x": x, "z": x}); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("extra input: %v", err)
	}
	bad := open(t, newGraph().input("x", Float, 2).initInts("s", []int{1}, []int64{3}).node("Reshape", []string{"x", "s", "->", "y"}).output("y"))
	if _, err := bad.Run(ctx, map[string]*tensor.Tensor{"x": x}); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("bad reshape: %v", err)
	}
	unsupported := open(t, newGraph().input("x", Float, 1, 1, 2, 2).node("Resize", []string{"x", "", "s", "->", "y"}, aString("mode", "cubic")).output("y").init("s", []int{4}, []float32{1, 1, 2, 2}))
	if _, err := unsupported.Run(ctx, map[string]*tensor.Tensor{"x": tensor.Zeros(tensor.Float32, 1, 1, 2, 2)}); !errors.As(err, &u) || u.Op != "Resize" {
		t.Errorf("cubic resize: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Run(ctx, map[string]*tensor.Tensor{"x": x}); err == nil {
		t.Error("run of a closed model")
	}
}

func TestExternalData(t *testing.T) {
	dir := t.TempDir()
	w := random(13, 2, 3)
	raw := make([]byte, 8)
	for _, f := range w.f32() {
		raw = append(raw, byte(math.Float32bits(f)), byte(math.Float32bits(f)>>8), byte(math.Float32bits(f)>>16), byte(math.Float32bits(f)>>24))
	}
	if err := os.WriteFile(filepath.Join(dir, "weights.bin"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	ext := func(loc, offset string) *pb {
		tp := (&pb{}).varint(1, 2).varint(1, 3).varint(2, int64(Float)).str(8, "w")
		tp.msg(13, (&pb{}).str(1, "location").str(2, loc))
		tp.msg(13, (&pb{}).str(1, "offset").str(2, offset))
		tp.msg(13, (&pb{}).str(1, "length").str(2, "24"))
		return tp.varint(14, 1)
	}
	write := func(t *pb) string {
		g := newGraph().input("x", Float, 2, 3).node("Add", []string{"x", "w", "->", "y"}).output("y")
		g.inits = append(g.inits, t)
		path := filepath.Join(dir, "model.onnx")
		if err := os.WriteFile(path, g.model(), 0o600); err != nil {
			panic(err)
		}
		return path
	}
	m, err := Open(write(ext("weights.bin", "8")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	res, err := m.Run(context.Background(), map[string]*tensor.Tensor{"x": tensor.Zeros(tensor.Float32, 2, 3)})
	if err != nil {
		t.Fatal(err)
	}
	got, _ := res["y"].Data[float32](context.Background())
	expect(t, got, res["y"].Shape(), w, 0)
	for _, bad := range []*pb{ext("../weights.bin", "8"), ext("/etc/passwd", "0"), ext("weights.bin", "16"), ext("weights.bin", "-1"), ext("missing.bin", "0")} {
		if m, err := Open(write(bad)); err == nil {
			m.Close()
			t.Errorf("external data %q opened", bad.b)
		}
	}
	if _, err := Parse(g0WithInit(ext("weights.bin", "8"))); !errors.Is(err, ErrFormat) {
		t.Errorf("Parse of external data: %v", err)
	}
}

func g0WithInit(tp *pb) []byte {
	g := newGraph().input("x", Float, 2, 3).node("Add", []string{"x", "w", "->", "y"}).output("y")
	g.inits = append(g.inits, tp)
	return g.model()
}

func TestInputs(t *testing.T) {
	g := newGraph().input("tokens", Int64, "batch", "seq").input("w", Float, 3).init("w", []int{3}, []float32{1, 2, 3}).
		node("Identity", []string{"tokens", "->", "y"}).output("y")
	m := open(t, g)
	in := m.Inputs()
	if len(in) != 1 || in[0].Name != "tokens" || in[0].Type != Int64 || in[0].DType() != tensor.Int32 ||
		!slices.Equal(in[0].Shape, []Dim{{Size: -1, Name: "batch"}, {Size: -1, Name: "seq"}}) {
		t.Errorf("inputs %v", in)
	}
	if got := in[0].String(); got != "tokens: int64[batch seq]" {
		t.Errorf("String() = %q", got)
	}
	if m.Opset() != 17 || m.Producer() != "mimir-test" {
		t.Errorf("opset %d, producer %q", m.Opset(), m.Producer())
	}
	res, err := m.Run(context.Background(), map[string]*tensor.Tensor{"tokens": tensor.FromSlice([]int32{1, 2, 3, 4, 5, 6}, 2, 3)})
	if err != nil {
		t.Fatal(err)
	}
	got, err := res["y"].Data[int32](context.Background())
	if err != nil || !slices.Equal(got, []int32{1, 2, 3, 4, 5, 6}) {
		t.Errorf("y = %v, %v", got, err)
	}
}
