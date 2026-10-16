package onnx

import (
	"context"
	"math"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// weights returns the values of p.
func weights(tb testing.TB, p *tensor.Param) []float32 {
	tb.Helper()
	v, err := p.Value().Data[float32](context.Background())
	if err != nil {
		tb.Fatal(err)
	}
	return v
}

// mnistCNN returns the CNN of the README, and the same network as an
// ONNX graph in the manner of torch.onnx.export.
func mnistCNN(tb testing.TB) (*nn.Sequential, *graph) {
	nn.ManualSeed(1)
	c1, c2 := nn.NewConv2d(1, 32, 3, nn.Padding(1)), nn.NewConv2d(32, 64, 3, nn.Padding(1))
	l1, l2 := nn.NewLinear(64*7*7, 128), nn.NewLinear(128, 10)
	model := nn.NewSequential(c1, nn.ReLU(), nn.MaxPool2d(2), c2, nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), l1, nn.ReLU(), l2)
	g := newGraph().input("x", Float, "batch", 1, 28, 28).
		init("c1.w", []int{32, 1, 3, 3}, weights(tb, c1.Weight)).init("c1.b", []int{32}, weights(tb, c1.Bias)).
		init("c2.w", []int{64, 32, 3, 3}, weights(tb, c2.Weight)).init("c2.b", []int{64}, weights(tb, c2.Bias)).
		init("l1.w", []int{128, 64 * 7 * 7}, weights(tb, l1.Weight)).init("l1.b", []int{128}, weights(tb, l1.Bias)).
		init("l2.w", []int{10, 128}, weights(tb, l2.Weight)).init("l2.b", []int{10}, weights(tb, l2.Bias))
	conv := []*pb{aInts("kernel_shape", 3, 3), aInts("pads", 1, 1, 1, 1)}
	pool := []*pb{aInts("kernel_shape", 2, 2), aInts("strides", 2, 2)}
	g.node("Conv", []string{"x", "c1.w", "c1.b", "->", "a"}, conv...).node("Relu", []string{"a", "->", "b"}).
		node("MaxPool", []string{"b", "->", "c"}, pool...).
		node("Conv", []string{"c", "c2.w", "c2.b", "->", "d"}, conv...).node("Relu", []string{"d", "->", "e"}).
		node("MaxPool", []string{"e", "->", "f"}, pool...).node("Flatten", []string{"f", "->", "g"}).
		node("Gemm", []string{"g", "l1.w", "l1.b", "->", "h"}, aInt("transB", 1)).node("Relu", []string{"h", "->", "i"}).
		node("Gemm", []string{"i", "l2.w", "l2.b", "->", "y"}, aInt("transB", 1)).output("y")
	return model, g
}

// encoder returns layers causal post-norm transformer encoder layers of
// dim features, and the same network as an ONNX graph in the manner of
// torch.onnx.export at opset 17: attention composed of MatMul, Softmax
// and a mask that Trilu computes, and head shapes that Shape, Gather and
// Concat compute.
func encoder(tb testing.TB, layers, dim, heads int) ([]*nn.TransformerEncoderLayer, *graph) {
	nn.ManualSeed(2)
	ls := make([]*nn.TransformerEncoderLayer, layers)
	g := newGraph().input("x", Float, "batch", "seq", dim)
	dh := dim / heads
	g.initInts("i0", []int{}, []int64{0}).initInts("i1", []int{}, []int64{1}).initInts("ax0", []int{1}, []int64{0}).
		initInts("k1", []int{}, []int64{1}).initInts("hd", []int{2}, []int64{int64(heads), int64(dh)}).
		initInts("m1", []int{1}, []int64{-1}).initInts("split", []int{3}, []int64{int64(dim), int64(dim), int64(dim)}).
		init("ninf", []int{}, []float32{float32(math.Inf(-1))}).init("scale", []int{}, []float32{float32(1 / math.Sqrt(float64(dh)))})
	g.node("Shape", []string{"x", "->", "shape"}).
		node("Gather", []string{"shape", "i0", "->", "b"}).node("Gather", []string{"shape", "i1", "->", "s"}).
		node("Unsqueeze", []string{"b", "ax0", "->", "b1"}).node("Unsqueeze", []string{"s", "ax0", "->", "s1"}).
		node("Concat", []string{"b1", "s1", "hd", "->", "heads"}, aInt("axis", 0)).
		node("Concat", []string{"b1", "s1", "m1", "->", "merged"}, aInt("axis", 0)).
		node("Concat", []string{"s1", "s1", "->", "square"}, aInt("axis", 0)).
		node("ConstantOfShape", []string{"square", "->", "ones"}, aTensor("value", tensorFloats("", []int{1}, []float32{1}))).
		node("Trilu", []string{"ones", "k1", "->", "tri"}).
		node("Cast", []string{"tri", "->", "above"}, aInt("to", int64(Bool)))
	h := "x"
	for i := range layers {
		l := nn.NewTransformerEncoderLayer(dim, heads, 4*dim)
		nn.Eval(l)
		ls[i] = l
		p := func(s string) string { return "l" + string(rune('0'+i)) + "." + s }
		w := weights(tb, l.SelfAttn.InProjWeight) // [3·dim, dim]
		wt := make([]float32, len(w))
		for r := range 3 * dim {
			for c := range dim {
				wt[c*3*dim+r] = w[r*dim+c]
			}
		}
		transposed := func(param *tensor.Param, out, in int) []float32 {
			v := weights(tb, param)
			t := make([]float32, len(v))
			for r := range out {
				for c := range in {
					t[c*out+r] = v[r*in+c]
				}
			}
			return t
		}
		g.init(p("qkv.w"), []int{dim, 3 * dim}, wt).init(p("qkv.b"), []int{3 * dim}, weights(tb, l.SelfAttn.InProjBias)).
			init(p("o.w"), []int{dim, dim}, transposed(l.SelfAttn.OutProj.Weight, dim, dim)).init(p("o.b"), []int{dim}, weights(tb, l.SelfAttn.OutProj.Bias)).
			init(p("f1.w"), []int{dim, 4 * dim}, transposed(l.Linear1.Weight, 4*dim, dim)).init(p("f1.b"), []int{4 * dim}, weights(tb, l.Linear1.Bias)).
			init(p("f2.w"), []int{4 * dim, dim}, transposed(l.Linear2.Weight, dim, 4*dim)).init(p("f2.b"), []int{dim}, weights(tb, l.Linear2.Bias)).
			init(p("n1.g"), []int{dim}, weights(tb, l.Norm1.Weight)).init(p("n1.b"), []int{dim}, weights(tb, l.Norm1.Bias)).
			init(p("n2.g"), []int{dim}, weights(tb, l.Norm2.Weight)).init(p("n2.b"), []int{dim}, weights(tb, l.Norm2.Bias))
		g.node("MatMul", []string{h, p("qkv.w"), "->", p("qkv0")}).node("Add", []string{p("qkv0"), p("qkv.b"), "->", p("qkv")}).
			node("Split", []string{p("qkv"), "split", "->", p("q0"), p("k0"), p("v0")}, aInt("axis", 2))
		for _, t := range []string{"q", "k", "v"} {
			g.node("Reshape", []string{p(t + "0"), "heads", "->", p(t + "1")}).
				node("Transpose", []string{p(t + "1"), "->", p(t)}, aInts("perm", 0, 2, 1, 3))
		}
		g.node("Transpose", []string{p("k"), "->", p("kt")}, aInts("perm", 0, 1, 3, 2)).
			node("MatMul", []string{p("q"), p("kt"), "->", p("s0")}).node("Mul", []string{p("s0"), "scale", "->", p("s1")}).
			node("Where", []string{"above", "ninf", p("s1"), "->", p("s")}).node("Softmax", []string{p("s"), "->", p("p")}, aInt("axis", -1)).
			node("MatMul", []string{p("p"), p("v"), "->", p("o0")}).node("Transpose", []string{p("o0"), "->", p("o1")}, aInts("perm", 0, 2, 1, 3)).
			node("Reshape", []string{p("o1"), "merged", "->", p("o2")}).
			node("MatMul", []string{p("o2"), p("o.w"), "->", p("o3")}).node("Add", []string{p("o3"), p("o.b"), "->", p("o")}).
			node("Add", []string{h, p("o"), "->", p("r1")}).
			node("LayerNormalization", []string{p("r1"), p("n1.g"), p("n1.b"), "->", p("x1")}).
			node("MatMul", []string{p("x1"), p("f1.w"), "->", p("f0")}).node("Add", []string{p("f0"), p("f1.b"), "->", p("f1")}).
			node("Relu", []string{p("f1"), "->", p("f2")}).
			node("MatMul", []string{p("f2"), p("f2.w"), "->", p("f3")}).node("Add", []string{p("f3"), p("f2.b"), "->", p("f")}).
			node("Add", []string{p("x1"), p("f"), "->", p("r2")}).
			node("LayerNormalization", []string{p("r2"), p("n2.g"), p("n2.b"), "->", p("out")})
		h = p("out")
	}
	g.node("Identity", []string{h, "->", "y"}).output("y")
	return ls, g
}

func encode(ls []*nn.TransformerEncoderLayer, x *tensor.Tensor) *tensor.Tensor {
	for _, l := range ls {
		x = l.Encode(x, tensor.Causal())
	}
	return x
}

// TestExportedModels checks the graphs of the benchmarks against the nn
// models they were written from.
func TestExportedModels(t *testing.T) {
	ctx := context.Background()
	check := func(t *testing.T, g *graph, x, want *tensor.Tensor, tol float64) {
		m := open(t, g)
		res, err := m.Run(ctx, map[string]*tensor.Tensor{"x": x})
		if err != nil {
			t.Fatal(err)
		}
		w, err := want.Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		a := &arr{shape: want.Shape(), data: make([]float64, len(w))}
		for i, v := range w {
			a.data[i] = float64(v)
		}
		got, err := res["y"].Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		expect(t, got, res["y"].Shape(), a, tol)
	}
	t.Run("cnn", func(t *testing.T) {
		model, g := mnistCNN(t)
		x := tensor.Randn(3, 2, 1, 28, 28)
		check(t, g, x, model.Forward(x), 1e-4)
	})
	t.Run("transformer", func(t *testing.T) {
		ls, g := encoder(t, 2, 32, 4)
		x := tensor.Randn(4, 2, 9, 32)
		check(t, g, x, encode(ls, x), 1e-4)
	})
}

// BenchmarkCNN runs the README's MNIST CNN on a batch of 64 through
// ONNX and through nn.
func BenchmarkCNN(b *testing.B) {
	ctx := context.Background()
	model, g := mnistCNN(b)
	x := tensor.Randn(5, 64, 1, 28, 28)
	if err := x.Realize(ctx); err != nil {
		b.Fatal(err)
	}
	b.Run("onnx", func(b *testing.B) {
		m := open(b, g)
		in := map[string]*tensor.Tensor{"x": x}
		for b.Loop() {
			if _, err := m.Run(ctx, in); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("nn", func(b *testing.B) {
		for b.Loop() {
			if err := model.Forward(x).Realize(ctx); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkSmallMLP runs a perceptron of two small layers on one example,
// where building and replaying the graph is most of the work, through
// ONNX and through nn.
func BenchmarkSmallMLP(b *testing.B) {
	ctx := context.Background()
	nn.ManualSeed(3)
	l1, l2 := nn.NewLinear(16, 32), nn.NewLinear(32, 10)
	model := nn.NewSequential(l1, nn.ReLU(), l2)
	g := newGraph().input("x", Float, "batch", 16).
		init("w1", []int{32, 16}, weights(b, l1.Weight)).init("b1", []int{32}, weights(b, l1.Bias)).
		init("w2", []int{10, 32}, weights(b, l2.Weight)).init("b2", []int{10}, weights(b, l2.Bias)).
		node("Gemm", []string{"x", "w1", "b1", "->", "h"}, aInt("transB", 1)).node("Relu", []string{"h", "->", "r"}).
		node("Gemm", []string{"r", "w2", "b2", "->", "y"}, aInt("transB", 1)).output("y")
	x := tensor.Randn(7, 1, 16)
	if err := x.Realize(ctx); err != nil {
		b.Fatal(err)
	}
	b.Run("onnx", func(b *testing.B) {
		m := open(b, g)
		in := map[string]*tensor.Tensor{"x": x}
		for b.Loop() {
			if _, err := m.Run(ctx, in); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("nn", func(b *testing.B) {
		for b.Loop() {
			if err := model.Forward(x).Realize(ctx); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkTransformer runs two causal encoder layers of 128 features
// in 4 heads over 8 sequences of 64 through ONNX and through nn.
func BenchmarkTransformer(b *testing.B) {
	ctx := context.Background()
	ls, g := encoder(b, 2, 128, 4)
	x := tensor.Randn(6, 8, 64, 128)
	if err := x.Realize(ctx); err != nil {
		b.Fatal(err)
	}
	b.Run("onnx", func(b *testing.B) {
		m := open(b, g)
		in := map[string]*tensor.Tensor{"x": x}
		for b.Loop() {
			if _, err := m.Run(ctx, in); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("nn", func(b *testing.B) {
		for b.Loop() {
			if err := encode(ls, x).Realize(ctx); err != nil {
				b.Fatal(err)
			}
		}
	})
}
