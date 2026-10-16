package onnx

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// TestAttentionFusion checks that attention composed of MatMul and
// Softmax runs fused and computes what the composition does, which a
// copy of the graph that also outputs the scores keeps unfused.
func TestAttentionFusion(t *testing.T) {
	ctx := context.Background()
	const b, h, n, d = 2, 2, 5, 4
	q, k, v := random(200, b, h, n, d), random(201, b, h, n, d), random(202, b, h, n, d)
	add := random(203, n, n)
	for i := range n {
		for j := range n {
			if (i+2*j)%4 == 3 {
				add.data[i*n+j] = math.Inf(-1)
			}
		}
	}
	keep := tensor.FromSlice([]int32{1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, n, n)
	inputs := map[string]*tensor.Tensor{"q": q.tensor(), "k": k.tensor(), "v": v.tensor(), "add": add.tensor(), "keep": keep}
	base := func() *graph {
		return newGraph().input("q", Float, b, h, "n", d).input("k", Float, b, h, "n", d).input("v", Float, b, h, "n", d).
			input("add", Float, n, n).input("keep", Bool, n, n).
			init("rsqrt", []int{}, []float32{float32(math.Pow(d, -0.25))}).init("sqrt", []int{}, []float32{float32(math.Sqrt(d))}).
			init("ninf", []int{}, []float32{float32(math.Inf(-1))}).initInts("one", []int{}, []int64{1})
	}
	cases := map[string]func(g *graph) *graph{
		"divided": func(g *graph) *graph {
			return g.node("Transpose", []string{"k", "->", "kt"}, aInts("perm", 0, 1, 3, 2)).
				node("MatMul", []string{"q", "kt", "->", "s0"}).node("Div", []string{"s0", "sqrt", "->", "s"})
		},
		"additive": func(g *graph) *graph {
			return g.node("Mul", []string{"q", "rsqrt", "->", "qs"}).
				node("Transpose", []string{"k", "->", "kt"}, aInts("perm", 0, 1, 3, 2)).node("Mul", []string{"kt", "rsqrt", "->", "kts"}).
				node("MatMul", []string{"qs", "kts", "->", "s0"}).node("Add", []string{"add", "s0", "->", "s"})
		},
		"causal": func(g *graph) *graph {
			return g.node("Shape", []string{"q", "->", "shape"}).
				initInts("tail", []int{1}, []int64{-2}).initInts("end", []int{1}, []int64{math.MaxInt64}).
				node("Slice", []string{"shape", "tail", "end", "->", "nd"}).
				initInts("i0", []int{1}, []int64{0}).node("Gather", []string{"nd", "i0", "->", "n1"}).
				node("Concat", []string{"n1", "n1", "->", "square"}, aInt("axis", 0)).
				node("ConstantOfShape", []string{"square", "->", "ones"}, aTensor("value", tensorInts("", []int{1}, []int64{1}))).
				node("Trilu", []string{"ones", "one", "->", "tri"}).node("Cast", []string{"tri", "->", "above"}, aInt("to", int64(Bool))).
				node("Transpose", []string{"k", "->", "kt"}, aInts("perm", 0, 1, 3, 2)).
				node("MatMul", []string{"q", "kt", "->", "s0"}).node("Where", []string{"above", "ninf", "s0", "->", "s"})
		},
		"kept": func(g *graph) *graph {
			return g.node("Transpose", []string{"k", "->", "kt"}, aInts("perm", 0, 1, 3, 2)).
				node("MatMul", []string{"q", "kt", "->", "s0"}).node("Where", []string{"keep", "s0", "ninf", "->", "s"})
		},
	}
	for name, scores := range cases {
		t.Run(name, func(t *testing.T) {
			var outs [2][]float32
			for i, fuse := range []bool{true, false} {
				g := scores(base()).node("Softmax", []string{"s", "->", "p"}, aInt("axis", -1)).
					node("MatMul", []string{"p", "v", "->", "y"}).output("y")
				if !fuse {
					g.output("s0")
				}
				m := open(t, g)
				fused := slices.ContainsFunc(m.nodes, func(n *node) bool { return n.op == "Attention" })
				if fused != fuse {
					t.Fatalf("fused %v, want %v", fused, fuse)
				}
				res, err := m.Run(ctx, inputs)
				if err != nil {
					t.Fatal(err)
				}
				if outs[i], err = res["y"].Data[float32](ctx); err != nil {
					t.Fatal(err)
				}
			}
			want := &arr{shape: []int{b, h, n, d}, data: make([]float64, len(outs[1]))}
			for i, x := range outs[1] {
				want.data[i] = float64(x)
			}
			expect(t, outs[0], want.shape, want, 1e-5)
		})
	}
}

func TestCausalOffset(t *testing.T) {
	mask := func(nq, nk, off int, kind maskKind) *value {
		v := hostInts(Bool, []int{nq, nk}, make([]int64, nq*nk))
		for i := range nq {
			for j := range nk {
				if (j <= i+off) == (kind == keepWhere) {
					v.ints[i*nk+j] = 1
				}
			}
		}
		return v
	}
	for _, tc := range []struct{ nq, nk, off int }{{4, 4, 0}, {2, 5, 3}, {3, 3, 1}} {
		for _, kind := range []maskKind{keepWhere, hideWhere} {
			if off, ok := causalOffset(mask(tc.nq, tc.nk, tc.off, kind), kind, tc.nq, tc.nk); !ok || off != tc.off {
				t.Errorf("%+v kind %d: offset %d, %v", tc, kind, off, ok)
			}
		}
	}
	m := mask(4, 4, 0, keepWhere)
	m.ints[2] = 1
	if _, ok := causalOffset(m, keepWhere, 4, 4); ok {
		t.Error("a mask with a hole is causal")
	}
	add := hostFloats(Float, []int{2, 2}, []float64{0, math.Inf(-1), 0, 0})
	if off, ok := causalOffset(add, additiveMask, 2, 2); !ok || off != 0 {
		t.Errorf("additive mask: %d, %v", off, ok)
	}
	add.floats[2] = -1
	if _, ok := causalOffset(add, additiveMask, 2, 2); ok {
		t.Error("an additive mask of -1 is causal")
	}
}
