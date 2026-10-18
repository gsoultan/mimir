package tensor_test

import (
	"math"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/tensor"
)

type goldenCase struct {
	fn  func(in ...*tensor.Tensor) *tensor.Tensor
	tol goldentest.Tol
}

var (
	tight = goldentest.Tol{Atol: 1e-6, Rtol: 1e-6}
	loose = goldentest.Tol{Atol: 1e-5, Rtol: 1e-5}
	exact = goldentest.Tol{}
)

func un(f func(*tensor.Tensor) *tensor.Tensor, tol goldentest.Tol) goldenCase {
	return goldenCase{func(in ...*tensor.Tensor) *tensor.Tensor { return f(in[0]) }, tol}
}

func bin(f func(x, y *tensor.Tensor) *tensor.Tensor, tol goldentest.Tol) goldenCase {
	return goldenCase{func(in ...*tensor.Tensor) *tensor.Tensor { return f(in[0], in[1]) }, tol}
}

// goldenCases mirrors the cases in tools/golden/gen.py.
var goldenCases = map[string]goldenCase{
	"neg":     un((*tensor.Tensor).Neg, exact),
	"exp":     un((*tensor.Tensor).Exp, goldentest.Tol{Atol: 1e-6, Rtol: 2e-6}),
	"exp2":    un((*tensor.Tensor).Exp2, tight),
	"log":     un((*tensor.Tensor).Log, tight),
	"log2":    un((*tensor.Tensor).Log2, tight),
	"sin":     un((*tensor.Tensor).Sin, tight),
	"cos":     un((*tensor.Tensor).Cos, goldentest.Tol{Atol: 1e-6, Rtol: 0}),
	"sqrt":    un((*tensor.Tensor).Sqrt, tight),
	"recip":   un((*tensor.Tensor).Recip, tight),
	"abs":     un((*tensor.Tensor).Abs, exact),
	"relu":    un((*tensor.Tensor).ReLU, exact),
	"sigmoid": un((*tensor.Tensor).Sigmoid, tight),
	"tanh":    un((*tensor.Tensor).Tanh, goldentest.Tol{Atol: 2e-7, Rtol: 1e-6}),
	"gelu":    un((*tensor.Tensor).GELU, tight),
	"silu":    un((*tensor.Tensor).SiLU, tight),

	"add":     bin((*tensor.Tensor).Add, tight),
	"sub":     bin((*tensor.Tensor).Sub, tight),
	"mul":     bin((*tensor.Tensor).Mul, tight),
	"div":     bin((*tensor.Tensor).Div, tight),
	"maximum": bin((*tensor.Tensor).Maximum, exact),
	"minimum": bin((*tensor.Tensor).Minimum, exact),
	"less":    bin((*tensor.Tensor).Less, exact),
	"greater": bin((*tensor.Tensor).Greater, exact),
	"equal":   bin((*tensor.Tensor).Equal, exact),
	"fmod":    bin((*tensor.Tensor).Fmod, tight),
	"where": {func(in ...*tensor.Tensor) *tensor.Tensor {
		return tensor.Where(in[0], in[1], in[2])
	}, exact},

	"add_int":  bin((*tensor.Tensor).Add, exact),
	"mul_int":  bin((*tensor.Tensor).Mul, exact),
	"fmod_int": bin((*tensor.Tensor).Fmod, exact),
	"sum_int":  un(func(x *tensor.Tensor) *tensor.Tensor { return x.Sum(1) }, exact),
	"cast_f2i": un(func(x *tensor.Tensor) *tensor.Tensor { return x.Cast(tensor.Int32) }, exact),
	"cast_i2f": un(func(x *tensor.Tensor) *tensor.Tensor { return x.Cast(tensor.Float32) }, exact),

	"sum_all":           un(func(x *tensor.Tensor) *tensor.Tensor { return x.Sum() }, loose),
	"sum_axis1":         un(func(x *tensor.Tensor) *tensor.Tensor { return x.Sum(1) }, loose),
	"sum_axes02":        un(func(x *tensor.Tensor) *tensor.Tensor { return x.Sum(0, 2) }, loose),
	"max_axis2":         un(func(x *tensor.Tensor) *tensor.Tensor { return x.Max(2) }, exact),
	"min_axis0":         un(func(x *tensor.Tensor) *tensor.Tensor { return x.Min(0) }, exact),
	"mean_last":         un(func(x *tensor.Tensor) *tensor.Tensor { return x.Mean(-1) }, loose),
	"var_axis1":         un(func(x *tensor.Tensor) *tensor.Tensor { return x.Var(1) }, loose),
	"softmax_axis1":     un(func(x *tensor.Tensor) *tensor.Tensor { return x.Softmax(1) }, tight),
	"log_softmax_large": un(func(x *tensor.Tensor) *tensor.Tensor { return x.LogSoftmax(-1) }, loose),

	"matmul_2d":      bin((*tensor.Tensor).MatMul, loose),
	"matmul_batched": bin((*tensor.Tensor).MatMul, loose),
	"matmul_vec_mat": bin((*tensor.Tensor).MatMul, loose),
	"matmul_mat_vec": bin((*tensor.Tensor).MatMul, loose),
	"matmul_transposed": bin(func(x, y *tensor.Tensor) *tensor.Tensor {
		return x.Transpose(0, 1).MatMul(y)
	}, loose),

	"permute_reshape": un(func(x *tensor.Tensor) *tensor.Tensor { return x.Permute(2, 0, 1).Reshape(4, 6) }, exact),
	"expand":          un(func(x *tensor.Tensor) *tensor.Tensor { return x.Expand(2, 3, 4) }, exact),
	"pad":             un(func(x *tensor.Tensor) *tensor.Tensor { return x.Pad([][2]int{{1, 0}, {2, 1}}, -1) }, exact),
	"shrink":          un(func(x *tensor.Tensor) *tensor.Tensor { return x.Shrink([][2]int{{1, 4}, {2, 6}}) }, exact),
	"flip":            un(func(x *tensor.Tensor) *tensor.Tensor { return x.Flip(0, 1) }, exact),
	"shrink_flip_add": un(func(x *tensor.Tensor) *tensor.Tensor {
		return x.Shrink([][2]int{{1, 4}, {2, 6}}).Flip(0).Add(x.Shrink([][2]int{{0, 3}, {0, 4}}))
	}, tight),

	"gather": {func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Gather(1, in[1]) }, exact},
	"scatter_add": {func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].ScatterAdd(0, in[1], in[2])
	}, tight},

	"rand_uniform": {func(...*tensor.Tensor) *tensor.Tensor { return tensor.Rand(7, 1001) }, exact},
	"rand_normal":  {func(...*tensor.Tensor) *tensor.Tensor { return tensor.Randn(7, 1001) }, tight},

	"layer_norm": {func(in ...*tensor.Tensor) *tensor.Tensor {
		x, g, b := in[0], in[1], in[2]
		m := x.Mean(-1).Unsqueeze(-1)
		d := x.Sub(m)
		v := d.Mul(d).Mean(-1).Unsqueeze(-1)
		return d.Div(v.Add(tensor.Scalar[float32](1e-5)).Sqrt()).Mul(g).Add(b)
	}, loose},
	"attention": {func(in ...*tensor.Tensor) *tensor.Tensor {
		q, k, v := in[0], in[1], in[2]
		scale := tensor.Scalar(float32(1 / math.Sqrt(8)))
		return q.MatMul(k.Transpose(-1, -2)).Mul(scale).Softmax(-1).MatMul(v)
	}, loose},
	"mlp": {func(in ...*tensor.Tensor) *tensor.Tensor {
		x, w1, b1, w2, b2 := in[0], in[1], in[2], in[3], in[4]
		return x.MatMul(w1).Add(b1).GELU().MatMul(w2).Add(b2).LogSoftmax(-1)
	}, loose},
}

// TestGolden checks every operation against outputs computed by PyTorch or
// NumPy (see tools/golden/gen.py).
func TestGolden(t *testing.T) {
	cases, meta := goldentest.Load(t, "ops")
	t.Logf("golden oracle: %s", meta["oracle"])
	for name := range goldenCases {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			gc, ok := goldenCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := make([]*tensor.Tensor, len(c.Inputs))
			for i, g := range c.Inputs {
				in[i] = goldentest.Tensor(t, g)
			}
			out := gc.fn(in...)
			if err := out.Err(); err != nil {
				t.Fatal(err)
			}
			switch c.Want.DType {
			case "f32":
				got, err := out.Data[float32](t.Context())
				if err != nil {
					t.Fatal(err)
				}
				goldentest.Compare(t, c.Want, got, out.Shape(), gc.tol)
			default:
				got, err := out.Data[int32](t.Context())
				if err != nil {
					t.Fatal(err)
				}
				goldentest.Compare(t, c.Want, got, out.Shape(), gc.tol)
			}
		})
	}
}

// TestGoldenSplit checks Split against the golden data of the slices of
// "shrink_flip_add", taking them as parts of splits instead of shrinks:
// the output and, through Split's gradient, the input's gradient.
func TestGoldenSplit(t *testing.T) {
	fn := func(x *tensor.Tensor) *tensor.Tensor {
		mid := x.Split(0, 1, 3, 1)[1].Split(1, 2, 4)[1]
		corner := x.Split(0, 3, 2)[0].Split(1, 4, 2)[0]
		return mid.Flip(0).Add(corner)
	}
	ops, _ := goldentest.Load(t, "ops")
	c, ok := ops["shrink_flip_add"]
	if !ok {
		t.Fatal(`no golden case "shrink_flip_add" in ops`)
	}
	goldentest.CompareTensor(t, c.Want, fn(goldentest.Tensor(t, c.Inputs[0])), tight)
	grads, _ := goldentest.Load(t, "grads")
	if c, ok = grads["shrink_flip_add"]; !ok {
		t.Fatal(`no golden case "shrink_flip_add" in grads`)
	}
	in := goldentest.Inputs(t, c)
	goldentest.CheckGradients(t, c, fn(in[0]), map[int]goldentest.Grader{0: in[0]}, goldentest.Tol{Atol: 1e-5, Rtol: 1e-4})
}

// gradOnlyCases are the gradient goldens without a forward-only
// counterpart in goldenCases.
var gradOnlyCases = map[string]func(in ...*tensor.Tensor) *tensor.Tensor{
	"log_softmax":    func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].LogSoftmax(-1) },
	"unfold_overlap": func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Unfold(1, 4, 2) },
	"unfold_gapped":  func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Unfold(1, 2, 3) },
	"cat":            func(in ...*tensor.Tensor) *tensor.Tensor { return tensor.Cat(1, in[0], in[1]) },
	"conv2d": func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].Conv2d(in[1], in[2], tensor.ConvOptions{Padding: 1})
	},
	"conv2d_grouped": func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].Conv2d(in[1], nil, tensor.ConvOptions{Stride: 2, Padding: 1, Dilation: 2, Groups: 2})
	},
	"max_pool2d": func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].MaxPool2d(2, tensor.ConvOptions{}) },
	"max_pool2d_padded": func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].MaxPool2d(3, tensor.ConvOptions{Stride: 2, Padding: 1})
	},
	"avg_pool2d": func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].AvgPool2d(3, tensor.ConvOptions{Stride: 1, Padding: 1})
	},
}

// TestGoldenGradients checks the output and input gradients of operations
// against PyTorch's autograd or NumPy central differences in float64.
func TestGoldenGradients(t *testing.T) {
	cases, meta := goldentest.Load(t, "grads")
	t.Logf("golden oracle: %s", meta["oracle"])
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			fn := gradOnlyCases[name]
			if fn == nil {
				gc, ok := goldenCases[name]
				if !ok {
					t.Fatalf("golden case %q has no Go counterpart", name)
				}
				fn = gc.fn
			}
			in := goldentest.Inputs(t, c)
			grads := map[int]goldentest.Grader{}
			for i := range c.Grads {
				grads[i] = in[i]
			}
			goldentest.CheckGradients(t, c, fn(in...), grads, goldentest.Tol{Atol: 1e-5, Rtol: 1e-4})
		})
	}
}
