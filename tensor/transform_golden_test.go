package tensor_test

import (
	"fmt"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/tensor"
)

// transformCase is a golden case of transforms.mim: a function of the
// case's inputs and the inputs it is differentiated with respect to.
type transformCase struct {
	kind string // hvp, jvp, jac, hess or perex, as in tools/golden/gen.py
	diff []int
	f    func(in ...*tensor.Tensor) *tensor.Tensor
}

// transformGoldenCases mirrors transform_cases in tools/golden/gen.py.
var transformGoldenCases = map[string]transformCase{
	"hvp_mlp": {"hvp", []int{0, 1, 2, 3}, func(in ...*tensor.Tensor) *tensor.Tensor {
		y := in[0].MatMul(in[1]).Add(in[2]).Tanh().MatMul(in[3])
		return y.Mul(y).Sum().Mul(tensor.Scalar[float32](0.25))
	}},
	"hvp_cross_entropy": {"hvp", []int{0, 1}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].MatMul(in[1]).LogSoftmax(-1).Gather(1, in[2].Unsqueeze(1)).Mean().Neg()
	}},
	"hvp_attention_causal": {"hvp", []int{0, 1, 2}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return tensor.ScaledDotProductAttention(in[0], in[1], in[2], tensor.CausalOffset(2)).Mul(in[3]).Sum()
	}},
	"hvp_conv_pool": {"hvp", []int{0, 1, 2}, func(in ...*tensor.Tensor) *tensor.Tensor {
		y := in[0].Conv2d(in[1], in[2], tensor.ConvOptions{Padding: 1}).MaxPool2d(2, tensor.ConvOptions{})
		return y.Mul(y).Mul(in[3]).Sum()
	}},
	"hvp_logcumsumexp": {"hvp", []int{0}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].LogCumSumExp(-1).Mul(in[1]).Sum()
	}},
	"jvp_mlp": {"jvp", []int{0, 1, 2}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].MatMul(in[1]).Add(in[2]).Tanh()
	}},
	"jvp_attention_gqa": {"jvp", []int{0, 1, 2}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return tensor.ScaledDotProductAttention(in[0], in[1], in[2])
	}},
	"jvp_softmax_cumsum": {"jvp", []int{0}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].Softmax(-1).CumSum(0)
	}},
	"jvp_conv_avg_pool": {"jvp", []int{0, 1}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].Conv2d(in[1], nil, tensor.ConvOptions{Padding: 1}).AvgPool2d(2, tensor.ConvOptions{})
	}},
	"jacobian_softmax": {"jac", []int{0}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].MatMul(in[1]).Softmax(-1)
	}},
	"jacobian_rows": {"jac", []int{0}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].MatMul(in[1]).Tanh().Sum(0)
	}},
	"hessian_logsumexp": {"hess", []int{0}, func(in ...*tensor.Tensor) *tensor.Tensor {
		return in[0].Sin().MatMul(in[1]).Exp().Sum().Log()
	}},
	"per_example_mlp": {"perex", []int{2, 3, 4}, func(in ...*tensor.Tensor) *tensor.Tensor {
		logits := in[0].MatMul(in[2]).Add(in[3]).ReLU().MatMul(in[4])
		return logits.LogSoftmax(-1).Gather(0, in[1].Unsqueeze(0)).Neg().Squeeze(0)
	}},
}

// TestGoldenTransforms checks second derivatives, forward derivatives,
// Jacobians, Hessians and per-example gradients against PyTorch's
// torch.func and torch.autograd.functional, or NumPy (see
// tools/golden/gen.py).
func TestGoldenTransforms(t *testing.T) {
	cases, meta := goldentest.Load(t, "transforms")
	t.Logf("golden oracle: %s", meta["oracle"])
	for name := range transformGoldenCases {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	tol := goldentest.Tol{Atol: 1e-5, Rtol: 1e-4}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			tc, ok := transformGoldenCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := goldentest.Inputs(t, c)
			// partial is f of the inputs in diff, the others fixed.
			partial := func(x ...*tensor.Tensor) *tensor.Tensor {
				args := append([]*tensor.Tensor(nil), in...)
				for k, i := range tc.diff {
					args[i] = x[k]
				}
				return tc.f(args...)
			}
			x := make([]*tensor.Tensor, len(tc.diff))
			for k, i := range tc.diff {
				x[k] = in[i]
			}
			extra := func(part string) *tensor.Tensor {
				g, ok := c.Extra[part]
				if !ok {
					t.Fatalf("golden case %q has no %s", name, part)
				}
				return goldentest.Tensor(t, g)
			}
			switch tc.kind {
			case "hvp":
				v := make([]*tensor.Tensor, len(tc.diff))
				for k, i := range tc.diff {
					v[k] = extra(fmt.Sprintf("vec%d", i))
				}
				_, hvp := tensor.JVP(tensor.Grad(partial), x, v)
				goldentest.CompareTensor(t, c.Want, partial(x...), tol)
				for k, i := range tc.diff {
					t.Run(fmt.Sprintf("hvp%d", i), func(t *testing.T) {
						goldentest.CompareTensor(t, c.Extra[fmt.Sprintf("hvp%d", i)], hvp[k], tol)
					})
				}
			case "jvp":
				v := make([]*tensor.Tensor, len(tc.diff))
				for k, i := range tc.diff {
					v[k] = extra(fmt.Sprintf("tan%d", i))
				}
				f := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{partial(x...)} }
				y, jvp := tensor.JVP(f, x, v)
				goldentest.CompareTensor(t, c.Want, y[0], tol)
				goldentest.CompareTensor(t, c.Extra["jvp"], jvp[0], tol)
			case "jac":
				goldentest.CompareTensor(t, c.Want, partial(x...), tol)
				goldentest.CompareTensor(t, c.Extra["jac"], tensor.Jacobian(func(x *tensor.Tensor) *tensor.Tensor { return partial(x) }, x[0]), tol)
			case "hess":
				goldentest.CompareTensor(t, c.Want, partial(x...), tol)
				goldentest.CompareTensor(t, c.Extra["hess"], tensor.Hessian(func(x *tensor.Tensor) *tensor.Tensor { return partial(x) }, x[0]), tol)
			case "perex":
				// Each example's loss and its gradients with respect to
				// the parameters, mapped over the batch.
				perExample := tensor.VMap(func(ex ...*tensor.Tensor) []*tensor.Tensor {
					loss := func(p ...*tensor.Tensor) *tensor.Tensor { return tc.f(append(ex[:2:2], p...)...) }
					return append([]*tensor.Tensor{loss(x...)}, tensor.Grad(loss)(x...)...)
				})
				got := perExample(in[0], in[1])
				goldentest.CompareTensor(t, c.Want, got[0], tol)
				for k, i := range tc.diff {
					t.Run(fmt.Sprintf("pergrad%d", i), func(t *testing.T) {
						goldentest.CompareTensor(t, c.Extra[fmt.Sprintf("pergrad%d", i)], got[1+k], tol)
					})
				}
			default:
				t.Fatalf("unknown kind %q", tc.kind)
			}
		})
	}
}
