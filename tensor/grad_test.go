package tensor_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// gradCase is a function of float32 inputs whose gradient is checked
// against central differences.
type gradCase struct {
	name string
	in   [][]float32 // values
	dims [][]int     // shapes
	f    func(x []*tensor.Tensor) *tensor.Tensor
}

// vals returns n values in [lo, hi) from r, kept at least gap away from the
// non-differentiable points in avoid.
func vals(r *rand.Rand, n int, lo, hi float32, avoid ...float32) []float32 {
	out := make([]float32, n)
	for i := range out {
	again:
		v := lo + (hi-lo)*r.Float32()
		for _, a := range avoid {
			if math.Abs(float64(v-a)) < 0.1 {
				goto again
			}
		}
		out[i] = v
	}
	return out
}

func gradCases() []gradCase {
	r := rand.New(rand.NewPCG(1, 2))
	n := func(shape ...int) int {
		p := 1
		for _, d := range shape {
			p *= d
		}
		return p
	}
	in := func(lo, hi float32, shape ...int) ([]float32, []int) {
		return vals(r, n(shape...), lo, hi, 0), shape
	}
	var cases []gradCase
	add := func(name string, f func(x []*tensor.Tensor) *tensor.Tensor, ins ...func() ([]float32, []int)) {
		c := gradCase{name: name, f: f}
		for _, mk := range ins {
			v, s := mk()
			c.in, c.dims = append(c.in, v), append(c.dims, s)
		}
		cases = append(cases, c)
	}
	u := func(lo, hi float32, shape ...int) func() ([]float32, []int) {
		return func() ([]float32, []int) { return in(lo, hi, shape...) }
	}
	unary := func(name string, lo, hi float32, f func(x *tensor.Tensor) *tensor.Tensor) {
		add(name, func(x []*tensor.Tensor) *tensor.Tensor { return f(x[0]) }, u(lo, hi, 3, 4))
	}
	unary("neg", -2, 2, (*tensor.Tensor).Neg)
	unary("exp", -2, 2, (*tensor.Tensor).Exp)
	unary("exp2", -2, 2, (*tensor.Tensor).Exp2)
	unary("log", 0.5, 3, (*tensor.Tensor).Log)
	unary("log2", 0.5, 3, (*tensor.Tensor).Log2)
	unary("sin", -3, 3, (*tensor.Tensor).Sin)
	unary("cos", -3, 3, (*tensor.Tensor).Cos)
	unary("sqrt", 0.5, 3, (*tensor.Tensor).Sqrt)
	unary("recip", 0.5, 3, (*tensor.Tensor).Recip)
	unary("abs", -2, 2, (*tensor.Tensor).Abs)
	unary("relu", -2, 2, (*tensor.Tensor).ReLU)
	unary("sigmoid", -3, 3, (*tensor.Tensor).Sigmoid)
	unary("tanh", -2, 2, (*tensor.Tensor).Tanh)
	unary("gelu", -3, 3, (*tensor.Tensor).GELU)
	unary("silu", -3, 3, (*tensor.Tensor).SiLU)

	bin := func(name string, f func(x, y *tensor.Tensor) *tensor.Tensor, ylo, yhi float32) {
		add(name, func(x []*tensor.Tensor) *tensor.Tensor { return f(x[0], x[1]) }, u(-2, 2, 3, 4), u(ylo, yhi, 4))
	}
	bin("add_broadcast", (*tensor.Tensor).Add, -2, 2)
	bin("sub", (*tensor.Tensor).Sub, -2, 2)
	bin("mul", (*tensor.Tensor).Mul, -2, 2)
	bin("div", (*tensor.Tensor).Div, 0.5, 2)
	// fmod jumps where x/y crosses an integer; keep x/y in (2, 3).
	add("fmod", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].Fmod(x[1]) }, u(2.2, 2.6, 3, 4), u(0.95, 1.05, 4))
	add("maximum", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].Maximum(x[1]) }, u(-2, 2, 3, 4), u(-2, 2, 3, 4))
	add("minimum", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].Minimum(x[1]) }, u(-2, 2, 3, 4), u(-2, 2, 3, 4))
	add("where", func(x []*tensor.Tensor) *tensor.Tensor {
		return tensor.Where(x[0].Greater(tensor.Scalar[float32](0)), x[0], x[1])
	}, u(-2, 2, 3, 4), u(-2, 2, 4))

	red := func(name string, f func(x *tensor.Tensor) *tensor.Tensor) {
		add(name, func(x []*tensor.Tensor) *tensor.Tensor { return f(x[0]) }, u(-2, 2, 2, 3, 4))
	}
	red("sum", func(x *tensor.Tensor) *tensor.Tensor { return x.Sum(0, 2) })
	red("max", func(x *tensor.Tensor) *tensor.Tensor { return x.Max(1) })
	red("min_all", func(x *tensor.Tensor) *tensor.Tensor { return x.Min() })
	red("mean", func(x *tensor.Tensor) *tensor.Tensor { return x.Mean(-1) })
	red("var", func(x *tensor.Tensor) *tensor.Tensor { return x.Var(1) })
	red("softmax", func(x *tensor.Tensor) *tensor.Tensor { return x.Softmax(1) })
	red("log_softmax", func(x *tensor.Tensor) *tensor.Tensor { return x.LogSoftmax(-1) })

	mov := func(name string, f func(x *tensor.Tensor) *tensor.Tensor) {
		add(name, func(x []*tensor.Tensor) *tensor.Tensor { return f(x[0]) }, u(-2, 2, 2, 3, 4))
	}
	mov("reshape", func(x *tensor.Tensor) *tensor.Tensor { return x.Reshape(4, -1) })
	mov("permute", func(x *tensor.Tensor) *tensor.Tensor { return x.Permute(2, 0, 1) })
	mov("expand", func(x *tensor.Tensor) *tensor.Tensor { return x.Unsqueeze(1).Expand(2, 5, 3, 4) })
	mov("pad", func(x *tensor.Tensor) *tensor.Tensor { return x.Pad([][2]int{{1, 0}, {0, 2}, {1, 1}}, 3) })
	mov("shrink", func(x *tensor.Tensor) *tensor.Tensor { return x.Shrink([][2]int{{1, 2}, {0, 2}, {1, 4}}) })
	mov("flip", func(x *tensor.Tensor) *tensor.Tensor { return x.Flip(0, 2) })
	mov("flatten", func(x *tensor.Tensor) *tensor.Tensor { return x.Flatten(1, 2) })
	mov("unfold_overlap", func(x *tensor.Tensor) *tensor.Tensor { return x.Unfold(2, 3, 1) })
	mov("unfold_strided", func(x *tensor.Tensor) *tensor.Tensor { return x.Unfold(-1, 3, 2) })
	mov("unfold_gapped", func(x *tensor.Tensor) *tensor.Tensor { return x.Unfold(1, 1, 2) })
	mov("transpose_matmul", func(x *tensor.Tensor) *tensor.Tensor { return x.MatMul(x.Transpose(1, 2)) })
	add("cat", func(x []*tensor.Tensor) *tensor.Tensor { return tensor.Cat(1, x[0], x[1]) }, u(-2, 2, 2, 3), u(-2, 2, 2, 2))
	add("stack", func(x []*tensor.Tensor) *tensor.Tensor { return tensor.Stack(0, x[0], x[1]) }, u(-2, 2, 3), u(-2, 2, 3))

	add("matmul_batched", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(x[1]) }, u(-1, 1, 2, 1, 3, 4), u(-1, 1, 3, 4, 2))
	add("matmul_vec", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(x[1]) }, u(-1, 1, 4), u(-1, 1, 4, 3))
	add("gather", func(x []*tensor.Tensor) *tensor.Tensor {
		return x[0].Gather(1, tensor.FromSlice([]int32{2, 0, 2, 1, 1, 1}, 2, 3))
	}, u(-2, 2, 2, 4))
	add("scatter_add", func(x []*tensor.Tensor) *tensor.Tensor {
		return x[0].ScatterAdd(0, tensor.FromSlice([]int32{1, 0, 1, 1}, 1, 4), x[1])
	}, u(-2, 2, 3, 4), u(-2, 2, 1, 4))

	add("conv2d", func(x []*tensor.Tensor) *tensor.Tensor {
		return x[0].Conv2d(x[1], x[2], tensor.ConvOptions{Padding: 1})
	}, u(-1, 1, 2, 2, 5, 5), u(-1, 1, 3, 2, 3, 3), u(-1, 1, 3))
	add("conv2d_strided_dilated_grouped", func(x []*tensor.Tensor) *tensor.Tensor {
		return x[0].Conv2d(x[1], nil, tensor.ConvOptions{Stride: 2, Dilation: 2, Groups: 2, Padding: 1})
	}, u(-1, 1, 1, 4, 7, 7), u(-1, 1, 2, 2, 2, 2))
	add("max_pool2d", func(x []*tensor.Tensor) *tensor.Tensor { return x[0].MaxPool2d(2, tensor.ConvOptions{}) }, u(-2, 2, 2, 2, 4, 5))
	add("max_pool2d_padded", func(x []*tensor.Tensor) *tensor.Tensor {
		return x[0].MaxPool2d(3, tensor.ConvOptions{Stride: 2, Padding: 1})
	}, u(-2, 2, 1, 2, 5, 5))
	add("avg_pool2d", func(x []*tensor.Tensor) *tensor.Tensor {
		return x[0].AvgPool2d(3, tensor.ConvOptions{Stride: 1, Padding: 1})
	}, u(-2, 2, 1, 2, 4, 4))
	add("custom_func", func(x []*tensor.Tensor) *tensor.Tensor { return square.Apply(x[0]) }, u(-2, 2, 3))
	add("reused_input", func(x []*tensor.Tensor) *tensor.Tensor {
		y := x[0].Mul(x[0]).Add(x[0].Exp())
		return y.Sum(-1).Add(y.Max(-1))
	}, u(-2, 2, 3, 4))
	return cases
}

// square is x², written as a Func with its gradient by hand.
var square = tensor.Func{
	Forward: func(in []*tensor.Tensor) *tensor.Tensor { return in[0].Mul(in[0]) },
	Backward: func(g *tensor.Tensor, in []*tensor.Tensor, _ *tensor.Tensor) []*tensor.Tensor {
		return []*tensor.Tensor{g.Mul(in[0]).Mul(tensor.Scalar[float32](2))}
	},
}

// weighted returns sum(f(x) * w) as a float64, with w fixed per output
// element so every output element contributes differently.
func weighted(ctx context.Context, f func([]*tensor.Tensor) *tensor.Tensor, x []*tensor.Tensor) (*tensor.Tensor, error) {
	out := f(x)
	if err := out.Err(); err != nil {
		return nil, err
	}
	w := tensor.Rand(99, out.Shape()...).Add(tensor.Scalar[float32](0.5))
	return out.Mul(w).Sum(), nil
}

func TestGradientsMatchFiniteDifferences(t *testing.T) {
	const h = 1e-2
	for _, c := range gradCases() {
		t.Run(c.name, func(t *testing.T) {
			ctx := t.Context()
			x := make([]*tensor.Tensor, len(c.in))
			for i, v := range c.in {
				x[i] = tensor.FromSlice(v, c.dims[i]...).RequiresGrad()
			}
			loss, err := weighted(ctx, c.f, x)
			if err != nil {
				t.Fatal(err)
			}
			if err := loss.Backward(ctx); err != nil {
				t.Fatal(err)
			}
			for i, v := range c.in {
				g := x[i].Grad()
				if g == nil {
					t.Fatalf("input %d has no gradient", i)
				}
				if !slices.Equal(g.Shape(), c.dims[i]) {
					t.Fatalf("input %d: gradient shape %v, want %v", i, g.Shape(), c.dims[i])
				}
				got := data[float32](t, g)
				for j := range v {
					at := func(d float32) float64 {
						p := slices.Clone(v)
						p[j] += d
						y := slices.Clone(x)
						y[i] = tensor.FromSlice(p, c.dims[i]...)
						for k := range y {
							if k != i {
								y[k] = y[k].Detach()
							}
						}
						l, err := weighted(ctx, c.f, y)
						if err != nil {
							t.Fatal(err)
						}
						s, err := l.Item[float32](ctx)
						if err != nil {
							t.Fatal(err)
						}
						return float64(s)
					}
					want := (at(h) - at(-h)) / (2 * h)
					if diff := math.Abs(float64(got[j]) - want); diff > 5e-3+1e-2*math.Abs(want) {
						t.Errorf("input %d element %d: gradient %v, finite difference %v", i, j, got[j], want)
					}
				}
			}
		})
	}
}

func TestGradientAccumulatesAndZeroes(t *testing.T) {
	ctx := t.Context()
	w := tensor.FromSlice([]float32{1, 2, 3}).RequiresGrad()
	for range 2 {
		if err := w.Mul(w).Sum().Backward(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if got := data[float32](t, w.Grad()); !slices.Equal(got, []float32{4, 8, 12}) {
		t.Errorf("accumulated gradient %v, want 2 * 2w", got)
	}
	w.ZeroGrad()
	if w.Grad() != nil {
		t.Error("gradient survived ZeroGrad")
	}
}

func TestBackwardErrors(t *testing.T) {
	ctx := t.Context()
	x := tensor.FromSlice([]float32{1, 2})
	if err := x.Sum().Backward(ctx); !errors.Is(err, tensor.ErrNoGrad) {
		t.Errorf("untracked: %v, want ErrNoGrad", err)
	}
	w := x.RequiresGrad()
	if err := w.Mul(w).Backward(ctx); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("two elements: %v, want ErrShape", err)
	}
	if err := tensor.FromSlice([]int32{1}).RequiresGrad().Err(); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("int gradient: %v, want ErrDType", err)
	}
	l := w.Exp().Sum()
	if err := l.Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if err := l.Backward(ctx); !errors.Is(err, tensor.ErrNoGrad) {
		t.Errorf("second backward: %v, want ErrNoGrad", err)
	}
	// The loss value survives its graph's release.
	if v, err := l.Item[float32](ctx); err != nil || math.Abs(float64(v)-(math.E+math.Exp(2))) > 1e-4 {
		t.Errorf("loss after backward = %v, %v", v, err)
	}

	bad := tensor.Func{
		Forward: func(in []*tensor.Tensor) *tensor.Tensor { return in[0].Sum() },
		Backward: func(g *tensor.Tensor, _ []*tensor.Tensor, _ *tensor.Tensor) []*tensor.Tensor {
			return []*tensor.Tensor{g}
		},
	}
	if err := bad.Apply(w).Backward(ctx); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("gradient of the wrong shape: %v, want ErrShape", err)
	}

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	v := tensor.FromSlice([]float32{1}).RequiresGrad()
	if err := v.Mul(v).Sum().Backward(cancelled); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: %v", err)
	}
	if v.Grad() != nil {
		t.Error("failed backward changed a gradient")
	}
}

func TestDetachStopsGradients(t *testing.T) {
	ctx := t.Context()
	w := tensor.FromSlice([]float32{3}).RequiresGrad()
	y := w.Mul(w.Detach()) // d/dw = w, not 2w
	if err := y.Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if got := data[float32](t, w.Grad()); got[0] != 3 {
		t.Errorf("gradient %v, want 3", got)
	}
}

func TestParam(t *testing.T) {
	ctx := t.Context()
	p := tensor.NewParam(tensor.Full(tensor.Float32, 2, 3))
	before := p.Value().Mul(tensor.Scalar[float32](10)) // built before Set
	if err := p.Value().Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if err := p.Set(tensor.Full(tensor.Float32, 5, 3)); err != nil {
		t.Fatal(err)
	}
	if got := data[float32](t, before); got[0] != 20 {
		t.Errorf("tensor built before Set reads %v, want the old value", got)
	}
	if got := data[float32](t, p.Value()); got[0] != 5 {
		t.Errorf("value after Set = %v", got)
	}
	if got := data[float32](t, p.Grad()); !slices.Equal(got, []float32{1, 1, 1}) {
		t.Errorf("gradient after Set = %v, want it kept", got)
	}
	if err := p.Set(tensor.Zeros(tensor.Float32, 4)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("Set of another shape: %v, want ErrShape", err)
	}
	b := tensor.NewBuffer(tensor.Zeros(tensor.Float32, 2))
	if b.Trainable() || b.Value().Tracked() {
		t.Error("buffer requires gradients")
	}
}

func TestWithEffectsRunsDuringBackward(t *testing.T) {
	ctx := t.Context()
	w := tensor.FromSlice([]float32{1, 2}).RequiresGrad()
	h := w.Mul(w)
	effect := h.Detach().Sum() // like a running statistic of h
	if err := h.WithEffects(effect).Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	v, err := effect.Item[float32](ctx)
	if err != nil || v != 5 {
		t.Errorf("effect = %v, %v", v, err)
	}
}

func TestUnfoldMatchesWindows(t *testing.T) {
	for _, tc := range []struct{ n, size, step int }{{7, 3, 1}, {7, 3, 2}, {8, 2, 2}, {9, 2, 3}, {5, 5, 1}, {6, 4, 3}, {10, 1, 4}} {
		v := make([]float32, 2*tc.n)
		for i := range v {
			v[i] = float32(i)
		}
		got := tensor.FromSlice(v, 2, tc.n).Unfold(1, tc.size, tc.step)
		o := (tc.n-tc.size)/tc.step + 1
		if !slices.Equal(got.Shape(), []int{2, o, tc.size}) {
			t.Fatalf("%+v: shape %v", tc, got.Shape())
		}
		var want []float32
		for r := range 2 {
			for i := range o {
				for j := range tc.size {
					want = append(want, v[r*tc.n+i*tc.step+j])
				}
			}
		}
		if g := data[float32](t, got); !slices.Equal(g, want) {
			t.Errorf("%+v: %v, want %v", tc, g, want)
		}
	}
}

func TestArgmax(t *testing.T) {
	nan := float32(math.NaN())
	x := tensor.FromSlice([]float32{1, 3, 3, 0, nan, 2, -1, -5, -2}, 3, 3)
	if got := data[int32](t, x.Argmax(1)); !slices.Equal(got, []int32{1, 1, 0}) {
		t.Errorf("argmax = %v, want first maximum and NaN", got)
	}
	if got := data[int32](t, tensor.FromSlice([]int32{4, 9, 9}).Argmax(0)); got[0] != 1 {
		t.Errorf("int argmax = %v", got)
	}
}

func TestConvShapesAndErrors(t *testing.T) {
	x := tensor.Zeros(tensor.Float32, 2, 4, 9, 9)
	tests := map[string]struct {
		got  *tensor.Tensor
		want []int
	}{
		"conv":         {x.Conv2d(tensor.Zeros(tensor.Float32, 6, 4, 3, 3), nil, tensor.ConvOptions{}), []int{2, 6, 7, 7}},
		"conv padded":  {x.Conv2d(tensor.Zeros(tensor.Float32, 6, 2, 3, 3), tensor.Zeros(tensor.Float32, 6), tensor.ConvOptions{Padding: 1, Groups: 2}), []int{2, 6, 9, 9}},
		"conv strided": {x.Conv2d(tensor.Zeros(tensor.Float32, 1, 4, 3, 3), nil, tensor.ConvOptions{Stride: 2, Dilation: 2}), []int{2, 1, 3, 3}},
		"max pool":     {x.MaxPool2d(2, tensor.ConvOptions{}), []int{2, 4, 4, 4}},
		"avg pool":     {x.AvgPool2d(3, tensor.ConvOptions{Stride: 1, Padding: 1}), []int{2, 4, 9, 9}},
	}
	for name, tt := range tests {
		if err := tt.got.Err(); err != nil {
			t.Errorf("%s: %v", name, err)
		} else if !slices.Equal(tt.got.Shape(), tt.want) {
			t.Errorf("%s: shape %v, want %v", name, tt.got.Shape(), tt.want)
		}
	}
	for name, bad := range map[string]*tensor.Tensor{
		"channels":     x.Conv2d(tensor.Zeros(tensor.Float32, 6, 3, 3, 3), nil, tensor.ConvOptions{}),
		"groups":       x.Conv2d(tensor.Zeros(tensor.Float32, 5, 2, 3, 3), nil, tensor.ConvOptions{Groups: 2}),
		"rank":         tensor.Zeros(tensor.Float32, 4, 9, 9).MaxPool2d(2, tensor.ConvOptions{}),
		"kernel":       x.Conv2d(tensor.Zeros(tensor.Float32, 6, 4, 10, 10), nil, tensor.ConvOptions{}),
		"bias":         x.Conv2d(tensor.Zeros(tensor.Float32, 6, 4, 3, 3), tensor.Zeros(tensor.Float32, 5), tensor.ConvOptions{}),
		"pool padding": x.MaxPool2d(2, tensor.ConvOptions{Padding: 2}),
		"negative":     x.AvgPool2d(2, tensor.ConvOptions{Stride: -1}),
	} {
		if err := bad.Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
}

func ExampleTensor_Backward() {
	ctx := context.Background()
	w := tensor.FromSlice([]float32{1, 2, 3}).RequiresGrad()
	loss := w.Mul(w).Sum() // sum of squares
	if err := loss.Backward(ctx); err != nil {
		fmt.Println(err)
		return
	}
	g, _ := w.Grad().Data[float32](ctx)
	fmt.Println(g)
	// Output: [2 4 6]
}
