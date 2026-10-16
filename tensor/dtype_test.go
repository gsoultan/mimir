package tensor_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

var halfTypes = []tensor.DType{tensor.BFloat16, tensor.Float16}

// floats returns the values of a float tensor of any dtype.
func floats(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	return data[float32](t, x.Cast(tensor.Float32))
}

// half returns v rounded to dt as a tensor of dt.
func half(dt tensor.DType, v []float32, shape ...int) *tensor.Tensor {
	return tensor.FromSlice(v, shape...).Cast(dt)
}

func sameBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool {
		return math.Float32bits(x) == math.Float32bits(y) || x != x && y != y
	})
}

func TestCastRoundsToNearestEven(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	ld := func(e int) float32 { return float32(math.Ldexp(1, e)) }
	in := []float32{1.0 / 3, 1 + 1.0/256, 1 + 3.0/256, 65520, 65504, 1e30, ld(-25), ld(-24), -2.5, 0, inf, -inf, nan}
	// 1e30 is 0x7149f2ca as a float32: its top 16 bits round up.
	want := map[tensor.DType][]float32{
		tensor.BFloat16: {0.333984375, 1, 1.015625, 65536, 65536, math.Float32frombits(0x714a0000), ld(-25), ld(-24), -2.5, 0, inf, -inf, nan},
		tensor.Float16:  {0.333251953125, 1.00390625, 1.01171875, inf, 65504, inf, 0, ld(-24), -2.5, 0, inf, -inf, nan},
	}
	for _, dt := range halfTypes {
		x := tensor.FromSlice(in)
		h := x.Cast(dt)
		if h.DType() != dt {
			t.Fatalf("cast to %s has dtype %s", dt, h.DType())
		}
		if got := floats(t, h); !sameBits(got, want[dt]) {
			t.Errorf("%s: cast %v\ngot  %v\nwant %v", dt, in, got, want[dt])
		}
		if h.Cast(dt) != h {
			t.Errorf("%s: cast to the same dtype is not a no-op", dt)
		}
		// A value of the type survives a round trip through the other
		// types, as far as they hold it.
		for _, via := range []tensor.DType{tensor.Float32, tensor.BFloat16, tensor.Float16} {
			v := []float32{1.5, -0.25, 3, 1024, 0.0625}
			if got := floats(t, half(dt, v).Cast(via).Cast(dt)); !slices.Equal(got, v) {
				t.Errorf("%s via %s: %v, want %v", dt, via, got, v)
			}
		}
		if got := data[int32](t, half(dt, []float32{2.5, -2.5, 7}).Cast(tensor.Int32)); !slices.Equal(got, []int32{2, -2, 7}) {
			t.Errorf("%s to int32: %v", dt, got)
		}
		if got := floats(t, tensor.FromSlice([]int32{3, -300, 257}).Cast(dt)); got[0] != 3 || got[1] != -300 || got[2] != 256 && got[2] != 257 {
			t.Errorf("int32 to %s: %v", dt, got)
		}
		if got := floats(t, tensor.Full(dt, 1.0/3, 2)); got[0] != want[dt][0] || got[1] != want[dt][0] {
			t.Errorf("%s constant 1/3 = %v, want %v", dt, got, want[dt][0])
		}
		_, err := h.Data[float32](t.Context())
		if !errors.Is(err, tensor.ErrDType) || !strings.Contains(err.Error(), "cast") {
			t.Errorf("%s: Data[float32] = %v, want ErrDType saying to cast", dt, err)
		}
		if _, err := h.Item[int32](t.Context()); !errors.Is(err, tensor.ErrShape) && !errors.Is(err, tensor.ErrDType) {
			t.Errorf("%s: Item[int32] = %v", dt, err)
		}
	}
	// bfloat16 to binary16 goes through float32: 1 + 1/128 is exact in
	// both, 257 rounds to 256 in bfloat16 and stays there.
	x := half(tensor.BFloat16, []float32{1 + 1.0/128, 257})
	if got := floats(t, x.Cast(tensor.Float16)); !slices.Equal(got, []float32{1 + 1.0/128, 256}) {
		t.Errorf("bfloat16 to float16: %v", got)
	}
}

func TestHalfPromotion(t *testing.T) {
	f := tensor.Ones(tensor.Float32, 2, 3)
	i := tensor.Ones(tensor.Int32, 2, 3)
	b := tensor.Ones(tensor.BFloat16, 2, 3)
	h := tensor.Ones(tensor.Float16, 2, 3)
	cond := tensor.Ones(tensor.Int32, 2, 3)
	cases := map[string]struct {
		got  *tensor.Tensor
		want tensor.DType
	}{
		"bf16 + bf16":           {b.Add(b), tensor.BFloat16},
		"f16 * f16":             {h.Mul(h), tensor.Float16},
		"bf16 + f32":            {b.Add(f), tensor.Float32},
		"f32 - f16":             {f.Sub(h), tensor.Float32},
		"bf16 max f16":          {b.Maximum(h), tensor.Float32},
		"bf16 < f32":            {b.Less(f), tensor.Float32},
		"bf16 == bf16":          {b.Equal(b), tensor.BFloat16},
		"f16 / f16":             {h.Div(h), tensor.Float16},
		"bf16 / f32":            {b.Div(f), tensor.Float32},
		"where bf16 f16":        {tensor.Where(cond, b, h), tensor.Float32},
		"where bf16 bf16":       {tensor.Where(b, b, b), tensor.BFloat16},
		"exp bf16":              {b.Exp(), tensor.BFloat16},
		"sigmoid f16":           {h.Sigmoid(), tensor.Float16},
		"gelu bf16":             {b.GELU(), tensor.BFloat16},
		"relu bf16":             {b.ReLU(), tensor.BFloat16},
		"sqrt f16":              {h.Sqrt(), tensor.Float16},
		"sum bf16":              {b.Sum(), tensor.Float32},
		"max f16":               {h.Max(1), tensor.Float32},
		"min bf16":              {b.Min(0), tensor.Float32},
		"mean bf16":             {b.Mean(1), tensor.Float32},
		"var f16":               {h.Var(1), tensor.Float32},
		"softmax bf16":          {b.Softmax(1), tensor.Float32},
		"log_softmax f16":       {h.LogSoftmax(1), tensor.Float32},
		"argmax bf16":           {b.Argmax(1), tensor.Int32},
		"matmul bf16":           {b.MatMul(b.Transpose(0, 1)), tensor.BFloat16},
		"matmul f16 f32":        {h.MatMul(f.Transpose(0, 1)), tensor.Float32},
		"matmul bf16 f16":       {b.MatMul(h.Transpose(0, 1)), tensor.Float32},
		"scatter_add bf16 f32":  {b.ScatterAdd(0, cond, f), tensor.Float32},
		"scatter_add bf16 bf16": {b.ScatterAdd(0, cond, b), tensor.BFloat16},
		"gather f16":            {h.Gather(1, cond), tensor.Float16},
		"pad bf16":              {b.Pad([][2]int{{1, 1}, {0, 0}}, 0.5), tensor.BFloat16},
		"cast chain":            {b.Cast(tensor.Float16).Cast(tensor.Int32), tensor.Int32},
		"cumsum bf16":           {b.CumSum(1), tensor.BFloat16},
		"cummax f16":            {firstOf(h.CumMax(0)), tensor.Float16},
		"cummax indices bf16":   {secondOf(b.CumMax(0)), tensor.Int32},
		"logcumsumexp f16":      {h.LogCumSumExp(1), tensor.Float16},
	}
	for name, c := range cases {
		if err := c.got.Err(); err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if c.got.DType() != c.want {
			t.Errorf("%s: dtype %s, want %s", name, c.got.DType(), c.want)
		}
	}
	for name, got := range map[string]*tensor.Tensor{
		"int32 + bf16":        i.Add(b),
		"f16 * int32":         h.Mul(i),
		"where int32 f16":     tensor.Where(cond, i, h),
		"scatter_add f16 i":   h.ScatterAdd(0, cond, i),
		"int32 requires grad": i.RequiresGrad(),
		"logcumsumexp int32":  i.LogCumSumExp(0),
	} {
		if err := got.Err(); !errors.Is(err, tensor.ErrDType) {
			t.Errorf("%s: %v, want ErrDType", name, err)
		}
	}
	if err := tensor.Cat(0, b, f).Err(); err == nil {
		t.Error("cat of bfloat16 and float32: no error")
	}
	for _, dt := range halfTypes {
		if err := tensor.Ones(dt, 2).RequiresGrad().Err(); err != nil {
			t.Errorf("%s requires grad: %v", dt, err)
		}
	}
}

func TestHalfOpsRoundOnce(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 9))
	v := make([]float32, 64)
	for i := range v {
		v[i] = rng.Float32()*6 - 3
	}
	for _, dt := range halfTypes {
		x := half(dt, v, 4, 16)
		f := x.Cast(tensor.Float32)
		for name, op := range map[string]func(*tensor.Tensor) *tensor.Tensor{
			"exp":     (*tensor.Tensor).Exp,
			"sigmoid": (*tensor.Tensor).Sigmoid,
			"tanh":    (*tensor.Tensor).Tanh,
			"gelu":    (*tensor.Tensor).GELU,
			"silu":    (*tensor.Tensor).SiLU,
			"cos":     (*tensor.Tensor).Cos,
			"exp2":    (*tensor.Tensor).Exp2,
			"neg":     (*tensor.Tensor).Neg,
			"abs":     (*tensor.Tensor).Abs,
			"relu":    (*tensor.Tensor).ReLU,
			"log":     func(x *tensor.Tensor) *tensor.Tensor { return x.Abs().Log() },
			"div":     func(x *tensor.Tensor) *tensor.Tensor { return x.Div(x.Flip(1)) },
			"mul":     func(x *tensor.Tensor) *tensor.Tensor { return x.Mul(x.Flip(0)) },
			"fmod":    func(x *tensor.Tensor) *tensor.Tensor { return x.Fmod(x.Flip(0)) },
			"sub":     func(x *tensor.Tensor) *tensor.Tensor { return x.Sub(x.Flip(0)) },
			"maximum": func(x *tensor.Tensor) *tensor.Tensor { return x.Maximum(x.Flip(1)) },
		} {
			got := op(x)
			if got.DType() != dt {
				t.Errorf("%s %s: dtype %s", dt, name, got.DType())
				continue
			}
			// The float32 result of the same values, rounded to dt once.
			want := floats(t, op(f).Cast(dt))
			if g := floats(t, got); !sameBits(g, want) {
				t.Errorf("%s %s:\ngot  %v\nwant %v", dt, name, g, want)
			}
		}
	}
	// 1 + 2^-9 rounds to 1 in bfloat16, and is exact in float32.
	one := half(tensor.BFloat16, []float32{1})
	small := tensor.FromSlice([]float32{1.0 / 512})
	if got := floats(t, one.Add(small.Cast(tensor.BFloat16))); got[0] != 1 {
		t.Errorf("bfloat16 1 + 2^-9 = %v, want 1", got)
	}
	if got := floats(t, one.Add(small)); got[0] != 1+1.0/512 {
		t.Errorf("bfloat16 1 + float32 2^-9 = %v, want exact", got)
	}
}

func TestHalfReductionsAccumulateInFloat32(t *testing.T) {
	for _, c := range []struct {
		dt  tensor.DType
		big float32
	}{{tensor.BFloat16, 256}, {tensor.Float16, 2048}} {
		v := slices.Concat([]float32{c.big}, slices.Repeat([]float32{1}, 8))
		x := half(c.dt, v, 3, 3)
		if got := floats(t, x.Sum()); got[0] != c.big+8 {
			t.Errorf("%s sum = %v, want %v", c.dt, got, c.big+8)
		}
		if got := floats(t, x.Sum(1)); !slices.Equal(got, []float32{c.big + 2, 3, 3}) {
			t.Errorf("%s row sums = %v", c.dt, got)
		}
		if got := floats(t, x.Max(0)); !slices.Equal(got, []float32{c.big, 1, 1}) {
			t.Errorf("%s column max = %v", c.dt, got)
		}
		if got := floats(t, x.Mean()); got[0] != (c.big+8)/9 {
			t.Errorf("%s mean = %v, want %v", c.dt, got, (c.big+8)/9)
		}
		s := floats(t, x.Softmax(1))
		if s[0] != 1 || s[1] != 0 || math.Abs(float64(s[3]-1.0/3)) > 1e-6 {
			t.Errorf("%s softmax = %v", c.dt, s)
		}
	}
}

func TestMatMulOnHalfRunsInFloat32(t *testing.T) {
	rng := rand.New(rand.NewPCG(10, 11))
	fill := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	for _, dt := range halfTypes {
		x, w := half(dt, fill(2*4*6), 2, 4, 6), half(dt, fill(6*5), 6, 5)
		got := x.MatMul(w)
		if got.DType() != dt {
			t.Fatalf("%s product has dtype %s", dt, got.DType())
		}
		want := floats(t, x.Cast(tensor.Float32).MatMul(w.Cast(tensor.Float32)).Cast(dt))
		if g := floats(t, got); !sameBits(g, want) {
			t.Errorf("%s matmul:\ngot  %v\nwant %v", dt, g, want)
		}
		// A vector operand and a float32 one.
		v := half(dt, fill(6), 6)
		if got := v.MatMul(w); got.DType() != dt || !slices.Equal(got.Shape(), []int{5}) {
			t.Errorf("%s vector product: %s%v", dt, got.DType(), got.Shape())
		}
		mixed := x.MatMul(w.Cast(tensor.Float32))
		if mixed.DType() != tensor.Float32 || !sameBits(floats(t, mixed), floats(t, x.Cast(tensor.Float32).MatMul(w.Cast(tensor.Float32)))) {
			t.Errorf("%s by float32 product differs from the float32 product", dt)
		}

		// A convolution with bias: the product rounds once and the bias
		// add once, so compare with the float32 convolution within an
		// element of precision.
		img, ker, bias := half(dt, fill(2*3*7*7), 2, 3, 7, 7), half(dt, fill(4*3*3*3), 4, 3, 3, 3), half(dt, fill(4), 4)
		conv := img.Conv2d(ker, bias, tensor.ConvOptions{Padding: 1})
		if conv.DType() != dt || !slices.Equal(conv.Shape(), []int{2, 4, 7, 7}) {
			t.Fatalf("%s conv2d: %s%v, %v", dt, conv.DType(), conv.Shape(), conv.Err())
		}
		ref := floats(t, img.Cast(tensor.Float32).Conv2d(ker.Cast(tensor.Float32), bias.Cast(tensor.Float32), tensor.ConvOptions{Padding: 1}))
		eps := 1.0 / 128
		if dt == tensor.Float16 {
			eps = 1.0 / 1024
		}
		for i, g := range floats(t, conv) {
			if math.Abs(float64(g-ref[i])) > eps*(1+math.Abs(float64(ref[i]))) {
				t.Fatalf("%s conv2d element %d = %v, float32 %v", dt, i, g, ref[i])
			}
		}
	}
}

func TestGradientsFlowThroughCasts(t *testing.T) {
	ctx := t.Context()
	xv := []float32{0.5, -1.25, 2, 3.5, -0.75, 1}
	wv := []float32{1.5, 2, -0.5, 0.25, 4, -3}
	for _, dt := range halfTypes {
		// The sum's Float32 gradient flows through the product's rule,
		// promoting, so both leaves get Float32 gradients: x through the
		// cast's rule, which casts to x's own dtype.
		x := tensor.FromSlice(xv, 2, 3).RequiresGrad()
		w := half(dt, wv, 2, 3).RequiresGrad()
		y := x.Cast(dt).Mul(w).Sum()
		if y.DType() != tensor.Float32 {
			t.Fatalf("%s: sum has dtype %s", dt, y.DType())
		}
		if err := y.Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if gx := x.Grad(); gx.DType() != tensor.Float32 || !slices.Equal(floats(t, gx), wv) {
			t.Errorf("%s: dy/dx = %s %v, want float32 %v", dt, gx.DType(), floats(t, gx), wv)
		}
		if gw := w.Grad(); gw.DType() != tensor.Float32 || !slices.Equal(floats(t, gw), xv) {
			t.Errorf("%s: dy/dw = %s %v, want float32 %v", dt, gw.DType(), floats(t, gw), xv)
		}
		// A cast of the product to Float32 casts the gradient back to dt
		// before the product's rule, so w's gradient is in dt.
		x = tensor.FromSlice(xv, 2, 3).RequiresGrad()
		w = half(dt, wv, 2, 3).RequiresGrad()
		if err := x.Cast(dt).Mul(w).Cast(tensor.Float32).Sum().Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if gx := x.Grad(); gx.DType() != tensor.Float32 || !slices.Equal(floats(t, gx), wv) {
			t.Errorf("%s: dy/dx = %s %v, want float32 %v", dt, gx.DType(), floats(t, gx), wv)
		}
		if gw := w.Grad(); gw.DType() != dt || !slices.Equal(floats(t, gw), xv) {
			t.Errorf("%s: dy/dw = %s %v, want %s %v", dt, gw.DType(), floats(t, gw), dt, xv)
		}
	}

	// The gradients of a small network in bfloat16 agree with the
	// float32 ones to the precision of the half type.
	rng := rand.New(rand.NewPCG(12, 13))
	fill := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	xs, ws, bs := fill(4*3), fill(3*2), fill(2)
	loss := func(dt tensor.DType) (*tensor.Tensor, []*tensor.Tensor) {
		x := half(dt, xs, 4, 3).RequiresGrad()
		w := half(dt, ws, 3, 2).RequiresGrad()
		b := half(dt, bs, 2).RequiresGrad()
		return x.MatMul(w).Add(b).Sigmoid().Mul(x.MatMul(w)).Sum(), []*tensor.Tensor{x, w, b}
	}
	l32, p32 := loss(tensor.Float32)
	if err := l32.Backward(ctx); err != nil {
		t.Fatal(err)
	}
	for _, dt := range halfTypes {
		l, p := loss(dt)
		if err := l.Backward(ctx); err != nil {
			t.Fatal(err)
		}
		for i := range p {
			want := floats(t, p32[i].Grad())
			g := p[i].Grad()
			if !g.DType().IsFloat() {
				t.Errorf("%s: gradient %d has dtype %s", dt, i, g.DType())
			}
			for j, v := range floats(t, g) {
				if math.Abs(float64(v-want[j])) > 0.02*(1+math.Abs(float64(want[j]))) {
					t.Errorf("%s: gradient %d element %d = %v, float32 %v", dt, i, j, v, want[j])
				}
			}
		}
	}
}

func TestHalfLeafAccumulatesFloat32Gradient(t *testing.T) {
	ctx := t.Context()
	for _, dt := range halfTypes {
		for _, floatFirst := range []bool{false, true} {
			w := half(dt, []float32{1, 2, 3}).RequiresGrad()
			// Sum's gradient is Float32, cast back to dt through Cast's;
			// the sum of a half tensor gives the leaf a Float32 gradient.
			own := func() *tensor.Tensor { return w.Cast(tensor.Float32).Mul(tensor.Scalar[float32](2)).Sum() }
			wide := func() *tensor.Tensor { return w.Sum() }
			first, second := own, wide
			if floatFirst {
				first, second = wide, own
			}
			if err := first().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			want := dt
			if floatFirst {
				want = tensor.Float32
			}
			if g := w.Grad(); g.DType() != want {
				t.Errorf("%s: first gradient has dtype %s, want %s", dt, g.DType(), want)
			}
			if err := second().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			g := w.Grad()
			if g.DType() != tensor.Float32 || !slices.Equal(floats(t, g), []float32{3, 3, 3}) {
				t.Errorf("%s (float32 first %v): accumulated gradient %s %v, want float32 [3 3 3]", dt, floatFirst, g.DType(), floats(t, g))
			}
			w.ZeroGrad()
			if err := wide().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			if g := w.Grad(); !slices.Equal(floats(t, g), []float32{1, 1, 1}) {
				t.Errorf("%s: gradient after ZeroGrad %v", dt, floats(t, g))
			}
		}
	}
}

func TestParamSetChangesFloatDType(t *testing.T) {
	p := tensor.NewParam(tensor.FromSlice([]float32{1, 2, 3}))
	for _, dt := range halfTypes {
		if err := p.Set(p.Value().Cast(dt)); err != nil {
			t.Fatalf("set to %s: %v", dt, err)
		}
		if got := p.Value(); got.DType() != dt || !got.Tracked() || !slices.Equal(floats(t, got), []float32{1, 2, 3}) {
			t.Errorf("after set to %s: %v, tracked %v", dt, got, got.Tracked())
		}
	}
	if err := p.Set(p.Value().Cast(tensor.Int32)); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("set trainable parameter to int32: %v, want ErrDType", err)
	}
	if err := p.Set(tensor.Ones(tensor.Float32, 4)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("set to another shape: %v, want ErrShape", err)
	}
	b := tensor.NewBuffer(tensor.FromSlice([]float32{1, 2, 3}))
	if err := b.Set(tensor.FromSlice([]int32{1, 2, 3})); err != nil {
		t.Errorf("set buffer to int32: %v", err)
	}
	// A gradient accumulated before the change is kept and added to.
	q := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	if err := q.Value().Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := q.Set(q.Value().Cast(tensor.BFloat16)); err != nil {
		t.Fatal(err)
	}
	if err := q.Value().Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := floats(t, q.Grad()); !slices.Equal(got, []float32{2, 2}) {
		t.Errorf("gradient across a dtype change: %v, want [2 2]", got)
	}
}
