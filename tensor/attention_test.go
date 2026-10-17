package tensor_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// composedAttention is the attention the fused kernels replace: the
// scores materialized, their softmax and its product with the values, with
// key heads repeated for grouped queries.
func composedAttention(q, k, v, mask *tensor.Tensor, scale float64, causal bool, offset int) *tensor.Tensor {
	r := q.Rank()
	repeat := func(x *tensor.Tensor) *tensor.Tensor {
		if r < 3 || q.Shape()[r-3] == x.Shape()[r-3] {
			return x
		}
		s := x.Shape()
		h := q.Shape()[r-3]
		grown := slices.Concat(s[:r-2], []int{h / s[r-3]}, s[r-2:])
		return x.Unsqueeze(r - 2).Expand(grown...).Reshape(slices.Concat(s[:r-3], []int{h}, s[r-2:])...)
	}
	k, v = repeat(k), repeat(v)
	s := q.MatMul(k.Transpose(-1, -2)).Mul(tensor.Scalar(float32(scale)))
	if mask != nil {
		s = s.Add(mask)
	}
	if causal {
		nq, nk := q.Shape()[r-2], k.Shape()[r-2]
		m := make([]float32, nq*nk)
		for i := range nq {
			for j := range nk {
				if j > i+offset {
					m[i*nk+j] = float32(math.Inf(-1))
				}
			}
		}
		s = s.Add(tensor.FromSlice(m, nq, nk))
	}
	return s.Softmax(-1).MatMul(v)
}

func randomTensor(r *rand.Rand, scale float32, shape ...int) *tensor.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = (r.Float32()*2 - 1) * scale
	}
	return tensor.FromSlice(v, shape...)
}

func TestAttentionMatchesComposition(t *testing.T) {
	type tc struct {
		name     string
		q, k, v  []int
		mask     []int // additive, or nil
		boolMask bool
		causal   bool
		offset   int
		scale    float64
		dt       tensor.DType
		tol      float64
	}
	cases := []tc{
		{name: "plain", q: []int{2, 3, 5, 8}, k: []int{2, 3, 7, 8}, v: []int{2, 3, 7, 4}},
		{name: "causal", q: []int{1, 2, 9, 4}, k: []int{1, 2, 9, 4}, v: []int{1, 2, 9, 4}, causal: true},
		{name: "causal offset", q: []int{2, 2, 3, 4}, k: []int{2, 2, 8, 4}, v: []int{2, 2, 8, 4}, causal: true, offset: 5},
		{name: "grouped", q: []int{2, 6, 5, 8}, k: []int{2, 2, 5, 8}, v: []int{2, 2, 5, 8}, causal: true},
		{name: "masked", q: []int{2, 2, 4, 8}, k: []int{2, 2, 6, 8}, v: []int{2, 2, 6, 8}, mask: []int{4, 6}},
		{name: "bool mask", q: []int{2, 2, 4, 8}, k: []int{2, 2, 6, 8}, v: []int{2, 2, 6, 8}, mask: []int{2, 1, 1, 6}, boolMask: true},
		{name: "broadcast batch", q: []int{3, 2, 4, 8}, k: []int{1, 2, 6, 8}, v: []int{1, 2, 6, 8}},
		{name: "rank 2", q: []int{4, 8}, k: []int{6, 8}, v: []int{6, 3}, scale: 0.3},
		{name: "long", q: []int{1, 1, 150, 16}, k: []int{1, 1, 300, 16}, v: []int{1, 1, 300, 16}, causal: true, offset: 150},
		{name: "bfloat16", q: []int{2, 2, 5, 8}, k: []int{2, 2, 5, 8}, v: []int{2, 2, 5, 8}, causal: true, dt: tensor.BFloat16, tol: 2e-2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ctx := t.Context()
			r := rand.New(rand.NewPCG(9, uint64(len(c.name))))
			if c.dt == 0 {
				c.dt = tensor.Float32
			}
			if c.tol == 0 {
				c.tol = 1e-5
			}
			mk := func(s []int) *tensor.Tensor { return randomTensor(r, 1.5, s...).Cast(c.dt).RequiresGrad() }
			q, k, v := mk(c.q), mk(c.k), mk(c.v)
			opts := []tensor.AttentionOption{}
			var mask *tensor.Tensor
			if c.mask != nil {
				if c.boolMask {
					n := 1
					for _, d := range c.mask {
						n *= d
					}
					keep := make([]float32, n)
					for i := range keep {
						if i%6 != 2 { // every row keeps a key
							keep[i] = 1
						}
					}
					m := tensor.FromSlice(keep, c.mask...)
					opts = append(opts, tensor.BoolMask(m))
					mask = tensor.Where(m, tensor.Scalar[float32](0), tensor.Scalar(float32(math.Inf(-1))))
				} else {
					mask = randomTensor(r, 2, c.mask...)
					opts = append(opts, tensor.AdditiveMask(mask))
				}
			}
			scale := c.scale
			if scale == 0 {
				scale = 1 / math.Sqrt(float64(c.q[len(c.q)-1]))
			} else {
				opts = append(opts, tensor.AttentionScale(scale))
			}
			if c.causal {
				opts = append(opts, tensor.CausalOffset(c.offset))
			}
			got := tensor.ScaledDotProductAttention(q, k, v, opts...)
			f32 := func(x *tensor.Tensor) *tensor.Tensor { return x.Cast(tensor.Float32) }
			want := composedAttention(f32(q), f32(k), f32(v), mask, scale, c.causal, c.offset)
			if c.dt != tensor.Float32 {
				want = want.Cast(c.dt)
			}
			if got.DType() != c.dt || !slices.Equal(got.Shape(), want.Shape()) {
				t.Fatalf("result %s%v, want %s%v", got.DType(), got.Shape(), c.dt, want.Shape())
			}
			w := randomTensor(r, 1, want.Shape()...)
			if err := f32(got).Mul(w).Sum().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			values := func(x *tensor.Tensor) []float32 {
				t.Helper()
				d, err := x.Cast(tensor.Float32).Data[float32](ctx)
				if err != nil {
					t.Fatal(err)
				}
				return d
			}
			// ZeroGrad reuses the gradients' memory: read them first.
			fused := [][]float32{values(got), values(q.Grad()), values(k.Grad()), values(v.Grad())}
			q.ZeroGrad()
			k.ZeroGrad()
			v.ZeroGrad()
			if err := f32(want).Mul(w).Sum().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			composed := [][]float32{values(want), values(q.Grad()), values(k.Grad()), values(v.Grad())}
			for n, what := range []string{"result", "query gradient", "key gradient", "value gradient"} {
				x, y := fused[n], composed[n]
				for i := range y {
					if d := math.Abs(float64(x[i] - y[i])); d > c.tol*math.Max(1, math.Abs(float64(y[i]))) || math.IsNaN(float64(x[i])) {
						t.Fatalf("%s[%d] = %v, want %v", what, i, x[i], y[i])
					}
				}
			}
		})
	}
}

func TestAttentionErrors(t *testing.T) {
	f := func(shape ...int) *tensor.Tensor { return tensor.Zeros(tensor.Float32, shape...) }
	cases := []struct {
		name    string
		q, k, v *tensor.Tensor
		opts    []tensor.AttentionOption
		want    error
	}{
		{"rank 1", f(4), f(4), f(4), nil, tensor.ErrShape},
		{"ranks differ", f(2, 3, 4), f(3, 4), f(3, 4), nil, tensor.ErrShape},
		{"heads do not divide", f(1, 3, 2, 4), f(1, 2, 2, 4), f(1, 2, 2, 4), nil, tensor.ErrShape},
		{"features differ", f(1, 2, 2, 4), f(1, 2, 2, 5), f(1, 2, 2, 4), nil, tensor.ErrShape},
		{"keys and values differ", f(1, 2, 2, 4), f(1, 2, 3, 4), f(1, 2, 2, 4), nil, tensor.ErrShape},
		{"batches differ", f(2, 2, 2, 4), f(3, 2, 2, 4), f(3, 2, 2, 4), nil, tensor.ErrShape},
		{"int inputs", tensor.Zeros(tensor.Int32, 1, 2, 4), tensor.Zeros(tensor.Int32, 1, 2, 4), tensor.Zeros(tensor.Int32, 1, 2, 4), nil, tensor.ErrDType},
		{"mask shape", f(1, 2, 3, 4), f(1, 2, 5, 4), f(1, 2, 5, 4), []tensor.AttentionOption{tensor.AdditiveMask(f(3, 4))}, tensor.ErrShape},
		{"int mask", f(1, 2, 3, 4), f(1, 2, 5, 4), f(1, 2, 5, 4), []tensor.AttentionOption{tensor.AdditiveMask(tensor.Zeros(tensor.Int32, 3, 5))}, tensor.ErrDType},
		{"failed mask", f(1, 2, 3, 4), f(1, 2, 5, 4), f(1, 2, 5, 4), []tensor.AttentionOption{tensor.AdditiveMask(f(2).Reshape(3))}, tensor.ErrShape},
	}
	for _, c := range cases {
		if got := tensor.ScaledDotProductAttention(c.q, c.k, c.v, c.opts...); !errors.Is(got.Err(), c.want) {
			t.Errorf("%s: got %v, want %v", c.name, got.Err(), c.want)
		}
	}
	if got := tensor.ScaledDotProductAttention(nil, f(1, 2, 3), f(1, 2, 3)); !errors.Is(got.Err(), tensor.ErrNilTensor) {
		t.Errorf("nil query: got %v", got.Err())
	}
}

func TestAttentionWithNoVisibleKeyIsZero(t *testing.T) {
	ctx := t.Context()
	q := tensor.Ones(tensor.Float32, 1, 1, 3, 2).RequiresGrad()
	kv := tensor.Ones(tensor.Float32, 1, 1, 3, 2).RequiresGrad()
	out := tensor.ScaledDotProductAttention(q, kv, kv, tensor.CausalOffset(-1))
	if err := out.Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	got, err := out.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if want := []float32{0, 0, 1, 1, 1, 1}; !slices.Equal(got, want) {
		t.Errorf("result %v, want %v", got, want)
	}
	g, err := q.Grad().Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range g {
		if x != 0 {
			t.Errorf("query gradient %v, want zeros", g)
			break
		}
	}
}

func TestSliceScatterWritesTheBlock(t *testing.T) {
	ctx := t.Context()
	x := tensor.FromSlice([]float32{0, 1, 2, 3, 4, 5}, 2, 3).RequiresGrad()
	src := tensor.FromSlice([]float32{-1, -2}, 2, 1).RequiresGrad()
	y := x.SliceScatter(src, [][2]int{{0, 2}, {1, 2}})
	got, err := y.Data[float32](ctx)
	if err != nil || !slices.Equal(got, []float32{0, -1, 2, 3, -2, 5}) {
		t.Fatalf("got %v, %v", got, err)
	}
	w := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err := y.Mul(w).Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	gx, _ := x.Grad().Data[float32](ctx)
	gs, _ := src.Grad().Data[float32](ctx)
	if !slices.Equal(gx, []float32{1, 0, 3, 4, 0, 6}) || !slices.Equal(gs, []float32{2, 5}) {
		t.Errorf("gradients %v and %v", gx, gs)
	}
	for _, bad := range []struct {
		src    *tensor.Tensor
		ranges [][2]int
		want   error
	}{
		{tensor.Zeros(tensor.Float32, 2, 2), [][2]int{{0, 2}, {1, 2}}, tensor.ErrShape},
		{tensor.Zeros(tensor.Float32, 2, 1), [][2]int{{0, 2}, {2, 4}}, tensor.ErrShape},
		{tensor.Zeros(tensor.Float32, 2, 1), [][2]int{{0, 2}}, tensor.ErrShape},
		{tensor.Zeros(tensor.Int32, 2, 1), [][2]int{{0, 2}, {1, 2}}, tensor.ErrDType},
	} {
		if err := x.SliceScatter(bad.src, bad.ranges).Err(); !errors.Is(err, bad.want) {
			t.Errorf("%v at %v: got %v, want %v", bad.src, bad.ranges, err, bad.want)
		}
	}
}

// TestKVCacheAppendsInPlace checks that decoding steps write their keys
// and values into the cache's memory instead of copying the cache.
func TestKVCacheAppendsInPlace(t *testing.T) {
	if tensor.Eager() {
		t.Skip("eager mode computes every operation into memory of its own")
	}
	ctx := t.Context()
	be := &allocCounter{Backend: cpu.New()}
	defer tensor.SetDevice(be)()
	const heads, capacity, dim = 2, 64, 8
	cache := nn.NewKVCache(1, heads, capacity, dim, tensor.Float32)
	for i := range 8 {
		before := be.elements
		k := tensor.Randn(uint64(i), 1, heads, 1, dim)
		keys, values, err := cache.Append(ctx, k, k.Neg())
		if err != nil {
			t.Fatal(err)
		}
		q := tensor.Randn(uint64(100+i), 1, 2*heads, 1, dim)
		if err := tensor.ScaledDotProductAttention(q, keys, values).Realize(ctx); err != nil {
			t.Fatal(err)
		}
		// The first append computes the cache; later ones write a position.
		if n := be.elements - before; i > 0 && n >= heads*capacity*dim {
			t.Errorf("step %d allocated %d elements, as much as the cache", i, n)
		}
	}
}

func ExampleScaledDotProductAttention() {
	// One head attending over three keys, causally.
	q := tensor.FromSlice([]float32{1, 0, 0, 1, 1, 1}, 1, 3, 2)
	k := tensor.FromSlice([]float32{1, 0, 0, 1, 1, 1}, 1, 3, 2)
	v := tensor.FromSlice([]float32{1, 2, 3}, 1, 3, 1)
	out, err := tensor.ScaledDotProductAttention(q, k, v, tensor.Causal()).Data[float32](context.Background())
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.3f\n", out)
	// Output: [1.000 1.670 2.255]
}
