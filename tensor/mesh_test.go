package tensor_test

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// meshes returns the meshes the sharding tests run over: four devices of
// CPUs as a 2×2 and a 1×4 mesh, with the axes "data" and "model".
func meshes(t *testing.T) map[string]*tensor.Mesh {
	t.Helper()
	devs := tensor.CPUs(4)
	out := map[string]*tensor.Mesh{}
	for name, shape := range map[string][]int{"2x2": {2, 2}, "1x4": {1, 4}, "4x1": {4, 1}} {
		m, err := tensor.NewMesh(shape, []string{"data", "model"}, devs...)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		out[name] = m
	}
	return out
}

// values returns n deterministic values in about [-1, 1].
func values(seed, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.Sin(float64(i*7+seed*131)+0.5)) * 0.9
	}
	return v
}

// input is a tensor of an op test and how to shard it.
type input struct {
	shape []int
	spec  tensor.PartitionSpec
}

func in(spec tensor.PartitionSpec, shape ...int) input { return input{shape, spec} }

// shardNear reports the first element where got and want differ by more than
// tol, relative to the larger of 1 and want's magnitude.
func shardNear(got, want []float32, tol float64) error {
	if len(got) != len(want) {
		return fmt.Errorf("%d elements, want %d", len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > tol*max(1, math.Abs(float64(want[i]))) || math.IsNaN(float64(got[i])) != math.IsNaN(float64(want[i])) {
			return fmt.Errorf("element %d is %v, want %v", i, got[i], want[i])
		}
	}
	return nil
}

// checkSharded checks f of sharded inputs, and the gradients of the sum
// of f's result times a fixed tensor with respect to them, against f of
// the same inputs on one device.
func checkSharded(t *testing.T, m *tensor.Mesh, f func(x ...*tensor.Tensor) *tensor.Tensor, ins ...input) {
	t.Helper()
	ctx := t.Context()
	plain := make([]*tensor.Tensor, len(ins))
	sharded := make([]*tensor.Tensor, len(ins))
	for i, x := range ins {
		v := tensor.FromSlice(values(i+1, prodOf(x.shape)), x.shape...)
		plain[i] = v.RequiresGrad()
		sharded[i] = tensor.Shard(v, m, x.spec).RequiresGrad()
		if err := sharded[i].Err(); err != nil {
			t.Fatalf("shard input %d: %v", i, err)
		}
	}
	want := f(plain...)
	got := f(sharded...)
	if err := got.Err(); err != nil {
		t.Fatalf("sharded: %v", err)
	}
	wv, err := want.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	gv, err := got.Data[float32](ctx)
	if err != nil {
		t.Fatalf("sharded: %v", err)
	}
	if err := shardNear(gv, wv, 1e-5); err != nil {
		t.Fatalf("result: %v", err)
	}
	// Gradients of a weighted sum, so that every element's gradient
	// differs.
	w := tensor.FromSlice(values(99, prodOf(want.Shape())), want.Shape()...)
	lossPlain := want.Mul(w).Sum()
	lossSharded := got.Mul(tensor.Shard(w, m, nil)).Sum()
	if err := lossPlain.Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if err := lossSharded.Backward(ctx); err != nil {
		t.Fatalf("sharded backward: %v", err)
	}
	for i := range ins {
		if plain[i].Grad() == nil {
			if sharded[i].Grad() != nil {
				t.Fatalf("gradient %d of an input without one", i)
			}
			continue
		}
		gw, err := plain[i].Grad().Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := sharded[i].Grad().Sharding(); !ok {
			t.Fatalf("gradient %d is not sharded", i)
		}
		gg, err := sharded[i].Grad().Data[float32](ctx)
		if err != nil {
			t.Fatalf("sharded gradient %d: %v", i, err)
		}
		if err := shardNear(gg, gw, 1e-4); err != nil {
			t.Fatalf("gradient %d: %v", i, err)
		}
	}
}

func prodOf(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func TestShardRoundTrip(t *testing.T) {
	ctx := t.Context()
	for name, m := range meshes(t) {
		for _, spec := range []tensor.PartitionSpec{nil, tensor.P("data"), tensor.P("", "model"), tensor.P("model", "data"), tensor.P("", "model").WithSections(1, 4, 8, 4)} {
			t.Run(name+"/"+spec.String(), func(t *testing.T) {
				data := values(1, 4*16)
				x := tensor.Shard(tensor.FromSlice(data, 4, 16), m, spec)
				if err := x.Err(); err != nil {
					t.Fatal(err)
				}
				s, ok := x.Sharding()
				if !ok || s.Mesh != m || s.Spec.String() != trim(m, spec).String() {
					t.Fatalf("Sharding() = %v, %v; want %v", s, ok, spec)
				}
				got, err := x.Data[float32](ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, data) {
					t.Fatalf("got %v, want %v", got, data)
				}
				back, err := x.To(tensor.CPU()).Data[float32](ctx)
				if err != nil || !slices.Equal(back, data) {
					t.Fatalf("To: %v, %v", back, err)
				}
			})
		}
	}
}

// trim returns spec as Sharding reports it: without splits over mesh axes
// of one device, which split nothing, and without trailing axes that are
// not split.
func trim(m *tensor.Mesh, spec tensor.PartitionSpec) tensor.PartitionSpec {
	spec = slices.Clone(spec)
	for i, s := range spec {
		if m.Size(s.Axis) == 1 {
			spec[i] = tensor.Split{}
		}
	}
	for len(spec) > 0 && spec[len(spec)-1].Axis == "" {
		spec = spec[:len(spec)-1]
	}
	return spec
}

func TestShardErrors(t *testing.T) {
	m := meshes(t)["2x2"]
	x := tensor.FromSlice(values(1, 12), 3, 4)
	for name, spec := range map[string]tensor.PartitionSpec{
		"uneven":   tensor.P("data"),
		"no axis":  tensor.P("", "pipe"),
		"twice":    tensor.P("model", "model"),
		"too many": tensor.P("", "", "data"),
		"sections": tensor.P("", "model").WithSections(1, 1, 3),
	} {
		if err := tensor.Shard(x, m, spec).Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
	if err := x.Add(tensor.Shard(x, m, nil)).Err(); !errors.Is(err, tensor.ErrDevice) {
		t.Errorf("sharded plus local: %v, want ErrDevice", err)
	}
}

func TestShardedOps(t *testing.T) {
	type opCase struct {
		f   func(x ...*tensor.Tensor) *tensor.Tensor
		ins []input
	}
	cases := map[string]opCase{
		"add same": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Add(x[1]).Exp() },
			[]input{in(tensor.P("data", "model"), 4, 8), in(tensor.P("data", "model"), 4, 8)}},
		"mul mixed": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Mul(x[1]).Tanh() },
			[]input{in(tensor.P("data"), 4, 8), in(tensor.P("", "model"), 4, 8)}},
		"broadcast": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Sub(x[1]) },
			[]input{in(tensor.P("data", "model"), 4, 8), in(nil, 8)}},
		"matmul rows": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(x[1]) },
			[]input{in(tensor.P("data"), 4, 8), in(nil, 8, 4)}},
		"matmul contracted": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(x[1]) },
			[]input{in(tensor.P("", "model"), 4, 8), in(tensor.P("model"), 8, 4)}},
		"matmul columns": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(x[1]).GELU() },
			[]input{in(tensor.P("data"), 4, 8), in(tensor.P("", "model"), 8, 4)}},
		"matmul conflict": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(x[1]) },
			[]input{in(tensor.P("data"), 4, 8), in(tensor.P("", "data"), 8, 4)}},
		"mlp": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(x[1]).ReLU().MatMul(x[2]) },
			[]input{in(tensor.P("data"), 4, 8), in(tensor.P("", "model"), 8, 16), in(tensor.P("model"), 16, 4)}},
		"batched matmul": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(x[1]) },
			[]input{in(tensor.P("data", "", "model"), 4, 4, 8), in(tensor.P("", "model"), 8, 4)}},
		"sum": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Sum(1) },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"sum all": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Mul(x[0]).Sum() },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"max": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Max(0) },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"softmax": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Softmax(-1) },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"logsoftmax rows": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].LogSoftmax(0) },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"reshape merge": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Reshape(32).Exp() },
			[]input{in(tensor.P("data"), 4, 8)}},
		"reshape split": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Reshape(4, 2, 4).Transpose(0, 2) },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"reshape inner": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Reshape(32) },
			[]input{in(tensor.P("", "model"), 4, 8)}},
		"transpose": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Transpose(0, 1).Sin() },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"slice": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Split(1, 3, 5)[1] },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"sections": {func(x ...*tensor.Tensor) *tensor.Tensor {
			p := x[0].Split(1, 4, 8, 4)
			return p[0].Add(p[2]).Mul(p[1].Split(1, 4, 4)[1])
		}, []input{in(tensor.P("", "model").WithSections(1, 4, 8, 4), 4, 16)}},
		"cat": {func(x ...*tensor.Tensor) *tensor.Tensor { return tensor.Cat(1, x[0], x[1]).Exp() },
			[]input{in(tensor.P("data", "model"), 4, 8), in(tensor.P("data", "model"), 4, 4)}},
		"pad": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Pad([][2]int{{0, 0}, {1, 2}}, 0) },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"gather": {func(x ...*tensor.Tensor) *tensor.Tensor {
			idx := x[1].Detach().Mul(tensor.Scalar[float32](2)).Add(tensor.Scalar[float32](2)).Cast(tensor.Int32)
			return x[0].Gather(0, idx)
		}, []input{in(tensor.P("", "model"), 4, 8), in(tensor.P("data", "model"), 8, 8)}},
		"cumsum": {func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].CumSum(1) },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"sort": {func(x ...*tensor.Tensor) *tensor.Tensor { s, _ := x[0].Sort(1, false); return s },
			[]input{in(tensor.P("data", "model"), 4, 8)}},
		"attention": {func(x ...*tensor.Tensor) *tensor.Tensor {
			return tensor.ScaledDotProductAttention(x[0], x[1], x[2], tensor.Causal())
		}, []input{in(tensor.P("data", "model"), 4, 4, 6, 8), in(tensor.P("data", "model"), 4, 4, 6, 8), in(tensor.P("data", "model"), 4, 4, 6, 8)}},
		"attention gqa": {func(x ...*tensor.Tensor) *tensor.Tensor {
			return tensor.ScaledDotProductAttention(x[0], x[1], x[2])
		}, []input{in(tensor.P("", "model"), 2, 8, 6, 8), in(tensor.P("", "model"), 2, 4, 6, 8), in(nil, 2, 4, 6, 8)}},
	}
	for mname, m := range meshes(t) {
		for name, c := range cases {
			t.Run(mname+"/"+name, func(t *testing.T) { checkSharded(t, m, c.f, c.ins...) })
		}
	}
}

// TestWithSharding checks that a product over a split contracted axis
// sums its partial results with an all-reduce, or with a reduce-scatter
// when the result is constrained to stay split, that Sharding reports
// the layout before anything is computed, and that the gradients agree
// either way.
func TestWithSharding(t *testing.T) {
	ctx := t.Context()
	m := meshes(t)["4x1"]
	xv, wv := values(1, 4*16), values(2, 16*8)
	want := tensor.FromSlice(xv, 4, 16).MatMul(tensor.FromSlice(wv, 16, 8))
	wantData, err := want.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]struct {
		spec tensor.PartitionSpec
		want tensor.Traffic
	}{
		"all-reduce":     {nil, tensor.Traffic{AllReduces: 1}},
		"reduce-scatter": {tensor.P("data"), tensor.Traffic{ReduceScatters: 1}},
	} {
		t.Run(name, func(t *testing.T) {
			x := tensor.Shard(tensor.FromSlice(xv, 4, 16), m, tensor.P("", "data")).RequiresGrad()
			w := tensor.Shard(tensor.FromSlice(wv, 16, 8), m, tensor.P("data")).RequiresGrad()
			y := tensor.WithSharding(x.MatMul(w), m, c.spec)
			s, ok := y.Sharding()
			if !ok || s.Spec.String() != c.spec.String() {
				t.Fatalf("Sharding() = %v, %v; want %v", s, ok, c.spec)
			}
			before := m.Traffic()
			if err := y.Realize(ctx); err != nil {
				t.Fatal(err)
			}
			after := m.Traffic()
			got := tensor.Traffic{
				AllReduces:     after.AllReduces - before.AllReduces,
				ReduceScatters: after.ReduceScatters - before.ReduceScatters,
				AllGathers:     after.AllGathers - before.AllGathers,
			}
			// Eager mode computes the product, its partial sums added up, before
			// the constraint exists.
			if got != c.want && !tensor.Eager() {
				t.Errorf("collectives %+v, want %+v", got, c.want)
			}
			data, err := y.Data[float32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			if err := shardNear(data, wantData, 1e-5); err != nil {
				t.Fatal(err)
			}
			if err := y.Mul(y).Sum().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			for i, g := range []*tensor.Tensor{x.Grad(), w.Grad()} {
				gs, ok := g.Sharding()
				if !ok {
					t.Fatalf("gradient %d is not sharded", i)
				}
				if want := []tensor.PartitionSpec{tensor.P("", "data"), tensor.P("data")}[i]; gs.Spec.String() != want.String() {
					t.Errorf("gradient %d laid out as %v, want %v", i, gs.Spec, want)
				}
			}
		})
	}
}

// TestShardAcrossMeshes moves a tensor between two meshes and back to one
// device and checks the values and the gradient.
func TestShardAcrossMeshes(t *testing.T) {
	ctx := t.Context()
	ms := meshes(t)
	v := values(3, 4*8)
	x := tensor.FromSlice(v, 4, 8).RequiresGrad()
	a := tensor.Shard(x, ms["2x2"], tensor.P("data", "model"))
	b := tensor.Shard(a.Mul(a), ms["1x4"], tensor.P("", "model"))
	c := b.To(tensor.CPU())
	got, err := c.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i := range v {
		if want := v[i] * v[i]; math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("element %d: %v, want %v", i, got[i], want)
		}
	}
	if err := c.Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if _, ok := x.Grad().Sharding(); ok {
		t.Fatal("the gradient of a local tensor is sharded")
	}
	g, err := x.Grad().Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i := range v {
		if want := 2 * v[i]; math.Abs(float64(g[i]-want)) > 1e-6 {
			t.Fatalf("gradient %d: %v, want %v", i, g[i], want)
		}
	}
}

func TestMeshErrors(t *testing.T) {
	devs := tensor.CPUs(4)
	for name, c := range map[string]struct {
		shape []int
		axes  []string
		devs  []tensor.Device
		want  error
	}{
		"axes":       {[]int{2, 2}, []string{"data"}, devs, tensor.ErrShape},
		"size":       {[]int{2, 3}, []string{"data", "model"}, devs, tensor.ErrShape},
		"names":      {[]int{2, 2}, []string{"data", "data"}, devs, tensor.ErrShape},
		"duplicates": {[]int{2, 2}, []string{"data", "model"}, []tensor.Device{devs[0], devs[1], devs[2], devs[0]}, tensor.ErrDevice},
	} {
		if _, err := tensor.NewMesh(c.shape, c.axes, c.devs...); !errors.Is(err, c.want) {
			t.Errorf("%s: %v, want %v", name, err, c.want)
		}
	}
	x := tensor.FromSlice(values(1, 8), 2, 4)
	if err := tensor.Shard(x, nil, nil).Err(); !errors.Is(err, tensor.ErrDevice) {
		t.Errorf("nil mesh: %v, want ErrDevice", err)
	}
	if err := tensor.WithSharding(x, nil, nil).Err(); !errors.Is(err, tensor.ErrDevice) {
		t.Errorf("nil mesh constraint: %v, want ErrDevice", err)
	}
	m, err := tensor.NewMesh([]int{4}, []string{"data"}, devs...)
	if err != nil {
		t.Fatal(err)
	}
	s := tensor.Shard(x, m, tensor.P("", "data"))
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Sum().Realize(t.Context()); !errors.Is(err, tensor.ErrDevice) {
		t.Errorf("sum on a closed mesh: %v, want ErrDevice", err)
	}
}

func TestShardedStepsReuseTheirAnalysis(t *testing.T) {
	m := meshes(t)["2x2"]
	w := tensor.Shard(tensor.Randn(1, 8, 8), m, tensor.P("", "model"))
	for step := range 4 {
		x := tensor.Shard(tensor.Randn(uint64(2+step), 4, 8), m, tensor.P("data"))
		// A scalar that changes every step, as a learning rate may, keeps
		// the graph's analysis.
		y := x.MatMul(w).ReLU().Mul(tensor.Scalar(float32(step + 1))).Sum()
		if _, err := y.Item[float32](t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	if got := tensor.PlanHits(m); got < 3 {
		t.Errorf("the steps found %d analyses in the cache, want at least 3", got)
	}
}
