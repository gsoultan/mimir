package tensor_test

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// groupCases are the groups the collective tests run over: devices of
// CPUs, which queue their parts, the CPU, whose parts run before the
// collective returns, with one of them, and the GPUs if there are
// several.
func groupCases() map[string][]tensor.Device {
	cases := map[string][]tensor.Device{
		"1":     tensor.CPUs(1),
		"2":     tensor.CPUs(2),
		"3":     tensor.CPUs(3),
		"4":     tensor.CPUs(4),
		"cpu+1": {tensor.CPU(), tensor.CPUs(1)[0]},
	}
	// Every GPU, through NCCL, where there are several CUDA devices.
	if gpus := tensor.GPUs(); len(gpus) > 1 {
		cases["gpus"] = gpus
	}
	return cases
}

// rankValues returns the values of rank r's tensor of n elements: small
// integers, so that sums are exact in any order, and their reductions.
func rankValues(r, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*7+r*13)%17 - 8)
	}
	return v
}

func TestGroupCollectives(t *testing.T) {
	ctx := t.Context()
	for name, devs := range groupCases() {
		for _, n := range []int{1, 7, 70001} {
			t.Run(fmt.Sprintf("%s/%d", name, n), func(t *testing.T) {
				g, err := tensor.NewGroup(devs...)
				if err != nil {
					t.Fatal(err)
				}
				defer g.Close()
				ranks := len(devs)
				vals := make([][]float32, ranks)
				ts := make([]*tensor.Tensor, ranks)
				for r := range ranks {
					vals[r] = rankValues(r, n*ranks)
					ts[r] = tensor.FromSlice(vals[r], ranks, n).To(devs[r])
				}
				reduce := func(op tensor.ReduceOp) []float32 {
					out := slices.Clone(vals[0])
					for _, v := range vals[1:] {
						for i, x := range v {
							switch op {
							case tensor.ReduceSum, tensor.ReduceMean:
								out[i] += x
							case tensor.ReduceMax:
								out[i] = max(out[i], x)
							case tensor.ReduceMin:
								out[i] = min(out[i], x)
							}
						}
					}
					if op == tensor.ReduceMean {
						for i := range out {
							out[i] *= 1 / float32(ranks)
						}
					}
					return out
				}
				check := func(what string, got []*tensor.Tensor, want func(r int) []float32) {
					t.Helper()
					for r, x := range got {
						if !x.Device().Equal(devs[r]) {
							t.Errorf("%s: rank %d result on %v", what, r, x.Device())
						}
						v := data[float32](t, x)
						if w := want(r); !slices.Equal(v, w) {
							t.Errorf("%s: rank %d: got %v..., want %v...", what, r, v[:min(len(v), 8)], w[:min(len(w), 8)])
						}
					}
				}
				for _, op := range []tensor.ReduceOp{tensor.ReduceSum, tensor.ReduceMean, tensor.ReduceMax, tensor.ReduceMin} {
					got, err := g.AllReduce(ctx, op, ts...)
					if err != nil {
						t.Fatal(err)
					}
					want := reduce(op)
					check("all-reduce "+op.String(), got, func(int) []float32 { return want })
				}
				root := ranks - 1
				got, err := g.Broadcast(ctx, root, ts...)
				if err != nil {
					t.Fatal(err)
				}
				check("broadcast", got, func(int) []float32 { return vals[root] })
				got, err = g.AllGather(ctx, ts...)
				if err != nil {
					t.Fatal(err)
				}
				gathered := slices.Concat(vals...)
				check("all-gather", got, func(int) []float32 { return gathered })
				got, err = g.ReduceScatter(ctx, tensor.ReduceSum, ts...)
				if err != nil {
					t.Fatal(err)
				}
				sum := reduce(tensor.ReduceSum)
				check("reduce-scatter", got, func(r int) []float32 { return sum[r*n : (r+1)*n] })
				// The inputs are untouched.
				for r, x := range ts {
					if v := data[float32](t, x); !slices.Equal(v, vals[r]) {
						t.Errorf("rank %d input changed", r)
					}
				}
			})
		}
	}
}

func TestGroupCollectivesOfOtherDTypes(t *testing.T) {
	ctx := t.Context()
	devs := tensor.CPUs(3)
	g, err := tensor.NewGroup(devs...)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	ints := make([]*tensor.Tensor, 3)
	halves := make([]*tensor.Tensor, 3)
	for r := range ints {
		v := make([]int32, 5)
		for i := range v {
			v[i] = int32(i*r - 3)
		}
		ints[r] = tensor.FromSlice(v, 5).To(devs[r])
		// A view, which the collective reads through a copy.
		halves[r] = tensor.Full(tensor.Float32, float64(r)+0.5, 4, 3).Cast(tensor.BFloat16).To(devs[r]).Transpose(0, 1)
	}
	got, err := g.AllReduce(ctx, tensor.ReduceMax, ints...)
	if err != nil {
		t.Fatal(err)
	}
	if v := data[int32](t, got[1]); !slices.Equal(v, []int32{-3, -1, 1, 3, 5}) {
		t.Errorf("int32 max: %v", v)
	}
	if _, err := g.AllReduce(ctx, tensor.ReduceMean, ints...); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("int32 mean: %v, want ErrDType", err)
	}
	got, err = g.AllReduce(ctx, tensor.ReduceSum, halves...)
	if err != nil {
		t.Fatal(err)
	}
	if got[2].DType() != tensor.BFloat16 || !slices.Equal(got[2].Shape(), []int{3, 4}) {
		t.Fatalf("bfloat16 sum: %v", got[2])
	}
	if v := data[float32](t, got[2].Cast(tensor.Float32)); v[0] != 4.5 || v[11] != 4.5 {
		t.Errorf("bfloat16 sum: %v", v)
	}
}

func TestGroupErrors(t *testing.T) {
	ctx := t.Context()
	devs := tensor.CPUs(2)
	if _, err := tensor.NewGroup(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("empty group: %v, want ErrShape", err)
	}
	if _, err := tensor.NewGroup(devs[0], devs[0]); !errors.Is(err, tensor.ErrDevice) {
		t.Errorf("repeated device: %v, want ErrDevice", err)
	}
	g, err := tensor.NewGroup(devs...)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	a := tensor.Ones(tensor.Float32, 3).To(devs[0])
	b := tensor.Ones(tensor.Float32, 3).To(devs[1])
	for name, c := range map[string]struct {
		ts   []*tensor.Tensor
		want error
	}{
		"one tensor":    {[]*tensor.Tensor{a}, tensor.ErrShape},
		"shapes":        {[]*tensor.Tensor{a, tensor.Ones(tensor.Float32, 4).To(devs[1])}, tensor.ErrShape},
		"dtypes":        {[]*tensor.Tensor{a, tensor.Ones(tensor.Int32, 3).To(devs[1])}, tensor.ErrShape},
		"wrong device":  {[]*tensor.Tensor{b, a}, tensor.ErrDevice},
		"failed tensor": {[]*tensor.Tensor{a, a.Reshape(7)}, tensor.ErrShape},
		"nil tensor":    {[]*tensor.Tensor{a, nil}, tensor.ErrNilTensor},
	} {
		_, err := g.AllReduce(ctx, tensor.ReduceSum, c.ts...)
		if !errors.Is(err, c.want) {
			t.Errorf("%s: %v, want %v", name, err, c.want)
		}
	}
	if _, err := g.ReduceScatter(ctx, tensor.ReduceSum, a, b); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("reduce-scatter of 3 over 2: %v, want ErrShape", err)
	}
	if _, err := g.Broadcast(ctx, 2, a, b); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("broadcast from rank 2: %v, want ErrShape", err)
	}
}

func TestReduceGradsAndBroadcastParams(t *testing.T) {
	ctx := t.Context()
	devs := tensor.CPUs(3)
	g, err := tensor.NewGroup(devs...)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	params := make([][]*tensor.Param, len(devs))
	for r, d := range devs {
		w := tensor.NewParam(tensor.Randn(uint64(r+1), 4, 3).To(d))
		bias := tensor.NewParam(tensor.Zeros(tensor.Float32, 3).To(d))
		stat := tensor.NewBuffer(tensor.Full(tensor.Float32, float64(r), 3).To(d))
		params[r] = []*tensor.Param{w, bias, stat}
		if r == 1 {
			continue // rank 1 has no gradients
		}
		x := tensor.Full(tensor.Float32, float64(r+1), 2, 4).To(d)
		if err := x.MatMul(w.Value()).Add(bias.Value()).Sum().Backward(ctx); err != nil {
			t.Fatal(err)
		}
	}
	old := params[0][0].Grad()
	if err := g.ReduceGrads(ctx, tensor.ReduceSum, 0, params...); err != nil {
		t.Fatal(err)
	}
	if _, err := old.Data[float32](ctx); !errors.Is(err, tensor.ErrOverwritten) {
		t.Errorf("old gradient: %v, want ErrOverwritten", err)
	}
	// d/dw of sum(x·w) is 2(r+1) for every element: 2 on rank 0, 6 on rank 2.
	if v := data[float32](t, params[0][0].Grad()); v[0] != 8 || v[11] != 8 {
		t.Errorf("reduced weight gradient %v, want 8s", v)
	}
	if v := data[float32](t, params[0][1].Grad()); v[0] != 4 {
		t.Errorf("reduced bias gradient %v, want 4s", v)
	}
	if v := data[float32](t, params[2][0].Grad()); v[0] != 6 {
		t.Errorf("rank 2's gradient %v, want 6s, untouched", v)
	}
	if params[1][0].Grad() != nil {
		t.Error("rank 1 got a gradient")
	}

	stale := params[2][0].Value()
	if err := g.BroadcastParams(ctx, 0, params...); err != nil {
		t.Fatal(err)
	}
	// The broadcast writes over the old value where it holds its memory
	// alone, as it does unless MIMIR_EAGER=1 shares it.
	if _, err := stale.Data[float32](ctx); err != nil && !errors.Is(err, tensor.ErrOverwritten) {
		t.Errorf("replaced value: %v, want ErrOverwritten or its values", err)
	}
	for i := range params[0] {
		want := data[float32](t, params[0][i].Value())
		for r := 1; r < len(devs); r++ {
			v := params[r][i].Value()
			if !v.Device().Equal(devs[r]) {
				t.Errorf("parameter %d of rank %d on %v", i, r, v.Device())
			}
			if got := data[float32](t, v); !slices.Equal(got, want) {
				t.Errorf("parameter %d of rank %d: %v, want %v", i, r, got, want)
			}
		}
	}
	// The broadcast values train on.
	if err := params[2][0].Value().Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if v := data[float32](t, params[2][0].Grad()); v[0] != 7 || math.IsNaN(float64(v[1])) {
		t.Errorf("gradient after broadcast: %v, want 7s", v)
	}
}
