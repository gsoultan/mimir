package optim_test

import (
	"slices"
	"testing"

	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// values returns x's elements, through float32 for a half tensor.
func values(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, err := x.Cast(tensor.Float32).Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// half returns v as a tensor of dt.
func half(dt tensor.DType, v ...float32) *tensor.Tensor {
	return tensor.FromSlice(v).Cast(dt)
}

// descend takes n steps of opt on the loss sum(p·slope), whose gradient
// is slope whatever p holds, so runs in different precisions see the same
// gradients.
func descend(t *testing.T, p *tensor.Param, opt *optim.Optimizer, slope *tensor.Tensor, n int) {
	t.Helper()
	for range n {
		if err := p.Value().Mul(slope).Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
	}
}

func TestStepRoundsToTheParameterDType(t *testing.T) {
	p := tensor.NewParam(half(tensor.BFloat16, 1, 2))
	// Adds 3/512: three quarters of the spacing of bfloat16 at 1, which
	// rounds up, and three eighths of the spacing at 2, which rounds down.
	rule := func(_ float64, _ int, p, _ *tensor.Tensor, _ optim.State) (*tensor.Tensor, optim.State) {
		return p.Cast(tensor.Float32).Add(tensor.Scalar(float32(3.0 / 512))), nil
	}
	opt := optim.New([]*tensor.Param{p}, 1, rule)
	descend(t, p, opt, tensor.FromSlice([]float32{1, 1}), 1)
	if got := p.Value().DType(); got != tensor.BFloat16 {
		t.Fatalf("parameter is %s after a step, want bfloat16", got)
	}
	if got, want := values(t, p.Value()), []float32{1 + 1.0/128, 2}; !slices.Equal(got, want) {
		t.Errorf("value after a float32 step = %v, want %v", got, want)
	}
}

func TestMasterWeightsKeepSmallUpdates(t *testing.T) {
	// Steps of 0.25 on a weight of 256, whose bfloat16 neighbours are a
	// whole unit away: rounded each time, the weight never moves; summed
	// in a float32 master copy, it reaches 254 after eight.
	slope := tensor.FromSlice([]float32{1})
	plain := tensor.NewParam(half(tensor.BFloat16, 256))
	descend(t, plain, optim.SGD([]*tensor.Param{plain}, optim.LR(0.25)), slope, 8)
	if got := values(t, plain.Value()); got[0] != 256 {
		t.Errorf("bfloat16 weight updated in bfloat16 = %v, want 256", got)
	}
	p := tensor.NewParam(half(tensor.BFloat16, 256))
	opt := optim.SGD([]*tensor.Param{p}, optim.LR(0.25), optim.MasterWeights())
	descend(t, p, opt, slope, 8)
	if got := values(t, p.Value()); got[0] != 254 || p.Value().DType() != tensor.BFloat16 {
		t.Errorf("bfloat16 weight updated through a master copy = %s %v, want bfloat16 254", p.Value().DType(), got)
	}
	state, _ := opt.Snapshot()
	m, ok := state["0.master"]
	if !ok || m.DType() != tensor.Float32 {
		t.Fatalf("state after master weight steps: %v, want a float32 master copy", state)
	}
	if got := values(t, m); got[0] != 254 {
		t.Errorf("master copy = %v, want 254", got)
	}
	// A float32 parameter has no master copy.
	f := tensor.NewParam(tensor.FromSlice([]float32{256}))
	optF := optim.SGD([]*tensor.Param{f}, optim.LR(0.25), optim.MasterWeights())
	descend(t, f, optF, slope, 8)
	if state, _ := optF.Snapshot(); len(state) != 0 {
		t.Errorf("float32 parameter has state %v under master weights", state)
	}
}

// TestMasterWeightsFollowFloat32 checks, for every optimizer, that a
// bfloat16 parameter updated through a master copy holds the float32
// run's weights rounded, since both see the same gradients.
func TestMasterWeightsFollowFloat32(t *testing.T) {
	slope := tensor.FromSlice([]float32{1, -2, 0.5})
	init := []float32{0.5, -1.25, 3}
	for name, mk := range map[string]func(ps []*tensor.Param, opts ...optim.Option) *optim.Optimizer{
		"sgd": func(ps []*tensor.Param, opts ...optim.Option) *optim.Optimizer {
			return optim.SGD(ps, append(opts, optim.LR(0.05), optim.Momentum(0.9), optim.WeightDecay(0.01))...)
		},
		"adam":    func(ps []*tensor.Param, opts ...optim.Option) *optim.Optimizer { return optim.Adam(ps, opts...) },
		"adamw":   func(ps []*tensor.Param, opts ...optim.Option) *optim.Optimizer { return optim.AdamW(ps, opts...) },
		"rmsprop": func(ps []*tensor.Param, opts ...optim.Option) *optim.Optimizer { return optim.RMSprop(ps, opts...) },
	} {
		t.Run(name, func(t *testing.T) {
			f := tensor.NewParam(tensor.FromSlice(init))
			descend(t, f, mk([]*tensor.Param{f}), slope, 5)
			for _, dt := range []tensor.DType{tensor.BFloat16, tensor.Float16} {
				p := tensor.NewParam(half(dt, init...))
				opt := mk([]*tensor.Param{p}, optim.MasterWeights())
				descend(t, p, opt, slope, 5)
				want := values(t, f.Value().Cast(dt))
				if got := values(t, p.Value()); !slices.Equal(got, want) {
					t.Errorf("%s weights %v, want the float32 weights rounded, %v", dt, got, want)
				}
				state, _ := opt.Snapshot()
				if got := values(t, state["0.master"]); !slices.Equal(got, values(t, f.Value())) {
					t.Errorf("%s master copy %v, want the float32 weights %v", dt, got, values(t, f.Value()))
				}
			}
		})
	}
}

func TestMasterWeightsSnapshotRestoreContinuesIdentically(t *testing.T) {
	slope := tensor.FromSlice([]float32{1, -2, 0.5})
	a := tensor.NewParam(half(tensor.BFloat16, 1, -2, 3))
	optA := optim.Adam([]*tensor.Param{a}, optim.LR(0.1), optim.MasterWeights())
	descend(t, a, optA, slope, 3)

	b := tensor.NewParam(a.Value())
	optB := optim.Adam([]*tensor.Param{b}, optim.LR(0.1), optim.MasterWeights())
	state, n := optA.Snapshot()
	if _, ok := state["0.master"]; !ok {
		t.Fatalf("snapshot %v has no master copy", state)
	}
	if err := optB.Restore(state, n); err != nil {
		t.Fatal(err)
	}
	descend(t, a, optA, slope, 2)
	descend(t, b, optB, slope, 2)
	if va, vb := values(t, a.Value()), values(t, b.Value()); !slices.Equal(va, vb) {
		t.Errorf("restored optimizer diverged: %v vs %v", va, vb)
	}
	// Without a master copy in the state, the next step starts one from
	// the parameter.
	c := tensor.NewParam(a.Value())
	optC := optim.Adam([]*tensor.Param{c}, optim.LR(0.1), optim.MasterWeights())
	state, n = optA.Snapshot()
	delete(state, "0.master")
	if err := optC.Restore(state, n); err != nil {
		t.Fatal(err)
	}
	descend(t, c, optC, slope, 1)
	if s, _ := optC.Snapshot(); s["0.master"] == nil || c.Value().DType() != tensor.BFloat16 {
		t.Errorf("step after a restore without a master copy: state %v, parameter %s", s, c.Value().DType())
	}
}
