package optim_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/internal/sched"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// steps mirrors STEPS in tools/golden/gen.py.
const steps = 5

var optimizers = map[string]func(ps []*tensor.Param) *optim.Optimizer{
	"sgd": func(ps []*tensor.Param) *optim.Optimizer { return optim.SGD(ps, optim.LR(0.05)) },
	"sgd_momentum": func(ps []*tensor.Param) *optim.Optimizer {
		return optim.SGD(ps, optim.LR(0.05), optim.Momentum(0.9), optim.WeightDecay(0.01))
	},
	"sgd_nesterov": func(ps []*tensor.Param) *optim.Optimizer {
		return optim.SGD(ps, optim.LR(0.05), optim.Momentum(0.9), optim.Nesterov())
	},
	"adam": func(ps []*tensor.Param) *optim.Optimizer { return optim.Adam(ps, optim.LR(0.1)) },
	"adam_l2": func(ps []*tensor.Param) *optim.Optimizer {
		return optim.Adam(ps, optim.LR(0.1), optim.WeightDecay(0.1))
	},
	"adamw": func(ps []*tensor.Param) *optim.Optimizer {
		return optim.AdamW(ps, optim.LR(0.1), optim.WeightDecay(0.1))
	},
	"rmsprop": func(ps []*tensor.Param) *optim.Optimizer { return optim.RMSprop(ps, optim.LR(0.01), optim.Alpha(0.9)) },
	"adagrad": func(ps []*tensor.Param) *optim.Optimizer {
		return optim.Adagrad(ps, optim.LR(0.1), optim.LRDecay(0.05), optim.InitialAccumulator(0.1))
	},
	"adadelta": func(ps []*tensor.Param) *optim.Optimizer { return optim.Adadelta(ps, optim.LR(1), optim.Rho(0.9)) },
	"adamax":   func(ps []*tensor.Param) *optim.Optimizer { return optim.Adamax(ps, optim.LR(0.1)) },
	"nadam":    func(ps []*tensor.Param) *optim.Optimizer { return optim.NAdam(ps, optim.LR(0.1)) },
	"lion": func(ps []*tensor.Param) *optim.Optimizer {
		return optim.Lion(ps, optim.LR(0.05), optim.WeightDecay(0.1))
	},
}

var schedules = map[string]optim.Schedule{
	"step":        optim.StepDecay(0.1, 2, 0.5),
	"cosine":      optim.Cosine(0.1, 0.01, 5),
	"warmup":      optim.Warmup(0.1, 0.25, 4),
	"exponential": optim.Exponential(0.1, 0.8),
	"polynomial":  optim.Polynomial(0.1, 5, 2),
	"onecycle":    optim.OneCycle(0.1, 10, 0.3),
}

// TestGolden runs each optimizer on sum(scale * (p - target)²) and each
// schedule over its epochs, against PyTorch or NumPy.
func TestGolden(t *testing.T) {
	cases, meta := goldentest.Load(t, "optim")
	t.Logf("golden oracle: %s", meta["oracle"])
	tol := goldentest.Tol{Atol: 1e-5, Rtol: 1e-4}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			if s, ok := schedules[name]; ok {
				lrs := make([]float32, c.Want.Shape[0])
				for e := range lrs {
					lrs[e] = float32(s(e))
				}
				goldentest.Compare(t, c.Want, lrs, c.Want.Shape, goldentest.Tol{Atol: 1e-7, Rtol: 1e-6})
				return
			}
			mk, ok := optimizers[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := goldentest.Inputs(t, c)
			p := tensor.NewParam(in[0])
			opt := mk([]*tensor.Param{p})
			for range steps {
				d := p.Value().Sub(in[1])
				if err := d.Mul(d).Mul(in[2]).Sum().Backward(t.Context()); err != nil {
					t.Fatal(err)
				}
				if err := opt.Step(t.Context()); err != nil {
					t.Fatal(err)
				}
				opt.ZeroGrad()
			}
			goldentest.CompareTensor(t, c.Want, p.Value(), tol)
		})
	}
}

func TestStepFailureChangesNothing(t *testing.T) {
	p := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	opt := optim.Adam([]*tensor.Param{p})
	if err := p.Value().Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := opt.Step(cancelled); err == nil {
		t.Fatal("step with a cancelled context succeeded")
	}
	if opt.Steps() != 0 {
		t.Error("failed step counted")
	}
	v, _ := p.Value().Data[float32](t.Context())
	if v[0] != 1 || v[1] != 2 {
		t.Errorf("failed step changed the parameter to %v", v)
	}
}

func TestSnapshotRestoreContinuesIdentically(t *testing.T) {
	run := func(p *tensor.Param, opt *optim.Optimizer, n int) {
		for range n {
			if err := p.Value().Mul(p.Value()).Sum().Backward(t.Context()); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(t.Context()); err != nil {
				t.Fatal(err)
			}
			opt.ZeroGrad()
		}
	}
	a := tensor.NewParam(tensor.FromSlice([]float32{1, -2, 3}))
	optA := optim.Adam([]*tensor.Param{a}, optim.LR(0.1))
	run(a, optA, 3)

	b := tensor.NewParam(a.Value())
	optB := optim.Adam([]*tensor.Param{b}, optim.LR(0.1))
	state, n := optA.Snapshot()
	if err := optB.Restore(state, n); err != nil {
		t.Fatal(err)
	}
	run(a, optA, 2)
	run(b, optB, 2)
	va, _ := a.Value().Data[float32](t.Context())
	vb, _ := b.Value().Data[float32](t.Context())
	for i := range va {
		if va[i] != vb[i] {
			t.Fatalf("restored optimizer diverged: %v vs %v", va, vb)
		}
	}
	if err := optB.Restore(map[string]*tensor.Tensor{"0.exp_avg": tensor.Zeros(tensor.Float32, 2)}, 1); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("restore of the wrong shape: %v, want ErrShape", err)
	}
}

func TestBadOptionsFailStep(t *testing.T) {
	ps := []*tensor.Param{tensor.NewParam(tensor.Zeros(tensor.Float32, 1))}
	for name, opt := range map[string]*optim.Optimizer{
		"negative lr":      optim.SGD(ps, optim.LR(-1)),
		"nesterov alone":   optim.SGD(ps, optim.Nesterov()),
		"beta of one":      optim.Adam(ps, optim.Betas(1, 0.999)),
		"negative decay":   optim.AdamW(ps, optim.WeightDecay(-0.1)),
		"NaN momentum":     optim.RMSprop(ps, optim.Momentum(math.NaN())),
		"negative epsilon": optim.Adam(ps, optim.Eps(-1)),
	} {
		if err := opt.Step(t.Context()); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
}

func TestSchedulesHoldAfterTheirLastEpoch(t *testing.T) {
	if got := optim.Cosine(1, 0.1, 10)(50); math.Abs(got-0.1) > 1e-12 {
		t.Errorf("cosine after its last epoch = %v, want 0.1", got)
	}
	if got := optim.Warmup(1, 0, 10)(50); got != 1 {
		t.Errorf("warmup after its last epoch = %v, want 1", got)
	}
}

func TestStepOverwritesTheOldValues(t *testing.T) {
	p := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	opt := optim.Adam([]*tensor.Param{p}, optim.LR(0.5))
	step := func() {
		t.Helper()
		if err := p.Value().Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
	}
	step()
	old := p.Value()
	state, _ := opt.Snapshot()
	step()
	if _, err := old.Data[float32](t.Context()); !errors.Is(err, tensor.ErrOverwritten) {
		t.Errorf("value from before a step: got %v, want ErrOverwritten", err)
	}
	for key, s := range state {
		if _, err := s.Data[float32](t.Context()); !errors.Is(err, tensor.ErrOverwritten) {
			t.Errorf("state %s from before a step: got %v, want ErrOverwritten", key, err)
		}
	}
	// Two Adam steps of 0.5 on a constant gradient move each value by 1.
	if got, err := p.Value().Data[float32](t.Context()); err != nil || math.Abs(float64(got[0])) > 1e-5 || math.Abs(float64(got[1]-1)) > 1e-5 {
		t.Errorf("value after two steps = %v, %v; want [0 1]", got, err)
	}
}

func TestRulesMayKeepTheGradientAsState(t *testing.T) {
	p := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	// The first step keeps g as it is; later steps add to it.
	keep := func(lr float64, step int, p, g *tensor.Tensor, s optim.State) (*tensor.Tensor, optim.State) {
		sum := g
		if old := s["sum"]; old != nil {
			sum = old.Add(g)
		}
		return p.Sub(sum.Mul(tensor.Scalar(float32(lr)))), optim.State{"sum": sum}
	}
	opt := optim.New([]*tensor.Param{p}, 0.5, keep)
	for range 2 {
		if err := p.Value().Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
	}
	// Steps of 0.5·1 and then 0.5·2.
	if got, err := p.Value().Data[float32](t.Context()); err != nil || got[0] != -0.5 || got[1] != 0.5 {
		t.Errorf("value after two steps = %v, %v; want [-0.5 0.5]", got, err)
	}
}

func TestGradScaleDividesTheGradients(t *testing.T) {
	p := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	opt := optim.SGD([]*tensor.Param{p}, optim.LR(0.5))
	if opt.GradScale() != 1 {
		t.Errorf("default gradient scale = %v, want 1", opt.GradScale())
	}
	// The gradient of 4·sum(p) is 4 everywhere; scaled down by 4 it is 1.
	opt.SetGradScale(4)
	if err := p.Value().Sum().Mul(tensor.Scalar[float32](4)).Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := opt.Step(t.Context()); err != nil {
		t.Fatal(err)
	}
	opt.ZeroGrad()
	if got, err := p.Value().Data[float32](t.Context()); err != nil || got[0] != 0.5 || got[1] != 1.5 {
		t.Errorf("value after a step at scale 4 = %v, %v; want [0.5 1.5]", got, err)
	}
	for _, scale := range []float64{0, -2, math.NaN(), math.Inf(1)} {
		opt.SetGradScale(scale)
		if err := p.Value().Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(t.Context()); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("step at gradient scale %v: %v, want ErrShape", scale, err)
		}
		opt.ZeroGrad()
	}
}

func TestGradsFinite(t *testing.T) {
	a := tensor.NewParam(tensor.FromSlice([]float32{1, 2}))
	b := tensor.NewParam(tensor.FromSlice([]float32{3, 4, 5}))
	opt := optim.SGD([]*tensor.Param{a, b})
	if ok, err := opt.GradsFinite(t.Context()); !ok || err != nil {
		t.Errorf("without gradients: %v, %v; want true", ok, err)
	}
	if err := a.Value().Sum().Add(b.Value().Sum()).Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if ok, err := opt.GradsFinite(t.Context()); !ok || err != nil {
		t.Errorf("with finite gradients: %v, %v; want true", ok, err)
	}
	opt.ZeroGrad()
	// Each kind of non-finite value, in either parameter.
	for _, bad := range []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		for i, p := range []*tensor.Param{a, b} {
			w := make([]float32, p.Value().Size())
			w[len(w)-1] = bad
			if err := p.Value().Mul(tensor.FromSlice(w)).Sum().Add(a.Value().Sum()).Add(b.Value().Sum()).Backward(t.Context()); err != nil {
				t.Fatal(err)
			}
			if ok, err := opt.GradsFinite(t.Context()); ok || err != nil {
				t.Errorf("with %v in the gradient of parameter %d: %v, %v; want false", bad, i, ok, err)
			}
			opt.ZeroGrad()
		}
	}
}

// TestStepsReplayTheirProgram checks that, once the optimizers' states
// exist, every step schedules the same graph and so replays the program
// compiled for the first, however many parameters and states there are.
func TestStepsReplayTheirProgram(t *testing.T) {
	for name, newOpt := range optimizers {
		t.Run(name, func(t *testing.T) {
			var ps []*tensor.Param
			for i := range 24 {
				ps = append(ps, tensor.NewParam(tensor.Full(tensor.Float32, float64(i), 3)))
			}
			opt := newOpt(ps)
			step := func() {
				t.Helper()
				l := ps[0].Value().Mul(ps[0].Value()).Sum()
				for _, p := range ps[1:] {
					l = l.Add(p.Value().Mul(p.Value()).Sum())
				}
				if err := l.Backward(t.Context()); err != nil {
					t.Fatal(err)
				}
				if err := opt.Step(t.Context()); err != nil {
					t.Fatal(err)
				}
				opt.ZeroGrad()
			}
			for range 3 {
				step()
			}
			before := sched.Compiled()
			for range 5 {
				step()
			}
			if n := sched.Compiled() - before; n > 0 {
				t.Errorf("5 steps compiled %d programs, want none", n)
			}
		})
	}
}

// TestRAdamRectifies follows torch.optim.RAdam past the steps where it
// takes plain momentum steps, which the five golden steps never leave.
func TestRAdamRectifies(t *testing.T) {
	p0, target := []float64{1.5, -0.5, 0.25}, []float64{-1, 1, 0}
	p := tensor.NewParam(tensor.FromSlice([]float32{1.5, -0.5, 0.25}, 3))
	tgt := tensor.FromSlice([]float32{-1, 1, 0}, 3)
	opt := optim.RAdam([]*tensor.Param{p}, optim.LR(0.05), optim.Betas(0.9, 0.9))
	want := slices.Clone(p0)
	m, v := make([]float64, 3), make([]float64, 3)
	b1, b2, eps, lr := 0.9, 0.9, 1e-8, 0.05
	rhoInf := 2/(1-b2) - 1
	for step := 1; step <= 30; step++ {
		d := p.Value().Sub(tgt)
		if err := d.Mul(d).Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
		b2t := math.Pow(b2, float64(step))
		bc1, bc2 := 1-math.Pow(b1, float64(step)), 1-b2t
		rho := rhoInf - 2*float64(step)*b2t/bc2
		for i := range want {
			g := 2 * (want[i] - target[i])
			m[i] = b1*m[i] + (1-b1)*g
			v[i] = b2*v[i] + (1-b2)*g*g
			mhat := m[i] / bc1
			if rho > 5 {
				r := math.Sqrt((rho - 4) * (rho - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rho))
				want[i] -= lr * mhat * r * math.Sqrt(bc2) / (math.Sqrt(v[i]) + eps)
			} else {
				want[i] -= lr * mhat
			}
		}
	}
	got, err := p.Value().Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > 1e-4 {
			t.Fatalf("after 30 steps %v, want %v", got, want)
		}
	}
}
