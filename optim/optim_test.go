package optim_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
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
}

var schedules = map[string]optim.Schedule{
	"step":   optim.StepDecay(0.1, 2, 0.5),
	"cosine": optim.Cosine(0.1, 0.01, 5),
	"warmup": optim.Warmup(0.1, 0.25, 4),
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
