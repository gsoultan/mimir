package train_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/train"
)

// parallelDevices returns n devices for data-parallel tests: GPUs under
// MIMIR_DEVICE=gpu, skipping t if there are fewer, and otherwise the
// devices of tensor.CPUs. It also returns the tolerance of comparisons
// with one device, which GPUs, summing in other orders, need larger.
func parallelDevices(t *testing.T, n int) ([]tensor.Device, float64) {
	t.Helper()
	if os.Getenv("MIMIR_DEVICE") == "gpu" {
		ds := tensor.GPUs()
		if len(ds) < n {
			t.Skipf("%d GPUs, want %d", len(ds), n)
		}
		return ds[:n], 1e-4
	}
	return tensor.CPUs(n), 1e-5
}

// paramValues returns the values of every parameter and buffer of m.
func paramValues(t *testing.T, m nn.Module) [][]float32 {
	t.Helper()
	var vs [][]float32
	for name, p := range nn.NamedParams(m) {
		v, err := p.Value().Data[float32](t.Context())
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		vs = append(vs, v)
	}
	return vs
}

// closeValues reports the first value of got that differs from want by
// more than tol relative to the largest magnitude of want's tensor.
func closeValues(got, want [][]float32, tol float64) error {
	if len(got) != len(want) {
		return fmt.Errorf("%d tensors, want %d", len(got), len(want))
	}
	for i := range want {
		scale := 1.0
		for _, w := range want[i] {
			scale = max(scale, math.Abs(float64(w)))
		}
		for j, w := range want[i] {
			if d := math.Abs(float64(got[i][j] - w)); !(d <= tol*scale) {
				return fmt.Errorf("tensor %d element %d: %v, want %v", i, j, got[i][j], w)
			}
		}
	}
	return nil
}

// TestDataParallelMatchesOneDevice trains the same model for a few steps
// on one device over whole batches and on two and four devices that split
// them, the batch of 10 unevenly over four, stepping the optimizer itself
// or through Parallel.Step, and checks that the losses and parameters
// agree, and that the replicas hold the model's parameters bit for bit.
func TestDataParallelMatchesOneDevice(t *testing.T) {
	ctx := t.Context()
	models := map[string]struct {
		build func() nn.Module
		input []int
	}{
		"mlp": {func() nn.Module {
			return nn.NewSequential(nn.Flatten(), nn.NewLinear(12, 16), nn.ReLU(), nn.NewLinear(16, 3))
		}, []int{3, 4}},
		"cnn": {func() nn.Module {
			return nn.NewSequential(
				nn.NewConv2d(1, 4, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
				nn.NewConv2d(4, 6, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
				nn.Flatten(), nn.NewLinear(6*2*2, 3))
		}, []int{1, 8, 8}},
	}
	opts := map[string]func([]*tensor.Param) *optim.Optimizer{
		"sgd":  func(ps []*tensor.Param) *optim.Optimizer { return optim.SGD(ps, optim.LR(0.1), optim.Momentum(0.9)) },
		"adam": func(ps []*tensor.Param) *optim.Optimizer { return optim.Adam(ps, optim.LR(0.01)) },
	}
	const batch, steps = 10, 3
	for mname, mc := range models {
		for oname, newOpt := range opts {
			for _, tc := range []struct {
				n     int
				local bool // step through Parallel.Step
			}{{2, false}, {4, false}, {2, true}, {4, true}} {
				n := tc.n
				name := fmt.Sprintf("%s/%s/%d", mname, oname, n)
				if tc.local {
					name += "/local"
				}
				t.Run(name, func(t *testing.T) {
					shape := append([]int{batch}, mc.input...)
					xs := make([]*tensor.Tensor, steps)
					ys := make([]*tensor.Tensor, steps)
					for s := range steps {
						xs[s] = tensor.Randn(uint64(10+s), shape...)
						labels := make([]int32, batch)
						for i := range labels {
							labels[i] = int32((i*7 + s) % 3)
						}
						ys[s] = tensor.FromSlice(labels, batch)
					}

					nn.ManualSeed(3)
					ref := mc.build()
					refOpt := newOpt(nn.Parameters(ref))
					var want []float64
					for s := range steps {
						l := loss.CrossEntropy(ref.Forward(xs[s]), ys[s])
						if err := l.Backward(ctx); err != nil {
							t.Fatal(err)
						}
						v, err := l.Item[float32](ctx)
						if err != nil {
							t.Fatal(err)
						}
						want = append(want, float64(v))
						if err := refOpt.Step(ctx); err != nil {
							t.Fatal(err)
						}
						refOpt.ZeroGrad()
					}

					devs, tol := parallelDevices(t, n)
					nn.ManualSeed(3)
					m := mc.build()
					if err := nn.To(m, devs[0]); err != nil {
						t.Fatal(err)
					}
					opt := newOpt(nn.Parameters(m))
					dp, err := train.NewParallel(m, devs...)
					if err != nil {
						t.Fatal(err)
					}
					defer dp.Close()
					for s := range steps {
						v, err := dp.Backward(ctx, loss.CrossEntropy, xs[s], ys[s])
						if err != nil {
							t.Fatal(err)
						}
						if math.Abs(v-want[s]) > tol*max(1, math.Abs(want[s])) {
							t.Errorf("step %d: loss %v, want %v", s, v, want[s])
						}
						// The first Parallel.Step gives the replicas the
						// optimizer's state, and the later ones step them.
						if got, want := train.AllReduced(dp), tc.local && s > 0; got != want {
							t.Errorf("step %d: gradients reduced into every replica: %v, want %v", s, got, want)
						}
						step := opt.Step
						if tc.local {
							step = func(ctx context.Context) error { return dp.Step(ctx, opt) }
						}
						if err := step(ctx); err != nil {
							t.Fatal(err)
						}
						opt.ZeroGrad()
					}
					if err := closeValues(paramValues(t, m), paramValues(t, ref), tol); err != nil {
						t.Error(err)
					}
					if tc.local {
						// The replicas stepped themselves to the model's
						// values.
						sameAsModel(t, m, dp)
						return
					}
					// The next Backward hands the replicas the last step's
					// values first.
					if _, err := dp.Backward(ctx, loss.CrossEntropy, xs[0], ys[0]); err != nil {
						t.Fatal(err)
					}
					opt.ZeroGrad()
					sameAsModel(t, m, dp)
				})
			}
		}
	}
}

// sameAsModel checks that dp's replicas hold the values of m's parameters
// bit for bit, and no gradients.
func sameAsModel(t *testing.T, m nn.Module, dp *train.Parallel) {
	t.Helper()
	values := paramValues(t, m)
	for r, rep := range dp.Replicas()[1:] {
		got := paramValues(t, rep)
		for i := range values {
			if !slices.Equal(got[i], values[i]) {
				t.Errorf("replica %d differs from the model in parameter %d", r+1, i)
			}
		}
		for _, p := range nn.Parameters(rep) {
			if p.Grad() != nil {
				t.Errorf("replica %d kept a gradient", r+1)
			}
		}
	}
}

func TestFitDataParallel(t *testing.T) {
	fit := func(opts ...train.Option) ([][]float32, []float64) {
		m := model()
		opt := optim.Adam(nn.Parameters(m), optim.LR(0.05))
		var losses []float64
		err := train.Fit(t.Context(), m, opt, blobs(t, 90, 1, data.BatchSize(32), data.Shuffle(3)),
			append([]train.Option{
				train.Loss(loss.CrossEntropy),
				train.Epochs(2),
				train.OnBatch(func(b train.Batch) error { losses = append(losses, b.Loss); return nil }),
			}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		return paramValues(t, m), losses
	}
	want, wantLosses := fit()
	two, _ := parallelDevices(t, 2)
	for _, opts := range [][]train.Option{
		{train.DataParallel(tensor.CPUs(3)...)},
		{train.DataParallel(two...), train.LossScale(1024)},
	} {
		got, losses := fit(opts...)
		if err := closeValues(got, want, 1e-4); err != nil {
			t.Error(err)
		}
		if len(losses) != len(wantLosses) {
			t.Fatalf("%d batches, want %d", len(losses), len(wantLosses))
		}
		for i, l := range losses {
			if math.Abs(l-wantLosses[i]) > 1e-4 {
				t.Errorf("batch %d: loss %v, want %v", i, l, wantLosses[i])
			}
		}
	}
	err := train.Fit(t.Context(), model(), optim.SGD(nil), blobs(t, 10, 1),
		train.Loss(loss.CrossEntropy), train.DataParallel(tensor.CPUs(2)...), train.MicroBatch(4))
	if !errors.Is(err, train.ErrConfig) {
		t.Errorf("DataParallel with MicroBatch: %v, want ErrConfig", err)
	}
}

func TestParallelErrors(t *testing.T) {
	m := model()
	if _, err := train.NewParallel(m, tensor.CPUs(2)...); !errors.Is(err, tensor.ErrDevice) {
		t.Errorf("model on another device: %v, want ErrDevice", err)
	}
	devs := tensor.CPUs(2)
	if err := nn.To(m, devs[0]); err != nil {
		t.Fatal(err)
	}
	dp, err := train.NewParallel(m, devs...)
	if err != nil {
		t.Fatal(err)
	}
	defer dp.Close()
	x := tensor.Ones(tensor.Float32, 4, 2)
	if _, err := dp.Backward(t.Context(), loss.CrossEntropy, x, tensor.FromSlice([]int32{0, 1, 2}, 3)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("3 targets for 4 examples: %v, want ErrShape", err)
	}
	// One example leaves the second device idle.
	v, err := dp.Backward(t.Context(), loss.CrossEntropy, x.Reshape(4, 2).Shrink([][2]int{{0, 1}, {0, 2}}), tensor.FromSlice([]int32{1}, 1))
	if err != nil || math.IsNaN(v) {
		t.Errorf("batch of one: %v, %v", v, err)
	}
	for _, p := range nn.Parameters(m) {
		if p.Grad() == nil {
			t.Fatal("no gradient from a batch of one")
		}
	}
}

// TestParallelStepsAfterOtherSteps steps the optimizer through
// Parallel.Step, then itself, then through Parallel.Step again, and
// checks that training matches one device's and that the replicas step
// themselves again once Parallel.Step has given them the optimizer's new
// state.
func TestParallelStepsAfterOtherSteps(t *testing.T) {
	ctx := t.Context()
	build := func() nn.Module {
		return nn.NewSequential(nn.NewLinear(5, 8), nn.ReLU(), nn.NewLinear(8, 2))
	}
	x, y := tensor.Randn(1, 6, 5), tensor.Randn(2, 6, 2)
	nn.ManualSeed(4)
	ref := build()
	refOpt := optim.Adam(nn.Parameters(ref), optim.LR(0.01))
	devs, tol := parallelDevices(t, 3)
	nn.ManualSeed(4)
	m := build()
	if err := nn.To(m, devs[0]); err != nil {
		t.Fatal(err)
	}
	opt := optim.Adam(nn.Parameters(m), optim.LR(0.01))
	dp, err := train.NewParallel(m, devs...)
	if err != nil {
		t.Fatal(err)
	}
	defer dp.Close()
	for s, local := range []bool{true, true, false, true, true, true} {
		l := loss.MSE(ref.Forward(x), y)
		if err := l.Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if err := refOpt.Step(ctx); err != nil {
			t.Fatal(err)
		}
		refOpt.ZeroGrad()
		if _, err := dp.Backward(ctx, loss.MSE, x, y); err != nil {
			t.Fatal(err)
		}
		// The replicas step themselves once a Parallel.Step has given
		// them the optimizer's state, until it steps by itself.
		if got, want := train.AllReduced(dp), s != 0 && s != 3; got != want {
			t.Errorf("step %d: gradients reduced into every replica: %v, want %v", s, got, want)
		}
		step := opt.Step
		if local {
			step = func(ctx context.Context) error { return dp.Step(ctx, opt) }
		}
		if err := step(ctx); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
	}
	if err := closeValues(paramValues(t, m), paramValues(t, ref), tol); err != nil {
		t.Error(err)
	}
	sameAsModel(t, m, dp)
}
