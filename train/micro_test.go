package train_test

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/train"
)

// values returns every trainable parameter of m, flattened.
func values(t *testing.T, m nn.Module) []float32 {
	t.Helper()
	var out []float32
	for _, p := range nn.Parameters(m) {
		v, err := p.Value().Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, v...)
	}
	return out
}

// sameValues fails the test unless got equals want to within 1e-4,
// relative to the larger value: training in chunks sums the gradients in
// another order than training on whole batches, and the rounding differs
// more on a GPU, whose reductions are trees.
func sameValues(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		if math.Abs(g-w) > 1e-4*max(math.Abs(g), math.Abs(w), 1e-3) {
			t.Fatalf("%s: value %d = %v, want %v", what, i, g, w)
		}
	}
}

// fit trains a fresh model on ds for epochs with a fresh optimizer and
// the options, and returns its parameters and the batches reported.
func fit(t *testing.T, newModel func() nn.Module, newOpt func([]*tensor.Param) *optim.Optimizer, ds data.Dataset, epochs int, opts ...train.Option) ([]float32, []train.Batch) {
	t.Helper()
	m := newModel()
	opt := newOpt(nn.Parameters(m))
	var batches []train.Batch
	opts = append([]train.Option{
		train.Loss(loss.CrossEntropy),
		train.Epochs(epochs),
		train.OnBatch(func(b train.Batch) error { batches = append(batches, b); return nil }),
	}, opts...)
	if err := train.Fit(t.Context(), m, opt, ds, opts...); err != nil {
		t.Fatal(err)
	}
	return values(t, m), batches
}

// adam returns the optimizer the small model trains with.
func adam(ps []*tensor.Param) *optim.Optimizer { return optim.Adam(ps, optim.LR(0.05)) }

func TestMicroBatchMatchesFullBatchTraining(t *testing.T) {
	ds := blobs(t, 96, 1, data.BatchSize(32), data.Shuffle(3))
	full, batches := fit(t, model, adam, ds, 2)
	if len(batches) != 6 {
		t.Fatalf("%d batches, want 6", len(batches))
	}
	// Chunks that divide the batch, that leave a remainder, and that
	// are the whole batch or more.
	for _, n := range []int{16, 10, 32, 100} {
		got, chunked := fit(t, model, adam, ds, 2, train.MicroBatch(n))
		sameValues(t, "parameters after micro-batches", got, full)
		if len(chunked) != len(batches) {
			t.Fatalf("micro-batches of %d reported %d batches, want %d", n, len(chunked), len(batches))
		}
		for i, b := range chunked {
			if math.Abs(b.Loss-batches[i].Loss) > 1e-5*batches[i].Loss || b.LossScale != 0 || b.Skipped {
				t.Errorf("micro-batches of %d: batch %d = %+v, want loss %v", n, i, b, batches[i].Loss)
			}
		}
	}
	m := model()
	err := train.Fit(t.Context(), m, optim.SGD(nn.Parameters(m)), ds, train.Loss(loss.CrossEntropy), train.MicroBatch(0))
	if !errors.Is(err, train.ErrConfig) {
		t.Errorf("micro-batches of 0: %v, want ErrConfig", err)
	}
}

// wide returns a model whose activations outweigh its parameters, so a
// batch's memory is mostly the batch's.
func wide() nn.Module {
	nn.ManualSeed(11)
	return nn.NewSequential(nn.NewLinear(2, 512), nn.ReLU(), nn.NewLinear(512, 3))
}

// fits reports whether a full-batch step of m on x, y fits in budget.
func fits(t *testing.T, m nn.Module, x, y *tensor.Tensor, budget int64) bool {
	t.Helper()
	dev := tensor.DefaultDevice()
	tensor.SetMemoryBudget(dev, budget)
	defer tensor.SetMemoryBudget(dev, 0)
	err := loss.CrossEntropy(m.Forward(x), y).Backward(t.Context())
	for _, p := range nn.Parameters(m) {
		p.ZeroGrad()
	}
	switch {
	case err == nil:
		return true
	case errors.Is(err, tensor.ErrMemoryBudget):
		return false
	default:
		t.Fatal(err)
		return false
	}
}

func TestMemoryBudgetSplitsBatchesThatDoNotFit(t *testing.T) {
	if os.Getenv("MIMIR_EAGER") == "1" {
		t.Skip("eager mode computes every operation into memory of its own, which only the garbage collector reclaims")
	}
	ds := blobs(t, 256, 1, data.BatchSize(256))
	var x, y *tensor.Tensor
	for b := range ds.All(t.Context()) {
		x, y = b.X, b.Y
	}
	// The smallest budget, to within a kilobyte, that a full-batch step
	// fits in.
	probe := wide()
	lo, hi := int64(0), int64(1<<30)
	if !fits(t, probe, x, y, hi) {
		t.Fatal("a full-batch step does not fit in a gigabyte")
	}
	for hi-lo > 1024 {
		if mid := (lo + hi) / 2; fits(t, probe, x, y, mid) {
			hi = mid
		} else {
			lo = mid
		}
	}
	budget := hi * 3 / 4
	m := wide()
	if fits(t, m, x, y, budget) {
		t.Fatalf("a full-batch step fits in %d bytes, three quarters of the %d it needs", budget, hi)
	}

	// Momentum SGD: Adam would amplify the rounding of a gradient that
	// nearly cancels over the batch, which the chunks sum differently.
	sgd := func(ps []*tensor.Param) *optim.Optimizer { return optim.SGD(ps, optim.LR(0.1), optim.Momentum(0.9)) }
	dev := tensor.DefaultDevice()
	full, _ := fit(t, wide, sgd, ds, 3)
	split, batches := fit(t, wide, sgd, ds, 3, train.MemoryBudget(budget))
	// The full batch does not fit, so the steps ran in chunks.
	sameValues(t, "parameters after training within the budget", split, full)
	if len(batches) != 3 {
		t.Errorf("%d batches, want 3", len(batches))
	}
	if got := tensor.MemoryBudget(dev); got != 0 {
		t.Errorf("budget after Fit = %d, want the previous 0", got)
	}

	// One example that does not fit fails training.
	err := train.Fit(t.Context(), wide(), optim.SGD(nil), ds, train.Loss(loss.CrossEntropy), train.MemoryBudget(1))
	if !errors.Is(err, tensor.ErrMemoryBudget) {
		t.Errorf("budget of one byte: %v, want ErrMemoryBudget", err)
	}
	if got := tensor.MemoryBudget(dev); got != 0 {
		t.Errorf("budget after a failed Fit = %d, want the previous 0", got)
	}
	err = train.Fit(t.Context(), wide(), optim.SGD(nil), ds, train.Loss(loss.CrossEntropy), train.MemoryBudget(0))
	if !errors.Is(err, train.ErrConfig) {
		t.Errorf("budget of 0: %v, want ErrConfig", err)
	}
}

func TestLossScaleMatchesUnscaledTraining(t *testing.T) {
	ds := blobs(t, 96, 1, data.BatchSize(32), data.Shuffle(3))
	plain, batches := fit(t, model, adam, ds, 2)
	scaled, reported := fit(t, model, adam, ds, 2, train.LossScale(1024))
	sameValues(t, "parameters after scaled losses", scaled, plain)
	for i, b := range reported {
		if b.LossScale != 1024 || b.Skipped || math.Abs(b.Loss-batches[i].Loss) > 1e-5*batches[i].Loss {
			t.Errorf("batch %d = %+v, want loss %v at scale 1024", i, b, batches[i].Loss)
		}
	}
	m := model()
	for _, scale := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		err := train.Fit(t.Context(), m, optim.SGD(nn.Parameters(m)), ds, train.Loss(loss.CrossEntropy), train.LossScale(scale))
		if !errors.Is(err, train.ErrConfig) {
			t.Errorf("loss scale %v: %v, want ErrConfig", scale, err)
		}
	}
}

func TestDynamicLossScaleSkipsStepsWithNonFiniteGradients(t *testing.T) {
	// Two batches of four: the first holds an infinite input, which makes
	// every gradient NaN.
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8}
	x[0] = float32(math.Inf(1))
	ds, err := data.FromTensors(tensor.FromSlice(x, 8, 2), tensor.FromSlice([]int32{0, 1, 2, 0, 1, 2, 0, 1}), data.BatchSize(4))
	if err != nil {
		t.Fatal(err)
	}
	m := model()
	before := values(t, m)
	var batches []train.Batch
	var epochs []train.Epoch
	err = train.Fit(t.Context(), m, optim.SGD(nn.Parameters(m), optim.LR(0.1)), ds,
		train.Loss(loss.CrossEntropy),
		train.DynamicLossScale(),
		train.OnBatch(func(b train.Batch) error {
			if b.Index == 0 {
				sameValues(t, "parameters after a skipped step", values(t, m), before)
			}
			batches = append(batches, b)
			return nil
		}),
		train.OnEpoch(func(e train.Epoch) error { epochs = append(epochs, e); return nil }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) != 2 {
		t.Fatalf("%d batches, want 2", len(batches))
	}
	if b := batches[0]; !b.Skipped || b.LossScale != 32768 {
		t.Errorf("batch with non-finite gradients = %+v, want it skipped at half the scale", b)
	}
	if b := batches[1]; b.Skipped || b.LossScale != 32768 || math.IsNaN(b.Loss) {
		t.Errorf("batch with finite gradients = %+v, want it taken at the same scale", b)
	}
	sameValues(t, "epoch loss excludes skipped steps", []float32{float32(epochs[0].Loss)}, []float32{float32(batches[1].Loss)})
	if epochs[0].Steps != 1 {
		t.Errorf("%d steps, want 1", epochs[0].Steps)
	}
	after := values(t, m)
	changed := false
	for i := range after {
		changed = changed || after[i] != before[i]
	}
	if !changed {
		t.Error("the step with finite gradients changed no parameter")
	}
}

func TestDynamicLossScaleGrowsAfterFiniteSteps(t *testing.T) {
	nn.ManualSeed(5)
	m := nn.NewLinear(2, 3)
	ds := blobs(t, 2001, 1, data.BatchSize(1))
	var scales []float64
	err := train.Fit(context.Background(), m, optim.SGD(nn.Parameters(m), optim.LR(0.01)), ds,
		train.Loss(loss.CrossEntropy),
		train.DynamicLossScale(),
		train.OnBatch(func(b train.Batch) error {
			if b.Skipped {
				return errors.New("a finite step was skipped")
			}
			scales = append(scales, b.LossScale)
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if scales[1998] != 65536 || scales[1999] != 131072 || scales[2000] != 131072 {
		t.Errorf("scale around the 2000th step: %v, %v, %v; want it doubled at the 2000th", scales[1998], scales[1999], scales[2000])
	}
}

// allValues returns every parameter and buffer of m, flattened, in name
// order.
func allValues(t *testing.T, m nn.Module) []float32 {
	t.Helper()
	var out []float32
	for _, p := range nn.NamedParams(m) {
		v, err := p.Value().Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, v...)
	}
	return out
}

// normalized returns a model whose batch normalization sets its running
// statistics during the forward pass, to values computed in Backward.
func normalized() nn.Module {
	nn.ManualSeed(11)
	return nn.NewSequential(nn.NewLinear(2, 512), nn.NewBatchNorm1d(512), nn.ReLU(), nn.NewLinear(512, 3))
}

// TestMemoryBudgetRestoresBuffersOfRefusedChunks checks that a chunk the
// budget refuses leaves no trace in the model: batch normalization had
// set its running statistics to values the refused Backward would have
// computed, which a retry that kept them would compute too, and would
// then be refused as well, down to a chunk of one example.
func TestMemoryBudgetRestoresBuffersOfRefusedChunks(t *testing.T) {
	if os.Getenv("MIMIR_EAGER") == "1" {
		t.Skip("eager mode computes every operation into memory of its own, which only the garbage collector reclaims")
	}
	ds := blobs(t, 256, 1, data.BatchSize(256))
	var x, y *tensor.Tensor
	for b := range ds.All(t.Context()) {
		x, y = b.X, b.Y
	}
	probe := normalized()
	lo, hi := int64(0), int64(1<<30)
	if !fits(t, probe, x, y, hi) {
		t.Fatal("a full-batch step does not fit in a gigabyte")
	}
	for hi-lo > 1024 {
		if mid := (lo + hi) / 2; fits(t, probe, x, y, mid) {
			hi = mid
		} else {
			lo = mid
		}
	}
	budget := hi * 3 / 4
	half := normalized()
	if !fits(t, half, x.Shrink([][2]int{{0, 128}, {0, 2}}), y.Shrink([][2]int{{0, 128}}), budget) {
		t.Fatalf("half a batch does not fit in %d bytes, three quarters of what the batch needs", budget)
	}
	sgd := func(ps []*tensor.Param) *optim.Optimizer { return optim.SGD(ps, optim.LR(0.1), optim.Momentum(0.9)) }
	// The batch does not fit and is halved once, so the steps and the
	// running statistics are those of chunks of 128.
	var split, chunked nn.Module
	for i, newModel := range []func() nn.Module{normalized, normalized} {
		m := newModel()
		opts := []train.Option{train.Loss(loss.CrossEntropy), train.Epochs(3), train.MemoryBudget(budget)}
		if i == 1 {
			opts = []train.Option{train.Loss(loss.CrossEntropy), train.Epochs(3), train.MicroBatch(128)}
		}
		if err := train.Fit(t.Context(), m, sgd(nn.Parameters(m)), ds, opts...); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			split = m
		} else {
			chunked = m
		}
	}
	sameValues(t, "parameters and running statistics after training within the budget", allValues(t, split), allValues(t, chunked))

	// A chunk of one example that does not fit fails with the budget
	// error, whatever the layer that fails on one example says.
	m := normalized()
	err := train.Fit(t.Context(), m, sgd(nn.Parameters(m)), ds, train.Loss(loss.CrossEntropy), train.MemoryBudget(1))
	if !errors.Is(err, tensor.ErrMemoryBudget) {
		t.Errorf("budget of one byte: %v, want ErrMemoryBudget", err)
	}
}

// TestDynamicLossScaleStopsAtOne checks that a batch whose gradients are
// never finite halves the scale down to 1 and no further: below 1 the
// scale protects nothing, and its reciprocal would overflow float32 and
// turn the next finite step's parameters into NaN.
func TestDynamicLossScaleStopsAtOne(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8}
	x[0] = float32(math.NaN())
	ds, err := data.FromTensors(tensor.FromSlice(x, 8, 2), tensor.FromSlice([]int32{0, 1, 2, 0, 1, 2, 0, 1}), data.BatchSize(4))
	if err != nil {
		t.Fatal(err)
	}
	m := model()
	least, taken := math.Inf(1), 0
	err = train.Fit(t.Context(), m, optim.SGD(nn.Parameters(m), optim.LR(0.1)), ds,
		train.Loss(loss.CrossEntropy),
		train.DynamicLossScale(),
		train.Epochs(24), // 65536 halves to 1 in 16
		train.OnBatch(func(b train.Batch) error {
			least = min(least, b.LossScale)
			if b.Index == 1 && !b.Skipped {
				taken++
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if least != 1 || taken != 24 {
		t.Errorf("least scale %v, finite batch taken %d times; want 1 and 24", least, taken)
	}
	for i, v := range values(t, m) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("parameter value %d is %v", i, v)
		}
	}
}

// TestFailedStepLeavesNoGradient checks that a step that fails after
// some of its chunks accumulated gradients clears them, so a later Fit
// with the same optimizer starts clean.
func TestFailedStepLeavesNoGradient(t *testing.T) {
	ds := blobs(t, 8, 1, data.BatchSize(8))
	m := model()
	boom := errors.New("boom")
	calls := 0
	failing := func(pred, target *tensor.Tensor) *tensor.Tensor {
		if calls++; calls == 2 {
			return tensor.Failed(boom)
		}
		return loss.CrossEntropy(pred, target)
	}
	err := train.Fit(t.Context(), m, optim.SGD(nn.Parameters(m)), ds, train.Loss(failing), train.MicroBatch(4))
	if !errors.Is(err, boom) {
		t.Fatalf("Fit: %v, want the loss's error", err)
	}
	for name, p := range nn.NamedParams(m) {
		if p.Grad() != nil {
			t.Errorf("%s keeps a gradient after the failed step", name)
		}
	}
}
