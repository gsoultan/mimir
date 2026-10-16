package nn_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/tensor"
)

// newBlock returns a model whose layers hold every kind of state a
// forward pass changes: batch normalization's running statistics and
// dropout's mask counter, around weights. Built after the same
// ManualSeed, two of them have the same weights and draw the same masks.
func newBlock() nn.Module {
	nn.ManualSeed(7)
	return nn.NewSequential(
		nn.NewLinear(6, 8), nn.NewBatchNorm1d(8), nn.ReLU(), nn.NewDropout(0.5), nn.NewLinear(8, 3),
	)
}

// namedParams returns the names and parameters of m, in order.
func namedParams(m nn.Module) (names []string, ps []*tensor.Param) {
	for name, p := range nn.NamedParams(m) {
		names, ps = append(names, name), append(ps, p)
	}
	return names, ps
}

func TestCheckpointMatchesPlainModule(t *testing.T) {
	ctx := t.Context()
	plain := newBlock()
	ref := nn.NewSequential(plain, nn.NewLinear(3, 3, nn.Seed(1)))
	model := nn.NewSequential(nn.Checkpoint(newBlock()), nn.NewLinear(3, 3, nn.Seed(1)))
	x := tensor.Randn(3, 16, 6)
	y := tensor.FromSlice(slices.Repeat([]int32{0, 1, 2, 1}, 4))

	wantOut, gotOut := ref.Forward(x), model.Forward(x)
	if err := loss.CrossEntropy(wantOut, y).Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if err := loss.CrossEntropy(gotOut, y).Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if got, want := data(t, gotOut), data(t, wantOut); !slices.Equal(got, want) {
		t.Errorf("output %v\nwant %v", got, want)
	}
	wantNames, want := namedParams(ref)
	gotNames, got := namedParams(model)
	if !slices.Equal(gotNames, []string{
		"Layers.0.Inner.Layers.0.Weight", "Layers.0.Inner.Layers.0.Bias",
		"Layers.0.Inner.Layers.1.Weight", "Layers.0.Inner.Layers.1.Bias",
		"Layers.0.Inner.Layers.1.RunningMean", "Layers.0.Inner.Layers.1.RunningVar",
		"Layers.0.Inner.Layers.4.Weight", "Layers.0.Inner.Layers.4.Bias",
		"Layers.1.Weight", "Layers.1.Bias",
	}) {
		t.Errorf("NamedParams = %v", gotNames)
	}
	// The same ops run on the same values in the same order, so the
	// gradients, and the running statistics the forward pass updated, are
	// equal bit for bit. The dropout mask the recompute draws is the one
	// the forward pass drew.
	for i, p := range want {
		if g := p.Grad(); g != nil {
			if got[i].Grad() == nil {
				t.Errorf("%s has no gradient", gotNames[i])
			} else if gv, wv := data(t, got[i].Grad()), data(t, g); !slices.Equal(gv, wv) {
				t.Errorf("%s gradient %v\nwant %v (%s)", gotNames[i], gv, wv, wantNames[i])
			}
		} else if got[i].Grad() != nil {
			t.Errorf("%s has a gradient, %s none", gotNames[i], wantNames[i])
		}
		if gv, wv := data(t, got[i].Value()), data(t, p.Value()); !slices.Equal(gv, wv) {
			t.Errorf("%s value %v\nwant %v", gotNames[i], gv, wv)
		}
	}
	bn := plain.(*nn.Sequential).Layers[1].(*nn.BatchNorm)
	if slices.Equal(data(t, bn.RunningMean.Value()), make([]float32, 8)) {
		t.Error("running mean was not updated")
	}
}

func TestCheckpointUpdatesRunningStatisticsOnce(t *testing.T) {
	bn := nn.NewBatchNorm1d(2)
	model := nn.Checkpoint(bn)
	x := tensor.FromSlice([]float32{1, 10, 3, 30}, 2, 2) // channel means 2 and 20
	for range 2 {
		if err := model.Forward(x).Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	// Two steps of momentum 0.1 toward the mean: 0.19 * mean, as
	// TestBatchNormRunningStatsUpdateOnBackward finds without Checkpoint.
	got := data(t, bn.RunningMean.Value())
	if abs(got[0]-0.38) > 1e-6 || abs(got[1]-3.8) > 1e-5 {
		t.Errorf("running mean = %v, want [0.38 3.8]", got)
	}
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func TestCheckpointFollowsTrainEvalAndTo(t *testing.T) {
	model := nn.Checkpoint(nn.NewSequential(nn.NewLinear(4, 4), nn.NewBatchNorm1d(4)))
	one := tensor.Ones(tensor.Float32, 1, 4)
	// Batch normalization of one row works only with running statistics.
	if err := model.Forward(one).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("training forward of one row: %v, want ErrShape", err)
	}
	nn.Eval(model)
	if y := model.Forward(one); y.Err() != nil || !y.Tracked() {
		t.Errorf("eval forward of one row: %v, tracked %v", y.Err(), y.Tracked())
	}
	nn.Train(model)
	if err := model.Forward(one).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("training forward of one row after Train: %v, want ErrShape", err)
	}
	if n := len(nn.Parameters(model)); n != 4 {
		t.Errorf("%d trainable parameters, want 4", n)
	}
	for _, d := range tensor.Devices() {
		if err := nn.To(model, d); err != nil {
			t.Fatal(err)
		}
		for name, p := range nn.NamedParams(model) {
			if !p.Value().Device().Equal(d) {
				t.Errorf("%s is on %v after To(%v)", name, p.Value().Device(), d)
			}
		}
	}
}

func TestCheckpointKeepsTheStateOfEveryForwardPassBeforeBackward(t *testing.T) {
	ctx := t.Context()
	x1 := tensor.FromSlice([]float32{1, 10, 3, 30}, 2, 2) // channel means 2 and 20
	x2 := tensor.FromSlice([]float32{5, 50, 7, 70}, 2, 2) // channel means 6 and 60
	plain := nn.NewBatchNorm1d(2)
	bn := nn.NewBatchNorm1d(2)
	model := nn.Checkpoint(bn)
	// Two forward passes, then one Backward, which recomputes the second
	// pass first and the first one last.
	for _, m := range []nn.Module{plain, model} {
		if err := m.Forward(x1).Sum().Add(m.Forward(x2).Sum()).Backward(ctx); err != nil {
			t.Fatal(err)
		}
	}
	// Momentum 0.1: 0.9 * (0.1 * [2 20]) + 0.1 * [6 60].
	want := data(t, plain.RunningMean.Value())
	if abs(want[0]-0.78) > 1e-6 || abs(want[1]-7.8) > 1e-5 {
		t.Fatalf("plain running mean = %v, want [0.78 7.8]", want)
	}
	if got := data(t, bn.RunningMean.Value()); !slices.Equal(got, want) {
		t.Errorf("running mean = %v, want %v", got, want)
	}
	if got, want := data(t, bn.RunningVar.Value()), data(t, plain.RunningVar.Value()); !slices.Equal(got, want) {
		t.Errorf("running variance = %v, want %v", got, want)
	}
}

func TestCheckpointDrawsTheMasksOfThePlainModuleAcrossSteps(t *testing.T) {
	ctx := t.Context()
	newDropBlock := func() nn.Module {
		nn.ManualSeed(11)
		return nn.NewSequential(nn.NewLinear(8, 8), nn.NewDropout(0.5))
	}
	plain, inner := newDropBlock(), newDropBlock()
	model := nn.Checkpoint(inner)
	x := tensor.Randn(5, 4, 8)
	// The block runs twice in one graph, so each step draws two masks,
	// and the next step must draw the two after them.
	for step := range 2 {
		wantOut, gotOut := plain.Forward(plain.Forward(x)), model.Forward(model.Forward(x))
		if err := wantOut.Sum().Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if err := gotOut.Sum().Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if got, want := data(t, gotOut), data(t, wantOut); !slices.Equal(got, want) {
			t.Errorf("step %d: output %v\nwant %v", step, got, want)
		}
		_, want := namedParams(plain)
		names, got := namedParams(inner)
		for i, p := range want {
			if gv, wv := data(t, got[i].Grad()), data(t, p.Grad()); !slices.Equal(gv, wv) {
				t.Errorf("step %d: %s gradient %v\nwant %v", step, names[i], gv, wv)
			}
			p.ZeroGrad()
			got[i].ZeroGrad()
		}
	}
}
