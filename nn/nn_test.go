package nn_test

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func set(t *testing.T, p *tensor.Param, v *tensor.Tensor) {
	t.Helper()
	if err := p.Set(v); err != nil {
		t.Fatal(err)
	}
}

func data(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, err := x.Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// nnCase builds a layer or loss from golden inputs and returns its output,
// where each input's gradient accumulates, and any extra results to check.
type nnCase func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, map[int]goldentest.Grader, map[string]*tensor.Tensor)

func lossCase(f loss.Func) nnCase {
	return func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, map[int]goldentest.Grader, map[string]*tensor.Tensor) {
		return f(in[0], in[1]), map[int]goldentest.Grader{0: in[0], 1: in[1]}, nil
	}
}

func batchNormCase(make func(c int, opts ...nn.Option) *nn.BatchNorm) nnCase {
	return func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, map[int]goldentest.Grader, map[string]*tensor.Tensor) {
		b := make(in[0].Shape()[1])
		set(t, b.Weight, in[1])
		set(t, b.Bias, in[2])
		out := b.Forward(in[0])
		// Running statistics are lazy until Backward computes them.
		return out, map[int]goldentest.Grader{0: in[0], 1: b.Weight, 2: b.Bias}, map[string]*tensor.Tensor{
			"running_mean": b.RunningMean.Value(),
			"running_var":  b.RunningVar.Value(),
		}
	}
}

var nnCases = map[string]nnCase{
	"linear": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, map[int]goldentest.Grader, map[string]*tensor.Tensor) {
		l := nn.NewLinear(6, 3)
		set(t, l.Weight, in[1])
		set(t, l.Bias, in[2])
		return l.Forward(in[0]), map[int]goldentest.Grader{0: in[0], 1: l.Weight, 2: l.Bias}, nil
	},
	"layer_norm": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, map[int]goldentest.Grader, map[string]*tensor.Tensor) {
		l := nn.NewLayerNorm(8)
		set(t, l.Weight, in[1])
		set(t, l.Bias, in[2])
		return l.Forward(in[0]), map[int]goldentest.Grader{0: in[0], 1: l.Weight, 2: l.Bias}, nil
	},
	"batch_norm2d": batchNormCase(nn.NewBatchNorm2d),
	"batch_norm1d": batchNormCase(nn.NewBatchNorm1d),
	"batch_norm_eval": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, map[int]goldentest.Grader, map[string]*tensor.Tensor) {
		b := nn.NewBatchNorm2d(3)
		set(t, b.RunningMean, in[1])
		set(t, b.RunningVar, in[2])
		set(t, b.Weight, in[3])
		set(t, b.Bias, in[4])
		nn.Eval(b)
		return b.Forward(in[0]), map[int]goldentest.Grader{0: in[0], 3: b.Weight, 4: b.Bias}, nil
	},
	"embedding": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, map[int]goldentest.Grader, map[string]*tensor.Tensor) {
		e := nn.NewEmbedding(5, 4)
		set(t, e.Weight, in[1])
		return e.Forward(in[0]), map[int]goldentest.Grader{1: e.Weight}, nil
	},
	"cross_entropy":      lossCase(loss.CrossEntropy),
	"cross_entropy_kdim": lossCase(loss.CrossEntropy),
	"nll":                lossCase(loss.NLL),
	"bce_with_logits":    lossCase(loss.BCEWithLogits),
	"mse":                lossCase(loss.MSE),
	"l1":                 lossCase(loss.L1),
	"huber":              lossCase(loss.Huber),
}

// TestGolden checks layers and losses, with their gradients, against
// PyTorch or NumPy (see tools/golden/gen.py).
func TestGolden(t *testing.T) {
	cases, meta := goldentest.Load(t, "nn")
	t.Logf("golden oracle: %s", meta["oracle"])
	for _, name := range slices.Sorted(maps.Keys(nnCases)) {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	tol := goldentest.Tol{Atol: 1e-5, Rtol: 1e-4}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			build, ok := nnCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := goldentest.Inputs(t, c)
			out, grads, extra := build(t, in)
			goldentest.CheckGradients(t, c, out, grads, tol)
			for part, want := range c.Extra {
				got, ok := extra[part]
				if !ok {
					t.Fatalf("no Go result for golden %q", part)
				}
				t.Run(part, func(t *testing.T) { goldentest.CompareTensor(t, want, got, tol) })
			}
		})
	}
}

// block is a custom module with parameters in unexported fields, a slice,
// a map and a shared layer.
type block struct {
	proj    *nn.Linear
	heads   []nn.Module
	norms   map[string]*nn.LayerNorm
	tied    *nn.Linear // the same layer as proj
	skipped *nn.Linear `nn:"-"`
	act     nn.Func
}

func (b *block) Forward(x *tensor.Tensor) *tensor.Tensor { return b.act(b.proj.Forward(x)) }

func TestParametersFindsEveryParam(t *testing.T) {
	proj := nn.NewLinear(4, 4)
	b := &block{
		proj:    proj,
		heads:   []nn.Module{nn.NewLinear(4, 2, nn.NoBias()), nn.NewDropout(0.1), nn.NewBatchNorm1d(2)},
		norms:   map[string]*nn.LayerNorm{"b": nn.NewLayerNorm(4), "a": nn.NewLayerNorm(4, nn.NoBias())},
		tied:    proj,
		skipped: nn.NewLinear(4, 4),
		act:     nn.ReLU(),
	}
	var names []string
	for name := range nn.NamedParams(b) {
		names = append(names, name)
	}
	want := []string{
		"proj.Weight", "proj.Bias", "heads.0.Weight",
		"heads.2.Weight", "heads.2.Bias", "heads.2.RunningMean", "heads.2.RunningVar",
		"norms.b.Weight", "norms.b.Bias",
	}
	if !slices.Equal(names, want) {
		t.Errorf("NamedParams = %v\nwant %v", names, want)
	}
	if got := len(nn.Parameters(b)); got != 7 {
		t.Errorf("%d trainable parameters, want 7 (buffers excluded)", got)
	}
}

func TestEvalSwitchesDropoutAndBatchNorm(t *testing.T) {
	d := nn.NewDropout(0.5)
	model := nn.NewSequential(nn.NewLinear(8, 8), d, nn.NewBatchNorm1d(8))
	x := tensor.Ones(tensor.Float32, 4, 8)
	dropped := data(t, d.Forward(tensor.Ones(tensor.Float32, 1000)))
	zeros := 0
	for _, v := range dropped {
		switch v {
		case 0:
			zeros++
		case 2:
		default:
			t.Fatalf("dropout produced %v, want 0 or 1/(1-p)", v)
		}
	}
	if zeros < 400 || zeros > 600 {
		t.Errorf("dropout zeroed %d of 1000, want about 500", zeros)
	}
	nn.Eval(model)
	if got := data(t, d.Forward(x)); slices.Contains(got, 0) {
		t.Error("dropout dropped in eval mode")
	}
	if err := model.Forward(tensor.Ones(tensor.Float32, 1, 8)).Err(); err != nil {
		t.Errorf("eval batch norm of one row: %v", err)
	}
	nn.Train(model)
	if err := model.Forward(tensor.Ones(tensor.Float32, 1, 8)).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("training batch norm of one row: %v, want ErrShape", err)
	}
}

func TestBatchNormRunningStatsUpdateOnBackward(t *testing.T) {
	b := nn.NewBatchNorm1d(2)
	x := tensor.FromSlice([]float32{1, 10, 3, 30}, 2, 2) // channel means 2 and 20
	for range 2 {
		if err := b.Forward(x).Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	// Two steps of momentum 0.1 toward the mean: 0.19 * mean.
	got := data(t, b.RunningMean.Value())
	if math.Abs(float64(got[0])-0.38) > 1e-6 || math.Abs(float64(got[1])-3.8) > 1e-5 {
		t.Errorf("running mean = %v, want [0.38 3.8]", got)
	}
}

func TestBatchNormRunningStatsSurviveAStepBeforeBackward(t *testing.T) {
	lin := nn.NewLinear(2, 2)
	set(t, lin.Weight, tensor.FromSlice([]float32{1, 0, 0, 1}, 2, 2))
	set(t, lin.Bias, tensor.Zeros(tensor.Float32, 2))
	b := nn.NewBatchNorm1d(2)
	x := tensor.FromSlice([]float32{1, 10, 3, 30}, 2, 2) // channel means 2 and 20
	// A training pass that no Backward reaches leaves the statistics to be
	// computed from the weights before the step below writes over them.
	if err := b.Forward(lin.Forward(x)).Err(); err != nil {
		t.Fatal(err)
	}
	opt := optim.SGD(nn.Parameters(lin), optim.LR(0.5))
	if err := lin.Forward(x).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := opt.Step(t.Context()); err != nil {
		t.Fatal(err)
	}
	got := data(t, b.RunningMean.Value())
	if math.Abs(float64(got[0])-0.2) > 1e-6 || math.Abs(float64(got[1])-2) > 1e-5 {
		t.Errorf("running mean = %v, want [0.2 2]", got)
	}
	if err := tensor.Realize(t.Context(), b.RunningMean.Value(), b.RunningVar.Value()); err != nil {
		t.Error(err)
	}
}

func TestBatchNormUpdatesAlongsideOtherSchedules(t *testing.T) {
	// One goroutine updates running statistics while another realizes
	// unrelated values. The race detector checks that the statistics
	// waiting to be computed are looked at only under the scheduler's
	// lock.
	b := nn.NewBatchNorm1d(8)
	x := tensor.Randn(1, 16, 8)
	y := tensor.Ones(tensor.Float32, 64)
	var wg sync.WaitGroup
	wg.Go(func() {
		for range 200 {
			if err := b.Forward(x).Err(); err != nil {
				t.Error(err)
				return
			}
			if err := tensor.Realize(t.Context(), b.RunningMean.Value(), b.RunningVar.Value()); err != nil {
				t.Error(err)
				return
			}
		}
	})
	wg.Go(func() {
		for range 200 {
			if err := tensor.RealizeInto(t.Context(), []*tensor.Tensor{nil}, []*tensor.Tensor{y.Neg()}); err != nil {
				t.Error(err)
				return
			}
		}
	})
	wg.Wait()
}

func TestManualSeedReproducesWeights(t *testing.T) {
	build := func() []float32 {
		nn.ManualSeed(42)
		return data(t, nn.NewLinear(3, 2).Weight.Value())
	}
	first, second := build(), build()
	if !slices.Equal(first, second) {
		t.Error("ManualSeed did not reproduce weights")
	}
	for _, w := range first {
		if math.Abs(float64(w)) > 1/math.Sqrt(3) {
			t.Errorf("weight %v outside ±1/√in", w)
		}
	}
}

func TestLayerShapes(t *testing.T) {
	x := tensor.Randn(1, 2, 3, 8, 8)
	tests := map[string]struct {
		got  *tensor.Tensor
		want []int
	}{
		"conv":      {nn.NewConv2d(3, 4, 3, nn.Padding(1), nn.Stride(2)).Forward(x), []int{2, 4, 4, 4}},
		"grouped":   {nn.NewConv2d(3, 6, 3, nn.Groups(3), nn.NoBias()).Forward(x), []int{2, 6, 6, 6}},
		"max pool":  {nn.MaxPool2d(2).Forward(x), []int{2, 3, 4, 4}},
		"avg pool":  {nn.AvgPool2d(3, nn.Stride(1), nn.Padding(1)).Forward(x), []int{2, 3, 8, 8}},
		"flatten":   {nn.Flatten().Forward(x), []int{2, 192}},
		"embedding": {nn.NewEmbedding(10, 5).Forward(tensor.FromSlice([]int32{1, 2, 9, 0, 0, 3}, 2, 3)), []int{2, 3, 5}},
		"softmax":   {nn.Softmax(-1).Forward(x), []int{2, 3, 8, 8}},
	}
	for name, tt := range tests {
		if err := tt.got.Err(); err != nil {
			t.Errorf("%s: %v", name, err)
		} else if !slices.Equal(tt.got.Shape(), tt.want) {
			t.Errorf("%s: shape %v, want %v", name, tt.got.Shape(), tt.want)
		}
	}
}

func TestLossShapeErrors(t *testing.T) {
	pred := tensor.Zeros(tensor.Float32, 4, 1)
	for name, err := range map[string]error{
		"mse broadcast":     loss.MSE(pred, tensor.Zeros(tensor.Float32, 4)).Err(),
		"float targets":     loss.CrossEntropy(tensor.Zeros(tensor.Float32, 4, 3), tensor.Zeros(tensor.Float32, 4)).Err(),
		"target shape":      loss.CrossEntropy(tensor.Zeros(tensor.Float32, 4, 3), tensor.Zeros(tensor.Int32, 3)).Err(),
		"logits rank":       loss.CrossEntropy(tensor.Zeros(tensor.Float32, 4), tensor.Zeros(tensor.Int32, 4)).Err(),
		"embedding floats":  nn.NewEmbedding(3, 2).Forward(tensor.Zeros(tensor.Float32, 2)).Err(),
		"batch norm rank":   nn.NewBatchNorm2d(3).Forward(tensor.Zeros(tensor.Float32, 2, 3)).Err(),
		"dropout above one": (&nn.Dropout{P: 2}).Forward(pred).Err(),
	} {
		if !errors.Is(err, tensor.ErrShape) && !errors.Is(err, tensor.ErrDType) {
			t.Errorf("%s: %v, want ErrShape or ErrDType", name, err)
		}
	}
}

func Example() {
	ctx := context.Background()
	nn.ManualSeed(0)
	model := nn.NewSequential(nn.NewLinear(4, 8), nn.ReLU(), nn.NewLinear(8, 3))
	x := tensor.Randn(1, 5, 4)
	y := tensor.FromSlice([]int32{0, 2, 1, 1, 0})

	l := loss.CrossEntropy(model.Forward(x), y)
	if err := l.Backward(ctx); err != nil {
		fmt.Println(err)
		return
	}
	for name, p := range nn.NamedParams(model) {
		fmt.Println(name, p.Value().Shape(), p.Grad().Shape())
	}
	// Output:
	// Layers.0.Weight [8 4] [8 4]
	// Layers.0.Bias [8] [8]
	// Layers.2.Weight [3 8] [3 8]
	// Layers.2.Bias [3] [3]
}
