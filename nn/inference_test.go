package nn_test

import (
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func TestInference(t *testing.T) {
	ctx := t.Context()
	rope := nn.NewRotaryEmbedding(8, 10000)
	for name, c := range map[string]struct {
		model func() nn.Module
		input []int
	}{
		"mlp": {func() nn.Module {
			return nn.NewSequential(nn.NewLinear(6, 5), nn.ReLU(), nn.NewDropout(0.5, nn.Seed(3)), nn.NewLinear(5, 2))
		}, []int{4, 6}},
		"cnn": {func() nn.Module {
			return nn.NewSequential(nn.NewConv2d(1, 3, 3, nn.Padding(1)), nn.NewBatchNorm2d(3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.NewLinear(3*4*4, 2))
		}, []int{2, 1, 8, 8}},
		"gpt2":  {func() nn.Module { return nn.NewGPT2Block(16, 2) }, []int{2, 5, 16}},
		"llama": {func() nn.Module { return nn.NewLlamaBlock(16, 2, 1, 24, rope) }, []int{2, 5, 16}},
	} {
		t.Run(name, func(t *testing.T) {
			nn.ManualSeed(7)
			m := c.model()
			nn.Train(m)
			v := nn.Inference(m)
			x := tensor.Randn(5, c.input...)

			var names, vnames []string
			var ps []*tensor.Param
			for n, p := range nn.NamedParams(m) {
				names, ps = append(names, n), append(ps, p)
			}
			for n, p := range nn.NamedParams(v) {
				vnames = append(vnames, n)
				if p.Trainable() || p.Value().Tracked() {
					t.Errorf("%s of the view requires gradients", n)
				}
			}
			if !slices.Equal(names, vnames) {
				t.Fatalf("view's parameters %v, want %v", vnames, names)
			}
			if len(nn.Parameters(v)) != 0 {
				t.Error("the view has trainable parameters")
			}

			// The view runs in evaluation mode, whatever m's.
			y := v.Forward(x)
			if y.Tracked() {
				t.Fatal("a forward pass through the view is tracked")
			}
			got := data(t, y)
			if again := data(t, v.Forward(x)); !slices.Equal(again, got) {
				t.Error("the view's forward pass is not deterministic")
			}
			nn.Eval(m)
			ym := m.Forward(x)
			if !ym.Tracked() {
				t.Error("a forward pass through m is not tracked")
			}
			if want := data(t, ym); !slices.Equal(got, want) {
				t.Errorf("view computes %v, want %v", got[:2], want[:2])
			}

			// Training m: the view reads the new values.
			nn.Train(m)
			opt := optim.SGD(nn.Parameters(m), optim.LR(0.1))
			if err := m.Forward(x).Sum().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(ctx); err != nil {
				t.Fatal(err)
			}
			nn.Eval(m)
			want := data(t, m.Forward(x))
			if after := data(t, v.Forward(x)); !slices.Equal(after, want) {
				t.Errorf("after a step the view computes %v, want %v", after[:2], want[:2])
			}
			if slices.Equal(want, got) {
				t.Error("the step changed nothing")
			}
			for i, p := range ps {
				if p.Trainable() && p.Grad() == nil {
					t.Errorf("%s has no gradient after training", names[i])
				}
			}
		})
	}
	if v := nn.Inference[nn.Module](nil); v != nil {
		t.Errorf("view of nil: %v", v)
	}
}

// Goroutines serve a model through views while another trains it.
func TestInferenceConcurrentWithTraining(t *testing.T) {
	ctx := t.Context()
	nn.ManualSeed(7)
	m := nn.NewGPT2Block(16, 2)
	opt := optim.SGD(nn.Parameters(m), optim.LR(0.01))
	x := tensor.Randn(5, 2, 5, 16)
	shared := nn.Inference(m)
	var wg sync.WaitGroup
	for i := range 4 {
		wg.Go(func() {
			v := shared
			if i%2 == 0 {
				v = nn.Inference(m)
			}
			for range 20 {
				y := v.Forward(x)
				if y.Tracked() {
					t.Error("inference is tracked")
					return
				}
				// A step may write over the weights a forward pass read.
				if err := y.Realize(ctx); err != nil && !errors.Is(err, tensor.ErrOverwritten) {
					t.Error(err)
					return
				}
			}
		})
	}
	wg.Go(func() {
		for range 20 {
			if err := m.Forward(x).Sum().Backward(ctx); err != nil {
				t.Error(err)
				return
			}
			if err := opt.Step(ctx); err != nil {
				t.Error(err)
				return
			}
			opt.ZeroGrad()
		}
	})
	wg.Wait()
}

// A view's mode and its model's are their own.
func TestInferenceModeIsItsOwn(t *testing.T) {
	m := nn.NewSequential(nn.NewDropout(0.5, nn.Seed(1)))
	nn.Train(m)
	v := nn.Inference(m)
	x := tensor.Full(tensor.Float32, 1, 256)
	if got := data(t, v.Forward(x)); slices.Contains(got, 0) {
		t.Error("the view drops elements: it is not in evaluation mode")
	}
	if got := data(t, m.Forward(x)); !slices.Contains(got, 0) {
		t.Error("the model drops nothing: Inference took it out of training")
	}
	nn.Eval(m)
	nn.Train(v)
	if got := data(t, m.Forward(x)); slices.Contains(got, 0) {
		t.Error("training the view put the model in training")
	}
}
