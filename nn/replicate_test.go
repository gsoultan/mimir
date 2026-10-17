package nn_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

func TestReplicate(t *testing.T) {
	dev := tensor.CPUs(1)[0]
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
			r, err := nn.Replicate(m, dev)
			if err != nil {
				t.Fatal(err)
			}
			var names, rnames []string
			var ps, rps []*tensor.Param
			for n, p := range nn.NamedParams(m) {
				names, ps = append(names, n), append(ps, p)
			}
			for n, p := range nn.NamedParams(r) {
				rnames, rps = append(rnames, n), append(rps, p)
			}
			if !slices.Equal(names, rnames) {
				t.Fatalf("replica's parameters %v, want %v", rnames, names)
			}
			for i, p := range rps {
				if p == ps[i] || p.Trainable() != ps[i].Trainable() {
					t.Errorf("%s: shared or of another kind", names[i])
				}
				if !p.Value().Device().Equal(dev) {
					t.Errorf("%s on %v, want %v", names[i], p.Value().Device(), dev)
				}
				if !slices.Equal(data(t, p.Value()), data(t, ps[i].Value())) {
					t.Errorf("%s: values differ", names[i])
				}
			}
			x := tensor.Randn(5, c.input...)
			want := data(t, m.Forward(x))
			got := data(t, r.Forward(x.To(dev)))
			if tensor.DefaultDevice().Equal(tensor.CPU()) {
				if !slices.Equal(got, want) {
					t.Errorf("replica computes %v, want %v", got[:4], want[:4])
				}
			} else {
				// The original runs on a GPU, whose kernels round otherwise.
				near(t, "replica's output", got, want, 1e-5)
			}
			// Training one leaves the other alone.
			if err := r.Forward(x.To(dev)).Sum().Backward(t.Context()); err != nil {
				t.Fatal(err)
			}
			for i, p := range ps {
				if p.Grad() != nil {
					t.Errorf("%s got the replica's gradient", names[i])
				}
			}
		})
	}
	if _, err := nn.Replicate(nil, dev); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("nil module: %v", err)
	}
}
