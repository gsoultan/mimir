package tensor_test

import (
	"slices"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// TestShardedStepsReplay checks that steps that replay their first
// step's recorded run compute what lowering each anew does, bit for bit:
// steps whose scalars and random seeds change, read a tensor on a
// device, and train a model in place.
func TestShardedStepsReplay(t *testing.T) {
	ctx := t.Context()
	run := func(t *testing.T, replay bool, step func(m *tensor.Mesh, i int) []*tensor.Tensor) [][]float32 {
		defer tensor.SetReplays(tensor.SetReplays(replay))
		m := meshes(t)["2x2"]
		var out [][]float32
		for i := range 4 {
			for _, x := range step(m, i) {
				v, err := x.Data[float32](ctx)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, v)
			}
		}
		if got := tensor.PlanReplays(m); replay && got == 0 {
			t.Error("no step replayed the first")
		} else if !replay && got != 0 {
			t.Errorf("%d steps replayed with replays off", got)
		}
		return out
	}
	for name, step := range map[string]func(t *testing.T) func(m *tensor.Mesh, i int) []*tensor.Tensor{
		"scalars": func(t *testing.T) func(m *tensor.Mesh, i int) []*tensor.Tensor {
			b := tensor.Scalar(float32(0.5)).Add(tensor.Scalar(float32(0.25)))
			var w *tensor.Tensor
			return func(m *tensor.Mesh, i int) []*tensor.Tensor {
				if w == nil {
					w = tensor.Shard(tensor.Randn(1, 8, 8), m, tensor.P("", "model"))
					if err := b.Realize(ctx); err != nil {
						t.Fatal(err)
					}
				}
				x := tensor.Shard(tensor.FromSlice(values(2+i, 4*8), 4, 8), m, tensor.P("data"))
				// A scale and a seed that change every step, and a bias
				// computed on a device, which each step copies to the mesh.
				y := x.MatMul(w).ReLU().Mul(tensor.Scalar(float32(i + 1))).Add(tensor.Rand(uint64(i), 4, 8)).Add(b)
				return []*tensor.Tensor{y, y.Sum()}
			}
		},
		"training": func(t *testing.T) func(m *tensor.Mesh, i int) []*tensor.Tensor {
			var model nn.Module
			var opt *optim.Optimizer
			var x, y *tensor.Tensor
			return func(m *tensor.Mesh, i int) []*tensor.Tensor {
				if model == nil {
					nn.ManualSeed(1)
					model = nn.NewGPT2Block(16, 2)
					if err := nn.Shard(model, m, nn.TensorParallel(model, m, "model")); err != nil {
						t.Fatal(err)
					}
					opt = optim.AdamW(nn.Parameters(model), optim.LR(1e-2))
					x = tensor.Shard(tensor.Randn(1, 4, 8, 16), m, tensor.P("data"))
					y = tensor.Shard(tensor.Randn(2, 4, 8, 16), m, tensor.P("data"))
				}
				l := loss.MSE(model.Forward(x), y)
				if err := l.Backward(ctx); err != nil {
					t.Fatal(err)
				}
				if err := opt.Step(ctx); err != nil {
					t.Fatal(err)
				}
				opt.ZeroGrad()
				out := []*tensor.Tensor{l}
				for _, p := range nn.Parameters(model) {
					out = append(out, p.Value())
				}
				return out
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			want := run(t, false, step(t))
			got := run(t, true, step(t))
			if len(got) != len(want) {
				t.Fatalf("%d results, want %d", len(got), len(want))
			}
			for i := range want {
				if !slices.Equal(got[i], want[i]) {
					t.Fatalf("result %d differs when replayed:\n%v\nwant\n%v", i, got[i], want[i])
				}
			}
		})
	}
}
