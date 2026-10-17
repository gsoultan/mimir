package train_test

import (
	"os"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/train"
)

// testMesh returns a mesh of the given shape over the axes "data" and
// "model": of GPUs under MIMIR_DEVICE=gpu, skipping t if there are too
// few, and otherwise of the devices of tensor.CPUs. It also returns the
// tolerance of comparisons with one device.
func testMesh(t *testing.T, shape ...int) (*tensor.Mesh, float64) {
	t.Helper()
	n := shape[0] * shape[1]
	devs, tol := tensor.CPUs(n), 2e-5
	if os.Getenv("MIMIR_DEVICE") == "gpu" {
		gpus := tensor.GPUs()
		if len(gpus) < n {
			t.Skipf("%d GPUs, want %d", len(gpus), n)
		}
		devs, tol = gpus[:n], 2e-4
	}
	m, err := tensor.NewMesh(shape, []string{"data", "model"}, devs...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m, tol
}

// shardedModel is a model of the sharding tests, the batches it trains
// on and its loss.
type shardedModel struct {
	build func() nn.Module
	x, y  func(step int) *tensor.Tensor
	loss  loss.Func
}

func shardedModels() map[string]shardedModel {
	return map[string]shardedModel{
		"mlp": {
			build: func() nn.Module {
				return nn.NewSequential(nn.NewLinear(12, 16), nn.Tanh(), nn.NewMLP(16, 32), nn.ReLU(), nn.NewLinear(16, 4))
			},
			x: func(s int) *tensor.Tensor { return tensor.Randn(uint64(20+s), 8, 12) },
			y: func(s int) *tensor.Tensor {
				labels := make([]int32, 8)
				for i := range labels {
					labels[i] = int32((i*5 + s) % 4)
				}
				return tensor.FromSlice(labels, 8)
			},
			loss: loss.CrossEntropy,
		},
		"gpt2 block": {
			build: func() nn.Module { return nn.NewGPT2Block(16, 4) },
			x:     func(s int) *tensor.Tensor { return tensor.Randn(uint64(40+s), 4, 6, 16) },
			y:     func(s int) *tensor.Tensor { return tensor.Randn(uint64(60+s), 4, 6, 16) },
			loss:  loss.MSE,
		},
	}
}

// strategies are the ways the sharding tests lay a model out: the shape
// of the mesh and the rule that places its parameters.
var strategies = map[string]struct {
	shape []int
	rule  func(m nn.Module, mesh *tensor.Mesh) nn.ShardRule
}{
	"dp":    {[]int{4, 1}, func(nn.Module, *tensor.Mesh) nn.ShardRule { return nil }},
	"tp":    {[]int{1, 4}, func(m nn.Module, mesh *tensor.Mesh) nn.ShardRule { return nn.TensorParallel(m, mesh, "model") }},
	"fsdp":  {[]int{4, 1}, func(_ nn.Module, mesh *tensor.Mesh) nn.ShardRule { return nn.FullyShard(mesh, "data") }},
	"dp×tp": {[]int{2, 2}, func(m nn.Module, mesh *tensor.Mesh) nn.ShardRule { return nn.TensorParallel(m, mesh, "model") }},
	"fsdp×tp": {[]int{2, 2}, func(m nn.Module, mesh *tensor.Mesh) nn.ShardRule {
		return nn.Rules(nn.TensorParallel(m, mesh, "model"), nn.FullyShard(mesh, "data"))
	}},
}

// TestShardedMatchesOneDevice trains each model for a few steps with Adam
// on one device and laid out over a mesh by each strategy, and checks
// that the losses and parameters agree.
func TestShardedMatchesOneDevice(t *testing.T) {
	ctx := t.Context()
	const steps = 3
	for mname, mc := range shardedModels() {
		for sname, st := range strategies {
			t.Run(mname+"/"+sname, func(t *testing.T) {
				run := func(mesh *tensor.Mesh) ([]float64, nn.Module) {
					nn.ManualSeed(5)
					m := mc.build()
					if mesh != nil {
						if err := nn.Shard(m, mesh, st.rule(m, mesh)); err != nil {
							t.Fatal(err)
						}
					}
					// An eps well above rounding noise keeps Adam from scaling up the
					// noise of gradients that are zero in exact arithmetic, such as
					// those of the keys' bias.
					opt := optim.AdamW(nn.Parameters(m), optim.LR(0.01), optim.WeightDecay(0.01), optim.Eps(1e-4))
					var losses []float64
					for s := range steps {
						x, y := mc.x(s), mc.y(s)
						if mesh != nil {
							spec := tensor.P("data")
							x, y = tensor.Shard(x, mesh, spec), tensor.Shard(y, mesh, spec)
						}
						l := mc.loss(m.Forward(x), y)
						if err := l.Backward(ctx); err != nil {
							t.Fatalf("step %d: %v", s, err)
						}
						v, err := l.Item[float32](ctx)
						if err != nil {
							t.Fatal(err)
						}
						losses = append(losses, float64(v))
						if err := opt.Step(ctx); err != nil {
							t.Fatalf("step %d: %v", s, err)
						}
						opt.ZeroGrad()
					}
					return losses, m
				}
				want, ref := run(nil)
				mesh, tol := testMesh(t, st.shape...)
				got, m := run(mesh)
				for s := range want {
					if d := got[s] - want[s]; d > tol*10 || d < -tol*10 {
						t.Errorf("step %d: loss %v, want %v", s, got[s], want[s])
					}
				}
				if err := closeValues(paramValues(t, m), paramValues(t, ref), tol); err != nil {
					t.Error(err)
				}
				for name, p := range nn.NamedParams(m) {
					if _, ok := p.Value().Sharding(); !ok {
						t.Errorf("%s is not on the mesh", name)
					}
				}
			})
		}
	}
}

// TestShardedFitMatchesDataParallel trains a model with Fit data-parallel
// on four devices, and over a mesh of the four with a "data" axis, and
// checks that the parameters agree.
func TestShardedFitMatchesDataParallel(t *testing.T) {
	ctx := t.Context()
	mc := shardedModels()["mlp"]
	xs := make([]*tensor.Tensor, 3)
	ys := make([]*tensor.Tensor, 3)
	for s := range xs {
		xs[s], ys[s] = mc.x(s), mc.y(s)
	}
	fit := func(opts ...train.Option) nn.Module {
		nn.ManualSeed(9)
		m := mc.build()
		opt := optim.SGD(nn.Parameters(m), optim.LR(0.1), optim.Momentum(0.9))
		ds, err := data.FromTensors(tensor.Cat(0, xs...), tensor.Cat(0, ys...), data.BatchSize(8))
		if err != nil {
			t.Fatal(err)
		}
		if err := train.Fit(ctx, m, opt, ds, append(opts, train.Loss(mc.loss), train.Epochs(2))...); err != nil {
			t.Fatal(err)
		}
		return m
	}
	devs, tol := parallelDevices(t, 4)
	want := paramValues(t, fit(train.DataParallel(devs...)))
	mesh, _ := testMesh(t, 4, 1)
	got := paramValues(t, fit(train.Sharded(mesh, nil)))
	if err := closeValues(got, want, tol); err != nil {
		t.Error(err)
	}
}

// TestShardedCommunication checks the collectives each strategy issues
// in a forward pass, a training step and an optimizer update: none
// forward under data parallelism, one all-reduce per row-parallel layer
// under tensor parallelism, one all-gather of each parameter a step and a
// reduce-scatter of each gradient under fully sharded data parallelism,
// and nothing in the update, whose state is laid out like the parameters.
func TestShardedCommunication(t *testing.T) {
	if os.Getenv("MIMIR_EAGER") == "1" {
		t.Skip("eager mode computes each operation alone, completing its collectives without seeing what reads it")
	}
	ctx := t.Context()
	rowParallel := map[string]int{"mlp": 1, "gpt2 block": 2}
	diff := func(a, b tensor.Traffic) tensor.Traffic {
		return tensor.Traffic{
			AllReduces:     a.AllReduces - b.AllReduces,
			ReduceScatters: a.ReduceScatters - b.ReduceScatters,
			AllGathers:     a.AllGathers - b.AllGathers,
			Bytes:          a.Bytes - b.Bytes,
		}
	}
	for mname, mc := range shardedModels() {
		for _, sname := range []string{"dp", "tp", "fsdp"} {
			st := strategies[sname]
			t.Run(mname+"/"+sname, func(t *testing.T) {
				mesh, _ := testMesh(t, st.shape...)
				nn.ManualSeed(5)
				m := mc.build()
				if err := nn.Shard(m, mesh, st.rule(m, mesh)); err != nil {
					t.Fatal(err)
				}
				params := len(nn.Parameters(m))
				opt := optim.AdamW(nn.Parameters(m), optim.LR(0.01))
				spec := tensor.P("data")
				x, y := tensor.Shard(mc.x(0), mesh, spec), tensor.Shard(mc.y(0), mesh, spec)
				t0 := mesh.Traffic()
				if err := m.Forward(x).Realize(ctx); err != nil {
					t.Fatal(err)
				}
				t1 := mesh.Traffic()
				if err := mc.loss(m.Forward(x), y).Backward(ctx); err != nil {
					t.Fatal(err)
				}
				t2 := mesh.Traffic()
				if err := opt.Step(ctx); err != nil {
					t.Fatal(err)
				}
				fwd, step, upd := diff(t1, t0), diff(t2, t1), diff(mesh.Traffic(), t2)
				t.Logf("forward %+v, step %+v", fwd, step)
				if upd != (tensor.Traffic{}) {
					t.Errorf("update: %+v, want no collectives", upd)
				}
				switch sname {
				case "dp":
					if fwd != (tensor.Traffic{}) {
						t.Errorf("forward: %+v, want no collectives", fwd)
					}
					if step.AllGathers != 0 || step.ReduceScatters != 0 {
						t.Errorf("step: %+v, want all-reduces only", step)
					}
				case "tp":
					if want := (tensor.Traffic{AllReduces: rowParallel[mname], Bytes: fwd.Bytes}); fwd != want {
						t.Errorf("forward: %+v, want %d all-reduces only", fwd, rowParallel[mname])
					}
				case "fsdp":
					if fwd.AllGathers != params || fwd.AllReduces != 0 || fwd.ReduceScatters != 0 {
						t.Errorf("forward: %+v, want %d all-gathers only", fwd, params)
					}
					// The backward pass reads the parameters the forward pass of the
					// same step gathered, and only the loss is all-reduced.
					if want := (tensor.Traffic{AllReduces: 1, ReduceScatters: params, AllGathers: params, Bytes: step.Bytes}); step != want {
						t.Errorf("step: %+v, want %d all-gathers and reduce-scatters", step, params)
					}
				}
			})
		}
	}
}

func TestShardedOptionErrors(t *testing.T) {
	mesh, _ := testMesh(t, 2, 2)
	m := nn.NewLinear(2, 2)
	opt := optim.SGD(nn.Parameters(m))
	ds, err := data.FromTensors(tensor.Zeros(tensor.Float32, 4, 2), tensor.Zeros(tensor.Float32, 4, 2))
	if err != nil {
		t.Fatal(err)
	}
	for name, o := range map[string]train.Option{
		"micro":    train.MicroBatch(2),
		"parallel": train.DataParallel(tensor.CPUs(2)...),
	} {
		err := train.Fit(t.Context(), m, opt, ds, train.Loss(loss.MSE), train.Sharded(mesh, nil), o)
		if err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
