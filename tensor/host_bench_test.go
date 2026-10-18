package tensor_test

import (
	"context"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// idle is a backend whose kernels do nothing, so that a benchmark on it
// times the host's work alone.
type idle struct{}

type idleBuffer struct {
	dt backend.DType
	n  int
}

func (b *idleBuffer) DType() backend.DType { return b.dt }
func (b *idleBuffer) Len() int             { return b.n }

func (idle) Name() string { return "idle" }
func (idle) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	return &idleBuffer{dt, n}, nil
}
func (idle) Free(backend.Buffer)                                    {}
func (idle) Upload(context.Context, backend.Buffer, []byte) error   { return nil }
func (idle) Download(context.Context, []byte, backend.Buffer) error { return nil }
func (idle) Run(context.Context, *backend.Kernel, []backend.Buffer, []backend.Buffer) error {
	return nil
}
func (idle) Sync(context.Context) error { return nil }

// BenchmarkHostStep times the host's work in a training step of a GPT-2
// block and of four, on a device whose kernels do nothing, without a
// mesh (n=1) and on a mesh of that one device (mesh=1): the difference
// is what sharding costs the host.
func BenchmarkHostStep(b *testing.B) {
	tensor.SetCheckPlans(false)
	defer tensor.SetCheckPlans(true)
	for _, bm := range []struct {
		name   string
		blocks int
	}{{"gpt2", 1}, {"transformer", 4}} {
		for _, sharded := range []bool{false, true} {
			name := bm.name + "/n=1"
			if sharded {
				name = bm.name + "/mesh=1"
			}
			b.Run(name, func(b *testing.B) {
				dev := tensor.DeviceOn(&idle{})
				nn.ManualSeed(1)
				blocks := make([]nn.Module, bm.blocks)
				for i := range blocks {
					blocks[i] = nn.NewGPT2Block(256, 8)
				}
				m := nn.NewSequential(blocks...)
				x, y := tensor.Randn(1, 8, 64, 256), tensor.Randn(2, 8, 64, 256)
				if sharded {
					mesh, err := tensor.NewMesh([]int{1}, []string{"data"}, dev)
					if err != nil {
						b.Fatal(err)
					}
					defer mesh.Close()
					if err := nn.Shard(m, mesh, nil); err != nil {
						b.Fatal(err)
					}
					x, y = tensor.Shard(x, mesh, tensor.P("data")), tensor.Shard(y, mesh, tensor.P("data"))
				} else {
					if err := nn.To(m, dev); err != nil {
						b.Fatal(err)
					}
					x, y = x.To(dev), y.To(dev)
				}
				opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-4))
				step := func() {
					l := loss.MSE(m.Forward(x), y)
					if err := l.Backward(b.Context()); err != nil {
						b.Fatal(err)
					}
					if _, err := l.Item[float32](b.Context()); err != nil {
						b.Fatal(err)
					}
					if err := opt.Step(b.Context()); err != nil {
						b.Fatal(err)
					}
					opt.ZeroGrad()
				}
				for range 2 {
					step()
				}
				for b.Loop() {
					step()
				}
			})
		}
	}
}
