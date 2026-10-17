package train_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkSharded times a training step, the forward pass, the loss, the
// backward pass and an AdamW update, of models laid out over a mesh of n
// devices by each strategy: "dp" splits the batch over a "data" axis,
// "fsdp" splits the parameters over it too, and "tp" splits the layers
// over a "model" axis (Megatron's column- then row-parallel layers). The
// models are a GPT-2 block of 256 features and 8 heads on sequences of
// 64, and "transformer", a stack of four GPT-2 blocks of 512 features and
// 8 heads on sequences of 64 (of 1024 features and 16 heads on sequences
// of 256 under MIMIR_DEVICE=gpu).
//
// The batch is fixed per device: 8 sequences per device of the "data"
// axis, so dp and fsdp grow it with n, and 8 sequences on every n for tp.
// Each reports examples per second and, against the same model on one
// device without a mesh (the sub-benchmarks n=1), the efficiency: the
// examples per second over n times those of one device for dp and fsdp,
// as the 8-GPU scaling gate measures it, and the speedup over n for tp.
// The sub-benchmark mesh=1 trains on a mesh of one device, whose cost
// over n=1 is the partitioner's ("overhead").
//
// The devices are GPUs under MIMIR_DEVICE=gpu, with n up to 8, and
// otherwise up to 4 devices of tensor.CPUs, which share the host's
// cores, so there efficiency can reach 1/n at most.
func BenchmarkSharded(b *testing.B) {
	gpu := os.Getenv("MIMIR_DEVICE") == "gpu"
	width, seq := 512, 64
	counts := []int{2, 4}
	if gpu {
		width, seq = 1024, 256
		counts = append(counts, 8)
	}
	const per = 8
	for _, bm := range []struct {
		name  string
		model func() nn.Module
		shape []int // of one example
	}{
		{"gpt2", func() nn.Module { return nn.NewGPT2Block(256, 8) }, []int{64, 256}},
		{"transformer", func() nn.Module {
			blocks := make([]nn.Module, 4)
			for i := range blocks {
				blocks[i] = nn.NewGPT2Block(width, width/64)
			}
			return nn.NewSequential(blocks...)
		}, []int{seq, width}},
	} {
		var single float64 // examples per second on one device
		run := func(name string, n int, strategy string) {
			b.Run(name, func(b *testing.B) {
				devs := benchDevices(b, n)
				nn.ManualSeed(1)
				m := bm.model()
				batch := per
				var mesh *tensor.Mesh
				if strategy == "" {
					if err := nn.To(m, devs[0]); err != nil {
						b.Fatal(err)
					}
				} else {
					shape := []int{n, 1}
					if strategy == "tp" {
						shape = []int{1, n}
					} else {
						batch = per * n
					}
					var err error
					mesh, err = tensor.NewMesh(shape, []string{"data", "model"}, devs...)
					if err != nil {
						b.Fatal(err)
					}
					defer mesh.Close()
					var rule nn.ShardRule
					switch strategy {
					case "tp":
						rule = nn.TensorParallel(m, mesh, "model")
					case "fsdp":
						rule = nn.FullyShard(mesh, "data")
					}
					if err := nn.Shard(m, mesh, rule); err != nil {
						b.Fatal(err)
					}
				}
				opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-4))
				shape := append([]int{batch}, bm.shape...)
				x, y := tensor.Randn(1, shape...), tensor.Randn(2, shape...)
				if mesh != nil {
					x, y = tensor.Shard(x, mesh, tensor.P("data")), tensor.Shard(y, mesh, tensor.P("data"))
				} else {
					x, y = x.To(devs[0]), y.To(devs[0])
				}
				if err := tensor.Realize(b.Context(), x, y); err != nil {
					b.Fatal(err)
				}
				// Reading each step's loss, as a training loop that logs it does,
				// waits for the step: the devices of tensor.CPUs, like GPUs, run
				// their work asynchronously.
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
				var before tensor.Traffic
				if mesh != nil {
					before = mesh.Traffic()
				}
				for b.Loop() {
					step()
				}
				if mesh != nil {
					b.ReportMetric(float64(mesh.Traffic().Bytes-before.Bytes)/float64(b.N)/(1<<20), "comm-MB/op")
				}
				rate := float64(batch*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(rate, "examples/s")
				switch {
				case strategy == "":
					single = rate
				case single == 0:
				case n == 1:
					b.ReportMetric(single/rate, "overhead")
				default:
					b.ReportMetric(rate/single/float64(n), "efficiency")
				}
			})
		}
		run(bm.name+"/n=1", 1, "")
		run(bm.name+"/mesh=1", 1, "dp")
		for _, strategy := range []string{"dp", "fsdp", "tp"} {
			for _, n := range counts {
				run(fmt.Sprintf("%s/%s/n=%d", bm.name, strategy, n), n, strategy)
			}
		}
	}
}
