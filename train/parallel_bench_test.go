package train_test

import (
	"fmt"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/train"
)

// BenchmarkDataParallel times a data-parallel training step, the
// gradients of a batch split between devices with Parallel and an AdamW
// step, on 1, 2 and 4 devices with the same examples per device, so the
// batch grows with the devices: the MNIST CNN of BenchmarkTrainStep on 64
// images per device, and a GPT-2 block of 256 features and 4 heads on 8
// sequences of 64 per device. The devices are GPUs under MIMIR_DEVICE=gpu,
// and otherwise the devices of tensor.CPUs, which share the host's cores.
// On GPUs, the sub-benchmarks named x8 repeat them with 8 times the
// examples per device, which tells whether a GPU's share of the small
// batches leaves it waiting on the host rather than computing. Each also
// reports the host's time per step in each phase of Parallel.Backward,
// those of one device numbered with its rank, and in the optimizer's
// step: a phase that waits for a device shows that device's work.
//
// Each reports examples per second, the speedup over one device and the
// efficiency, the speedup divided by the devices: 1 when n devices
// process n times the examples in the same time. Devices of tensor.CPUs
// split the same cores, so there the ideal speedup is 1, not n, and the
// sub-benchmarks named n/cpu train the batch of n devices on the CPU
// itself, one device with every core, for the speedup that splitting
// gives over the same work on one device ("over-cpu" in those of n).
func BenchmarkDataParallel(b *testing.B) {
	for _, bm := range []struct {
		name  string
		model func() nn.Module
		batch func(n int) (x, y *tensor.Tensor)
		loss  func(pred, target *tensor.Tensor) *tensor.Tensor
	}{
		{"cnn", mnistCNN, func(n int) (x, y *tensor.Tensor) {
			labels := make([]int32, n)
			for i := range labels {
				labels[i] = int32(i % 10)
			}
			return tensor.Rand(1, n, 1, 28, 28), tensor.FromSlice(labels, n)
		}, loss.CrossEntropy},
		{"gpt2", func() nn.Module { return nn.NewGPT2Block(256, 4) }, func(n int) (x, y *tensor.Tensor) {
			return tensor.Randn(1, n, 64, 256), tensor.Randn(2, n, 64, 256)
		}, loss.MSE},
	} {
		scales := []int{1}
		if os.Getenv("MIMIR_DEVICE") == "gpu" {
			scales = append(scales, 8)
		}
		for _, scale := range scales {
			per := scale * map[string]int{"cnn": 64, "gpt2": 8}[bm.name]
			var single float64         // examples per second on one device
			whole := map[int]float64{} // on the CPU, of the batch of n devices
			for _, n := range []int{1, 2, 4} {
				for _, onCPU := range []bool{true, false} {
					name := fmt.Sprintf("%s/%d", bm.name, n)
					if scale > 1 {
						name = fmt.Sprintf("%s/x%d/%d", bm.name, scale, n)
					}
					if onCPU {
						if n == 1 || os.Getenv("MIMIR_DEVICE") == "gpu" {
							continue
						}
						name += "/cpu"
					}
					b.Run(name, func(b *testing.B) {
						devs := []tensor.Device{tensor.CPU()}
						if !onCPU {
							devs = benchDevices(b, n)
						}
						nn.ManualSeed(1)
						m := bm.model()
						if err := nn.To(m, devs[0]); err != nil {
							b.Fatal(err)
						}
						opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-4))
						dp, err := train.NewParallel(m, devs...)
						if err != nil {
							b.Fatal(err)
						}
						defer dp.Close()
						x, y := bm.batch(per * n)
						if err := tensor.Realize(b.Context(), x, y); err != nil {
							b.Fatal(err)
						}
						var spent map[string]time.Duration // host time per phase
						step := func() {
							if _, err := dp.Backward(b.Context(), bm.loss, x, y); err != nil {
								b.Fatal(err)
							}
							start := time.Now()
							if err := opt.Step(b.Context()); err != nil {
								b.Fatal(err)
							}
							opt.ZeroGrad()
							if spent != nil {
								spent["step"] += time.Since(start)
							}
						}
						for range 3 {
							step()
						}
						spent = map[string]time.Duration{}
						defer train.SetPhases(func(phase string, rank int, d time.Duration) {
							if rank >= 0 {
								phase += strconv.Itoa(rank)
							}
							spent[phase] += d
						})()
						for b.Loop() {
							step()
						}
						for phase, d := range spent {
							b.ReportMetric(float64(d.Microseconds())/float64(b.N), phase+"-µs/op")
						}
						rate := float64(per*n*b.N) / b.Elapsed().Seconds()
						b.ReportMetric(rate, "examples/s")
						switch {
						case onCPU:
							whole[n] = rate
						case n == 1:
							single = rate
						case single > 0:
							b.ReportMetric(rate/single, "speedup")
							b.ReportMetric(rate/single/float64(n), "efficiency")
							if whole[n] > 0 {
								b.ReportMetric(rate/whole[n], "over-cpu")
							}
						}
					})
				}
			}
		}
	}
}

var gpuDevices = sync.OnceValue(tensor.GPUs)

// benchDevices returns n devices of the kind the benchmarks run on,
// skipping b if there are fewer.
func benchDevices(b *testing.B, n int) []tensor.Device {
	if os.Getenv("MIMIR_DEVICE") == "gpu" {
		ds := gpuDevices()
		if len(ds) < n {
			b.Skipf("%d GPUs, want %d", len(ds), n)
		}
		return ds[:n]
	}
	return tensor.CPUs(n)
}
