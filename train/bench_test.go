package train_test

import (
	"os"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkTrainStep times one training step: the forward pass, the loss,
// the backward pass and an AdamW update. The mlp and cnn models are those
// of examples/mnist, on a batch of 128 MNIST-sized images. deep is a deep
// network for table rows, on a batch of 512. resnet18 is
// ResNet-18 on a batch of 16 ImageNet-sized images, a step of seconds,
// which runs only when MIMIR_BENCH_RESNET=1. tools/bench times the same
// steps in PyTorch.
//
// Each model trains on one batch over and over, and, in the upload and
// donated sub-benchmarks, on a new batch every step, uploaded from host
// memory as a dataset's batches are; donated hands each batch's memory
// to its step (see tensor.Tensor.Donate). Each reports the page faults
// of a step, where the platform counts them.
func BenchmarkTrainStep(b *testing.B) {
	for _, bm := range []struct {
		name    string
		model   func() nn.Module
		batch   int
		input   []int // of one example
		classes int
	}{
		{"mlp", mnistMLP, 128, []int{1, 28, 28}, 10},
		{"cnn", mnistCNN, 128, []int{1, 28, 28}, 10},
		{"deep", deepMLP, 512, []int{20}, 3},
		{"resnet18", resnet18, 16, []int{3, 224, 224}, 1000},
	} {
		for _, feed := range []string{"", "/upload", "/donated"} {
			b.Run(bm.name+feed, func(b *testing.B) {
				if bm.name == "resnet18" && os.Getenv("MIMIR_BENCH_RESNET") != "1" {
					b.Skip("set MIMIR_BENCH_RESNET=1 to run")
				}
				trainStep(b, bm.model, bm.batch, bm.input, bm.classes, feed)
			})
		}
	}
}

// trainStep times the training step of BenchmarkTrainStep of the model
// that model returns, on batches of the given size and examples of the
// shape input, fed as feed says: one batch, or a new one every step,
// uploaded or uploaded and donated.
func trainStep(b *testing.B, model func() nn.Module, batch int, input []int, classes int, feed string) {
	nn.ManualSeed(1)
	m := model()
	opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
	shape := append([]int{batch}, input...)
	x := tensor.Rand(1, shape...)
	labels := make([]int32, batch)
	for i := range labels {
		labels[i] = int32(i % classes)
	}
	y := tensor.FromSlice(labels, batch)
	if err := tensor.Realize(b.Context(), x, y); err != nil {
		b.Fatal(err)
	}
	// A few batches of host memory, which the upload feeds go through in
	// turn.
	var images [4][]float32
	for i := range images {
		var err error
		if images[i], err = tensor.Rand(uint64(i+2), shape...).Data[float32](b.Context()); err != nil {
			b.Fatal(err)
		}
	}
	n := 0
	next := func() (x, y *tensor.Tensor) {
		n++
		x, y = tensor.FromSlice(images[n%len(images)], shape...), tensor.FromSlice(labels, batch)
		if feed == "/donated" {
			x, y = x.Donate(), y.Donate()
		}
		return x, y
	}
	step := func() {
		x, y := x, y
		if feed != "" {
			x, y = next()
		}
		l := loss.CrossEntropy(m.Forward(x), y)
		if err := l.Backward(b.Context()); err != nil {
			b.Fatal(err)
		}
		if err := opt.Step(b.Context()); err != nil {
			b.Fatal(err)
		}
		opt.ZeroGrad()
	}
	// The first steps compile the GPU kernels, plan the schedules,
	// create the optimizer's state and, on CUDA, capture the step's
	// graph: work a training run does once, which is not timed.
	for range 3 {
		step()
	}
	faults, counted := pageFaults()
	for b.Loop() {
		step()
	}
	if after, _ := pageFaults(); counted {
		b.ReportMetric(float64(after-faults)/float64(b.N), "faults/op")
	}
}

func mnistMLP() nn.Module {
	return nn.NewSequential(
		nn.Flatten(),
		nn.NewLinear(28*28, 256), nn.ReLU(),
		nn.NewLinear(256, 10),
	)
}

// deepMLP is a deep, wide network for rows of a table, like those
// mimir-server builds for a designed table model: 20 numbers in, three
// answers out, 2.8 million parameters.
func deepMLP() nn.Module {
	return nn.NewSequential(
		nn.NewLinear(20, 1024), nn.ReLU(),
		nn.NewLinear(1024, 1024), nn.ReLU(),
		nn.NewLinear(1024, 1024), nn.ReLU(),
		nn.NewLinear(1024, 512), nn.ReLU(),
		nn.NewLinear(512, 256), nn.ReLU(),
		nn.NewLinear(256, 3),
	)
}

func mnistCNN() nn.Module {
	return nn.NewSequential(
		nn.NewConv2d(1, 32, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
		nn.NewConv2d(32, 64, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
		nn.Flatten(),
		nn.NewLinear(64*7*7, 128), nn.ReLU(),
		nn.NewLinear(128, 10),
	)
}

// resnet18 returns ResNet-18 for 224×224 images and 1000 classes, like
// torchvision.models.resnet18.
func resnet18() nn.Module {
	layers := []nn.Module{
		nn.NewConv2d(3, 64, 7, nn.Stride(2), nn.Padding(3), nn.NoBias()), nn.NewBatchNorm2d(64), nn.ReLU(),
		nn.MaxPool2d(3, nn.Stride(2), nn.Padding(1)),
	}
	in := 64
	for _, stage := range []struct{ out, stride int }{{64, 1}, {128, 2}, {256, 2}, {512, 2}} {
		layers = append(layers, newBasicBlock(in, stage.out, stage.stride), newBasicBlock(stage.out, stage.out, 1))
		in = stage.out
	}
	layers = append(layers, nn.AvgPool2d(7), nn.Flatten(), nn.NewLinear(512, 1000))
	return nn.NewSequential(layers...)
}

// basicBlock is ResNet's residual block of two 3×3 convolutions.
type basicBlock struct {
	conv1, conv2 *nn.Conv2d
	bn1, bn2     *nn.BatchNorm
	down         *nn.Sequential // projects the input when the shape changes
}

func newBasicBlock(in, out, stride int) *basicBlock {
	b := &basicBlock{
		conv1: nn.NewConv2d(in, out, 3, nn.Stride(stride), nn.Padding(1), nn.NoBias()),
		bn1:   nn.NewBatchNorm2d(out),
		conv2: nn.NewConv2d(out, out, 3, nn.Padding(1), nn.NoBias()),
		bn2:   nn.NewBatchNorm2d(out),
	}
	if stride != 1 || in != out {
		b.down = nn.NewSequential(nn.NewConv2d(in, out, 1, nn.Stride(stride), nn.NoBias()), nn.NewBatchNorm2d(out))
	}
	return b
}

func (b *basicBlock) Forward(x *tensor.Tensor) *tensor.Tensor {
	y := b.bn1.Forward(b.conv1.Forward(x)).ReLU()
	y = b.bn2.Forward(b.conv2.Forward(y))
	skip := x
	if b.down != nil {
		skip = b.down.Forward(x)
	}
	return y.Add(skip).ReLU()
}
