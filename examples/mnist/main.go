// Command mnist trains a classifier on the MNIST handwritten digits with
// mimir, on the CPU or the GPU.
//
//	go run ./examples/mnist -model cnn -epochs 3 -device auto
//
// It downloads MNIST to -data on first use, checking each file's SHA-256.
// A multilayer perceptron reaches about 98% test accuracy and the
// convolutional network over 99% in 10 epochs.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/gsoultan/mimir"
	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/data/datasets"
	"github.com/gsoultan/mimir/metrics"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/train"
)

func main() {
	var (
		kind   = flag.String("model", "mlp", "mlp or cnn")
		epochs = flag.Int("epochs", 3, "passes over the training set")
		batch  = flag.Int("batch", 128, "examples per step")
		lr     = flag.Float64("lr", 1e-3, "AdamW learning rate")
		dir    = flag.String("data", "data/mnist", "where MNIST is kept")
		ckpt   = flag.String("checkpoint", "", "directory to checkpoint to and resume from")
		device = flag.String("device", "cpu", "cpu, gpu or auto")
	)
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	dev, err := pick(*device)
	if err != nil {
		log.Fatal(err)
	}
	if err := run(ctx, dev, *kind, *epochs, *batch, *lr, *dir, *ckpt); err != nil {
		log.Fatal(err)
	}
}

// pick returns the device that name selects.
func pick(name string) (mimir.Device, error) {
	switch name {
	case "cpu":
		return mimir.CPU(), nil
	case "gpu":
		return mimir.GPU()
	case "auto":
		return mimir.Auto(), nil
	}
	return mimir.Device{}, fmt.Errorf("unknown device %q, want cpu, gpu or auto", name)
}

func run(ctx context.Context, dev mimir.Device, kind string, epochs, batch int, lr float64, dir, ckpt string) error {
	if err := datasets.DownloadMNIST(ctx, dir); err != nil {
		return err
	}
	trainSet, testSet, err := datasets.MNIST(dir, data.BatchSize(batch), data.Shuffle(42))
	if err != nil {
		return err
	}

	var model nn.Module
	switch kind {
	case "mlp":
		model = nn.NewSequential(
			nn.Flatten(),
			nn.NewLinear(28*28, 256), nn.ReLU(),
			nn.NewLinear(256, 10),
		)
	case "cnn":
		model = nn.NewSequential(
			nn.NewConv2d(1, 32, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2), // 32×14×14
			nn.NewConv2d(32, 64, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2), // 64×7×7
			nn.Flatten(),
			nn.NewLinear(64*7*7, 128), nn.ReLU(),
			nn.NewLinear(128, 10),
		)
	default:
		return fmt.Errorf("unknown model %q, want mlp or cnn", kind)
	}
	if err := nn.To(model, dev); err != nil {
		return err
	}
	opt := optim.AdamW(nn.Parameters(model), optim.LR(lr), optim.WeightDecay(0.01))

	opts := []train.Option{
		train.Loss(loss.CrossEntropy),
		train.Epochs(epochs),
		train.Schedule(optim.Cosine(lr, 0, epochs)),
		train.Eval(testSet, metrics.Accuracy),
		train.Progress(os.Stdout),
	}
	if ckpt != "" {
		opts = append(opts, train.Checkpoint(ckpt))
	}
	fmt.Printf("training %s on %v with %d images, testing on %d\n", kind, dev, trainSet.Len(), testSet.Len())
	return train.Fit(ctx, model, opt, trainSet, opts...)
}
