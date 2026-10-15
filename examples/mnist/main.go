// Command mnist trains a classifier on the MNIST handwritten digits, on
// the CPU, with mimir.
//
//	go run ./examples/mnist -model cnn -epochs 3
//
// It downloads MNIST to -data on first use, checking each file's SHA-256.
// A multilayer perceptron reaches about 97% test accuracy in a few
// epochs and the small convolutional network about 98%.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

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
	)
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *kind, *epochs, *batch, *lr, *dir, *ckpt); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, kind string, epochs, batch int, lr float64, dir, ckpt string) error {
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
			nn.NewConv2d(1, 16, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2), // 16×14×14
			nn.NewConv2d(16, 32, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2), // 32×7×7
			nn.Flatten(),
			nn.NewLinear(32*7*7, 10),
		)
	default:
		return fmt.Errorf("unknown model %q, want mlp or cnn", kind)
	}
	opt := optim.AdamW(nn.Parameters(model), optim.LR(lr), optim.WeightDecay(0.01))

	opts := []train.Option{
		train.Loss(loss.CrossEntropy),
		train.Epochs(epochs),
		train.Eval(testSet, metrics.Accuracy),
		train.Progress(os.Stdout),
	}
	if ckpt != "" {
		opts = append(opts, train.Checkpoint(ckpt))
	}
	fmt.Printf("training %s on %d images, testing on %d\n", kind, trainSet.Len(), testSet.Len())
	return train.Fit(ctx, model, opt, trainSet, opts...)
}
