// Command gan trains a generative adversarial network of multilayer
// perceptrons on the MNIST handwritten digits, printing the
// discriminator's and the generator's losses every epoch and writing a
// PNG grid of digits generated from the same noise after each.
//
//	go run ./examples/gan -epochs 20 -out gan
//
// It downloads MNIST to -data on first use. Digit-like strokes appear
// after an epoch or two and recognizable digits within about ten.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/gsoultan/mimir"
	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/data/datasets"
	"github.com/gsoultan/mimir/examples/internal/grid"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

func main() {
	var (
		epochs = flag.Int("epochs", 20, "passes over the training set")
		batch  = flag.Int("batch", 64, "real images per step")
		lr     = flag.Float64("lr", 2e-4, "Adam learning rate of both networks")
		hidden = flag.Int("hidden", 256, "units of the narrower hidden layer; the wider has twice as many")
		noise  = flag.Int("noise", 64, "dimensions of the generator's noise")
		seed   = flag.Uint64("seed", 1, "seed of the weights and noise")
		dir    = flag.String("data", "data/mnist", "where MNIST is kept")
		out    = flag.String("out", "gan", "directory for the PNG grids")
		device = flag.String("device", "cpu", "cpu, gpu or auto")
	)
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	dev, err := pick(*device)
	if err != nil {
		log.Fatal(err)
	}
	cfg := Config{Pixels: 28 * 28, Noise: *noise, Hidden: *hidden}
	if err := run(ctx, dev, cfg, *epochs, *batch, *lr, *seed, *dir, *out); err != nil {
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

func run(ctx context.Context, dev mimir.Device, cfg Config, epochs, batch int, lr float64, seed uint64, dir, out string) error {
	if err := datasets.DownloadMNIST(ctx, dir); err != nil {
		return err
	}
	trainSet, _, err := datasets.MNIST(dir, data.BatchSize(batch), data.Shuffle(seed), data.DropLast())
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	model, err := NewGAN(cfg, seed)
	if err != nil {
		return err
	}
	if err := nn.To(model, dev); err != nil {
		return err
	}
	tr := NewTrainer(model, lr, seed)
	fmt.Printf("training a GAN on %v with %d images\n", dev, trainSet.Len())
	// Pixels in [0, 1] become [-1, 1], the range of the generator's tanh.
	two, one := tensor.Scalar[float32](2), tensor.Scalar[float32](1)
	path := filepath.Join(out, "samples.png")
	for epoch := 1; epoch <= epochs; epoch++ {
		start := time.Now()
		var sum Losses
		n := 0
		for b, err := range trainSet.All(ctx) {
			if err != nil {
				return err
			}
			x := b.X.Reshape(-1, cfg.Pixels).Mul(two).Sub(one).To(dev)
			l, err := tr.Step(ctx, x)
			if err != nil {
				return err
			}
			sum.D, sum.G, n = sum.D+l.D, sum.G+l.G, n+1
		}
		fmt.Printf("epoch %2d  discriminator loss %.3f  generator loss %.3f  %.1fs\n",
			epoch, sum.D/float64(n), sum.G/float64(n), time.Since(start).Seconds())
		if err := grid.Write(ctx, path, model.Generate(64, seed), 28, 28, 8, -1, 1); err != nil {
			return err
		}
	}
	fmt.Printf("wrote %s\n", path)
	return nil
}
