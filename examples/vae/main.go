// Command vae trains a variational autoencoder on the MNIST handwritten
// digits with the reparameterization trick, printing the evidence lower
// bound (ELBO) of the training and test images every epoch and writing
// PNG grids of digits decoded from the prior and of test reconstructions.
//
//	go run ./examples/vae -epochs 10 -out vae
//
// It downloads MNIST to -data on first use. With the default 400 hidden
// units and 20 latent dimensions the test ELBO is about -117 nats after
// two epochs and -106 after ten, each 20 to 60 seconds on 4 CPU cores.
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
		epochs = flag.Int("epochs", 10, "passes over the training set")
		batch  = flag.Int("batch", 128, "images per step")
		lr     = flag.Float64("lr", 1e-3, "Adam learning rate")
		hidden = flag.Int("hidden", 400, "hidden units of the encoder and decoder")
		latent = flag.Int("latent", 20, "latent dimensions")
		seed   = flag.Uint64("seed", 1, "seed of the weights and samples")
		dir    = flag.String("data", "data/mnist", "where MNIST is kept")
		out    = flag.String("out", "vae", "directory for the PNG grids")
		device = flag.String("device", "cpu", "cpu, gpu or auto")
	)
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	dev, err := pick(*device)
	if err != nil {
		log.Fatal(err)
	}
	cfg := Config{Pixels: 28 * 28, Hidden: *hidden, Latent: *latent}
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
	trainSet, testSet, err := datasets.MNIST(dir, data.BatchSize(batch), data.Shuffle(seed))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	model, err := NewVAE(cfg, seed)
	if err != nil {
		return err
	}
	if err := nn.To(model, dev); err != nil {
		return err
	}
	tr := NewTrainer(model, lr, seed)
	fmt.Printf("training a VAE with %d latent dimensions on %v with %d images, testing on %d\n", cfg.Latent, dev, trainSet.Len(), testSet.Len())
	flat := func(x *tensor.Tensor) *tensor.Tensor { return x.Reshape(-1, cfg.Pixels).To(dev) }
	for epoch := 1; epoch <= epochs; epoch++ {
		start := time.Now()
		var sum float64
		n := 0
		for b, err := range trainSet.All(ctx) {
			if err != nil {
				return err
			}
			v, err := tr.Step(ctx, flat(b.X))
			if err != nil {
				return err
			}
			k := b.X.Shape()[0]
			sum, n = sum+v*float64(k), n+k
		}
		var testSum float64
		testN := 0
		var first *tensor.Tensor
		for b, err := range testSet.All(ctx) {
			if err != nil {
				return err
			}
			if first == nil {
				first = flat(b.X)
			}
			v, err := tr.Evaluate(ctx, flat(b.X))
			if err != nil {
				return err
			}
			k := b.X.Shape()[0]
			testSum, testN = testSum+v*float64(k), testN+k
		}
		fmt.Printf("epoch %2d  train ELBO %8.2f  test ELBO %8.2f nats  %.1fs\n",
			epoch, sum/float64(n), testSum/float64(testN), time.Since(start).Seconds())
		if err := pictures(ctx, model, first, seed, out, 28); err != nil {
			return err
		}
	}
	fmt.Printf("wrote %s and %s\n", filepath.Join(out, "samples.png"), filepath.Join(out, "reconstructions.png"))
	return nil
}

// pictures writes 64 images of side×side pixels decoded from the prior to
// samples.png in out, and the first 32 images of x above their
// reconstructions to reconstructions.png.
func pictures(ctx context.Context, model *VAE, x *tensor.Tensor, seed uint64, out string, side int) error {
	if err := grid.Write(ctx, filepath.Join(out, "samples.png"), model.Generate(64, seed), side, side, 8, 0, 1); err != nil {
		return err
	}
	k := min(32, x.Shape()[0])
	x = x.Shrink([][2]int{{0, k}, {0, x.Shape()[1]}})
	pair := tensor.Cat(0, x, model.Forward(x).Detach())
	return grid.Write(ctx, filepath.Join(out, "reconstructions.png"), pair, side, side, 8, 0, 1)
}
