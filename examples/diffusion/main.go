// Command diffusion trains a small denoising diffusion probabilistic
// model (DDPM) on the MNIST handwritten digits, or on a toy cloud of 2-D
// points, printing the denoising loss every epoch and writing a PNG of
// samples drawn with the ancestral sampler after each: a grid of digits,
// or the points drawn over the data.
//
//	go run ./examples/diffusion -out diffusion
//	go run ./examples/diffusion -dataset moons -out moons
//
// It downloads MNIST to -data on first use. The denoiser is a multilayer
// perceptron with residual connections, which keeps an MNIST epoch to
// one or two minutes on 4 CPU cores; digit shapes appear within a few
// epochs, though without a convolutional U-Net they keep some speckle.
// The two moons train in a few minutes and come out sharp. -ddim adds
// samples drawn with the deterministic DDIM sampler in a tenth of the
// steps. Flags left at 0 take the data set's defaults.
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

// options are the command's settings.
type options struct {
	dataset       string
	epochs, batch int
	lr            float64
	hidden, steps int
	seed          uint64
	ddim, every   int
	dir, out      string
	dev           mimir.Device
}

func main() {
	var o options
	flag.StringVar(&o.dataset, "dataset", "mnist", "mnist or moons")
	flag.IntVar(&o.epochs, "epochs", 0, "passes over the training set; 0 for 20 on MNIST and 200 on the moons")
	flag.IntVar(&o.batch, "batch", 0, "examples per step; 0 for 128 on MNIST and 256 on the moons")
	flag.Float64Var(&o.lr, "lr", 0, "Adam learning rate; 0 for 5e-4 on MNIST and 3e-3 on the moons")
	flag.IntVar(&o.every, "every", 0, "epochs between pictures; 0 for 1 on MNIST and 20 on the moons")
	flag.IntVar(&o.hidden, "hidden", 0, "units of each hidden layer of the denoiser; 0 for 1024 on MNIST and 128 on the moons")
	flag.IntVar(&o.steps, "steps", 0, "diffusion steps; 0 for 500 on MNIST and 100 on the moons")
	flag.Uint64Var(&o.seed, "seed", 1, "seed of the weights and noise")
	flag.IntVar(&o.ddim, "ddim", 10, "stride of the DDIM sampler's steps; 0 skips it")
	flag.StringVar(&o.dir, "data", "data/mnist", "where MNIST is kept")
	flag.StringVar(&o.out, "out", "diffusion", "directory for the PNGs")
	device := flag.String("device", "cpu", "cpu, gpu or auto")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var err error
	if o.dev, err = pick(*device); err != nil {
		log.Fatal(err)
	}
	if err := run(ctx, o); err != nil {
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

// task is a data set to train on and how to picture samples of it.
type task struct {
	set *data.Tensors
	cfg Config
	// scale maps a batch of examples to rows of values in [-1, 1].
	scale func(x *tensor.Tensor) *tensor.Tensor
	// picture writes samples, [n, cfg.Pixels], to path.
	picture func(ctx context.Context, path string, x *tensor.Tensor) error
}

// load returns the task that o names, setting the options o left at 0
// to its defaults.
func load(ctx context.Context, o *options) (task, error) {
	switch o.dataset {
	case "mnist":
		o.defaults(20, 128, 1, 5e-4)
		if err := datasets.DownloadMNIST(ctx, o.dir); err != nil {
			return task{}, err
		}
		set, _, err := datasets.MNIST(o.dir, data.BatchSize(o.batch), data.Shuffle(o.seed))
		if err != nil {
			return task{}, err
		}
		cfg := Config{Pixels: 28 * 28, Hidden: or(o.hidden, 1024), Time: 64, Steps: or(o.steps, 500)}
		two, one := tensor.Scalar[float32](2), tensor.Scalar[float32](1)
		return task{
			set: set, cfg: cfg,
			scale: func(x *tensor.Tensor) *tensor.Tensor { return x.Reshape(-1, cfg.Pixels).Mul(two).Sub(one) },
			picture: func(ctx context.Context, path string, x *tensor.Tensor) error {
				return grid.Write(ctx, path, x, 28, 28, 8, -1, 1)
			},
		}, nil
	case "moons":
		o.defaults(200, 256, 20, 3e-3)
		points := moons(1<<14, 0.05, o.seed)
		set, err := data.FromTensors(points, tensor.Zeros(tensor.Float32, 1<<14), data.BatchSize(o.batch), data.Shuffle(o.seed))
		if err != nil {
			return task{}, err
		}
		shown := moons(2000, 0.05, ^o.seed)
		return task{
			set: set, cfg: Config{Pixels: 2, Hidden: or(o.hidden, 128), Time: 32, Steps: or(o.steps, 100)},
			scale: func(x *tensor.Tensor) *tensor.Tensor { return x },
			picture: func(ctx context.Context, path string, x *tensor.Tensor) error {
				return grid.Scatter(ctx, path, 400, -1, 1, shown, x)
			},
		}, nil
	}
	return task{}, fmt.Errorf("unknown data set %q, want mnist or moons", o.dataset)
}

// defaults sets the options left at 0.
func (o *options) defaults(epochs, batch, every int, lr float64) {
	o.epochs, o.batch, o.every = or(o.epochs, epochs), or(o.batch, batch), or(o.every, every)
	if o.lr == 0 {
		o.lr = lr
	}
}

// or returns v, or def if v is 0.
func or(v, def int) int {
	if v == 0 {
		return def
	}
	return v
}

func run(ctx context.Context, o options) error {
	tk, err := load(ctx, &o)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(o.out, 0o755); err != nil {
		return err
	}
	model, err := NewDDPM(tk.cfg, o.seed)
	if err != nil {
		return err
	}
	if err := nn.To(model, o.dev); err != nil {
		return err
	}
	tr, err := NewTrainer(ctx, model, o.lr, o.seed)
	if err != nil {
		return err
	}
	n := 64
	if tk.cfg.Pixels == 2 {
		n = 2000
	}
	fmt.Printf("training a DDPM with %d steps on %v with %d examples of %s\n", tk.cfg.Steps, o.dev, tk.set.Len(), o.dataset)
	path, ddimPath := filepath.Join(o.out, "samples.png"), filepath.Join(o.out, "samples-ddim.png")
	for epoch := 1; epoch <= o.epochs; epoch++ {
		start := time.Now()
		var sum float64
		batches := 0
		for b, err := range tk.set.All(ctx) {
			if err != nil {
				return err
			}
			v, err := tr.Step(ctx, tk.scale(b.X).To(o.dev))
			if err != nil {
				return err
			}
			sum, batches = sum+v, batches+1
		}
		trained := time.Since(start)
		if epoch%o.every != 0 && epoch != o.epochs {
			fmt.Printf("epoch %2d  loss %.4f  %.1fs training\n", epoch, sum/float64(batches), trained.Seconds())
			continue
		}
		x, err := tr.EMA.Sample(ctx, n, o.seed)
		if err != nil {
			return err
		}
		if err := tk.picture(ctx, path, x); err != nil {
			return err
		}
		if o.ddim > 0 {
			x, err := tr.EMA.SampleDDIM(ctx, n, o.seed, o.ddim)
			if err != nil {
				return err
			}
			if err := tk.picture(ctx, ddimPath, x); err != nil {
				return err
			}
		}
		fmt.Printf("epoch %2d  loss %.4f  %.1fs training, %.1fs sampling\n",
			epoch, sum/float64(batches), trained.Seconds(), (time.Since(start) - trained).Seconds())
	}
	fmt.Printf("wrote %s\n", path)
	if o.ddim > 0 {
		fmt.Printf("wrote %s\n", ddimPath)
	}
	return nil
}
