package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

var errConfig = errors.New("gan: invalid configuration")

// Config sets the sizes of a GAN.
type Config struct {
	Pixels int // pixels an image, each in [-1, 1]
	Noise  int // dimensions of the generator's input noise
	Hidden int // units of the narrower hidden layer; the wider has twice as many
}

// Generator maps noise to images through two hidden layers of leaky
// ReLUs and a tanh.
type Generator struct {
	In, Mid, Out *nn.Linear
}

// Forward maps noise, [batch, noise], to images, [batch, pixels].
func (g *Generator) Forward(z *tensor.Tensor) *tensor.Tensor {
	h := leaky(g.In.Forward(z))
	h = leaky(g.Mid.Forward(h))
	return g.Out.Forward(h).Tanh()
}

// Discriminator maps images to the log-odds that they are real.
type Discriminator struct {
	In, Mid, Out *nn.Linear
}

// Forward maps images, [batch, pixels], to log-odds, [batch, 1].
func (d *Discriminator) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := leaky(d.In.Forward(x))
	h = leaky(d.Mid.Forward(h))
	return d.Out.Forward(h)
}

// leaky is the leaky ReLU with slope 0.2 below zero that GANs favor.
func leaky(x *tensor.Tensor) *tensor.Tensor { return x.Maximum(x.Mul(tensor.Scalar[float32](0.2))) }

// GAN is the generative adversarial network of Goodfellow et al. with
// multilayer perceptrons for its generator and discriminator.
type GAN struct {
	Config Config
	G      *Generator
	D      *Discriminator
}

// NewGAN returns a GAN initialized from seed.
func NewGAN(cfg Config, seed uint64) (*GAN, error) {
	if cfg.Pixels <= 0 || cfg.Noise <= 0 || cfg.Hidden <= 0 {
		return nil, fmt.Errorf("gan: configuration %+v: %w", cfg, errConfig)
	}
	nn.ManualSeed(seed)
	h := cfg.Hidden
	return &GAN{
		Config: cfg,
		G:      &Generator{In: nn.NewLinear(cfg.Noise, h), Mid: nn.NewLinear(h, 2*h), Out: nn.NewLinear(2*h, cfg.Pixels)},
		D:      &Discriminator{In: nn.NewLinear(cfg.Pixels, 2*h), Mid: nn.NewLinear(2*h, h), Out: nn.NewLinear(h, 1)},
	}, nil
}

// Forward maps noise to images with the generator.
func (m *GAN) Forward(z *tensor.Tensor) *tensor.Tensor { return m.G.Forward(z) }

// Generate returns n images, [n, pixels], from noise drawn with seed.
func (m *GAN) Generate(n int, seed uint64) *tensor.Tensor {
	return m.G.Forward(m.noise(n, seed)).Detach()
}

// noise returns n standard normal inputs to the generator.
func (m *GAN) noise(n int, seed uint64) *tensor.Tensor {
	return tensor.Randn(seed, n, m.Config.Noise).To(m.G.In.Weight.Value().Device())
}

// Trainer alternates steps of the discriminator and the generator with
// Adam, as in DCGAN: learning rate 2e-4 and β₁ = 0.5.
type Trainer struct {
	Model      *GAN
	OptG, OptD *optim.Optimizer
	seed       uint64
}

// NewTrainer returns a trainer of m with learning rate lr whose noise
// derives from seed.
func NewTrainer(m *GAN, lr float64, seed uint64) *Trainer {
	opts := []optim.Option{optim.LR(lr), optim.Betas(0.5, 0.999)}
	return &Trainer{
		Model: m,
		OptG:  optim.Adam(nn.Parameters(m.G), opts...),
		OptD:  optim.Adam(nn.Parameters(m.D), opts...),
		seed:  seed,
	}
}

// Losses are the losses of one training step.
type Losses struct {
	D float64 // the discriminator's binary cross-entropy on real and generated images, summed
	G float64 // the generator's non-saturating loss, -log D(G(z))
}

// Step trains the discriminator on the real images x, [batch, pixels],
// and as many generated ones, then the generator on a fresh batch of
// noise, and returns their losses before the steps.
func (t *Trainer) Step(ctx context.Context, x *tensor.Tensor) (Losses, error) {
	defer t.OptD.ZeroGrad()
	defer t.OptG.ZeroGrad()
	m, n := t.Model, x.Shape()[0]
	seed := mix(t.seed, t.OptD.Steps())
	dev := x.Device()
	real, fake := tensor.Ones(tensor.Float32, n, 1).To(dev), tensor.Zeros(tensor.Float32, n, 1).To(dev)

	generated := m.G.Forward(m.noise(n, seed)).Detach()
	dl := loss.BCEWithLogits(m.D.Forward(x), real).Add(loss.BCEWithLogits(m.D.Forward(generated), fake))
	if err := dl.Backward(ctx); err != nil {
		return Losses{}, err
	}
	d, err := dl.Item[float32](ctx) // computed by Backward
	if err != nil {
		return Losses{}, err
	}
	if err := t.OptD.Step(ctx); err != nil {
		return Losses{}, err
	}
	t.OptD.ZeroGrad()

	gl := loss.BCEWithLogits(m.D.Forward(m.G.Forward(m.noise(n, ^seed))), real)
	if err := gl.Backward(ctx); err != nil {
		return Losses{}, err
	}
	g, err := gl.Item[float32](ctx)
	if err != nil {
		return Losses{}, err
	}
	return Losses{D: float64(d), G: float64(g)}, t.OptG.Step(ctx)
}

// mix returns a seed for step that is unrelated to those of other steps.
func mix(seed uint64, step int) uint64 {
	z := seed + uint64(step+1)*0x9e3779b97f4a7c15
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}
