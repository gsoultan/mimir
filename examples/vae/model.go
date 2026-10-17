package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/gsoultan/mimir/distributions"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

var errConfig = errors.New("vae: invalid configuration")

// Config sets the sizes of a VAE.
type Config struct {
	Pixels int // pixels an image, each in [0, 1]
	Hidden int // units of the encoder's and the decoder's hidden layer
	Latent int // dimensions of the code
}

// VAE is the variational autoencoder of Kingma and Welling: an encoder
// from an image to a diagonal normal posterior over codes, and a decoder
// from a code to independent Bernoulli pixels, under a standard normal
// prior.
type VAE struct {
	Config Config

	Enc        *nn.Linear // pixels to hidden
	Loc, Scale *nn.Linear // hidden to the posterior's mean and log standard deviation
	Dec, Out   *nn.Linear // code to hidden to pixel log-odds
}

// NewVAE returns a VAE initialized from seed.
func NewVAE(cfg Config, seed uint64) (*VAE, error) {
	if cfg.Pixels <= 0 || cfg.Hidden <= 0 || cfg.Latent <= 0 {
		return nil, fmt.Errorf("vae: configuration %+v: %w", cfg, errConfig)
	}
	nn.ManualSeed(seed)
	return &VAE{
		Config: cfg,
		Enc:    nn.NewLinear(cfg.Pixels, cfg.Hidden),
		Loc:    nn.NewLinear(cfg.Hidden, cfg.Latent),
		Scale:  nn.NewLinear(cfg.Hidden, cfg.Latent),
		Dec:    nn.NewLinear(cfg.Latent, cfg.Hidden),
		Out:    nn.NewLinear(cfg.Hidden, cfg.Pixels),
	}, nil
}

// Encode returns the posterior q(z | x) of images x, [batch, pixels].
func (m *VAE) Encode(x *tensor.Tensor) *distributions.Normal {
	h := m.Enc.Forward(x).ReLU()
	return distributions.NewNormal(m.Loc.Forward(h), m.Scale.Forward(h).Exp())
}

// Decode returns the pixel log-odds of p(x | z) for codes z,
// [batch, latent].
func (m *VAE) Decode(z *tensor.Tensor) *tensor.Tensor {
	return m.Out.Forward(m.Dec.Forward(z).ReLU())
}

// Forward returns the pixel probabilities that x decodes to from the mean
// of its posterior.
func (m *VAE) Forward(x *tensor.Tensor) *tensor.Tensor {
	return m.Decode(m.Encode(x).Loc).Sigmoid()
}

// ELBO returns the evidence lower bound of each image of x,
// E_q[log p(x | z)] - KL(q(z | x) ‖ p(z)), estimated from one code drawn
// from the posterior with the reparameterization trick, so that its
// gradient reaches the encoder.
func (m *VAE) ELBO(x *tensor.Tensor, seed uint64) *tensor.Tensor {
	q := m.Encode(x)
	z := q.RSample(seed)
	logits := m.Decode(z)
	recon := distributions.NewBernoulliLogits(logits).LogProb(x).Sum(-1)
	prior := distributions.NewNormal(tensor.Scalar[float32](0), tensor.Scalar[float32](1))
	return recon.Sub(distributions.KL(q, prior).Sum(-1))
}

// Generate decodes n codes drawn from the prior into pixel probabilities,
// [n, pixels].
func (m *VAE) Generate(n int, seed uint64) *tensor.Tensor {
	z := tensor.Randn(seed, n, m.Config.Latent).To(m.Enc.Weight.Value().Device())
	return m.Decode(z).Sigmoid().Detach()
}

// Trainer fits a VAE by maximizing the ELBO with Adam.
type Trainer struct {
	Model *VAE
	Opt   *optim.Optimizer
	seed  uint64
}

// NewTrainer returns a trainer of m with learning rate lr whose posterior
// samples derive from seed.
func NewTrainer(m *VAE, lr float64, seed uint64) *Trainer {
	return &Trainer{Model: m, Opt: optim.Adam(nn.Parameters(m), optim.LR(lr)), seed: seed}
}

// Step takes one optimizer step on the images x, [batch, pixels], and
// returns their mean ELBO before it.
func (t *Trainer) Step(ctx context.Context, x *tensor.Tensor) (float64, error) {
	defer t.Opt.ZeroGrad()
	elbo := t.Model.ELBO(x, mix(t.seed, t.Opt.Steps())).Mean()
	if err := elbo.Neg().Backward(ctx); err != nil {
		return 0, err
	}
	v, err := elbo.Item[float32](ctx)
	if err != nil {
		return 0, err
	}
	return float64(v), t.Opt.Step(ctx)
}

// Evaluate returns the mean ELBO of the images x, with posterior samples
// that do not change between calls.
func (t *Trainer) Evaluate(ctx context.Context, x *tensor.Tensor) (float64, error) {
	v, err := t.Model.ELBO(x, ^t.seed).Mean().Detach().Item[float32](ctx)
	return float64(v), err
}

// mix returns a seed for step that is unrelated to those of other steps.
func mix(seed uint64, step int) uint64 {
	z := seed + uint64(step+1)*0x9e3779b97f4a7c15
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}
