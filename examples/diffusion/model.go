package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

var errConfig = errors.New("diffusion: invalid configuration")

// Config sets the sizes of a DDPM.
type Config struct {
	Pixels int // pixels an image, each in [-1, 1]
	Hidden int // units of each hidden layer of the denoiser
	Time   int // dimensions of the sinusoidal embedding of the step; even
	Steps  int // diffusion steps, T
}

// DDPM is the denoising diffusion probabilistic model of Ho et al.: a
// fixed forward process that adds Gaussian noise to an image over Steps
// steps, and a network trained to predict the noise in an image at any
// step, which runs the process backwards from pure noise to an image.
//
// The network is a multilayer perceptron with residual connections, and
// a sinusoidal embedding of the step that a small perceptron of its own
// maps to a vector added to the input of every hidden layer. It predicts
// the velocity v = √ᾱ_t·ε - √(1-ᾱ_t)·x₀ of Salimans and Ho rather than the
// noise ε itself: the noise then follows as √(1-ᾱ_t)·x_t + √ᾱ_t·v, which
// is nearly the identity in x_t at the early steps, where a perceptron
// without the skip connections of a U-Net predicts ε poorly and leaves
// the samples speckled.
type DDPM struct {
	Config Config

	Time1, Time2 *nn.Linear // step embedding to hidden to the vector added to each layer
	In           *nn.Linear // noisy pixels to hidden
	Res1, Res2   *nn.Linear // hidden to hidden, added to its input
	Out          *nn.Linear // hidden to predicted noise
	schedule     schedule
}

// schedule holds the forward process's noise variances β_t, linear in t
// from 1e-4 to 0.02 at T = 1000 and scaled by 1000/T for other T as in
// Nichol and Dhariwal, and quantities derived from them.
type schedule struct {
	beta, alpha, alphaBar []float64 // β_t, α_t = 1 - β_t and ᾱ_t = α_1⋯α_t
}

func newSchedule(steps int) schedule {
	s := schedule{beta: make([]float64, steps), alpha: make([]float64, steps), alphaBar: make([]float64, steps)}
	scale := 1000 / float64(steps)
	lo, hi := 1e-4*scale, min(0.02*scale, 0.999)
	prod := 1.0
	for t := range steps {
		b := lo
		if steps > 1 {
			b += (hi - lo) * float64(t) / float64(steps-1)
		}
		prod *= 1 - b
		s.beta[t], s.alpha[t], s.alphaBar[t] = b, 1-b, prod
	}
	return s
}

// NewDDPM returns a DDPM initialized from seed.
func NewDDPM(cfg Config, seed uint64) (*DDPM, error) {
	if cfg.Pixels <= 0 || cfg.Hidden <= 0 || cfg.Time <= 0 || cfg.Time%2 != 0 || cfg.Steps <= 0 {
		return nil, fmt.Errorf("diffusion: configuration %+v: %w", cfg, errConfig)
	}
	nn.ManualSeed(seed)
	return &DDPM{
		Config:   cfg,
		Time1:    nn.NewLinear(cfg.Time, cfg.Hidden),
		Time2:    nn.NewLinear(cfg.Hidden, cfg.Hidden),
		In:       nn.NewLinear(cfg.Pixels, cfg.Hidden),
		Res1:     nn.NewLinear(cfg.Hidden, cfg.Hidden),
		Res2:     nn.NewLinear(cfg.Hidden, cfg.Hidden),
		Out:      nn.NewLinear(cfg.Hidden, cfg.Pixels),
		schedule: newSchedule(cfg.Steps),
	}, nil
}

// Forward maps noisy images beside the embeddings of their steps,
// [batch, pixels+time], to the noise predicted in them, [batch, pixels].
func (m *DDPM) Forward(in *tensor.Tensor) *tensor.Tensor {
	n, p := in.Shape()[0], m.Config.Pixels
	x := in.Shrink([][2]int{{0, n}, {0, p}})
	emb := in.Shrink([][2]int{{0, n}, {p, p + m.Config.Time}})
	step := m.Time2.Forward(m.Time1.Forward(emb).SiLU())
	h := m.In.Forward(x).Add(step).SiLU()
	h = h.Add(m.Res1.Forward(h.Add(step)).SiLU())
	h = h.Add(m.Res2.Forward(h.Add(step)).SiLU())
	return m.Out.Forward(h)
}

// Predict returns the noise predicted in the images x at steps ts, one a
// row of x, counted from 0.
func (m *DDPM) Predict(x *tensor.Tensor, ts []int) *tensor.Tensor {
	signal, spread := m.coefficients(ts)
	return x.Mul(spread).Add(m.velocity(x, ts).Mul(signal))
}

// velocity returns the velocity the network predicts for the images x at
// steps ts.
func (m *DDPM) velocity(x *tensor.Tensor, ts []int) *tensor.Tensor {
	return m.Forward(tensor.Cat(1, x, m.embed(ts)))
}

// coefficients returns √ᾱ_t and √(1-ᾱ_t) for the steps ts, each a
// [len(ts), 1] tensor: the scales of the image and of the noise in x_t.
func (m *DDPM) coefficients(ts []int) (signal, spread *tensor.Tensor) {
	ab := m.schedule.alphaBar
	signal = m.column(ts, func(t int) float64 { return math.Sqrt(ab[t]) })
	spread = m.column(ts, func(t int) float64 { return math.Sqrt(1 - ab[t]) })
	return signal, spread
}

// embed returns the sinusoidal embeddings of steps ts, [len(ts), Time]:
// sines and cosines of the step at frequencies from 1 down to 1/10000.
func (m *DDPM) embed(ts []int) *tensor.Tensor {
	d := m.Config.Time
	half := d / 2
	e := make([]float32, len(ts)*d)
	for i, t := range ts {
		for k := range half {
			f := math.Pow(10000, -float64(k)/float64(max(half-1, 1)))
			s, c := math.Sincos(float64(t) * f)
			e[i*d+k], e[i*d+half+k] = float32(s), float32(c)
		}
	}
	return tensor.FromSlice(e, len(ts), d).To(m.device())
}

func (m *DDPM) device() tensor.Device { return m.In.Weight.Value().Device() }

// column returns f(t) for each step of ts as a [len(ts), 1] tensor.
func (m *DDPM) column(ts []int, f func(t int) float64) *tensor.Tensor {
	v := make([]float32, len(ts))
	for i, t := range ts {
		v[i] = float32(f(t))
	}
	return tensor.FromSlice(v, len(ts), 1).To(m.device())
}

// Loss returns the mean squared error of the velocity predicted for the
// images x, [batch, pixels], noised to the steps ts with standard normal
// noise drawn from seed. It is the simplified objective of Ho et al.
// weighted by 1 + √(ᾱ_t/(1-ᾱ_t)), the signal-to-noise ratio plus one.
func (m *DDPM) Loss(x *tensor.Tensor, ts []int, seed uint64) *tensor.Tensor {
	noise := tensor.Randn(seed, x.Shape()...).To(m.device())
	signal, spread := m.coefficients(ts)
	noisy := x.Mul(signal).Add(noise.Mul(spread))
	v := noise.Mul(signal).Sub(x.Mul(spread))
	return loss.MSE(m.velocity(noisy, ts), v)
}

// Sample runs the reverse process from standard normal noise to n
// images, [n, pixels], with ancestral sampling, drawing its noise from
// seed. Each step estimates the clean image from the predicted noise,
// clips it to [-1, 1] and draws from the forward process's posterior
// given it, q(x_{t-1} | x_t, x_0), as the reference implementation does;
// without the clipping the errors of a briefly trained denoiser grow
// through the steps.
func (m *DDPM) Sample(ctx context.Context, n int, seed uint64) (*tensor.Tensor, error) {
	return m.sample(ctx, n, seed, m.Predict)
}

// sample runs the reverse process with the noise predictions of predict.
func (m *DDPM) sample(ctx context.Context, n int, seed uint64, predict func(x *tensor.Tensor, ts []int) *tensor.Tensor) (*tensor.Tensor, error) {
	s := m.schedule
	f := func(v float64) *tensor.Tensor { return tensor.Scalar(float32(v)) }
	lo, hi := f(-1), f(1)
	x := tensor.Randn(seed, n, m.Config.Pixels).To(m.device())
	ts := make([]int, n)
	for t := m.Config.Steps - 1; t >= 0; t-- {
		for i := range ts {
			ts[i] = t
		}
		eps := predict(x, ts).Detach()
		ab, prev := s.alphaBar[t], 1.0
		if t > 0 {
			prev = s.alphaBar[t-1]
		}
		// x₀ = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t, and the posterior's mean is
		// β_t·√ᾱ_{t-1}/(1-ᾱ_t)·x₀ + (1-ᾱ_{t-1})·√α_t/(1-ᾱ_t)·x_t.
		x0 := x.Sub(eps.Mul(f(math.Sqrt(1 - ab)))).Mul(f(1 / math.Sqrt(ab))).Maximum(lo).Minimum(hi)
		x = x0.Mul(f(s.beta[t] * math.Sqrt(prev) / (1 - ab))).Add(x.Mul(f((1 - prev) * math.Sqrt(s.alpha[t]) / (1 - ab))))
		if t > 0 {
			z := tensor.Randn(mix(seed, t), n, m.Config.Pixels).To(m.device())
			x = x.Add(z.Mul(f(math.Sqrt(s.beta[t] * (1 - prev) / (1 - ab)))))
		}
		if err := x.Realize(ctx); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// SampleDDIM draws n images, [n, pixels], with the deterministic sampler
// of Song et al. (DDIM with η = 0) over every stride-th step, starting
// from standard normal noise drawn from seed. It needs stride times fewer
// network evaluations than Sample, and adds no noise along the way.
func (m *DDPM) SampleDDIM(ctx context.Context, n int, seed uint64, stride int) (*tensor.Tensor, error) {
	return m.sampleDDIM(ctx, n, seed, stride, m.Predict)
}

func (m *DDPM) sampleDDIM(ctx context.Context, n int, seed uint64, stride int, predict func(x *tensor.Tensor, ts []int) *tensor.Tensor) (*tensor.Tensor, error) {
	if stride <= 0 {
		return nil, fmt.Errorf("diffusion: DDIM stride %d: %w", stride, errConfig)
	}
	s := m.schedule
	f := func(v float64) *tensor.Tensor { return tensor.Scalar(float32(v)) }
	lo, hi := f(-1), f(1)
	x := tensor.Randn(seed, n, m.Config.Pixels).To(m.device())
	ts := make([]int, n)
	for t := m.Config.Steps - 1; t >= 0; t -= stride {
		for i := range ts {
			ts[i] = t
		}
		eps := predict(x, ts).Detach()
		ab, prev := s.alphaBar[t], 1.0
		if t-stride >= 0 {
			prev = s.alphaBar[t-stride]
		}
		// The clean image the noise implies, clipped, and the noise that
		// takes it back to x_t; then the same noise at the earlier step.
		x0 := x.Sub(eps.Mul(f(math.Sqrt(1 - ab)))).Mul(f(1 / math.Sqrt(ab))).Maximum(lo).Minimum(hi)
		eps = x.Sub(x0.Mul(f(math.Sqrt(ab)))).Mul(f(1 / math.Sqrt(1-ab)))
		x = x0.Mul(f(math.Sqrt(prev))).Add(eps.Mul(f(math.Sqrt(1 - prev))))
		if err := x.Realize(ctx); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// Trainer fits a DDPM with Adam and keeps an exponential moving average
// of its weights, which samples better than the weights of any one step,
// as in Ho et al.
type Trainer struct {
	Model *DDPM
	EMA   *DDPM // the moving average of Model's weights
	Opt   *optim.Optimizer
	seed  uint64
}

// emaDecay is the largest decay of the moving average per step; early
// steps decay it faster, (1+n)/(10+n) after n steps, so the average soon
// forgets the initial weights.
const emaDecay = 0.999

// NewTrainer returns a trainer of m with learning rate lr whose steps and
// noise derive from seed. It copies m's weights to start the average.
func NewTrainer(ctx context.Context, m *DDPM, lr float64, seed uint64) (*Trainer, error) {
	ema := &DDPM{
		Config: m.Config, Time1: copyLinear(m.Time1), Time2: copyLinear(m.Time2),
		In: copyLinear(m.In), Res1: copyLinear(m.Res1), Res2: copyLinear(m.Res2), Out: copyLinear(m.Out),
		schedule: m.schedule,
	}
	var copies []*tensor.Tensor
	for _, p := range nn.NamedParams(ema) {
		copies = append(copies, p.Value())
	}
	if err := tensor.Realize(ctx, copies...); err != nil {
		return nil, err
	}
	return &Trainer{Model: m, EMA: ema, Opt: optim.Adam(nn.Parameters(m), optim.LR(lr)), seed: seed}, nil
}

// copyLinear returns a layer with copies of l's weights, which no
// optimizer trains.
func copyLinear(l *nn.Linear) *nn.Linear {
	return &nn.Linear{Weight: tensor.NewBuffer(l.Weight.Value().Detach().Clone()), Bias: tensor.NewBuffer(l.Bias.Value().Detach().Clone())}
}

// Step takes one optimizer step on the images x, [batch, pixels], each
// noised to a step drawn uniformly, updates the moving average, and
// returns the loss before the step.
func (t *Trainer) Step(ctx context.Context, x *tensor.Tensor) (float64, error) {
	defer t.Opt.ZeroGrad()
	seed := mix(t.seed, t.Opt.Steps())
	r := rand.New(rand.NewPCG(seed, 0))
	ts := make([]int, x.Shape()[0])
	for i := range ts {
		ts[i] = r.IntN(t.Model.Config.Steps)
	}
	l := t.Model.Loss(x, ts, seed)
	if err := l.Backward(ctx); err != nil {
		return 0, err
	}
	v, err := l.Item[float32](ctx) // computed by Backward
	if err != nil {
		return 0, err
	}
	if err := t.Opt.Step(ctx); err != nil {
		return 0, err
	}
	n := float64(t.Opt.Steps())
	d := min(emaDecay, (1+n)/(10+n))
	keep, take := tensor.Scalar(float32(d)), tensor.Scalar(float32(1-d))
	var avg []*tensor.Param
	for _, e := range nn.NamedParams(t.EMA) {
		avg = append(avg, e)
	}
	dsts, srcs := make([]*tensor.Tensor, len(avg)), make([]*tensor.Tensor, len(avg))
	for i, p := range nn.Parameters(t.Model) {
		dsts[i] = avg[i].Value()
		srcs[i] = dsts[i].Mul(keep).Add(p.Value().Detach().Mul(take))
	}
	// Like an optimizer step, write the new averages over the old.
	if err := tensor.RealizeInto(ctx, dsts, srcs); err != nil {
		return 0, err
	}
	for i, e := range avg {
		if err := e.Set(srcs[i]); err != nil {
			return 0, err
		}
	}
	return float64(v), nil
}

// mix returns a seed for step that is unrelated to those of other steps.
func mix(seed uint64, step int) uint64 {
	z := seed + uint64(step+1)*0x9e3779b97f4a7c15
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}
