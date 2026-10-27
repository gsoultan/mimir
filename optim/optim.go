// Package optim updates parameters from their gradients, like torch.optim.
//
//	opt := optim.AdamW(nn.Parameters(model), optim.LR(1e-3), optim.WeightDecay(0.01))
//	for batch, err := range trainSet.All(ctx) {
//		...
//		if err := l.Backward(ctx); err != nil { ... }
//		if err := opt.Step(ctx); err != nil { ... }
//		opt.ZeroGrad()
//	}
//
// Step computes every parameter's new value and optimizer state in one
// schedule, writing them over the old ones in place like PyTorch, so
// parameters always hold computed values, never a growing recipe, and a
// training step reuses the same memory every time. A tensor taken from a
// parameter or the optimizer state before a step is stale after it:
// reading it, or computing a tensor built from it, returns
// tensor.ErrOverwritten.
package optim

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/gsoultan/mimir/tensor"
)

// State is one parameter's optimizer state, such as Adam's moment
// estimates, by name.
type State map[string]*tensor.Tensor

// Rule computes one step for one parameter. From the parameter's value p,
// its gradient g and its state after the previous step, which is empty
// before the first, it returns the new value and state. step counts from
// 1 and lr is the current learning rate. ZeroGrad releases g's memory, so
// a state that keeps g's values holds a copy, such as g.Clone(); Step
// copies a state that is g itself.
type Rule func(lr float64, step int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State)

// Optimizer applies a Rule to every trainable parameter that has a
// gradient. It is not safe for concurrent use.
type Optimizer struct {
	params []*tensor.Param
	rule   Rule
	lr     float64
	scale  float64 // the factor the gradients carry, which Step divides out
	steps  int
	states []State
	clip   float64 // the most the gradients' norm may be, or 0
	err    error   // a configuration error, reported by Step
}

// New returns an optimizer that updates params with rule at learning rate
// lr. SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax, NAdam, RAdam
// and Lion are built on it.
func New(params []*tensor.Param, lr float64, rule Rule) *Optimizer {
	o := &Optimizer{params: params, rule: rule, lr: lr, scale: 1, states: make([]State, len(params))}
	if lr < 0 || math.IsNaN(lr) {
		o.err = fmt.Errorf("optim: learning rate %v: %w", lr, tensor.ErrShape)
	}
	return o
}

// Params returns the parameters the optimizer updates, in order.
func (o *Optimizer) Params() []*tensor.Param { return slices.Clone(o.params) }

// Over returns an optimizer that updates params with o's rule, learning
// rate and gradient scale, and no state: one for a replica of o's
// parameters, such as train.Parallel steps alongside o, each over its
// own device's copies. params[i] stands for o's i-th parameter. Restore
// gives it o's state.
func (o *Optimizer) Over(params []*tensor.Param) *Optimizer {
	c := New(params, o.lr, o.rule)
	c.scale, c.clip, c.err = o.scale, o.clip, cmp.Or(c.err, o.err)
	if len(params) != len(o.params) {
		c.err = fmt.Errorf("optim: %d parameters for an optimizer of %d: %w", len(params), len(o.params), tensor.ErrShape)
	}
	return c
}

// LR returns the learning rate.
func (o *Optimizer) LR() float64 { return o.lr }

// SetLR sets the learning rate for later steps, as a Schedule does.
func (o *Optimizer) SetLR(lr float64) { o.lr = lr }

// GradScale returns the factor that the gradients carry; see SetGradScale.
func (o *Optimizer) GradScale() float64 { return o.scale }

// SetGradScale tells the optimizer that the gradients carry scale times
// the gradient of the loss, as they do when the loss is multiplied by
// scale before Backward to keep small gradients from underflowing (see
// train.LossScale). Step divides each gradient by scale as it reads it,
// folded into the rule's input, so the division fuses with the update
// and costs no memory. The default is 1. A scale whose reciprocal is not
// a positive, finite float32 is a configuration error that Step reports.
func (o *Optimizer) SetGradScale(scale float64) { o.scale = scale }

// ClipNorm returns the most the gradients' global norm may be, 0 for no
// limit; see SetClipNorm.
func (o *Optimizer) ClipNorm() float64 { return o.clip }

// SetClipNorm limits the global L2 norm of the gradients, all parameters'
// together, to n before each step, scaling them down alike when it is
// larger, like torch.nn.utils.clip_grad_norm_: the guard against the one
// huge step that throws away what a network learned, which RNNs,
// transformers and reinforcement learning want. The norm is computed in
// the step's own schedule, so it costs no extra pass or host read. 0, the
// default, turns it off.
func (o *Optimizer) SetClipNorm(n float64) { o.clip = n }

// GradsFinite reports whether every gradient that Step would use is
// finite. It sums the gradients in one schedule and tests the sum on the
// host: the sum is NaN or infinite exactly when a gradient is, or when
// the gradients are so large that their sum overflows float32, which is
// as good as infinite. train.DynamicLossScale skips such steps.
func (o *Optimizer) GradsFinite(ctx context.Context) (bool, error) {
	var total *tensor.Tensor
	for _, p := range o.params {
		g := p.Grad()
		if g == nil || !p.Trainable() {
			continue
		}
		if total == nil {
			total = g.Sum()
		} else {
			total = total.Add(g.Sum())
		}
	}
	if total == nil {
		return true, nil
	}
	v, err := total.Item[float32](ctx)
	if err != nil {
		return false, err
	}
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0), nil
}

// Steps returns the number of steps taken.
func (o *Optimizer) Steps() int { return o.steps }

// ZeroGrad discards every parameter's gradient.
func (o *Optimizer) ZeroGrad() {
	for _, p := range o.params {
		p.ZeroGrad()
	}
}

// Step updates every trainable parameter that has a gradient, divided by
// the gradient scale (see SetGradScale), writing the new values and
// states over the old ones (see tensor.RealizeInto). The rule's value is
// cast to the parameter's dtype, so a BFloat16 or Float16 parameter takes
// an update computed in float32 rounded to its dtype (see MasterWeights).
// If Step fails before it starts writing, nothing changes; cancelling ctx
// after that does not stop it.
func (o *Optimizer) Step(ctx context.Context) error {
	if o.err != nil {
		return o.err
	}
	if inv := float32(1 / o.scale); !(o.scale > 0) || !(inv > 0) || math.IsInf(float64(inv), 0) {
		return fmt.Errorf("optim: gradient scale %v: %w", o.scale, tensor.ErrShape)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	type update struct {
		i     int
		value *tensor.Tensor
		state State
	}
	if o.clip < 0 || math.IsNaN(o.clip) {
		return fmt.Errorf("optim: clip norm %v: %w", o.clip, tensor.ErrShape)
	}
	step := o.steps + 1
	var updates []update
	var targets, dsts []*tensor.Tensor
	var factor *tensor.Tensor // the clipping's scale, when clipping
	if o.clip > 0 {
		var sq *tensor.Tensor
		for _, p := range o.params {
			if g := p.Grad(); g != nil && p.Trainable() {
				g = g.Cast(tensor.Float32)
				s := g.Mul(g).Sum()
				if sq == nil {
					sq = s
				} else {
					sq = sq.Add(s)
				}
			}
		}
		if sq != nil {
			norm := sq.Sqrt().Mul(f32(1 / o.scale))
			factor = f32(o.clip).Div(norm.Add(f32(1e-6))).Minimum(f32(1))
		}
	}
	for i, p := range o.params {
		g := p.Grad()
		if g == nil || !p.Trainable() {
			continue
		}
		if o.scale != 1 {
			g = g.Mul(f32(1 / o.scale))
		}
		if factor != nil {
			g = g.Mul(factor.Cast(g.DType()))
		}
		cur := p.Value().Detach()
		v, s := o.rule(o.lr, step, cur, g, o.states[i])
		if err := v.Err(); err != nil {
			return fmt.Errorf("optim: parameter %d: %w", i, err)
		}
		if !slices.Equal(v.Shape(), cur.Shape()) {
			return fmt.Errorf("optim: parameter %d: rule returned %v for %v: %w", i, v.Shape(), cur.Shape(), tensor.ErrShape)
		}
		if v.DType() != cur.DType() {
			v = v.Cast(cur.DType())
		}
		for name, t := range s {
			if t == g {
				s[name] = g.Clone()
			}
		}
		updates = append(updates, update{i, v, s})
		targets, dsts = append(targets, v), append(dsts, cur)
		// In the order of their names, not the map's, so that each step
		// schedules the same graph and replays its program.
		for _, name := range slices.Sorted(maps.Keys(s)) {
			targets, dsts = append(targets, s[name]), append(dsts, o.states[i][name])
		}
	}
	if err := tensor.RealizeInto(ctx, dsts, targets); err != nil {
		return err
	}
	for _, u := range updates {
		if err := o.params[u.i].Set(u.value); err != nil {
			return fmt.Errorf("optim: parameter %d: %w", u.i, err)
		}
		o.states[u.i] = u.state
	}
	o.steps = step
	return nil
}

// Snapshot returns the state of every parameter, named "<index>.<name>"
// by the parameter's position, and the number of steps taken, for
// checkpoints. The tensors are the optimizer's own, which the next step
// writes over: encode them, or Clone them and Realize the clones, before
// it.
func (o *Optimizer) Snapshot() (map[string]*tensor.Tensor, int) {
	out := map[string]*tensor.Tensor{}
	for i, s := range o.states {
		for name, t := range s {
			out[strconv.Itoa(i)+"."+name] = t
		}
	}
	return out, o.steps
}

// Restore replaces the optimizer's state with a copy of one from Snapshot,
// taken from an optimizer of the same kind over parameters of the same
// shapes.
func (o *Optimizer) Restore(state map[string]*tensor.Tensor, steps int) error {
	states := make([]State, len(o.params))
	var copies []*tensor.Tensor
	for _, key := range slices.SortedFunc(maps.Keys(state), cmp.Compare) {
		idx, name, ok := strings.Cut(key, ".")
		i, err := strconv.Atoi(idx)
		if !ok || err != nil || i < 0 || i >= len(o.params) {
			return fmt.Errorf("optim: restore state %q for %d parameters: %w", key, len(o.params), tensor.ErrShape)
		}
		if want := o.params[i].Value().Shape(); !slices.Equal(state[key].Shape(), want) {
			return fmt.Errorf("optim: restore state %q of shape %v for parameter %v: %w", key, state[key].Shape(), want, tensor.ErrShape)
		}
		if states[i] == nil {
			states[i] = State{}
		}
		c := state[key].Clone()
		states[i][name] = c
		copies = append(copies, c)
	}
	if err := tensor.Realize(context.Background(), copies...); err != nil {
		return fmt.Errorf("optim: restore: %w", err)
	}
	o.states, o.steps = states, steps
	return nil
}

// Option configures an optimizer. Options an optimizer does not use are
// ignored.
type Option func(*config)

type config struct {
	lr, momentum, dampening, weightDecay float64
	beta1, beta2, eps, alpha             float64
	nesterov, master                     bool
	clip                                 float64
	lrDecay, initial, rho                float64
	momentumDecay                        float64
}

// LR sets the learning rate.
func LR(lr float64) Option { return func(c *config) { c.lr = lr } }

// Momentum sets SGD's or RMSprop's momentum factor.
func Momentum(m float64) Option { return func(c *config) { c.momentum = m } }

// Dampening scales the gradient SGD adds to its momentum by 1-d.
func Dampening(d float64) Option { return func(c *config) { c.dampening = d } }

// Nesterov makes SGD use Nesterov momentum.
func Nesterov() Option { return func(c *config) { c.nesterov = true } }

// WeightDecay sets the weight decay: an L2 penalty added to the gradient
// for SGD, Adam and RMSprop, and a decoupled shrinking of the weights for
// AdamW.
func WeightDecay(wd float64) Option { return func(c *config) { c.weightDecay = wd } }

// Betas sets Adam's decay rates for its first and second moment estimates.
func Betas(b1, b2 float64) Option { return func(c *config) { c.beta1, c.beta2 = b1, b2 } }

// Eps sets the term added to denominators for numerical stability.
func Eps(eps float64) Option { return func(c *config) { c.eps = eps } }

// Alpha sets RMSprop's smoothing constant.
func Alpha(a float64) Option { return func(c *config) { c.alpha = a } }

// ClipNorm limits the gradients' global norm before each step; see
// Optimizer.SetClipNorm.
func ClipNorm(n float64) Option { return func(c *config) { c.clip = n } }

// LRDecay sets Adagrad's learning rate decay: step t uses lr/(1+(t-1)·d).
func LRDecay(d float64) Option { return func(c *config) { c.lrDecay = d } }

// InitialAccumulator sets the value Adagrad's sum of squared gradients
// starts at.
func InitialAccumulator(v float64) Option { return func(c *config) { c.initial = v } }

// Rho sets Adadelta's decay rate for its running averages.
func Rho(r float64) Option { return func(c *config) { c.rho = r } }

// MomentumDecay sets NAdam's momentum decay, ψ.
func MomentumDecay(d float64) Option { return func(c *config) { c.momentumDecay = d } }

// MasterWeights keeps a float32 master copy of every BFloat16 and Float16
// parameter in its state, named "master", and updates that: the rule runs
// on the master copy with the gradient cast to float32, and the parameter
// takes the result rounded to its dtype. Without it a half parameter is
// updated in its own precision, where an update smaller than half the
// spacing of its values is lost: in BFloat16 a step of less than 1/512 of
// a weight leaves it unchanged. Float32 parameters are updated as usual.
// The master copy is saved and restored with the rest of the state.
func MasterWeights() Option { return func(c *config) { c.master = true } }

// master returns rule applied to the float32 master copy of each half
// parameter, kept in the parameter's state as "master".
func master(rule Rule) Rule {
	return func(lr float64, step int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
		if p.DType() == tensor.Float32 {
			return rule(lr, step, p, g, s)
		}
		m, ok := s["master"]
		if !ok {
			m = p.Cast(tensor.Float32)
		}
		v, next := rule(lr, step, m, g.Cast(tensor.Float32), s)
		if next == nil {
			next = State{}
		}
		next["master"] = v
		return v.Cast(p.DType()), next
	}
}

func configure(opts []Option, c config) (config, error) {
	for _, o := range opts {
		o(&c)
	}
	bad := func(what string, v float64) error {
		return fmt.Errorf("optim: %s %v: %w", what, v, tensor.ErrShape)
	}
	switch {
	case c.momentum < 0 || math.IsNaN(c.momentum):
		return c, bad("momentum", c.momentum)
	case c.weightDecay < 0 || math.IsNaN(c.weightDecay):
		return c, bad("weight decay", c.weightDecay)
	case c.eps < 0 || math.IsNaN(c.eps):
		return c, bad("eps", c.eps)
	case !(c.beta1 >= 0 && c.beta1 < 1):
		return c, bad("beta1", c.beta1)
	case !(c.beta2 >= 0 && c.beta2 < 1):
		return c, bad("beta2", c.beta2)
	case !(c.alpha >= 0 && c.alpha < 1):
		return c, bad("alpha", c.alpha)
	case c.lrDecay < 0 || math.IsNaN(c.lrDecay):
		return c, bad("learning rate decay", c.lrDecay)
	case c.initial < 0 || math.IsNaN(c.initial):
		return c, bad("initial accumulator", c.initial)
	case !(c.rho >= 0 && c.rho < 1):
		return c, bad("rho", c.rho)
	case c.momentumDecay < 0 || math.IsNaN(c.momentumDecay):
		return c, bad("momentum decay", c.momentumDecay)
	case c.nesterov && (c.momentum == 0 || c.dampening != 0):
		return c, fmt.Errorf("optim: Nesterov momentum needs momentum > 0 and no dampening: %w", tensor.ErrShape)
	}
	return c, nil
}

// build returns an optimizer for rule, or one whose Step reports err.
func build(params []*tensor.Param, c config, err error, rule func(c config) Rule) *Optimizer {
	r := rule(c)
	if c.master {
		r = master(r)
	}
	o := New(params, c.lr, r)
	o.clip = c.clip
	if err != nil {
		o.err = err
	}
	return o
}

func f32(v float64) *tensor.Tensor { return tensor.Full(tensor.Float32, v) }

// zeroIfMissing returns s[name], or zeros shaped like p before the first step.
func zeroIfMissing(s State, name string, p *tensor.Tensor) *tensor.Tensor {
	if t, ok := s[name]; ok {
		return t
	}
	return tensor.Zeros(tensor.Float32, p.Shape()...)
}

// SGD returns stochastic gradient descent, like torch.optim.SGD: the
// default learning rate is 1e-3, with optional Momentum, Dampening,
// Nesterov and WeightDecay.
func SGD(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 1e-3})
	return build(params, c, err, func(c config) Rule {
		return func(lr float64, _ int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
			if c.weightDecay != 0 {
				g = g.Add(p.Mul(f32(c.weightDecay)))
			}
			var next State
			if c.momentum != 0 {
				buf := g.Clone() // memory of its own, which later steps update
				if prev, ok := s["momentum_buffer"]; ok {
					buf = prev.Mul(f32(c.momentum)).Add(g.Mul(f32(1 - c.dampening)))
				}
				next = State{"momentum_buffer": buf}
				if c.nesterov {
					g = g.Add(buf.Mul(f32(c.momentum)))
				} else {
					g = buf
				}
			}
			return p.Sub(g.Mul(f32(lr))), next
		}
	})
}

// Adam returns the Adam optimizer, like torch.optim.Adam: learning rate
// 1e-3, Betas 0.9 and 0.999, Eps 1e-8, and an optional L2 WeightDecay.
func Adam(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8})
	return build(params, c, err, func(c config) Rule { return adam(c, false) })
}

// AdamW returns Adam with decoupled weight decay, like torch.optim.AdamW:
// learning rate 1e-3, WeightDecay 0.01, Betas 0.9 and 0.999, Eps 1e-8.
func AdamW(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01})
	return build(params, c, err, func(c config) Rule { return adam(c, true) })
}

func adam(c config, decoupled bool) Rule {
	return func(lr float64, step int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
		if c.weightDecay != 0 {
			if decoupled {
				p = p.Mul(f32(1 - lr*c.weightDecay))
			} else {
				g = g.Add(p.Mul(f32(c.weightDecay)))
			}
		}
		m := zeroIfMissing(s, "exp_avg", p).Mul(f32(c.beta1)).Add(g.Mul(f32(1 - c.beta1)))
		v := zeroIfMissing(s, "exp_avg_sq", p).Mul(f32(c.beta2)).Add(g.Mul(g).Mul(f32(1 - c.beta2)))
		bc1 := 1 - math.Pow(c.beta1, float64(step))
		bc2 := 1 - math.Pow(c.beta2, float64(step))
		denom := v.Sqrt().Mul(f32(1 / math.Sqrt(bc2))).Add(f32(c.eps))
		return p.Sub(m.Div(denom).Mul(f32(lr / bc1))), State{"exp_avg": m, "exp_avg_sq": v}
	}
}

// RMSprop returns RMSprop, like torch.optim.RMSprop: learning rate 1e-2,
// Alpha 0.99, Eps 1e-8, with optional Momentum and L2 WeightDecay.
func RMSprop(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 1e-2, alpha: 0.99, eps: 1e-8})
	return build(params, c, err, func(c config) Rule {
		return func(lr float64, _ int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
			if c.weightDecay != 0 {
				g = g.Add(p.Mul(f32(c.weightDecay)))
			}
			v := zeroIfMissing(s, "square_avg", p).Mul(f32(c.alpha)).Add(g.Mul(g).Mul(f32(1 - c.alpha)))
			step := g.Div(v.Sqrt().Add(f32(c.eps)))
			next := State{"square_avg": v}
			if c.momentum != 0 {
				step = zeroIfMissing(s, "momentum_buffer", p).Mul(f32(c.momentum)).Add(step)
				next["momentum_buffer"] = step
			}
			return p.Sub(step.Mul(f32(lr))), next
		}
	})
}

// Adagrad returns Adagrad, like torch.optim.Adagrad: learning rate 1e-2,
// Eps 1e-10, with optional LRDecay, InitialAccumulator and L2
// WeightDecay. Each weight's step shrinks with the sum of its squared
// gradients, so rare features keep learning fast.
func Adagrad(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 1e-2, eps: 1e-10})
	return build(params, c, err, func(c config) Rule {
		return func(lr float64, step int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
			if c.weightDecay != 0 {
				g = g.Add(p.Mul(f32(c.weightDecay)))
			}
			sum, ok := s["sum"]
			if !ok {
				sum = tensor.Full(tensor.Float32, c.initial).Expand(p.Shape()...)
			}
			sum = sum.Add(g.Mul(g))
			clr := lr / (1 + float64(step-1)*c.lrDecay)
			return p.Sub(g.Div(sum.Sqrt().Add(f32(c.eps))).Mul(f32(clr))), State{"sum": sum}
		}
	})
}

// Adadelta returns Adadelta, like torch.optim.Adadelta: learning rate 1,
// Rho 0.9, Eps 1e-6, with optional L2 WeightDecay. It scales each step
// by the ratio of the running sizes of past steps and gradients, so the
// learning rate matters little.
func Adadelta(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 1, rho: 0.9, eps: 1e-6})
	return build(params, c, err, func(c config) Rule {
		return func(lr float64, _ int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
			if c.weightDecay != 0 {
				g = g.Add(p.Mul(f32(c.weightDecay)))
			}
			sq := zeroIfMissing(s, "square_avg", p).Mul(f32(c.rho)).Add(g.Mul(g).Mul(f32(1 - c.rho)))
			acc := zeroIfMissing(s, "acc_delta", p)
			delta := acc.Add(f32(c.eps)).Sqrt().Div(sq.Add(f32(c.eps)).Sqrt()).Mul(g)
			acc = acc.Mul(f32(c.rho)).Add(delta.Mul(delta).Mul(f32(1 - c.rho)))
			return p.Sub(delta.Mul(f32(lr))), State{"square_avg": sq, "acc_delta": acc}
		}
	})
}

// Adamax returns Adamax, Adam with the infinity norm, like
// torch.optim.Adamax: learning rate 2e-3, Betas 0.9 and 0.999, Eps 1e-8,
// with optional L2 WeightDecay.
func Adamax(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 2e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8})
	return build(params, c, err, func(c config) Rule {
		return func(lr float64, step int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
			if c.weightDecay != 0 {
				g = g.Add(p.Mul(f32(c.weightDecay)))
			}
			m := zeroIfMissing(s, "exp_avg", p).Mul(f32(c.beta1)).Add(g.Mul(f32(1 - c.beta1)))
			u := zeroIfMissing(s, "exp_inf", p).Mul(f32(c.beta2)).Maximum(g.Abs().Add(f32(c.eps)))
			bc := 1 - math.Pow(c.beta1, float64(step))
			return p.Sub(m.Div(u).Mul(f32(lr / bc))), State{"exp_avg": m, "exp_inf": u}
		}
	})
}

// NAdam returns Adam with Nesterov momentum, like torch.optim.NAdam:
// learning rate 2e-3, Betas 0.9 and 0.999, Eps 1e-8, MomentumDecay 4e-3,
// with optional L2 WeightDecay.
func NAdam(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 2e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, momentumDecay: 4e-3})
	return build(params, c, err, func(c config) Rule {
		mu := func(t int) float64 { return c.beta1 * (1 - 0.5*math.Pow(0.96, float64(t)*c.momentumDecay)) }
		return func(lr float64, step int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
			if c.weightDecay != 0 {
				g = g.Add(p.Mul(f32(c.weightDecay)))
			}
			// The product of the momentum factors so far is a function of
			// the step alone, so it needs no state.
			prod := 1.0
			for t := 1; t <= step; t++ {
				prod *= mu(t)
			}
			mt, mnext := mu(step), mu(step+1)
			m := zeroIfMissing(s, "exp_avg", p).Mul(f32(c.beta1)).Add(g.Mul(f32(1 - c.beta1)))
			v := zeroIfMissing(s, "exp_avg_sq", p).Mul(f32(c.beta2)).Add(g.Mul(g).Mul(f32(1 - c.beta2)))
			bc2 := 1 - math.Pow(c.beta2, float64(step))
			denom := v.Mul(f32(1 / bc2)).Sqrt().Add(f32(c.eps))
			upd := g.Mul(f32(lr * (1 - mt) / (1 - prod))).Add(m.Mul(f32(lr * mnext / (1 - prod*mnext))))
			return p.Sub(upd.Div(denom)), State{"exp_avg": m, "exp_avg_sq": v}
		}
	})
}

// RAdam returns Rectified Adam, like torch.optim.RAdam: learning rate
// 1e-3, Betas 0.9 and 0.999, Eps 1e-8, with optional L2 WeightDecay. It
// takes plain momentum steps until there are enough gradients to trust
// Adam's variance estimate, so it needs no warmup.
func RAdam(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8})
	return build(params, c, err, func(c config) Rule {
		rhoInf := 2/(1-c.beta2) - 1
		return func(lr float64, step int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
			if c.weightDecay != 0 {
				g = g.Add(p.Mul(f32(c.weightDecay)))
			}
			m := zeroIfMissing(s, "exp_avg", p).Mul(f32(c.beta1)).Add(g.Mul(f32(1 - c.beta1)))
			v := zeroIfMissing(s, "exp_avg_sq", p).Mul(f32(c.beta2)).Add(g.Mul(g).Mul(f32(1 - c.beta2)))
			next := State{"exp_avg": m, "exp_avg_sq": v}
			b2t := math.Pow(c.beta2, float64(step))
			bc1, bc2 := 1-math.Pow(c.beta1, float64(step)), 1-b2t
			rho := rhoInf - 2*float64(step)*b2t/bc2
			mhat := m.Mul(f32(1 / bc1))
			if rho <= 5 {
				return p.Sub(mhat.Mul(f32(lr))), next
			}
			r := math.Sqrt((rho - 4) * (rho - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rho))
			adapt := v.Sqrt().Add(f32(c.eps)).Recip().Mul(f32(math.Sqrt(bc2)))
			return p.Sub(mhat.Mul(adapt).Mul(f32(lr * r))), next
		}
	})
}

// Lion returns Lion, the optimizer Chen et al. found by search, like
// lion-pytorch: learning rate 1e-4, Betas 0.9 and 0.99, decoupled
// WeightDecay. Every weight moves by exactly lr each step, in the
// direction of the sign of a blend of momentum and gradient; it keeps
// one moment instead of Adam's two, and wants a learning rate 3 to 10
// times smaller than AdamW's.
func Lion(params []*tensor.Param, opts ...Option) *Optimizer {
	c, err := configure(opts, config{lr: 1e-4, beta1: 0.9, beta2: 0.99})
	return build(params, c, err, func(c config) Rule {
		return func(lr float64, _ int, p, g *tensor.Tensor, s State) (*tensor.Tensor, State) {
			if c.weightDecay != 0 {
				p = p.Mul(f32(1 - lr*c.weightDecay))
			}
			m := zeroIfMissing(s, "exp_avg", p)
			blend := m.Mul(f32(c.beta1)).Add(g.Mul(f32(1 - c.beta1)))
			// sign(x), and 0 at 0.
			sign := blend.Greater(f32(0)).Cast(tensor.Float32).Sub(blend.Less(f32(0)).Cast(tensor.Float32))
			m = m.Mul(f32(c.beta2)).Add(g.Mul(f32(1 - c.beta2)))
			return p.Sub(sign.Mul(f32(lr))), State{"exp_avg": m}
		}
	})
}
