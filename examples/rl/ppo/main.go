// Command ppo learns to balance CartPole with proximal policy
// optimization (Schulman et al.): an actor-critic collects rollouts from
// several environments side by side, estimates advantages with
// generalized advantage estimation, and takes several epochs of
// minibatch steps on the clipped surrogate objective, with a value loss
// and an entropy bonus.
//
//	go run ./examples/rl/ppo
//
// Training stops once the greedy policy's mean return over 100 episodes
// reaches 195, the task's definition of solved.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"time"

	"github.com/gsoultan/mimir/examples/rl/cartpole"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func main() {
	cfg := defaults
	flag.IntVar(&cfg.Updates, "updates", cfg.Updates, "most rollouts and updates")
	flag.Float64Var(&cfg.LR, "lr", cfg.LR, "Adam learning rate")
	flag.Uint64Var(&cfg.Seed, "seed", cfg.Seed, "seed of the environments, the networks and sampling")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := run(ctx, cfg, os.Stdout)
	if err != nil {
		log.Fatal(err)
	}
	if !res.Solved {
		log.Fatalf("not solved in %d updates (%d episodes, %d steps, %v): greedy mean return %.1f",
			res.Updates, res.Episodes, res.Steps, res.Time.Round(time.Millisecond), res.Eval)
	}
	fmt.Printf("solved in %d updates, %d episodes, %d steps, %v: greedy mean return %.1f over 100 episodes\n",
		res.Updates, res.Episodes, res.Steps, res.Time.Round(time.Millisecond), res.Eval)
}

// config sets up training.
type config struct {
	Envs, Steps   int     // environments side by side, and steps of each per rollout
	Updates       int     // rollouts, each followed by an update, at most
	Epochs        int     // passes over a rollout per update
	Minibatch     int     // transitions per optimizer step
	Hidden        int     // units of each hidden layer of the actor and the critic
	LR            float64 // Adam's learning rate
	Gamma, Lambda float64 // discount and GAE's λ
	Clip          float64 // the surrogate's clipping of the probability ratio
	ValueCoef     float64
	EntropyCoef   float64
	Seed          uint64
}

var defaults = config{
	Envs: 8, Steps: 128, Updates: 200, Epochs: 4, Minibatch: 256, Hidden: 64,
	LR: 1e-3, Gamma: 0.99, Lambda: 0.95, Clip: 0.2, ValueCoef: 0.5, EntropyCoef: 0.01,
	Seed: 1,
}

// result is how training went: whether the greedy policy solved the
// task, its mean return over 100 episodes when last evaluated, the
// updates, finished training episodes and environment steps it took,
// and the time.
type result struct {
	Solved                   bool
	Eval                     float64
	Updates, Episodes, Steps int
	Returns                  []float64 // of each finished training episode
	Losses                   []float64 // the mean loss of each update
	Time                     time.Duration
}

// policy is the actor, which maps observations to the logits of the
// actions, and the critic, which maps them to the value of the state.
type policy struct {
	Actor, Critic *nn.Sequential
}

func newPolicy(hidden int) *policy {
	mlp := func(out int) *nn.Sequential {
		return nn.NewSequential(
			nn.NewLinear(4, hidden), nn.Tanh(),
			nn.NewLinear(hidden, hidden), nn.Tanh(),
			nn.NewLinear(hidden, out),
		)
	}
	return &policy{Actor: mlp(2), Critic: mlp(1)}
}

// Forward returns the logits of the actions for observations [n, 4].
func (p *policy) Forward(x *tensor.Tensor) *tensor.Tensor { return p.Actor.Forward(x) }

// rollout holds the transitions of one rollout, step by step and, within
// a step, environment by environment, in memory reused from rollout to
// rollout.
type rollout struct {
	obs          []float32 // [steps*envs*4]
	action       []int32
	logp, value  []float32 // under the policy that acted
	reward, done []float32 // done is 1 where the episode ended after the step
	adv, ret     []float32 // GAE advantages and the returns the critic learns
	lastValue    []float32 // of the observations after the last step
	finalValue   []float32 // of the last observation of an episode truncated at the step, or 0
}

func newRollout(steps, envs int) *rollout {
	n := steps * envs
	return &rollout{
		obs: make([]float32, 4*n), action: make([]int32, n),
		logp: make([]float32, n), value: make([]float32, n),
		reward: make([]float32, n), done: make([]float32, n),
		adv: make([]float32, n), ret: make([]float32, n),
		lastValue: make([]float32, envs), finalValue: make([]float32, n),
	}
}

// advantages computes the GAE advantages and returns of r, gathered by
// envs environments. A truncated episode bootstraps from the value of
// the state it stopped in.
func (r *rollout) advantages(envs int, gamma, lambda float64) {
	n := len(r.action)
	for e := range envs {
		var gae float64
		for i := n - envs + e; i >= 0; i -= envs {
			next := float64(r.lastValue[e])
			if i+envs < n {
				next = float64(r.value[i+envs])
			}
			if r.done[i] != 0 {
				next = float64(r.finalValue[i])
				gae = 0
			}
			delta := float64(r.reward[i]) + gamma*next - float64(r.value[i])
			gae = delta + gamma*lambda*gae
			r.adv[i] = float32(gae)
			r.ret[i] = float32(gae + float64(r.value[i]))
		}
	}
}

// evaluate returns the mean return of the greedy policy over episodes
// episodes, run side by side so that each step is one batched forward
// pass.
func evaluate(ctx context.Context, p *policy, episodes int, seed uint64) (float64, error) {
	envs := make([]*cartpole.Env, episodes)
	obs := make([]float32, 4*episodes)
	for i := range envs {
		envs[i] = cartpole.New(seed + uint64(i))
		o := envs[i].Reset()
		copy(obs[4*i:], o[:])
	}
	done := make([]bool, episodes)
	var total float64
	for running := episodes; running > 0; {
		actions, err := p.Forward(tensor.FromSlice(obs, episodes, 4)).Argmax(1).Data[int32](ctx)
		if err != nil {
			return 0, err
		}
		for i, e := range envs {
			if done[i] {
				continue
			}
			o, r, term, trunc := e.Step(int(actions[i]))
			copy(obs[4*i:], o[:])
			total += r
			if term || trunc {
				done[i] = true
				running--
			}
		}
	}
	return total / float64(episodes), nil
}

// act returns the logits and values of the policy for observations obs,
// [n*4], computed in one schedule.
func act(ctx context.Context, p *policy, obs []float32, n int) (logits, values []float32, err error) {
	x := tensor.FromSlice(obs, n, 4)
	l, v := p.Actor.Forward(x), p.Critic.Forward(x)
	if err := tensor.Realize(ctx, l, v); err != nil {
		return nil, nil, err
	}
	if logits, err = l.Data[float32](ctx); err != nil {
		return nil, nil, err
	}
	values, err = v.Data[float32](ctx)
	return logits, values, err
}

// sample draws an action from the softmax of two logits and returns it
// with its log-probability. A package of distributions would serve
// here; the examples keep to what they need.
func sample(rng *rand.Rand, logits []float32) (int, float32) {
	m := math.Max(float64(logits[0]), float64(logits[1]))
	e0, e1 := math.Exp(float64(logits[0])-m), math.Exp(float64(logits[1])-m)
	lse := m + math.Log(e0+e1)
	a := 0
	if rng.Float64()*(e0+e1) >= e0 {
		a = 1
	}
	return a, float32(float64(logits[a]) - lse)
}

// minibatch holds the transitions of one optimizer step.
type minibatch struct {
	obs            []float32
	action         []int32
	logp, adv, ret []float32
}

// loss returns the PPO loss of p on m: the clipped surrogate objective,
// negated, plus ValueCoef times the critic's squared error, minus
// EntropyCoef times the policy's entropy.
func loss(cfg config, p *policy, m *minibatch) *tensor.Tensor {
	n := len(m.action)
	x := tensor.FromSlice(m.obs, n, 4)
	logp := p.Actor.Forward(x).LogSoftmax(1)
	newLogp := logp.Gather(1, tensor.FromSlice(m.action, n, 1)).Reshape(n)
	adv := tensor.FromSlice(m.adv)
	ratio := newLogp.Sub(tensor.FromSlice(m.logp)).Exp()
	lo, hi := tensor.Scalar(float32(1-cfg.Clip)), tensor.Scalar(float32(1+cfg.Clip))
	surrogate := ratio.Mul(adv).Minimum(ratio.Maximum(lo).Minimum(hi).Mul(adv))
	entropy := logp.Exp().Mul(logp).Sum(1).Neg()
	v := p.Critic.Forward(x).Reshape(n)
	d := v.Sub(tensor.FromSlice(m.ret))
	scalar := func(f float64) *tensor.Tensor { return tensor.Scalar(float32(f)) }
	return surrogate.Mean().Neg().
		Add(d.Mul(d).Mean().Mul(scalar(cfg.ValueCoef))).
		Sub(entropy.Mean().Mul(scalar(cfg.EntropyCoef)))
}

// update takes cfg.Epochs passes over the rollout r in shuffled
// minibatches, normalizing the advantages of each, and returns the mean
// loss.
func update(ctx context.Context, cfg config, p *policy, opt *optim.Optimizer, r *rollout, rng *rand.Rand, m *minibatch) (float64, error) {
	n := len(r.action)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	var sum float64
	var steps int
	for range cfg.Epochs {
		rng.Shuffle(n, func(i, j int) { order[i], order[j] = order[j], order[i] })
		for start := 0; start+cfg.Minibatch <= n; start += cfg.Minibatch {
			var mean, sq float64
			for k, i := range order[start : start+cfg.Minibatch] {
				copy(m.obs[4*k:4*k+4], r.obs[4*i:4*i+4])
				m.action[k], m.logp[k], m.adv[k], m.ret[k] = r.action[i], r.logp[i], r.adv[i], r.ret[i]
				mean += float64(r.adv[i])
				sq += float64(r.adv[i]) * float64(r.adv[i])
			}
			mean /= float64(cfg.Minibatch)
			std := math.Sqrt(max(sq/float64(cfg.Minibatch)-mean*mean, 0)) + 1e-8
			for k := range m.adv {
				m.adv[k] = float32((float64(m.adv[k]) - mean) / std)
			}
			l := loss(cfg, p, m)
			if err := l.Backward(ctx); err != nil {
				return 0, err
			}
			if err := opt.Step(ctx); err != nil {
				return 0, err
			}
			opt.ZeroGrad()
			v, err := l.Item[float32](ctx)
			if err != nil {
				return 0, err
			}
			sum += float64(v)
			steps++
		}
	}
	return sum / float64(max(steps, 1)), nil
}

func newMinibatch(n int) *minibatch {
	return &minibatch{obs: make([]float32, 4*n), action: make([]int32, n), logp: make([]float32, n), adv: make([]float32, n), ret: make([]float32, n)}
}

func run(ctx context.Context, cfg config, progress io.Writer) (result, error) {
	if cfg.Envs <= 0 || cfg.Steps <= 0 || cfg.Minibatch <= 0 || cfg.Minibatch > cfg.Envs*cfg.Steps || cfg.Updates <= 0 {
		return result{}, errors.New("ppo: invalid configuration")
	}
	start := time.Now()
	nn.ManualSeed(cfg.Seed)
	p := newPolicy(cfg.Hidden)
	opt := optim.Adam(nn.Parameters(p), optim.LR(cfg.LR), optim.Eps(1e-5))
	rng := rand.New(rand.NewPCG(cfg.Seed, 2))
	envs := make([]*cartpole.Env, cfg.Envs)
	obs := make([]float32, 4*cfg.Envs)
	rets := make([]float64, cfg.Envs)
	for e := range envs {
		envs[e] = cartpole.New(cfg.Seed*1000 + uint64(e))
		o := envs[e].Reset()
		copy(obs[4*e:], o[:])
	}
	r := newRollout(cfg.Steps, cfg.Envs)
	m := newMinibatch(cfg.Minibatch)
	var res result
	for res.Updates < cfg.Updates {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		// Anneal the learning rate linearly to zero over the updates.
		opt.SetLR(cfg.LR * (1 - float64(res.Updates)/float64(cfg.Updates)))
		for s := range cfg.Steps {
			logits, values, err := act(ctx, p, obs, cfg.Envs)
			if err != nil {
				return res, err
			}
			for e, env := range envs {
				i := s*cfg.Envs + e
				a, lp := sample(rng, logits[2*e:2*e+2])
				copy(r.obs[4*i:4*i+4], obs[4*e:4*e+4])
				r.action[i], r.logp[i], r.value[i] = int32(a), lp, values[e]
				o, rew, term, trunc := env.Step(a)
				r.reward[i], r.done[i], r.finalValue[i] = float32(rew), 0, 0
				rets[e] += rew
				if term || trunc {
					r.done[i] = 1
					if trunc {
						// Bootstrap from the state the episode stopped in.
						_, v, err := act(ctx, p, o[:], 1)
						if err != nil {
							return res, err
						}
						r.finalValue[i] = v[0]
					}
					res.Returns = append(res.Returns, rets[e])
					res.Episodes++
					rets[e] = 0
					o = env.Reset()
				}
				copy(obs[4*e:], o[:])
			}
		}
		res.Steps += cfg.Steps * cfg.Envs
		_, last, err := act(ctx, p, obs, cfg.Envs)
		if err != nil {
			return res, err
		}
		copy(r.lastValue, last)
		r.advantages(cfg.Envs, cfg.Gamma, cfg.Lambda)
		l, err := update(ctx, cfg, p, opt, r, rng, m)
		if err != nil {
			return res, err
		}
		res.Updates++
		res.Losses = append(res.Losses, l)
		recent := mean(res.Returns[max(0, len(res.Returns)-20):])
		if progress != nil && res.Updates%5 == 0 {
			fmt.Fprintf(progress, "update %3d  steps %6d  episodes %4d  last 20 returns %5.1f  loss %7.4f  %v\n",
				res.Updates, res.Steps, res.Episodes, recent, l, time.Since(start).Round(time.Millisecond))
		}
		if len(res.Returns) >= 20 && recent >= 195 {
			v, err := evaluate(ctx, p, 100, cfg.Seed+1_000_000)
			if err != nil {
				return res, err
			}
			res.Eval = v
			if progress != nil {
				fmt.Fprintf(progress, "update %3d  greedy mean return over 100 episodes %.1f\n", res.Updates, v)
			}
			if v >= 195 {
				res.Solved = true
				break
			}
		}
	}
	res.Time = time.Since(start)
	return res, nil
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	var s float64
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}
