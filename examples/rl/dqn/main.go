// Command dqn learns to balance CartPole with a deep Q-network, as Mnih
// et al. played Atari: a network estimates the return of each action, the
// agent acts on it epsilon-greedily, and the network learns from
// minibatches of past transitions drawn from a replay buffer, against
// targets from a copy of itself that follows it every few hundred steps.
// The targets are double Q-learning's: the network picks the next
// action and the copy values it.
//
//	go run ./examples/rl/dqn
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
	"math/rand/v2"
	"os"
	"os/signal"
	"time"

	"github.com/gsoultan/mimir/examples/rl/cartpole"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func main() {
	cfg := defaults
	flag.IntVar(&cfg.Episodes, "episodes", cfg.Episodes, "most training episodes")
	flag.Float64Var(&cfg.LR, "lr", cfg.LR, "Adam learning rate")
	flag.Uint64Var(&cfg.Seed, "seed", cfg.Seed, "seed of the environment, the network and exploration")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := run(ctx, cfg, os.Stdout)
	if err != nil {
		log.Fatal(err)
	}
	if !res.Solved {
		log.Fatalf("not solved in %d episodes (%d steps, %v): greedy mean return %.1f",
			res.Episodes, res.Steps, res.Time.Round(time.Millisecond), res.Eval)
	}
	fmt.Printf("solved in %d episodes, %d steps, %v: greedy mean return %.1f over 100 episodes\n",
		res.Episodes, res.Steps, res.Time.Round(time.Millisecond), res.Eval)
}

// config sets up training.
type config struct {
	Episodes     int     // training episodes at most
	Hidden       int     // units of each of the network's two hidden layers
	LR           float64 // Adam's learning rate
	Gamma        float64 // discount
	Batch        int     // transitions per update
	Buffer       int     // transitions the replay buffer keeps
	Start        int     // steps of random play before the first update
	EpsilonSteps int     // steps over which epsilon falls from 1 to EpsilonEnd
	EpsilonEnd   float64
	SyncEvery    int // steps between copies of the network into the target network
	Seed         uint64
}

var defaults = config{
	Episodes: 1000, Hidden: 128, LR: 5e-4, Gamma: 0.99,
	Batch: 64, Buffer: 50_000, Start: 1000,
	EpsilonSteps: 10_000, EpsilonEnd: 0.05, SyncEvery: 500, Seed: 1,
}

// result is how training went: whether the greedy policy solved the
// task, its mean return over 100 episodes when last evaluated, the
// training episodes and environment steps it took, and the time.
type result struct {
	Solved          bool
	Eval            float64
	Episodes, Steps int
	Returns         []float64 // of each training episode
	Time            time.Duration
}

// qnet returns a network from observations to the value of each action.
func qnet(hidden int) *nn.Sequential {
	return nn.NewSequential(
		nn.NewLinear(4, hidden), nn.ReLU(),
		nn.NewLinear(hidden, hidden), nn.ReLU(),
		nn.NewLinear(hidden, 2),
	)
}

// replay is a ring buffer of transitions.
type replay struct {
	obs, next     []cartpole.Observation
	action        []int32
	reward, final []float32 // final is 1 where the episode terminated
	n, at         int
}

func newReplay(size int) *replay {
	return &replay{
		obs: make([]cartpole.Observation, size), next: make([]cartpole.Observation, size),
		action: make([]int32, size), reward: make([]float32, size), final: make([]float32, size),
	}
}

func (b *replay) add(obs, next cartpole.Observation, action int, reward float64, terminated bool) {
	b.obs[b.at], b.next[b.at], b.action[b.at], b.reward[b.at] = obs, next, int32(action), float32(reward)
	b.final[b.at] = 0
	if terminated {
		b.final[b.at] = 1
	}
	b.at = (b.at + 1) % len(b.obs)
	b.n = min(b.n+1, len(b.obs))
}

// batch is a minibatch of transitions, in memory reused from draw to draw.
type batch struct {
	obs, next     []float32
	action        []int32
	reward, final []float32
}

func newBatch(n int) *batch {
	return &batch{
		obs: make([]float32, 4*n), next: make([]float32, 4*n),
		action: make([]int32, n), reward: make([]float32, n), final: make([]float32, n),
	}
}

// sample draws len(m.action) transitions uniformly into m.
func (b *replay) sample(rng *rand.Rand, m *batch) {
	for i := range m.action {
		j := rng.IntN(b.n)
		copy(m.obs[4*i:], b.obs[j][:])
		copy(m.next[4*i:], b.next[j][:])
		m.action[i], m.reward[i], m.final[i] = b.action[j], b.reward[j], b.final[j]
	}
}

// agent is the Q-network, its target copy and its optimizer.
type agent struct {
	cfg            config
	online, target *nn.Sequential
	opt            *optim.Optimizer
}

// newAgent returns an agent with a new network, from nn's seed, and an
// uninitialized target network: sync it before use.
func newAgent(cfg config) *agent {
	a := &agent{cfg: cfg, online: qnet(cfg.Hidden), target: qnet(cfg.Hidden)}
	a.opt = optim.Adam(nn.Parameters(a.online), optim.LR(cfg.LR))
	return a
}

// sync copies the online network's parameters into the target network.
func (a *agent) sync(ctx context.Context) error {
	src, dst := nn.Parameters(a.online), nn.Parameters(a.target)
	copies := make([]*tensor.Tensor, len(src))
	for i, p := range src {
		copies[i] = p.Value().Clone()
	}
	if err := tensor.Realize(ctx, copies...); err != nil {
		return err
	}
	for i, p := range dst {
		if err := p.Set(copies[i]); err != nil {
			return err
		}
	}
	return nil
}

// greedy returns the action of largest value for each of the n
// observations in obs, [n*4].
func (a *agent) greedy(ctx context.Context, obs []float32, n int) ([]int32, error) {
	return a.online.Forward(tensor.FromSlice(obs, n, 4)).Argmax(1).Data[int32](ctx)
}

// update takes one optimizer step on a minibatch, minimizing the Huber
// loss between the values of the actions taken and their double
// Q-learning targets, and returns the loss.
func (a *agent) update(ctx context.Context, m *batch) (float32, error) {
	n := len(m.action)
	obs, next := tensor.FromSlice(m.obs, n, 4), tensor.FromSlice(m.next, n, 4)
	action := tensor.FromSlice(m.action, n, 1)
	q := a.online.Forward(obs).Gather(1, action).Reshape(n)
	best := a.online.Forward(next).Detach().Argmax(1).Reshape(n, 1)
	nextQ := a.target.Forward(next).Detach().Gather(1, best).Reshape(n)
	keep := tensor.Scalar[float32](1).Sub(tensor.FromSlice(m.final))
	y := tensor.FromSlice(m.reward).Add(keep.Mul(nextQ).Mul(tensor.Scalar(float32(a.cfg.Gamma))))
	l := loss.Huber(q, y)
	if err := l.Backward(ctx); err != nil {
		return 0, err
	}
	if err := a.opt.Step(ctx); err != nil {
		return 0, err
	}
	a.opt.ZeroGrad()
	return l.Item[float32](ctx)
}

// evaluate returns the mean return of the greedy policy over episodes
// episodes, run side by side so that each step is one batched forward
// pass.
func (a *agent) evaluate(ctx context.Context, episodes int, seed uint64) (float64, error) {
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
		actions, err := a.greedy(ctx, obs, episodes)
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

func run(ctx context.Context, cfg config, progress io.Writer) (result, error) {
	if cfg.Batch <= 0 || cfg.Buffer < cfg.Batch || cfg.SyncEvery <= 0 || cfg.Episodes <= 0 {
		return result{}, errors.New("dqn: invalid configuration")
	}
	start := time.Now()
	nn.ManualSeed(cfg.Seed)
	a := newAgent(cfg)
	if err := a.sync(ctx); err != nil {
		return result{}, err
	}
	env := cartpole.New(cfg.Seed)
	rng := rand.New(rand.NewPCG(cfg.Seed, 1))
	buf := newReplay(cfg.Buffer)
	m := newBatch(cfg.Batch)
	var res result
	var lastLoss float32
	for res.Episodes < cfg.Episodes {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		obs := env.Reset()
		var ret float64
		for {
			eps := max(cfg.EpsilonEnd, 1-(1-cfg.EpsilonEnd)*float64(res.Steps)/float64(cfg.EpsilonSteps))
			action := rng.IntN(2)
			if res.Steps >= cfg.Start && rng.Float64() >= eps {
				act, err := a.greedy(ctx, obs[:], 1)
				if err != nil {
					return res, err
				}
				action = int(act[0])
			}
			next, r, term, trunc := env.Step(action)
			buf.add(obs, next, action, r, term)
			obs, ret = next, ret+r
			res.Steps++
			if res.Steps >= cfg.Start {
				buf.sample(rng, m)
				l, err := a.update(ctx, m)
				if err != nil {
					return res, err
				}
				lastLoss = l
			}
			if res.Steps%cfg.SyncEvery == 0 {
				if err := a.sync(ctx); err != nil {
					return res, err
				}
			}
			if term || trunc {
				break
			}
		}
		res.Episodes++
		res.Returns = append(res.Returns, ret)
		recent := mean(res.Returns[max(0, len(res.Returns)-10):])
		if progress != nil && res.Episodes%20 == 0 {
			fmt.Fprintf(progress, "episode %4d  steps %6d  return %5.1f  last 10 %5.1f  loss %.4f  %v\n",
				res.Episodes, res.Steps, ret, recent, lastLoss, time.Since(start).Round(time.Millisecond))
		}
		// Evaluate the greedy policy once training episodes go well.
		if len(res.Returns) >= 10 && recent >= 195 && res.Episodes%5 == 0 {
			v, err := a.evaluate(ctx, 100, cfg.Seed+1_000_000)
			if err != nil {
				return res, err
			}
			res.Eval = v
			if progress != nil {
				fmt.Fprintf(progress, "episode %4d  greedy mean return over 100 episodes %.1f\n", res.Episodes, v)
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
