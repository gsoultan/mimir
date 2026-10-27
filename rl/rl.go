// Package rl teaches agents to act by trial and error, as Stable
// Baselines3 does: an agent acts in an environment, is rewarded, and
// learns which actions earn the most over time.
//
//	agent, err := rl.TrainPPO(ctx, func(seed uint64) rl.Env { return rl.NewCartPole() }, rl.PPOConfig{Steps: 100_000})
//	if err != nil { ... }
//	score, err := rl.Evaluate(ctx, agent, rl.NewCartPole(), 10, 1) // mean reward of 10 episodes
//
// PPO, proximal policy optimization with generalized advantage
// estimation, is the strong default for discrete actions; DQN, deep
// Q-learning with a replay buffer, double Q targets and a target network,
// learns more from each step and suits environments that are slow to
// run. Both read observations with small networks of package nn, so an
// agent saves, loads and runs on any device as a network does.
//
// An environment is anything with the Env interface; CartPole, the
// classic pole on a cart, and Chain, a corridor whose reward is at the
// far end, come with the package.
package rl

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Step is what an environment returns for an action.
type Step struct {
	Obs    []float32 // the observation after the action
	Reward float64
	// Done means the episode ended in a terminal state, such as the pole
	// falling: nothing follows. Truncated means it was cut off, by a step
	// limit, in a state that had a future.
	Done, Truncated bool
}

// Env is an environment of discrete actions.
type Env interface {
	// Observation is how many numbers each observation holds.
	Observation() int
	// Actions is how many actions there are, numbered from 0.
	Actions() int
	// Reset starts an episode, seeded, and returns its first observation.
	Reset(seed uint64) []float32
	// Step takes an action.
	Step(action int) Step
}

// Policy picks an action for an observation.
type Policy interface {
	Act(obs []float32) int
}

// Evaluate runs policy for episodes episodes of env, seeded from seed,
// and returns the mean of their total rewards.
func Evaluate(ctx context.Context, policy Policy, env Env, episodes int, seed uint64) (float64, error) {
	var total float64
	for e := range episodes {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		obs := env.Reset(seed + uint64(e))
		for {
			s := env.Step(policy.Act(obs))
			total += s.Reward
			if s.Done || s.Truncated {
				break
			}
			obs = s.Obs
		}
	}
	return total / float64(max(episodes, 1)), nil
}

// mlp returns a network of Linear layers through hidden, each followed by
// act, from in numbers to out.
func mlp(in, out int, hidden []int, act nn.Func) *nn.Sequential {
	s := &nn.Sequential{}
	for _, h := range hidden {
		s.Layers = append(s.Layers, nn.NewLinear(in, h), act)
		in = h
	}
	s.Layers = append(s.Layers, nn.NewLinear(in, out))
	return s
}

// rows runs m on observations, n of size f, and returns its outputs.
func rows(ctx context.Context, m nn.Module, obs []float32, n, f int) ([]float32, error) {
	return m.Forward(tensor.FromSlice(obs, n, f)).Data[float32](ctx)
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// CartPole is the classic control problem of Barto, Sutton and Anderson,
// as Gymnasium's CartPole-v1: push a cart left or right to keep the pole
// on it upright. The observation is the cart's position and speed and
// the pole's angle and spin; every step the pole stays up earns 1, and an
// episode ends when it falls past 12°, the cart leaves the track, or after
// MaxSteps.
type CartPole struct {
	MaxSteps int // 500 unless set
	x, v     float64
	th, w    float64
	steps    int
}

// NewCartPole returns a CartPole of 500 steps an episode.
func NewCartPole() *CartPole { return &CartPole{MaxSteps: 500} }

func (c *CartPole) Observation() int { return 4 }
func (c *CartPole) Actions() int     { return 2 }

func (c *CartPole) obs() []float32 {
	return []float32{float32(c.x), float32(c.v), float32(c.th), float32(c.w)}
}

func (c *CartPole) Reset(seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 0x6361727470))
	u := func() float64 { return r.Float64()*0.1 - 0.05 }
	c.x, c.v, c.th, c.w, c.steps = u(), u(), u(), u(), 0
	return c.obs()
}

func (c *CartPole) Step(action int) Step {
	const (
		g, mc, mp, l, force, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
		total, pml               = mc + mp, mp * l
	)
	f := force
	if action == 0 {
		f = -force
	}
	cos, sin := math.Cos(c.th), math.Sin(c.th)
	temp := (f + pml*c.w*c.w*sin) / total
	thAcc := (g*sin - cos*temp) / (l * (4.0/3.0 - mp*cos*cos/total))
	xAcc := temp - pml*thAcc*cos/total
	c.x += tau * c.v
	c.v += tau * xAcc
	c.th += tau * c.w
	c.w += tau * thAcc
	c.steps++
	done := c.x < -2.4 || c.x > 2.4 || c.th < -12*2*math.Pi/360 || c.th > 12*2*math.Pi/360
	limit := c.MaxSteps
	if limit <= 0 {
		limit = 500
	}
	return Step{Obs: c.obs(), Reward: 1, Done: done, Truncated: !done && c.steps >= limit}
}

// Chain is a corridor of N cells: the agent starts in the second, may
// step left or right, earns 0.01 for reaching the first cell, which ends
// the episode, and 1 for reaching the last. A greedy agent settles for
// the near reward; one that explores and plans ahead walks to the far
// one. The observation is the one-hot of the cell.
type Chain struct {
	N        int
	MaxSteps int // 4·N unless set
	at       int
	steps    int
}

func (c *Chain) Observation() int { return c.N }
func (c *Chain) Actions() int     { return 2 }

func (c *Chain) obs() []float32 {
	o := make([]float32, c.N)
	o[c.at] = 1
	return o
}

func (c *Chain) Reset(uint64) []float32 {
	c.at, c.steps = 1, 0
	return c.obs()
}

func (c *Chain) Step(action int) Step {
	if action == 0 {
		c.at--
	} else {
		c.at++
	}
	c.steps++
	limit := c.MaxSteps
	if limit <= 0 {
		limit = 4 * c.N
	}
	switch {
	case c.at <= 0:
		c.at = 0
		return Step{Obs: c.obs(), Reward: 0.01, Done: true}
	case c.at >= c.N-1:
		c.at = c.N - 1
		return Step{Obs: c.obs(), Reward: 1, Done: true}
	}
	return Step{Obs: c.obs(), Truncated: c.steps >= limit}
}

func checkEnv(env Env) error {
	if env == nil || env.Observation() < 1 || env.Actions() < 2 {
		return fmt.Errorf("rl: an environment needs observations and at least 2 actions: %w", tensor.ErrShape)
	}
	return nil
}
