package rl

import (
	"context"
	"math/rand/v2"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// DQNConfig configures TrainDQN. Zero fields take the defaults.
type DQNConfig struct {
	Steps       int     // environment steps; the default is 50,000
	Buffer      int     // transitions remembered; the default is 50,000
	Batch       int     // transitions a step learns from; the default is 64
	StartAfter  int     // steps of acting at random before learning; the default is 1,000
	TrainEvery  int     // environment steps between learning steps; the default is 1
	TargetEvery int     // learning steps between copies to the target network; the default is 500
	LR          float64 // Adam's learning rate; the default is 1e-3
	Gamma       float64 // the discount; the default is 0.99
	// EpsStart and EpsEnd bound the chance of a random action, which falls
	// from the one to the other over the first Explore share of the steps.
	// The defaults are 1, 0.05 and 0.3.
	EpsStart, EpsEnd, Explore float64
	MaxNorm                   float64 // the gradients' largest norm; the default is 10
	Hidden                    []int   // the network's layers; the default is 64, 64
	Seed                      uint64
	// Progress, if set, is told the steps so far and the total reward of
	// each episode as it ends.
	Progress func(steps int, reward float64)
}

func (c DQNConfig) resolve() DQNConfig {
	set := func(v *int, d int) {
		if *v <= 0 {
			*v = d
		}
	}
	setf := func(v *float64, d float64) {
		if *v == 0 {
			*v = d
		}
	}
	set(&c.Steps, 50_000)
	set(&c.Buffer, 50_000)
	set(&c.Batch, 64)
	set(&c.StartAfter, 1000)
	set(&c.TrainEvery, 1)
	set(&c.TargetEvery, 500)
	setf(&c.LR, 1e-3)
	setf(&c.Gamma, 0.99)
	setf(&c.EpsStart, 1)
	setf(&c.EpsEnd, 0.05)
	setf(&c.Explore, 0.3)
	setf(&c.MaxNorm, 10)
	if c.Hidden == nil {
		c.Hidden = []int{64, 64}
	}
	return c
}

// DQN is an agent of Mnih et al.'s deep Q-learning, with van Hasselt's
// double Q targets: a network that values each action of an observation.
type DQN struct {
	Net *nn.Sequential
	obs int
}

// Act returns the action valued highest.
func (q *DQN) Act(obs []float32) int {
	out, err := rows(context.Background(), q.Net, obs, 1, q.obs)
	if err != nil {
		return 0
	}
	return argmax(out)
}

// Values returns the network's value of each action of obs.
func (q *DQN) Values(ctx context.Context, obs []float32) ([]float32, error) {
	return rows(ctx, q.Net, obs, 1, q.obs)
}

// TrainDQN trains a DQN agent on env.
func TrainDQN(ctx context.Context, env Env, c DQNConfig) (*DQN, error) {
	if err := checkEnv(env); err != nil {
		return nil, err
	}
	c = c.resolve()
	f, k := env.Observation(), env.Actions()
	q := &DQN{Net: mlp(f, k, c.Hidden, nn.ReLU()), obs: f}
	target := mlp(f, k, c.Hidden, nn.ReLU())
	sync := func() error {
		src := nn.Parameters(q.Net)
		for i, p := range nn.Parameters(target) {
			if err := p.Set(src[i].Value().Clone()); err != nil {
				return err
			}
		}
		return tensor.Realize(ctx, valuesOf(target)...)
	}
	if err := sync(); err != nil {
		return nil, err
	}
	opt := optim.Adam(nn.Parameters(q.Net), optim.LR(c.LR), optim.ClipNorm(c.MaxNorm))
	r := rand.New(rand.NewPCG(c.Seed, 0x64716e))

	size := min(c.Buffer, c.Steps)
	bObs, bNext := make([]float32, size*f), make([]float32, size*f)
	bAct, bRew, bEnd := make([]int32, size), make([]float32, size), make([]float32, size)
	filled, at := 0, 0

	obs := env.Reset(c.Seed)
	episode, episodes, learned := 0.0, uint64(1), 0
	for step := range c.Steps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		eps := c.EpsEnd
		if span := c.Explore * float64(c.Steps); float64(step) < span {
			eps = c.EpsStart + (c.EpsEnd-c.EpsStart)*float64(step)/span
		}
		var a int
		if step < c.StartAfter || r.Float64() < eps {
			a = r.IntN(k)
		} else {
			vals, err := rows(ctx, q.Net, obs, 1, f)
			if err != nil {
				return nil, err
			}
			a = argmax(vals)
		}
		st := env.Step(a)
		copy(bObs[at*f:], obs)
		copy(bNext[at*f:], st.Obs)
		bAct[at], bRew[at] = int32(a), float32(st.Reward)
		// A truncated episode had a future: its next value still counts.
		bEnd[at] = 0
		if st.Done {
			bEnd[at] = 1
		}
		at = (at + 1) % size
		filled = min(filled+1, size)
		episode += st.Reward
		obs = st.Obs
		if st.Done || st.Truncated {
			if c.Progress != nil {
				c.Progress(step+1, episode)
			}
			episode = 0
			obs = env.Reset(c.Seed + episodes)
			episodes++
		}
		if step < c.StartAfter || step%c.TrainEvery != 0 || filled < c.Batch {
			continue
		}
		if err := q.learn(ctx, opt, target, r, c, f, filled, bObs, bNext, bAct, bRew, bEnd); err != nil {
			return nil, err
		}
		learned++
		if learned%c.TargetEvery == 0 {
			if err := sync(); err != nil {
				return nil, err
			}
		}
	}
	return q, nil
}

func valuesOf(m nn.Module) []*tensor.Tensor {
	var out []*tensor.Tensor
	for _, p := range nn.Parameters(m) {
		out = append(out, p.Value())
	}
	return out
}

// learn takes one step on a batch drawn from the buffer.
func (q *DQN) learn(ctx context.Context, opt *optim.Optimizer, target *nn.Sequential, r *rand.Rand, c DQNConfig, f, filled int, bObs, bNext []float32, bAct []int32, bRew, bEnd []float32) error {
	b := c.Batch
	o, nx := make([]float32, 0, b*f), make([]float32, 0, b*f)
	a, rew, end := make([]int32, b), make([]float32, b), make([]float32, b)
	for j := range b {
		i := r.IntN(filled)
		o = append(o, bObs[i*f:(i+1)*f]...)
		nx = append(nx, bNext[i*f:(i+1)*f]...)
		a[j], rew[j], end[j] = bAct[i], bRew[i], bEnd[i]
	}
	next := tensor.FromSlice(nx, b, f)
	// Double Q: the online network picks the next action, the target
	// network values it.
	pick := q.Net.Forward(next).Detach().Argmax(1).Cast(tensor.Int32).Reshape(b, 1)
	nextQ := target.Forward(next).Detach().Gather(1, pick).Reshape(b)
	keep := tensor.Scalar[float32](1).Sub(tensor.FromSlice(end, b))
	y := tensor.FromSlice(rew, b).Add(nextQ.Mul(keep).Mul(tensor.Scalar(float32(c.Gamma)))).Detach()
	got := q.Net.Forward(tensor.FromSlice(o, b, f)).Gather(1, tensor.FromSlice(a, b, 1)).Reshape(b)
	l := loss.Huber(got, y)
	if err := l.Backward(ctx); err != nil {
		return err
	}
	if err := opt.Step(ctx); err != nil {
		return err
	}
	opt.ZeroGrad()
	return nil
}
