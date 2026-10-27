package rl

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// PPOConfig configures TrainPPO. Zero fields take the defaults, Stable
// Baselines3's for PPO.
type PPOConfig struct {
	Steps     int     // environment steps in all; the default is 100,000
	Envs      int     // environments run side by side; the default is 8
	Rollout   int     // steps of each environment between updates; the default is 256
	Epochs    int     // passes over each rollout; the default is 10
	Minibatch int     // the default is 64
	LR        float64 // Adam's learning rate; the default is 3e-4
	Gamma     float64 // the discount; the default is 0.99
	Lambda    float64 // GAE's λ; the default is 0.95
	Clip      float64 // how far the policy may move in an update; the default is 0.2
	Value     float64 // the value loss's weight; the default is 0.5
	Entropy   float64 // the entropy bonus's weight; the default is 0
	MaxNorm   float64 // the gradients' largest norm; the default is 0.5
	Hidden    []int   // the networks' layers; the default is 64, 64
	Seed      uint64
	// Progress, if set, is told the steps so far and the mean reward of
	// the episodes that ended in the last rollout.
	Progress func(steps int, reward float64)
}

func (c PPOConfig) resolve() PPOConfig {
	def := func(v *float64, d float64) {
		if *v == 0 {
			*v = d
		}
	}
	if c.Steps <= 0 {
		c.Steps = 100_000
	}
	if c.Envs <= 0 {
		c.Envs = 8
	}
	if c.Rollout <= 0 {
		c.Rollout = 256
	}
	if c.Epochs <= 0 {
		c.Epochs = 10
	}
	if c.Minibatch <= 0 {
		c.Minibatch = 64
	}
	def(&c.LR, 3e-4)
	def(&c.Gamma, 0.99)
	def(&c.Lambda, 0.95)
	def(&c.Clip, 0.2)
	def(&c.Value, 0.5)
	def(&c.MaxNorm, 0.5)
	if c.Hidden == nil {
		c.Hidden = []int{64, 64}
	}
	return c
}

// PPO is an agent of Schulman et al.'s proximal policy optimization: an
// actor that scores each action and a critic that values each
// observation.
type PPO struct {
	Actor  *nn.Sequential
	Critic *nn.Sequential
	obs    int
}

// Act returns the action the actor scores highest.
func (p *PPO) Act(obs []float32) int {
	out, err := rows(context.Background(), p.Actor, obs, 1, p.obs)
	if err != nil {
		return 0
	}
	return argmax(out)
}

// Probabilities returns the actor's chance of each action.
func (p *PPO) Probabilities(ctx context.Context, obs []float32) ([]float32, error) {
	return p.Actor.Forward(tensor.FromSlice(obs, 1, p.obs)).Softmax(1).Data[float32](ctx)
}

// NewPPO returns an untrained agent for env with c's networks.
func NewPPO(env Env, c PPOConfig) (*PPO, error) {
	if err := checkEnv(env); err != nil {
		return nil, err
	}
	c = c.resolve()
	return &PPO{Actor: mlp(env.Observation(), env.Actions(), c.Hidden, nn.Tanh()), Critic: mlp(env.Observation(), 1, c.Hidden, nn.Tanh()), obs: env.Observation()}, nil
}

// TrainPPO trains a PPO agent on environments made by newEnv, one per
// Envs, each seeded apart.
func TrainPPO(ctx context.Context, newEnv func(seed uint64) Env, c PPOConfig) (*PPO, error) {
	c = c.resolve()
	envs := make([]Env, c.Envs)
	for i := range envs {
		envs[i] = newEnv(c.Seed + uint64(i))
	}
	agent, err := NewPPO(envs[0], c)
	if err != nil {
		return nil, err
	}
	return agent, agent.Train(ctx, envs, c)
}

// Train continues training p on envs.
func (p *PPO) Train(ctx context.Context, envs []Env, c PPOConfig) error {
	c = c.resolve()
	for _, e := range envs {
		if err := checkEnv(e); err != nil {
			return err
		}
		if e.Observation() != p.obs {
			return fmt.Errorf("rl: environments of %d and %d numbers: %w", e.Observation(), p.obs, tensor.ErrShape)
		}
	}
	n, f, t := len(envs), p.obs, c.Rollout
	k := envs[0].Actions()
	r := rand.New(rand.NewPCG(c.Seed, 0x70706f))
	params := append(nn.Parameters(p.Actor), nn.Parameters(p.Critic)...)
	opt := optim.Adam(params, optim.LR(c.LR), optim.Eps(1e-5), optim.ClipNorm(c.MaxNorm))

	obs := make([]float32, n*f)
	for i, e := range envs {
		copy(obs[i*f:], e.Reset(c.Seed*1000+uint64(i)))
	}
	episode := make([]float64, n)
	resets := uint64(n)
	size := n * t
	bObs := make([]float32, size*f)
	bAct := make([]int32, size)
	bLogp := make([]float32, size)
	bVal := make([]float32, size+n)
	bRew := make([]float64, size)
	bDone := make([]bool, size)

	for done := 0; done < c.Steps; {
		var ended []float64
		for s := range t {
			if err := ctx.Err(); err != nil {
				return err
			}
			logits, err := rows(ctx, p.Actor, obs, n, f)
			if err != nil {
				return err
			}
			vals, err := rows(ctx, p.Critic, obs, n, f)
			if err != nil {
				return err
			}
			copy(bObs[s*n*f:], obs)
			copy(bVal[s*n:], vals)
			for i, e := range envs {
				l := logits[i*k : (i+1)*k]
				a, logp := sample(l, r)
				at := s*n + i
				bAct[at], bLogp[at] = int32(a), float32(logp)
				st := e.Step(a)
				rew := st.Reward
				episode[i] += st.Reward
				if st.Truncated && !st.Done {
					// Cut off with a future: count the value of where it was.
					v, err := rows(ctx, p.Critic, st.Obs, 1, f)
					if err != nil {
						return err
					}
					rew += c.Gamma * float64(v[0])
				}
				bRew[at], bDone[at] = rew, st.Done || st.Truncated
				next := st.Obs
				if st.Done || st.Truncated {
					ended = append(ended, episode[i])
					episode[i] = 0
					next = e.Reset(c.Seed*1000 + resets)
					resets++
				}
				copy(obs[i*f:], next)
			}
		}
		last, err := rows(ctx, p.Critic, obs, n, f)
		if err != nil {
			return err
		}
		copy(bVal[size:], last)
		adv, ret := gae(bRew, bVal, bDone, n, t, c.Gamma, c.Lambda)
		if err := p.update(ctx, opt, bObs, bAct, bLogp, adv, ret, f, c, r); err != nil {
			return err
		}
		done += size
		if c.Progress != nil && len(ended) > 0 {
			var sum float64
			for _, e := range ended {
				sum += e
			}
			c.Progress(done, sum/float64(len(ended)))
		}
	}
	return nil
}

// sample draws an action from logits' softmax and returns it with its log
// probability.
func sample(logits []float32, r *rand.Rand) (int, float64) {
	top := math.Inf(-1)
	for _, v := range logits {
		top = max(top, float64(v))
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v) - top)
	}
	u := r.Float64() * sum
	a := len(logits) - 1
	for i, v := range logits {
		u -= math.Exp(float64(v) - top)
		if u <= 0 {
			a = i
			break
		}
	}
	return a, float64(logits[a]) - top - math.Log(sum)
}

// gae returns the generalized advantage estimates and returns of a
// rollout of t steps of n environments, laid out step by step, values
// holding one more step for the observations after the last.
func gae(rew []float64, values []float32, done []bool, n, t int, gamma, lambda float64) (adv, ret []float32) {
	adv, ret = make([]float32, n*t), make([]float32, n*t)
	for i := range n {
		var a float64
		for s := t - 1; s >= 0; s-- {
			at := s*n + i
			next := float64(values[at+n])
			keep := 1.0
			if done[at] {
				keep, next = 0, 0
			}
			delta := rew[at] + gamma*next - float64(values[at])
			a = delta + gamma*lambda*keep*a
			adv[at] = float32(a)
			ret[at] = float32(a + float64(values[at]))
		}
	}
	return adv, ret
}

func (p *PPO) update(ctx context.Context, opt *optim.Optimizer, obs []float32, act []int32, oldLogp, adv, ret []float32, f int, c PPOConfig, r *rand.Rand) error {
	size := len(act)
	idx := make([]int, size)
	for i := range idx {
		idx[i] = i
	}
	for range c.Epochs {
		r.Shuffle(size, func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
		for lo := 0; lo+c.Minibatch <= size; lo += c.Minibatch {
			if err := ctx.Err(); err != nil {
				return err
			}
			m := c.Minibatch
			mo := make([]float32, 0, m*f)
			ma, ml, mr := make([]int32, m), make([]float32, m), make([]float32, m)
			madv := make([]float32, m)
			var mean, sq float64
			for j, i := range idx[lo : lo+m] {
				mo = append(mo, obs[i*f:(i+1)*f]...)
				ma[j], ml[j], mr[j], madv[j] = act[i], oldLogp[i], ret[i], adv[i]
				mean += float64(adv[i])
			}
			mean /= float64(m)
			for _, a := range madv {
				sq += (float64(a) - mean) * (float64(a) - mean)
			}
			std := math.Sqrt(sq/float64(m)) + 1e-8
			for j := range madv {
				madv[j] = float32((float64(madv[j]) - mean) / std)
			}
			x := tensor.FromSlice(mo, m, f)
			logp := p.Actor.Forward(x).LogSoftmax(1)
			chosen := logp.Gather(1, tensor.FromSlice(ma, m, 1)).Reshape(m)
			ratio := chosen.Sub(tensor.FromSlice(ml, m)).Exp()
			a := tensor.FromSlice(madv, m)
			clipped := ratio.Maximum(tensor.Scalar(float32(1 - c.Clip))).Minimum(tensor.Scalar(float32(1 + c.Clip)))
			policy := ratio.Mul(a).Minimum(clipped.Mul(a)).Mean().Neg()
			v := p.Critic.Forward(x).Reshape(m)
			d := v.Sub(tensor.FromSlice(mr, m))
			loss := policy.Add(d.Mul(d).Mean().Mul(tensor.Scalar(float32(c.Value))))
			if c.Entropy != 0 {
				ent := logp.Exp().Mul(logp).Sum(1).Mean().Neg()
				loss = loss.Sub(ent.Mul(tensor.Scalar(float32(c.Entropy))))
			}
			if err := loss.Backward(ctx); err != nil {
				return err
			}
			if err := opt.Step(ctx); err != nil {
				return err
			}
			opt.ZeroGrad()
		}
	}
	return nil
}
