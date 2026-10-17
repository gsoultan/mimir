package main

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
)

func TestPPOImprovesOnASmallBudget(t *testing.T) {
	cfg := defaults
	cfg.Envs, cfg.Steps, cfg.Minibatch, cfg.Updates, cfg.Seed = 4, 128, 128, 6, 3
	res, err := run(t.Context(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, l := range res.Losses {
		if math.IsNaN(l) || math.IsInf(l, 0) {
			t.Fatalf("update %d: loss %v", i, l)
		}
	}
	// Random pushes last about 22 steps.
	early, late := mean(res.Returns[:10]), mean(res.Returns[len(res.Returns)-10:])
	if late < 1.5*early {
		t.Errorf("mean return %.1f in the first 10 episodes and %.1f in the last 10", early, late)
	}
}

func TestAdvantagesAreDiscountedReturnsWithLambdaOne(t *testing.T) {
	// Two environments, five steps; the first ends its episode at step 2,
	// the second is truncated at step 3 and bootstraps from 4.
	const envs, steps, gamma = 2, 5, 0.9
	r := newRollout(steps, envs)
	rng := rand.New(rand.NewPCG(1, 1))
	for i := range r.reward {
		r.reward[i], r.value[i] = rng.Float32(), rng.Float32()
	}
	r.done[2*envs+0], r.done[3*envs+1], r.finalValue[3*envs+1] = 1, 1, 4
	r.lastValue[0], r.lastValue[1] = 2, 3
	r.advantages(envs, gamma, 1)
	for e := range envs {
		for s := range steps {
			// The discounted rewards to the end of the episode or rollout,
			// then the bootstrapped value.
			var g, f float64 = 0, 1
			for u := s; u < steps; u++ {
				i := u*envs + e
				g += f * float64(r.reward[i])
				f *= gamma
				if r.done[i] != 0 {
					g += f * float64(r.finalValue[i])
					break
				}
				if u == steps-1 {
					g += f * float64(r.lastValue[e])
				}
			}
			i := s*envs + e
			if math.Abs(float64(r.ret[i])-g) > 1e-5 || math.Abs(float64(r.adv[i]+r.value[i])-g) > 1e-5 {
				t.Errorf("env %d step %d: return %v, advantage plus value %v, want %v", e, s, r.ret[i], r.adv[i]+r.value[i], g)
			}
		}
	}
}

func TestSampleFollowsTheSoftmax(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	logits := []float32{0.3, 1.5}
	p1 := 1 / (1 + math.Exp(0.3-1.5))
	ones := 0
	const n = 20000
	for range n {
		a, lp := sample(rng, logits)
		want := math.Log(1 - p1)
		if a == 1 {
			ones++
			want = math.Log(p1)
		}
		if math.Abs(float64(lp)-want) > 1e-6 {
			t.Fatalf("log-probability %v of action %d, want %v", lp, a, want)
		}
	}
	if got := float64(ones) / n; math.Abs(got-p1) > 0.01 {
		t.Errorf("action 1 drawn %.3f of the time, want %.3f", got, p1)
	}
}

// BenchmarkUpdate times one PPO update: four epochs of minibatches of
// 256 over a rollout of 8 environments and 128 steps.
func BenchmarkUpdate(b *testing.B) {
	cfg := defaults
	nn.ManualSeed(1)
	p := newPolicy(cfg.Hidden)
	opt := optim.Adam(nn.Parameters(p), optim.LR(1e-4))
	r := newRollout(cfg.Steps, cfg.Envs)
	rng := rand.New(rand.NewPCG(1, 2))
	for i := range r.action {
		for k := range 4 {
			r.obs[4*i+k] = rng.Float32()*0.2 - 0.1
		}
		r.action[i], r.logp[i] = int32(rng.IntN(2)), float32(math.Log(0.5))
		r.adv[i], r.ret[i] = rng.Float32()*2-1, rng.Float32()*10
	}
	m := newMinibatch(cfg.Minibatch)
	for b.Loop() {
		if _, err := update(b.Context(), cfg, p, opt, r, rng, m); err != nil {
			b.Fatal(err)
		}
	}
}
