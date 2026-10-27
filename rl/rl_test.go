package rl_test

import (
	"context"
	"math"
	"testing"

	"github.com/gsoultan/mimir/rl"
)

func TestCartPolePhysics(t *testing.T) {
	c := rl.NewCartPole()
	obs := c.Reset(1)
	for _, v := range obs {
		if math.Abs(float64(v)) > 0.05 {
			t.Fatalf("start %v", obs)
		}
	}
	// Pushing one way the whole time drops the pole within 100 steps.
	steps := 0
	for {
		steps++
		if s := c.Step(1); s.Done || s.Truncated {
			if !s.Done || steps > 100 {
				t.Errorf("pushed right: done %v after %d steps", s.Done, steps)
			}
			break
		}
	}
}

type always int

func (a always) Act([]float32) int { return int(a) }

func TestChainRewards(t *testing.T) {
	ctx := context.Background()
	env := &rl.Chain{N: 6}
	left, _ := rl.Evaluate(ctx, always(0), env, 3, 1)
	right, _ := rl.Evaluate(ctx, always(1), env, 3, 1)
	if left != 0.01 || right != 1 {
		t.Errorf("left %v, right %v", left, right)
	}
}

func TestPPOBalancesCartPole(t *testing.T) {
	if testing.Short() {
		t.Skip("trains an agent")
	}
	ctx := context.Background()
	var last float64
	agent, err := rl.TrainPPO(ctx, func(uint64) rl.Env { return rl.NewCartPole() }, rl.PPOConfig{
		Steps: 40_000, LR: 1e-3, Seed: 1, Progress: func(_ int, r float64) { last = r },
	})
	if err != nil {
		t.Fatal(err)
	}
	score, err := rl.Evaluate(ctx, agent, rl.NewCartPole(), 5, 100)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("last rollout's mean reward %.1f, greedy score %.1f", last, score)
	if score < 300 {
		t.Errorf("PPO keeps the pole up %.0f steps, want at least 300 of 500", score)
	}
}

func TestDQNWalksTheChain(t *testing.T) {
	if testing.Short() {
		t.Skip("trains an agent")
	}
	ctx := context.Background()
	env := &rl.Chain{N: 8}
	agent, err := rl.TrainDQN(ctx, env, rl.DQNConfig{Steps: 4000, StartAfter: 500, TargetEvery: 100, Hidden: []int{32}, Seed: 2})
	if err != nil {
		t.Fatal(err)
	}
	score, err := rl.Evaluate(ctx, agent, env, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if score != 1 {
		t.Errorf("DQN scores %v on the chain, want 1 for the far end", score)
	}
}
