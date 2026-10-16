package main

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/examples/rl/cartpole"
)

func TestDQNImprovesOnASmallBudget(t *testing.T) {
	cfg := defaults
	cfg.Episodes, cfg.Hidden, cfg.Batch, cfg.Start, cfg.EpsilonSteps, cfg.SyncEvery, cfg.Seed = 80, 64, 32, 300, 1500, 200, 2
	res, err := run(t.Context(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Episodes != cfg.Episodes && !res.Solved {
		t.Fatalf("%d episodes, want %d", res.Episodes, cfg.Episodes)
	}
	// The first episodes act at random, and last about 22 steps.
	early, late := mean(res.Returns[:20]), mean(res.Returns[len(res.Returns)-20:])
	if late < 2*early {
		t.Errorf("mean return %.1f in the first 20 episodes and %.1f in the last 20", early, late)
	}
}

func TestUpdateLowersTheLossOnAFixedBatch(t *testing.T) {
	cfg := defaults
	a, m := newTestAgent(t, cfg)
	var first, last float32
	for i := range 50 {
		l, err := a.update(t.Context(), m)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			t.Fatalf("update %d: loss %v", i, l)
		}
		if i == 0 {
			first = l
		}
		last = l
	}
	if last >= first {
		t.Errorf("loss %v after 50 updates, %v at first", last, first)
	}
}

// newTestAgent returns an agent and a batch of random transitions.
func newTestAgent(t testing.TB, cfg config) (*agent, *batch) {
	t.Helper()
	a := newAgent(cfg)
	if err := a.sync(t.Context()); err != nil {
		t.Fatal(err)
	}
	buf := newReplay(1000)
	env := cartpole.New(1)
	obs := env.Reset()
	rng := rand.New(rand.NewPCG(1, 2))
	for range 1000 {
		action := rng.IntN(2)
		next, r, term, trunc := env.Step(action)
		buf.add(obs, next, action, r, term)
		obs = next
		if term || trunc {
			obs = env.Reset()
		}
	}
	m := newBatch(cfg.Batch)
	buf.sample(rng, m)
	return a, m
}

// BenchmarkUpdate times one optimizer step on a minibatch.
func BenchmarkUpdate(b *testing.B) {
	a, m := newTestAgent(b, defaults)
	for b.Loop() {
		if _, err := a.update(b.Context(), m); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAct times choosing one greedy action.
func BenchmarkAct(b *testing.B) {
	a, _ := newTestAgent(b, defaults)
	obs := []float32{0.01, -0.02, 0.03, 0.04}
	for b.Loop() {
		if _, err := a.greedy(b.Context(), obs, 1); err != nil {
			b.Fatal(err)
		}
	}
}
