package main

import (
	"testing"

	"github.com/gsoultan/mimir"
)

// small is a citation graph a few epochs learn quickly.
var small = Citations{
	Papers: 600, Links: 1200, Words: 200, Topics: 4,
	Homophily: 0.8, Focus: 0.5, WordsPerPaper: 10,
	Train: 20, Val: 100, Test: 200,
}

func TestGCNLearnsTopics(t *testing.T) {
	res, err := run(t.Context(), mimir.CPU(), config{Citations: small, Epochs: 30, Hidden: 16, LR: 0.01, Seed: 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Chance is a quarter.
	if res.Test < 0.6 {
		t.Errorf("test accuracy %.2f after 30 epochs", res.Test)
	}
}

func TestGenerateIsDeterministicAndSplitsApart(t *testing.T) {
	a, err := generate(small, 5)
	if err != nil {
		t.Fatal(err)
	}
	b, err := generate(small, 5)
	if err != nil {
		t.Fatal(err)
	}
	if a.graph.Edges() != 2*small.Links || a.graph.Edges() != b.graph.Edges() {
		t.Fatalf("%d and %d edges, want %d", a.graph.Edges(), b.graph.Edges(), 2*small.Links)
	}
	seen := map[int32]bool{}
	for _, split := range [][]int32{a.train, a.val, a.test} {
		for _, p := range split {
			if seen[p] {
				t.Fatalf("paper %d in two splits", p)
			}
			seen[p] = true
		}
	}
	if len(a.train) != small.Topics*small.Train || len(a.val) != small.Val || len(a.test) != small.Test {
		t.Errorf("splits of %d, %d and %d", len(a.train), len(a.val), len(a.test))
	}
	if _, err := generate(Citations{Papers: 10, Topics: 2, Words: 5, Train: 20}, 1); err == nil {
		t.Error("too few papers for the splits")
	}
}

// BenchmarkTrain times 50 epochs of training and evaluation on the
// Cora-sized graph.
func BenchmarkTrain(b *testing.B) {
	cfg := config{Citations: cora, Epochs: 50, Hidden: 16, LR: 0.01, Seed: 1}
	for b.Loop() {
		res, err := run(b.Context(), mimir.CPU(), cfg, nil)
		if err != nil {
			b.Fatal(err)
		}
		b.ReportMetric(float64(res.Time.Microseconds())/1e3/50, "ms/epoch")
	}
}
