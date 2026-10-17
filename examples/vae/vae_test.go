package main

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// bars returns n binary 8×8 images, each one lit row or column, which a
// small VAE learns to code in a few dozen steps.
func bars(n int, seed uint64) *tensor.Tensor {
	r := rand.New(rand.NewPCG(seed, 0))
	px := make([]float32, n*64)
	for i := range n {
		k := r.IntN(8)
		vertical := r.IntN(2) == 1
		for j := range 8 {
			at := k*8 + j
			if vertical {
				at = j*8 + k
			}
			px[i*64+at] = 1
		}
	}
	return tensor.FromSlice(px, n, 64)
}

// TestTrainingRaisesTheELBO trains a tiny VAE on bars and checks that the
// ELBO of held-out bars climbs well above its starting value of about
// 64·ln ½ ≈ -44 nats.
func TestTrainingRaisesTheELBO(t *testing.T) {
	ctx := t.Context()
	m, err := NewVAE(Config{Pixels: 64, Hidden: 64, Latent: 4}, 3)
	if err != nil {
		t.Fatal(err)
	}
	tr := NewTrainer(m, 1e-2, 5)
	x, held := bars(256, 1), bars(64, 2)
	before, err := tr.Evaluate(ctx, held)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 60 {
		lo := i % 8 * 32
		v, err := tr.Step(ctx, x.Shrink([][2]int{{lo, lo + 32}, {0, 64}}))
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(v) {
			t.Fatalf("step %d: ELBO is NaN", i)
		}
	}
	after, err := tr.Evaluate(ctx, held)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("held-out ELBO %.2f before, %.2f after 60 steps", before, after)
	if before > -35 || after < before+20 {
		t.Errorf("held-out ELBO %.2f before and %.2f after training, want a rise of over 20 nats", before, after)
	}

	dir := t.TempDir()
	if err := pictures(ctx, m, held, 1, dir, 8); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"samples.png", "reconstructions.png"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Error(err)
		}
	}
	if _, err := NewVAE(Config{Pixels: 64}, 1); err == nil {
		t.Error("NewVAE without hidden units succeeded")
	}
}

// BenchmarkStep times one training step on a batch of 128 MNIST-sized
// images with the command's default sizes.
func BenchmarkStep(b *testing.B) {
	ctx := b.Context()
	m, err := NewVAE(Config{Pixels: 784, Hidden: 400, Latent: 20}, 1)
	if err != nil {
		b.Fatal(err)
	}
	tr := NewTrainer(m, 1e-3, 1)
	x := tensor.Rand(1, 128, 784)
	if err := x.Realize(ctx); err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := tr.Step(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}
