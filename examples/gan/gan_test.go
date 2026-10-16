package main

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// bars returns n 8×8 images in [-1, 1], each one lit row or column on a
// dark background.
func bars(n int, seed uint64) *tensor.Tensor {
	r := rand.New(rand.NewPCG(seed, 0))
	px := make([]float32, n*64)
	for i := range px {
		px[i] = -1
	}
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

// gap returns the mean squared difference between the mean images of a
// and b.
func gap(t *testing.T, a, b *tensor.Tensor) float64 {
	t.Helper()
	d := a.Mean(0).Sub(b.Mean(0))
	v, err := d.Mul(d).Mean().Item[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return float64(v)
}

// TestTrainingLowersTheLosses trains a tiny GAN on bars and checks that
// the discriminator's loss falls below its value for a coin toss, 2 ln 2,
// as it first learns to tell bars from noise, and that the generator
// learns their statistics meanwhile: the gap between the mean generated
// image and the mean bar falls over threefold. (On data this simple the
// discriminator soon wins, so the test stops early.)
func TestTrainingLowersTheLosses(t *testing.T) {
	ctx := t.Context()
	m, err := NewGAN(Config{Pixels: 64, Noise: 8, Hidden: 32}, 3)
	if err != nil {
		t.Fatal(err)
	}
	tr := NewTrainer(m, 5e-4, 5)
	x := bars(256, 1)
	before := gap(t, m.Generate(256, 9), x)
	var first, least Losses
	for i := range 100 {
		lo := i % 8 * 32
		l, err := tr.Step(ctx, x.Shrink([][2]int{{lo, lo + 32}, {0, 64}}))
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(l.D) || math.IsNaN(l.G) {
			t.Fatalf("step %d: losses %+v", i, l)
		}
		if i == 0 {
			first, least = l, l
		}
		least.D = min(least.D, l.D)
	}
	after := gap(t, m.Generate(256, 9), x)
	t.Logf("discriminator loss %.3f at first, least %.3f; mean-image gap %.4f before, %.4f after", first.D, least.D, before, after)
	if math.Abs(first.D-2*math.Ln2) > 0.3 || least.D > first.D-0.2 {
		t.Errorf("discriminator loss %.3f at first and %.3f at least, want about %.3f falling by over 0.2", first.D, least.D, 2*math.Ln2)
	}
	if after > before/3 {
		t.Errorf("mean-image gap %.4f before and %.4f after training, want a threefold fall", before, after)
	}
	if _, err := NewGAN(Config{Pixels: 64}, 1); err == nil {
		t.Error("NewGAN without noise succeeded")
	}
}

// BenchmarkStep times one training step, of the discriminator and of the
// generator, on a batch of 64 MNIST-sized images with the command's
// default sizes.
func BenchmarkStep(b *testing.B) {
	ctx := b.Context()
	m, err := NewGAN(Config{Pixels: 784, Noise: 64, Hidden: 256}, 1)
	if err != nil {
		b.Fatal(err)
	}
	tr := NewTrainer(m, 2e-4, 1)
	x := tensor.Rand(1, 64, 784)
	if err := x.Realize(ctx); err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := tr.Step(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}
