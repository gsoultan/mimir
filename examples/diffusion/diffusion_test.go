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

// TestScheduleEndsInNoise checks that the forward process keeps almost
// all of the signal after its first step and almost none after its last,
// whatever the number of steps.
func TestScheduleEndsInNoise(t *testing.T) {
	for _, steps := range []int{1000, 500, 50} {
		s := newSchedule(steps)
		if first, last := s.alphaBar[0], s.alphaBar[steps-1]; first < 0.99 || last > 0.01 {
			t.Errorf("%d steps: ᾱ %g after the first and %g after the last, want over 0.99 and under 0.01", steps, first, last)
		}
	}
}

// TestSamplerRecoversAPointMass runs the ancestral and DDIM samplers with
// the exact noise prediction for data that are always the same image c,
// ε = (x_t - √ᾱ_t·c) / √(1-ᾱ_t), and checks that they return c.
func TestSamplerRecoversAPointMass(t *testing.T) {
	ctx := t.Context()
	m, err := NewDDPM(Config{Pixels: 4, Hidden: 4, Time: 2, Steps: 100}, 1)
	if err != nil {
		t.Fatal(err)
	}
	c := []float32{-0.9, -0.2, 0.3, 0.8}
	exact := func(x *tensor.Tensor, ts []int) *tensor.Tensor {
		ab := m.schedule.alphaBar[ts[0]]
		return x.Sub(tensor.FromSlice(c, 1, 4).Mul(tensor.Scalar(float32(math.Sqrt(ab))))).Mul(tensor.Scalar(float32(1 / math.Sqrt(1-ab))))
	}
	ancestral, err := m.sample(ctx, 3, 5, exact)
	if err != nil {
		t.Fatal(err)
	}
	ddim, err := m.sampleDDIM(ctx, 3, 5, 7, exact)
	if err != nil {
		t.Fatal(err)
	}
	for name, s := range map[string]*tensor.Tensor{"ancestral": ancestral, "DDIM": ddim} {
		got, err := s.Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if math.Abs(float64(v-c[i%4])) > 1e-4 {
				t.Errorf("%s: pixel %d of image %d is %v, want %v", name, i%4, i/4, v, c[i%4])
			}
		}
	}
	if _, err := m.SampleDDIM(ctx, 3, 5, 0); err == nil {
		t.Error("DDIM with stride 0 succeeded")
	}
}

// TestTrainingLowersTheLoss trains a tiny DDPM on bars and checks that
// its loss falls by over a quarter, to well below 1, the loss of
// predicting no velocity for pixels of ±1, and that the sampler then
// draws finite images whose pixels lean towards the dark background as
// the bars' do.
func TestTrainingLowersTheLoss(t *testing.T) {
	ctx := t.Context()
	m, err := NewDDPM(Config{Pixels: 64, Hidden: 128, Time: 16, Steps: 50}, 3)
	if err != nil {
		t.Fatal(err)
	}
	tr, err := NewTrainer(ctx, m, 3e-3, 5)
	if err != nil {
		t.Fatal(err)
	}
	x := bars(256, 1)
	var first, last float64
	for i := range 240 {
		lo := i % 4 * 64
		v, err := tr.Step(ctx, x.Shrink([][2]int{{lo, lo + 64}, {0, 64}}))
		if err != nil {
			t.Fatal(err)
		}
		if i < 10 {
			first += v / 10
		}
		if i >= 230 {
			last += v / 10
		}
	}
	t.Logf("mean loss %.4f over the first 10 steps, %.4f over the last 10", first, last)
	if last > 0.6 || last > first*3/4 {
		t.Errorf("mean loss %.4f over the first 10 steps and %.4f over the last 10, want a fall by over a quarter to under 0.6", first, last)
	}

	s, err := m.Sample(ctx, 32, 7)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Shape(); len(got) != 2 || got[0] != 32 || got[1] != 64 {
		t.Fatalf("samples of shape %v, want [32 64]", got)
	}
	px, err := s.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	var mean float64
	for _, v := range px {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("sample pixel %v", v)
		}
		mean += float64(v) / float64(len(px))
	}
	if mean > -0.3 {
		t.Errorf("mean sample pixel %.3f, want near the bars' -0.75", mean)
	}
	if _, err := NewDDPM(Config{Pixels: 64, Hidden: 8, Time: 3, Steps: 10}, 1); err == nil {
		t.Error("NewDDPM with an odd embedding succeeded")
	}
}

// BenchmarkStep times one training step on a batch of 128 MNIST-sized
// images with the command's default sizes.
func BenchmarkStep(b *testing.B) {
	ctx := b.Context()
	m, err := NewDDPM(Config{Pixels: 784, Hidden: 1024, Time: 64, Steps: 500}, 1)
	if err != nil {
		b.Fatal(err)
	}
	tr, err := NewTrainer(ctx, m, 1e-3, 1)
	if err != nil {
		b.Fatal(err)
	}
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
