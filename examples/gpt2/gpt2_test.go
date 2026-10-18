package main

import (
	"bytes"
	"io"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// corpus is a little text with plenty of repetition, which a tiny model
// learns to predict within a few dozen steps.
var corpus = strings.Repeat(`To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to.
`, 8)

func tinyConfig() Config {
	return Config{Vocab: byteVocab, Context: 32, Embed: 32, Heads: 4, Layers: 2}
}

func newTrainer(t *testing.T, half tensor.DType) *Trainer {
	t.Helper()
	g, err := NewGPT(tinyConfig(), 7)
	if err != nil {
		t.Fatal(err)
	}
	tr, err := NewTrainer(g, encode([]byte(corpus)), TrainConfig{
		Batch: 8, Steps: 40, LR: 3e-3, MinLR: 3e-4, Warmup: 5, Clip: 1, Decay: 0.1,
		Half: half, Seed: 1, Log: 10, EvalIter: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

// TestTrainingLowersTheLoss trains a tiny model for 40 steps, in float32
// and with bfloat16 weights, and checks that the loss falls well below
// that of guessing among the 256 bytes, ln 256 ≈ 5.5, on training and
// held-out batches alike.
func TestTrainingLowersTheLoss(t *testing.T) {
	for _, half := range []tensor.DType{tensor.Float32, tensor.BFloat16} {
		t.Run(half.String(), func(t *testing.T) {
			ctx := t.Context()
			tr := newTrainer(t, half)
			if dt := tr.Model.Blocks[0].Attn.QKV.Weight.Value().DType(); dt != half {
				t.Fatalf("weights of %s, want %s", dt, half)
			}
			before, err := tr.Validate(ctx)
			if err != nil {
				t.Fatal(err)
			}
			var first, last float64
			for i := range tr.Cfg.Steps {
				v, norm, skipped, err := tr.Step(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if skipped || math.IsNaN(norm) {
					t.Fatalf("step %d: skipped %v, norm %v", i, skipped, norm)
				}
				if i == 0 {
					first = v
				}
				last = v
			}
			after, err := tr.Validate(ctx)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("training loss %.3f → %.3f, validation %.3f → %.3f", first, last, before, after)
			if first < 5 || last > first-1.5 || after > before-1.5 {
				t.Errorf("training loss %.3f → %.3f, validation %.3f → %.3f: want a fall of 1.5 from about ln 256", first, last, before, after)
			}
		})
	}
}

// TestScheduleWarmsUpAndDecays checks the learning rate's shape.
func TestScheduleWarmsUpAndDecays(t *testing.T) {
	c := TrainConfig{Steps: 100, LR: 1, MinLR: 0.1, Warmup: 10}
	if lr := c.LRAt(0); lr != 0.1 {
		t.Errorf("step 0: %v, want 0.1", lr)
	}
	if lr := c.LRAt(9); lr != 1 {
		t.Errorf("step 9: %v, want 1", lr)
	}
	if lr := c.LRAt(55); math.Abs(lr-0.55) > 1e-12 {
		t.Errorf("halfway through the decay: %v, want 0.55", lr)
	}
	if lr := c.LRAt(100); math.Abs(lr-0.1) > 1e-12 {
		t.Errorf("step 100: %v, want 0.1", lr)
	}
}

// TestGradNormMatchesTheGradients checks the norm that clipping divides
// the gradients by against one computed from their values.
func TestGradNormMatchesTheGradients(t *testing.T) {
	ctx := t.Context()
	tr := newTrainer(t, tensor.Float32)
	x, y := tr.batch(tr.train, rand.New(rand.NewPCG(3, 4)))
	if err := tr.lossOf(x, y).Backward(ctx); err != nil {
		t.Fatal(err)
	}
	got, err := tr.gradNorm(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var want float64
	for _, p := range nn.Parameters(tr.Model) {
		g, err := p.Grad().Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range g {
			want += float64(v) * float64(v)
		}
	}
	want = math.Sqrt(want)
	if math.Abs(got-want) > 1e-4*want {
		t.Errorf("gradient norm %v, want %v", got, want)
	}
}

// TestCachedGenerationMatchesUncached generates greedily from a trained
// model with the KV cache and by running the whole sequence every step,
// and checks that both produce the same tokens, and that a checkpoint
// reproduces the model.
func TestCachedGenerationMatchesUncached(t *testing.T) {
	ctx := t.Context()
	tr := newTrainer(t, tensor.Float32)
	for range 20 {
		if _, _, _, err := tr.Step(ctx); err != nil {
			t.Fatal(err)
		}
	}
	prompt := encode([]byte("To be, or "))
	greedy := &Sampler{}
	n := tr.Model.Config.Context - len(prompt)
	cached, err := Generate(ctx, tr.Model, prompt, n, greedy, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	uncached, err := Generate(ctx, tr.Model, prompt, n, greedy, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cached, uncached) {
		t.Errorf("cached generation %q differs from uncached %q", decodeBytes(cached), decodeBytes(uncached))
	}

	path := filepath.Join(t.TempDir(), "gpt2.mim")
	if err := tr.Save(ctx, path); err != nil {
		t.Fatal(err)
	}
	g, err := LoadGPT(path)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := Generate(ctx, g, prompt, n, greedy, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(loaded, cached) {
		t.Errorf("generation after loading the checkpoint %q differs from %q", decodeBytes(loaded), decodeBytes(cached))
	}
}

// TestCachedLogitsMatchUncached checks the logits of every position
// decoded one at a time against those of one pass over the sequence.
func TestCachedLogitsMatchUncached(t *testing.T) {
	ctx := t.Context()
	g, err := NewGPT(tinyConfig(), 3)
	if err != nil {
		t.Fatal(err)
	}
	seq := encode([]byte("that is the question"))
	all, err := g.Forward(tensor.FromSlice(seq, 1, len(seq))).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	caches := g.NewCaches(1)
	v := g.Config.Vocab
	for i, tok := range seq {
		logits, err := g.Decode(ctx, tensor.FromSlice([]int32{tok}, 1, 1), caches)
		if err != nil {
			t.Fatal(err)
		}
		got, err := logits.Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for j, w := range all[i*v : (i+1)*v] {
			if math.Abs(float64(got[j]-w)) > 1e-4*(1+math.Abs(float64(w))) {
				t.Fatalf("position %d, token %d: cached logit %v, uncached %v", i, j, got[j], w)
			}
		}
	}
}

// TestResumeContinuesTraining checks that a trainer resumed from a
// checkpoint takes the same next step as the one that saved it.
func TestResumeContinuesTraining(t *testing.T) {
	ctx := t.Context()
	a := newTrainer(t, tensor.Float32)
	for range 3 {
		if _, _, _, err := a.Step(ctx); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "gpt2.mim")
	if err := a.Save(ctx, path); err != nil {
		t.Fatal(err)
	}
	b := newTrainer(t, tensor.Float32)
	if ok, err := b.Resume(path); !ok || err != nil {
		t.Fatalf("resume: %v, %v", ok, err)
	}
	if b.Opt.Steps() != 3 {
		t.Fatalf("resumed at step %d, want 3", b.Opt.Steps())
	}
	b.rng = rand.New(rand.NewPCG(9, 9))
	a.rng = rand.New(rand.NewPCG(9, 9))
	la, _, _, errA := a.Step(ctx)
	lb, _, _, errB := b.Step(ctx)
	if errA != nil || errB != nil {
		t.Fatal(errA, errB)
	}
	if la != lb {
		t.Errorf("loss after resuming %v, want %v", lb, la)
	}
	if err := a.Run(ctx, io.Discard, ""); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	b.Cfg.Steps = 6
	if err := b.Run(ctx, &buf, ""); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "tokens/s") {
		t.Errorf("progress %q does not report tokens/s", buf.String())
	}
}

func decodeBytes(toks []int32) string {
	b := make([]byte, len(toks))
	for i, t := range toks {
		b[i] = byte(t)
	}
	return string(b)
}
