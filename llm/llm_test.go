package llm_test

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/llm"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// The tiny models, converted by mim-import -graph -a8=false -kv f32 -ctx
// 64 from the GGUF files of mimir-model's testdata: their graphs decode
// the weights and multiply in float32, as Model does with its weights
// dequantized, so the two agree to float32's rounding.
var models = []struct{ name, path string }{
	{"llama", filepath.Join("..", "io", "mim", "testdata", "llama", "tiny-q4km-exact.mim")},
	{"qwen2", filepath.Join("testdata", "qwen2-exact.mim")},
}

var prompt = []int32{1, 79, 110, 99, 101, 363, 264, 315, 307, 280}

func open(t *testing.T, path string) *mim.Model {
	t.Helper()
	f, err := mim.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

// load builds the model in f, with its weights dequantized to float32 if
// deq is set.
func load(t *testing.T, f *mim.Model, deq bool) *llm.Model {
	t.Helper()
	md, err := f.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	m, err := llm.Load(f, md, tensor.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if deq {
		for _, p := range m.Params() {
			v := p.Value().Cast(tensor.Float32)
			if err := v.Realize(t.Context()); err != nil {
				t.Fatal(err)
			}
			if err := p.Set(v); err != nil {
				t.Fatal(err)
			}
		}
	}
	return m
}

// graphLogits runs ids through the graph of f from position 0 and returns
// the logits of every position.
func graphLogits(t *testing.T, f *mim.Model, ids []int32) []float32 {
	t.Helper()
	p := make([]int32, len(ids))
	for i := range p {
		p[i] = int32(i)
	}
	out, err := f.NewSession().Run(t.Context(), map[string]*tensor.Tensor{
		"tokens":    tensor.FromSlice(ids, 1, len(ids)),
		"positions": tensor.FromSlice(p, 1, len(ids)),
	})
	if err != nil {
		t.Fatal(err)
	}
	return data(t, out["logits"])
}

// forward returns the model's logits of every position of ids.
func forward(t *testing.T, m *llm.Model, ids []int32) []float32 {
	t.Helper()
	return data(t, nn.Inference(m).Forward(tensor.FromSlice(ids, 1, len(ids))))
}

func data(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, err := x.Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// near fails unless got is within tol of want, relative to want's
// largest magnitude.
func near(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	var scale, worst float64
	for i := range want {
		scale = max(scale, math.Abs(float64(want[i])))
		worst = max(worst, math.Abs(float64(got[i]-want[i])))
	}
	if worst > tol*scale {
		t.Errorf("%s: off by %.3g, %.2g%% of the largest value %.3g", what, worst, 100*worst/scale, scale)
	}
}

// TestMatchesGraph checks Model against the graph mim-import built for
// the same file: all positions at once with Forward, and one token at a
// time through the caches with Logits.
func TestMatchesGraph(t *testing.T) {
	for _, tc := range models {
		t.Run(tc.name, func(t *testing.T) {
			f := open(t, tc.path)
			m := load(t, f, true)
			if m.Config.Arch != tc.name {
				t.Errorf("architecture %q", m.Config.Arch)
			}
			want := graphLogits(t, f, prompt)
			near(t, "forward", forward(t, m, prompt), want, 2e-5)

			vocab := m.Config.Vocab
			caches := m.NewCaches(len(prompt))
			for i, id := range prompt {
				l, err := m.Logits(t.Context(), []int32{id}, caches, false)
				if err != nil {
					t.Fatal(err)
				}
				near(t, "step", data(t, l), want[i*vocab:(i+1)*vocab], 2e-5)
			}
		})
	}
}

// train fine-tunes m's pairs on seq, learning every token, for steps
// steps, and returns the loss before and after.
func train(t *testing.T, m *llm.Model, seq []int32, steps int) (before, after float64) {
	t.Helper()
	learn := make([]bool, len(seq))
	for i := range learn {
		learn[i] = true
	}
	seqs, masks := [][]int32{seq}, [][]bool{learn}
	before, _, _, err := m.Score(t.Context(), seqs, masks)
	if err != nil {
		t.Fatal(err)
	}
	opt := optim.AdamW(m.Adapters(), optim.LR(1e-2), optim.WeightDecay(0))
	for range steps {
		loss := m.Loss(seqs, masks)
		if err := loss.Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
	}
	after, _, _, err = m.Score(t.Context(), seqs, masks)
	if err != nil {
		t.Fatal(err)
	}
	return before, after
}

// TestLoRALearns fine-tunes the pairs of each model, whose own weights
// stay quantized, until it repeats a sequence it found unlikely.
func TestLoRALearns(t *testing.T) {
	seq := []int32{1, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150}
	for _, tc := range models {
		t.Run(tc.name, func(t *testing.T) {
			m := load(t, open(t, tc.path), false)
			if got := forward(t, m, seq); len(got) == 0 {
				t.Fatal("no logits")
			}
			base := forward(t, m, seq)
			if err := m.AddLoRA(llm.LoRAConfig{Rank: 4, Targets: llm.Projections}); err != nil {
				t.Fatal(err)
			}
			near(t, "fresh pairs", forward(t, m, seq), base, 0)
			if n := len(m.Adapters()); n != 2*7*m.Config.Layers {
				t.Errorf("%d adapter parameters", n)
			}
			before, after := train(t, m, seq, 100)
			t.Logf("loss %.3f -> %.3f", before, after)
			if !(after < before/4) {
				t.Errorf("loss %.3f -> %.3f, want it to fall to a quarter", before, after)
			}
			learn := make([]bool, len(seq))
			for i := range learn {
				learn[i] = true
			}
			_, right, total, err := m.Score(t.Context(), [][]int32{seq}, [][]bool{learn})
			if err != nil {
				t.Fatal(err)
			}
			if right < total-1 {
				t.Errorf("%d of %d tokens right after training", right, total)
			}
			if err := m.AddLoRA(llm.LoRAConfig{}); err == nil {
				t.Error("pairs added twice")
			}
		})
	}
}

// TestAdapterRoundTrip writes a fine-tune's pairs as an adapter file and
// checks that the logits come back with them: on a fresh Model with
// ReadAdapter, and in the base graph with mim.Model.ApplyLoRA.
func TestAdapterRoundTrip(t *testing.T) {
	seq := []int32{1, 50, 60, 70, 80, 90, 100, 110}
	const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	for _, tc := range models {
		t.Run(tc.name, func(t *testing.T) {
			m := load(t, open(t, tc.path), true)
			if err := m.AddLoRA(llm.LoRAConfig{Rank: 2, Alpha: 3, Targets: []string{"attn_q", "attn_v", "ffn_down"}}); err != nil {
				t.Fatal(err)
			}
			train(t, m, seq, 5)
			want := forward(t, m, prompt)

			var buf bytes.Buffer
			if err := m.WriteAdapter(t.Context(), &buf, sha, "test"); err != nil {
				t.Fatal(err)
			}
			a, err := mim.Parse(buf.Bytes())
			if err != nil {
				t.Fatal(err)
			}
			defer a.Close()
			l, err := a.LoRA()
			if err != nil {
				t.Fatal(err)
			}
			if l.Alpha != 3 || l.Base != sha || len(l.Layers) != 3*m.Config.Layers {
				t.Errorf("adapter of alpha %v, base %q, %d layers", l.Alpha, l.Base, len(l.Layers))
			}

			fresh := load(t, open(t, tc.path), true)
			if err := fresh.ReadAdapter(a); err != nil {
				t.Fatal(err)
			}
			near(t, "ReadAdapter", forward(t, fresh, prompt), want, 1e-6)

			g := open(t, tc.path)
			if err := g.ApplyLoRA(a, "f"+sha[1:]); err == nil {
				t.Error("adapter of another base applied")
			}
			if err := g.ApplyLoRA(a, sha); err != nil {
				t.Fatal(err)
			}
			near(t, "ApplyLoRA", graphLogits(t, g, prompt), want, 2e-5)
			if err := g.ApplyLoRA(a, sha); err == nil {
				t.Error("adapter applied twice")
			}
		})
	}
}

func TestEmbedding(t *testing.T) {
	m := load(t, open(t, models[1].path), false)
	ctx := context.Background()
	a, err := m.Embedding(ctx, []int32{1, 50, 60, 70})
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != m.Config.Embed {
		t.Fatalf("%d features, want %d", len(a), m.Config.Embed)
	}
	var norm float64
	for _, x := range a {
		norm += float64(x) * float64(x)
	}
	if math.Abs(norm-1) > 1e-4 {
		t.Errorf("squared length %v, want 1", norm)
	}
	b, err := m.Embedding(ctx, []int32{1, 50, 60, 70})
	if err != nil {
		t.Fatal(err)
	}
	near(t, "same text", b, a, 0)
	if _, err := m.Embedding(ctx, nil); err == nil {
		t.Error("embedding of no tokens")
	}
	if _, err := m.Embedding(ctx, []int32{int32(m.Config.Vocab)}); err == nil {
		t.Error("embedding of a token past the vocabulary")
	}
}

// TestStepsMatchForward checks the steps that decode one token at a time
// against the forward pass over the whole sequence, at positions on both
// sides of a step's window growing, with the weights quantized and not,
// and after a parameter's value changes under the recorded steps. With
// quantized weights a step and the forward pass quantize the activations
// in blocks of different sizes, and agree only to a few percent.
func TestStepsMatchForward(t *testing.T) {
	for _, tc := range models {
		for _, deq := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/deq=%v", tc.name, deq), func(t *testing.T) {
				f := open(t, tc.path)
				m := load(t, f, deq)
				seq := make([]int32, 300)
				for i := range seq {
					seq[i] = int32((i*37 + 11) % m.Config.Vocab)
				}
				vocab := m.Config.Vocab
				tol := 1e-4
				if !deq {
					tol = 3e-2
				}
				want := forward(t, m, seq)
				caches := m.NewCaches(len(seq))
				for i, id := range seq {
					if i == 200 {
						// The model's norm takes a new value, equal to the
						// old: the steps must read the new one.
						p := m.Norm.Weight
						if err := p.Set(p.Value().Mul(tensor.Scalar[float32](1))); err != nil {
							t.Fatal(err)
						}
					}
					l, err := m.Logits(t.Context(), []int32{id}, caches, false)
					if err != nil {
						t.Fatal(err)
					}
					switch i {
					case 0, 1, 199, 200, 254, 255, 256, 257, 299:
						near(t, fmt.Sprintf("step %d", i), data(t, l), want[i*vocab:(i+1)*vocab], tol)
					}
				}
				if caches[0].Len() != len(seq) {
					t.Errorf("caches hold %d positions, want %d", caches[0].Len(), len(seq))
				}
			})
		}
	}
}
