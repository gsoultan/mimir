package main

import (
	"bytes"
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/io/gguf"
	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/tensor"
)

// reference holds the tokens of a test model's prompt and the logits a
// NumPy implementation of the model computes for them in float64.
type reference struct {
	tokens []int32
	logits []float32 // [len(tokens), vocab]
}

func readReference(t *testing.T, path string) reference {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	tt, _ := f.Tensor("tokens")
	lt, _ := f.Tensor("logits")
	var r reference
	if r.tokens, err = tt.Int32s(); err != nil {
		t.Fatal(err)
	}
	if r.logits, err = lt.Float32s(); err != nil {
		t.Fatal(err)
	}
	return r
}

func loadModel(t *testing.T, path string) *Model {
	t.Helper()
	return load(t, path, false)
}

// load reads the model at path, with its weights dequantized to float32
// if deq is set.
func load(t *testing.T, path string, deq bool) *Model {
	t.Helper()
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	m, err := Load(f, tensor.DefaultDevice())
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

// compareLogits fails unless got is within tol of want relative to the
// largest logit, and picks the same tokens but for near ties.
func compareLogits(t *testing.T, what string, got, want []float32, vocab int, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d logits, want %d", what, len(got), len(want))
	}
	var scale float64
	for _, w := range want {
		scale = max(scale, math.Abs(float64(w)))
	}
	var overall float64
	for p := range len(want) / vocab {
		g, w := got[p*vocab:(p+1)*vocab], want[p*vocab:(p+1)*vocab]
		var worst float64
		for j := range w {
			worst = max(worst, math.Abs(float64(g[j]-w[j])))
		}
		overall = max(overall, worst)
		if worst > tol*scale {
			t.Errorf("%s: position %d differs by %.4g, %.2g%% of the largest logit %.3g", what, p, worst, 100*worst/scale, scale)
		}
		if a, b := argmax(g), argmax(w); a != b && math.Abs(float64(w[a]-w[b])) > tol*scale {
			t.Errorf("%s: position %d picks token %d, want %d", what, p, a, b)
		}
	}
	t.Logf("%s: logits within %.2g of %.3g", what, overall, scale)
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// TestLogitsMatchNumPy runs the tiny models of testdata, built by
// tools/gguf/llama.py, over their prompts, in one batched pass and one
// token at a time through the KV caches, and checks the logits of every
// position against those of the script's NumPy implementation of Llama.
// spm.gguf has grouped-query attention, linear rope scaling, an output
// projection of its own, and Q8_0, F16, BF16 and F32 weights; bpe.gguf has
// four query heads to one key head, Llama 3.1's rope frequency factors,
// tied embeddings, and Q4_K and Q6_K weights.
//
// With the weights dequantized to float32 the logits match to float32's
// rounding. The CPU multiplies by quantized weights with activations
// rounded to 8 bits, as llama.cpp does, which moves the logits by a few
// percent of the largest: llama.cpp's own logits for these models differ
// from NumPy's by up to 0.5% and 2.5%.
func TestLogitsMatchNumPy(t *testing.T) {
	for _, c := range []struct {
		name string
		deq  bool
		tol  float64
	}{{"spm", true, 1e-4}, {"bpe", true, 1e-4}, {"spm", false, 0.02}, {"bpe", false, 0.04}} {
		name := c.name
		if c.deq {
			name += "/dequantized"
		}
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			m := load(t, filepath.Join("testdata", c.name+".gguf"), c.deq)
			ref := readReference(t, filepath.Join("testdata", c.name+".logits.safetensors"))
			v := m.Config.Vocab
			logits, err := m.Logits(ctx, ref.tokens, m.NewCaches(len(ref.tokens)), true)
			if err != nil {
				t.Fatal(err)
			}
			got, err := logits.Data[float32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			compareLogits(t, "batched", got, ref.logits, v, c.tol)

			caches := m.NewCaches(len(ref.tokens))
			var stepped []float32
			for _, tok := range ref.tokens {
				l, err := m.Logits(ctx, []int32{tok}, caches, false)
				if err != nil {
					t.Fatal(err)
				}
				d, err := l.Data[float32](ctx)
				if err != nil {
					t.Fatal(err)
				}
				stepped = append(stepped, d...)
			}
			compareLogits(t, "one token at a time", stepped, ref.logits, v, c.tol)
			compareLogits(t, "one token at a time against batched", stepped, got, v, 1e-4)
		})
	}
}

// TestGenerate checks greedy generation against the argmax of a pass over
// the whole sequence, the stop condition and the timings.
func TestGenerate(t *testing.T) {
	ctx := t.Context()
	m := loadModel(t, "testdata/bpe.gguf")
	tok := openTokenizer(t, "testdata/bpe.gguf")
	prompt := tok.Encode("Hello world", true)
	out, st, err := Generate(ctx, m, prompt, 8, &Sampler{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 8 || st.PromptTokens != len(prompt) || st.Tokens != 7 || st.PromptTime <= 0 || st.Time <= 0 {
		t.Fatalf("generated %v with %+v", out, st)
	}
	seq := append(slices.Clone(prompt), out...)
	logits, err := m.Logits(ctx, seq, m.NewCaches(len(seq)), true)
	if err != nil {
		t.Fatal(err)
	}
	all, err := logits.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	v := m.Config.Vocab
	for i, id := range out {
		p := len(prompt) - 1 + i
		if want := argmax(all[p*v : (p+1)*v]); int(id) != want {
			t.Errorf("token %d: %d, want the argmax %d", i, id, want)
		}
	}
	last := out[len(out)-1]
	want := out[:slices.Index(out, last)]
	stopped, _, err := Generate(ctx, m, prompt, 8, &Sampler{}, func(id int32) bool { return id == last }, nil)
	if err != nil || !slices.Equal(stopped, want) {
		t.Errorf("stopping at %d: %v, %v; want %v", last, stopped, err, want)
	}
}

// TestSampler checks greedy picks, top-k and top-p.
func TestSampler(t *testing.T) {
	logits := []float32{1, 5, 3, 4, 2}
	if got := (&Sampler{}).Sample(logits); got != 1 {
		t.Errorf("greedy: %d", got)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	seen := map[int32]int{}
	for range 2000 {
		seen[(&Sampler{Temperature: 1, TopK: 3, Rand: rng}).Sample(logits)]++
	}
	if len(seen) != 3 || seen[0] != 0 || seen[4] != 0 || seen[1] < seen[3] || seen[3] < seen[2] {
		t.Errorf("top-3 samples: %v", seen)
	}
	// Probabilities of 1, 3 and 2 are about 0.64, 0.24 and 0.09: top-p
	// of 0.8 keeps the first two.
	clear(seen)
	for range 2000 {
		seen[(&Sampler{Temperature: 1, TopP: 0.8, Rand: rng}).Sample(logits)]++
	}
	if len(seen) != 2 || seen[1] == 0 || seen[3] == 0 {
		t.Errorf("top-p samples: %v", seen)
	}
}

// TestLoadRejectsBadFiles checks that a file with missing tensors,
// unknown architecture or impossible hyperparameters is an error.
func TestLoadRejectsBadFiles(t *testing.T) {
	b, err := os.ReadFile("testdata/spm.gguf")
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	rewrite := func(edit func(kvs []gguf.KV, ts []gguf.Tensor) ([]gguf.KV, []gguf.Tensor)) *gguf.File {
		var kvs []gguf.KV
		for _, k := range f.Keys() {
			v, _ := f.Metadata(k)
			kvs = append(kvs, gguf.KV{Key: k, Value: v})
		}
		kvs, ts := edit(kvs, f.Tensors())
		var buf bytes.Buffer
		if err := gguf.Write(&buf, kvs, ts); err != nil {
			t.Fatal(err)
		}
		g, err := gguf.Parse(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	set := func(key string, v any) func([]gguf.KV, []gguf.Tensor) ([]gguf.KV, []gguf.Tensor) {
		return func(kvs []gguf.KV, ts []gguf.Tensor) ([]gguf.KV, []gguf.Tensor) {
			for i := range kvs {
				if kvs[i].Key == key {
					kvs[i].Value = gguf.ValueOf(v)
				}
			}
			return kvs, ts
		}
	}
	for name, edit := range map[string]func([]gguf.KV, []gguf.Tensor) ([]gguf.KV, []gguf.Tensor){
		"architecture": set("general.architecture", "gpt2"),
		"heads":        set("llama.attention.head_count", uint32(3)),
		"layers":       set("llama.block_count", uint32(1<<30)),
		"base":         set("llama.rope.freq_base", float32(-1)),
		"scaling":      set("llama.rope.scaling.type", "yarn"),
		"missing": func(kvs []gguf.KV, ts []gguf.Tensor) ([]gguf.KV, []gguf.Tensor) {
			return kvs, slices.DeleteFunc(ts, func(x gguf.Tensor) bool { return x.Name == "blk.1.ffn_up.weight" })
		},
	} {
		if _, err := Load(rewrite(edit), tensor.CPU()); err == nil {
			t.Errorf("%s: loaded", name)
		}
	}
	bad := rewrite(func(kvs []gguf.KV, ts []gguf.Tensor) ([]gguf.KV, []gguf.Tensor) {
		return set("tokenizer.ggml.model", "wordpiece")(kvs, ts)
	})
	if _, err := NewTokenizer(bad); !errors.Is(err, errTokenizer) {
		t.Errorf("tokenizer of an unknown model: %v", err)
	}
	m := loadModel(t, "testdata/spm.gguf")
	if _, err := m.Logits(t.Context(), []int32{int32(m.Config.Vocab)}, m.NewCaches(4), false); !errors.Is(err, tensor.ErrIndex) {
		t.Errorf("token outside the vocabulary: %v", err)
	}
}
