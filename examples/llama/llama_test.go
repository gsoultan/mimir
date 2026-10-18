package main

import (
	"bytes"
	"errors"
	"maps"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"testing"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

// openFile opens the .mim file at path, until the test ends, and returns
// it with its metadata.
func openFile(tb testing.TB, path string) (*mim.Model, mimfile.Metadata) {
	tb.Helper()
	f, err := mim.Open(path)
	return opened(tb, f, err)
}

// openBytes is openFile for the bytes of a file.
func openBytes(tb testing.TB, b []byte) (*mim.Model, mimfile.Metadata) {
	tb.Helper()
	f, err := mim.Parse(b)
	return opened(tb, f, err)
}

func opened(tb testing.TB, f *mim.Model, err error) (*mim.Model, mimfile.Metadata) {
	tb.Helper()
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { f.Close() })
	md, err := f.Metadata()
	if err != nil {
		tb.Fatal(err)
	}
	return f, md
}

// reference holds the tokens of a test model's prompt and the logits a
// NumPy implementation of the model computes for them in float64.
type reference struct {
	tokens []int32
	logits []float32 // [len(tokens), vocab]
}

func readReference(t *testing.T, path string) reference {
	t.Helper()
	f, _ := openFile(t, path)
	tokens, err := f.Tensor("tokens")
	if err != nil {
		t.Fatal(err)
	}
	logits, err := f.Tensor("logits")
	if err != nil {
		t.Fatal(err)
	}
	var r reference
	if r.tokens, err = tokens.Data[int32](t.Context()); err != nil {
		t.Fatal(err)
	}
	if r.logits, err = logits.Data[float32](t.Context()); err != nil {
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
	f, md := openFile(t, path)
	m, err := Load(f, md, tensor.DefaultDevice())
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
// tools/gguf/llama.py and converted with mim-import, over their prompts,
// in one batched pass and one token at a time through the KV caches, and
// checks the logits of every position against those of the script's
// NumPy implementation of Llama, <name>.logits.mim. spm.mim has
// grouped-query attention, linear rope scaling, an output projection of
// its own, and Q8_0, F16, BF16 and F32 weights; bpe.mim has
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
			m := load(t, filepath.Join("testdata", c.name+".mim"), c.deq)
			ref := readReference(t, filepath.Join("testdata", c.name+".logits.mim"))
			if !c.deq {
				// The weights keep the file's types.
				want := map[string][2]tensor.DType{"spm": {tensor.Q8_0, tensor.BFloat16}, "bpe": {tensor.Q4_K, tensor.Q6_K}}[c.name]
				b := m.Blocks[0]
				if q, down := b.Q.Weight.Value().DType(), b.Down.Weight.Value().DType(); q != want[0] || down != want[1] {
					t.Errorf("query and down projections of %v and %v, want the file's %v and %v", q, down, want[0], want[1])
				}
			}
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
	m := loadModel(t, "testdata/bpe.mim")
	tok := openTokenizer(t, "testdata/bpe.mim")
	prompt := tok.Encode("Hello world", true)
	out, st, err := Generate(ctx, m, prompt, 8, &Sampler{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 8 || st.PromptTokens != len(prompt) || st.Tokens != 7 || st.PromptTime < 0 || st.Time < 0 {
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
	f, md := openFile(t, "testdata/spm.mim")
	// rewrite writes the file again with its metadata and tensors as edit
	// changes them, and reads it back.
	rewrite := func(edit func(md mimfile.Metadata, ts map[string]*tensor.Tensor)) (*mim.Model, mimfile.Metadata) {
		md := maps.Clone(md)
		ts := map[string]*tensor.Tensor{}
		for _, name := range f.Names() {
			v, err := f.Tensor(name)
			if err != nil {
				t.Fatal(err)
			}
			ts[name] = v
		}
		edit(md, ts)
		var buf bytes.Buffer
		if err := mim.WriteTensors(t.Context(), &buf, ts, &mim.SaveOptions{Metadata: md}); err != nil {
			t.Fatal(err)
		}
		return openBytes(t, buf.Bytes())
	}
	set := func(key string, v any) func(mimfile.Metadata, map[string]*tensor.Tensor) {
		return func(md mimfile.Metadata, _ map[string]*tensor.Tensor) { md[key] = v }
	}
	for name, edit := range map[string]func(mimfile.Metadata, map[string]*tensor.Tensor){
		"architecture": set("general.architecture", "gpt2"),
		"heads":        set("llama.attention.head_count", uint64(3)),
		"layers":       set("llama.block_count", uint64(1<<30)),
		"base":         set("llama.rope.freq_base", -1.0),
		"scaling":      set("llama.rope.scaling.type", "yarn"),
		"missing": func(_ mimfile.Metadata, ts map[string]*tensor.Tensor) {
			delete(ts, "blk.1.ffn_up.weight")
		},
	} {
		g, gmd := rewrite(edit)
		if _, err := Load(g, gmd, tensor.CPU()); err == nil {
			t.Errorf("%s: loaded", name)
		}
	}
	_, bad := rewrite(set("tokenizer.ggml.model", "wordpiece"))
	if _, err := NewTokenizer(bad); !errors.Is(err, errTokenizer) {
		t.Errorf("tokenizer of an unknown model: %v", err)
	}
	m := loadModel(t, "testdata/spm.mim")
	if _, err := m.Logits(t.Context(), []int32{int32(m.Config.Vocab)}, m.NewCaches(4), false); !errors.Is(err, tensor.ErrIndex) {
		t.Errorf("token outside the vocabulary: %v", err)
	}
}
