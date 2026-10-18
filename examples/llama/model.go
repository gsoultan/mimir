package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	mimfile "github.com/gsoultan/mim"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Config holds a Llama model's hyperparameters, as the metadata of a
// Mimir Model file converted from GGUF names them under the
// architecture's prefix: "llama.block_count" and so on.
type Config struct {
	Vocab     int
	Context   int // positions the model was trained on
	Embed     int
	Hidden    int // features of the MLP
	Layers    int
	Heads     int
	KVHeads   int
	HeadDim   int
	RopeDim   int // leading features of each head that rotate
	RopeBase  float64
	RopeScale float64 // linear rope scaling: positions are divided by it
	Eps       float64 // of the RMSNorms
}

// Limits on what a file may ask for, which keep a corrupt or hostile file
// from building a model too large to describe.
const (
	maxLayers = 1 << 10
	maxDim    = 1 << 20
)

// metaInt returns md[key] as an integer, which the metadata holds as an
// int64 or a uint64, and whether it is one.
func metaInt(md mimfile.Metadata, key string) (int64, bool) {
	switch v := md[key].(type) {
	case int64:
		return v, true
	case uint64:
		return int64(v), v <= math.MaxInt64
	}
	return 0, false
}

// metaFloat returns md[key] as a number, and whether it is one.
func metaFloat(md mimfile.Metadata, key string) (float64, bool) {
	if x, ok := md[key].(float64); ok {
		return x, true
	}
	n, ok := metaInt(md, key)
	return float64(n), ok
}

// readConfig reads the hyperparameters of a model of architecture arch
// from its metadata md.
func readConfig(md mimfile.Metadata, arch string) (Config, error) {
	var errs []error
	integer := func(key string, def int, required bool) int {
		key = arch + "." + key
		v, ok := md[key]
		if !ok {
			if required {
				errs = append(errs, fmt.Errorf("no %s", key))
			}
			return def
		}
		n, ok := metaInt(md, key)
		if !ok || n <= 0 || n > maxDim {
			errs = append(errs, fmt.Errorf("%s = %v", key, v))
			return def
		}
		return int(n)
	}
	float := func(key string, def float64) float64 {
		key = arch + "." + key
		v, ok := md[key]
		if !ok {
			return def
		}
		x, ok := metaFloat(md, key)
		if !ok || !(x > 0) || math.IsInf(x, 0) {
			errs = append(errs, fmt.Errorf("%s = %v", key, v))
			return def
		}
		return x
	}
	c := Config{
		Context:  integer("context_length", 2048, false),
		Embed:    integer("embedding_length", 0, true),
		Hidden:   integer("feed_forward_length", 0, true),
		Layers:   integer("block_count", 0, true),
		Heads:    integer("attention.head_count", 0, true),
		RopeBase: float("rope.freq_base", 10000),
		Eps:      float("attention.layer_norm_rms_epsilon", 1e-5),
	}
	c.KVHeads = integer("attention.head_count_kv", c.Heads, false)
	c.HeadDim = integer("attention.key_length", c.Embed/max(c.Heads, 1), false)
	c.RopeDim = integer("rope.dimension_count", c.HeadDim, false)
	c.RopeScale = 1
	scaling := "none"
	if v, ok := md[arch+".rope.scaling.type"]; ok {
		scaling, _ = v.(string)
	}
	switch scaling {
	case "none":
		c.RopeScale = float("rope.scale_linear", 1) // the key older files used
	case "linear":
		c.RopeScale = float("rope.scaling.factor", 1)
	default:
		errs = append(errs, fmt.Errorf("rope scaling %q", scaling))
	}
	if integer("attention.value_length", c.HeadDim, false) != c.HeadDim {
		errs = append(errs, fmt.Errorf("values of other than %d features", c.HeadDim))
	}
	if !(c.RopeBase > 1) {
		errs = append(errs, fmt.Errorf("rope base %v", c.RopeBase))
	}
	if c.Layers > maxLayers || c.Heads%max(c.KVHeads, 1) != 0 || c.RopeDim > c.HeadDim || c.RopeDim%2 != 0 {
		errs = append(errs, fmt.Errorf("%d layers, %d heads and %d key heads of %d features, %d rotated", c.Layers, c.Heads, c.KVHeads, c.HeadDim, c.RopeDim))
	}
	if err := errors.Join(errs...); err != nil {
		return c, fmt.Errorf("llama: model hyperparameters: %w: %w", err, tensor.ErrShape)
	}
	return c, nil
}

// Block is one decoder block of Llama.
type Block struct {
	AttnNorm, MLPNorm *nn.RMSNorm
	Q, K, V, O        *nn.Linear
	Gate, Up, Down    *nn.Linear
}

// Model is a Llama-family decoder: Llama 2 and 3, TinyLlama, SmolLM and
// others of the "llama" architecture of llama.cpp.
type Model struct {
	Config Config `nn:"-"`
	Embed  *nn.Embedding
	Blocks []*Block
	Norm   *nn.RMSNorm
	Output *nn.Linear // shares Embed's weight when the file ties them
	Rope   *nn.RotaryEmbedding
}

// Load reads the model in f, whose metadata is md, to dev. Its weights
// keep the file's types, quantized or not; on the CPU they read the
// file's mapping in place.
func Load(f *mim.Model, md mimfile.Metadata, dev tensor.Device) (*Model, error) {
	if arch, _ := md["general.architecture"].(string); arch != "llama" {
		return nil, fmt.Errorf("llama: architecture %q, want llama: %w", arch, errors.ErrUnsupported)
	}
	c, err := readConfig(md, "llama")
	if err != nil {
		return nil, err
	}
	names := f.Names() // sorted
	has := func(name string) bool {
		_, ok := slices.BinarySearch(names, name)
		return ok
	}
	load := func(name string, shape ...int) (*tensor.Tensor, error) {
		w, err := f.Tensor(name)
		if err != nil {
			return nil, fmt.Errorf("llama: %w", err)
		}
		if got := w.Shape(); !slices.Equal(got, shape) {
			return nil, fmt.Errorf("llama: %s of shape %v, want %v: %w", name, got, shape, tensor.ErrShape)
		}
		return w.To(dev), nil
	}
	buffer := func(name string, shape ...int) (*tensor.Param, error) {
		w, err := load(name, shape...)
		if err != nil {
			return nil, err
		}
		return tensor.NewBuffer(w), nil
	}
	linear := func(name string, out, in int) (*nn.Linear, error) {
		w, err := buffer(name, out, in)
		return &nn.Linear{Weight: w}, err
	}
	norm := func(name string) (*nn.RMSNorm, error) {
		w, err := buffer(name, c.Embed)
		return &nn.RMSNorm{Weight: w, Eps: c.Eps}, err
	}

	emb, err := f.Tensor("token_embd.weight")
	if err != nil {
		return nil, fmt.Errorf("llama: %w", err)
	}
	if emb.Rank() != 2 {
		return nil, fmt.Errorf("llama: token_embd.weight of shape %v: %w", emb.Shape(), tensor.ErrShape)
	}
	c.Vocab = emb.Shape()[0]
	m := &Model{Config: c}
	var errs []error
	keep := func(err error) { errs = append(errs, err) }
	embed, err := buffer("token_embd.weight", c.Vocab, c.Embed)
	keep(err)
	m.Embed = &nn.Embedding{Weight: embed}
	m.Norm, err = norm("output_norm.weight")
	keep(err)
	if has("output.weight") {
		m.Output, err = linear("output.weight", c.Vocab, c.Embed)
		keep(err)
	} else {
		m.Output = &nn.Linear{Weight: embed}
	}
	q, kv := c.Heads*c.HeadDim, c.KVHeads*c.HeadDim
	for i := range c.Layers {
		p := fmt.Sprintf("blk.%d.", i)
		b := &Block{}
		var e [9]error
		b.AttnNorm, e[0] = norm(p + "attn_norm.weight")
		b.Q, e[1] = linear(p+"attn_q.weight", q, c.Embed)
		b.K, e[2] = linear(p+"attn_k.weight", kv, c.Embed)
		b.V, e[3] = linear(p+"attn_v.weight", kv, c.Embed)
		b.O, e[4] = linear(p+"attn_output.weight", c.Embed, q)
		b.MLPNorm, e[5] = norm(p + "ffn_norm.weight")
		b.Gate, e[6] = linear(p+"ffn_gate.weight", c.Hidden, c.Embed)
		b.Up, e[7] = linear(p+"ffn_up.weight", c.Hidden, c.Embed)
		b.Down, e[8] = linear(p+"ffn_down.weight", c.Embed, c.Hidden)
		errs = append(errs, e[:]...)
		m.Blocks = append(m.Blocks, b)
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	if m.Rope, err = rope(f, has("rope_freqs.weight"), c); err != nil {
		return nil, err
	}
	return m, nil
}

// rope returns the rotary embedding of c. llama.cpp rotates the "llama"
// architecture's interleaved pairs of features, which its converter
// permutes Hugging Face's weights for. Llama 3.1 and later store a
// divisor of each pair's frequency in rope_freqs.weight, which f holds if
// freqs is set.
func rope(f *mim.Model, freqs bool, c Config) (*nn.RotaryEmbedding, error) {
	r := &nn.RotaryEmbedding{Dim: c.RopeDim, Base: c.RopeBase, Interleaved: true}
	var factors []float32
	if freqs {
		w, err := f.Tensor("rope_freqs.weight")
		if err != nil {
			return nil, fmt.Errorf("llama: %w", err)
		}
		if !slices.Equal(w.Shape(), []int{c.RopeDim / 2}) || w.DType() != tensor.Float32 {
			return nil, fmt.Errorf("llama: rope_freqs.weight of %v %v, want %d float32: %w", w.Shape(), w.DType(), c.RopeDim/2, tensor.ErrShape)
		}
		if factors, err = w.Data[float32](context.Background()); err != nil {
			return nil, err
		}
		for _, x := range factors {
			if !(x > 0) || math.IsInf(float64(x), 0) {
				return nil, fmt.Errorf("llama: rope frequency factor %v: %w", x, tensor.ErrShape)
			}
		}
	}
	if c.RopeScale == 1 && factors == nil {
		return r, nil
	}
	// Scaling sees each pair's inverse frequency base^(-2i/Dim); recover
	// i from it to find the pair's factor.
	r.Scaling = func(inv float64) float64 {
		out := inv / c.RopeScale
		if factors != nil {
			i := int(math.Round(-math.Log(inv) / math.Log(c.RopeBase) * float64(c.RopeDim) / 2))
			out /= float64(factors[min(max(i, 0), len(factors)-1)])
		}
		return out
	}
	return r, nil
}

// Params returns the model's weights, each once.
func (m *Model) Params() []*tensor.Param {
	ps := []*tensor.Param{m.Embed.Weight, m.Norm.Weight}
	if m.Output.Weight != m.Embed.Weight {
		ps = append(ps, m.Output.Weight)
	}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Weight, b.MLPNorm.Weight)
		for _, l := range []*nn.Linear{b.Q, b.K, b.V, b.O, b.Gate, b.Up, b.Down} {
			ps = append(ps, l.Weight)
		}
	}
	return ps
}

// NewCaches returns a key and value cache for each block, for up to
// capacity positions.
func (m *Model) NewCaches(capacity int) []*nn.KVCache {
	c := m.Config
	caches := make([]*nn.KVCache, c.Layers)
	for i := range caches {
		caches[i] = nn.NewKVCache(1, c.KVHeads, capacity, c.HeadDim, tensor.Float32)
	}
	return caches
}

// Logits runs tokens, which follow the positions the caches hold,
// appends their keys and values to the caches, and returns the logits of
// the last token, [Vocab], or of every token, [len(tokens), Vocab], if
// all is set.
func (m *Model) Logits(ctx context.Context, tokens []int32, caches []*nn.KVCache, all bool) (*tensor.Tensor, error) {
	c := m.Config
	if len(tokens) == 0 || len(caches) != len(m.Blocks) {
		return nil, fmt.Errorf("llama: %d tokens with %d caches for %d blocks: %w", len(tokens), len(caches), len(m.Blocks), tensor.ErrShape)
	}
	for _, t := range tokens {
		if t < 0 || int(t) >= c.Vocab {
			return nil, fmt.Errorf("llama: token %d of a vocabulary of %d: %w", t, c.Vocab, tensor.ErrIndex)
		}
	}
	dev := m.Embed.Weight.Value().Device()
	s := len(tokens)
	x := m.Embed.Forward(tensor.FromSlice(tokens, 1, s).To(dev)) // [1, S, embed]
	for i, b := range m.Blocks {
		past := caches[i].Len()
		q, k, v := b.project(b.AttnNorm.Forward(x), c)
		q, k = m.Rope.Apply(q, past), m.Rope.Apply(k, past)
		// The cache's Append computes the keys and values alone. Computing
		// the block's input, its queries, keys and values in one schedule
		// first keeps each block's products from running once more for
		// every later block.
		if err := tensor.Realize(ctx, x, q, k, v); err != nil {
			return nil, err
		}
		keys, values, err := caches[i].Append(ctx, k, v)
		if err != nil {
			return nil, err
		}
		a := tensor.ScaledDotProductAttention(q, keys, values, tensor.CausalOffset(past))
		x = x.Add(b.O.Forward(a.Transpose(1, 2).Reshape(1, s, c.Heads*c.HeadDim)))
		x = x.Add(b.mlp(b.MLPNorm.Forward(x)))
	}
	if !all {
		x = x.Shrink([][2]int{{0, 1}, {s - 1, s}, {0, c.Embed}})
		s = 1
	}
	logits := m.Output.Forward(m.Norm.Forward(x)).Reshape(s, c.Vocab)
	if !all {
		logits = logits.Reshape(c.Vocab)
	}
	return logits, nil
}

// project returns the queries, keys and values of h, [1, S, Embed], each
// [1, heads, S, HeadDim].
func (b *Block) project(h *tensor.Tensor, c Config) (q, k, v *tensor.Tensor) {
	return heads(b.Q.Forward(h), c.Heads, c.HeadDim), heads(b.K.Forward(h), c.KVHeads, c.HeadDim), heads(b.V.Forward(h), c.KVHeads, c.HeadDim)
}

// mlp returns the SwiGLU feed-forward layer of h: Down(SiLU(Gate(h))·Up(h)).
func (b *Block) mlp(h *tensor.Tensor) *tensor.Tensor {
	return b.Down.Forward(b.Gate.Forward(h).SiLU().Mul(b.Up.Forward(h)))
}

// heads maps [1, S, n·dim] to [1, n, S, dim].
func heads(x *tensor.Tensor, n, dim int) *tensor.Tensor {
	s := x.Shape()
	return x.Reshape(s[0], s[1], n, dim).Transpose(1, 2)
}
