// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mimir, a pure-Go machine learning framework,
// created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Package llm runs and fine-tunes decoder-only language models read from
// Mimir Model (.mim) files, as mim-import converts them from GGUF: the
// "llama" architecture of llama.cpp (Llama 1 to 3, TinyLlama, Mistral,
// SmolLM and others in its layout), "qwen2" (Qwen 2 and 2.5), "qwen3"
// (Qwen 3), "gemma2" and "gemma3" (Gemma 2 and 3) and "phi3" (Phi 3, 3.5
// and 4, whose fused projections mim-import splits apart).
//
// Load builds a Model from a file's weights and metadata, keeping the
// weights in the file's types, quantized or not. Logits runs it one block
// of tokens at a time with key-value caches, for generation. Hidden runs
// whole sequences at once, recording gradients when the model has
// trainable parameters, and Loss, Score and Embedding build on it.
//
// Fine-tuning is LoRA: AddLoRA puts a trainable low-rank pair next to the
// chosen projections of every block (see nn.LoRA), whose gradients flow
// through the frozen weights, quantized ones included, and WriteAdapter
// saves the pairs alone as an adapter file of the format
// (spec/adapters.md), a few megabytes next to a base of gigabytes. A
// runtime applies it to the base with mim.Model.ApplyLoRA, and ReadAdapter
// puts it back on a Model to train it further or embed with it.
package llm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Architectures lists the values of general.architecture Load reads.
var Architectures = []string{"llama", "qwen2", "qwen3", "gemma2", "gemma3", "phi3"}

// Config holds a model's hyperparameters, as the metadata of a Mimir Model
// file converted from GGUF names them under the architecture's prefix:
// "llama.block_count" and so on. The fields after Eps are zero for an
// architecture that lacks them.
type Config struct {
	Arch      string // one of Architectures
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

	// RopeAttnFactor scales the rotated features, as Phi 3's LongRoPE
	// does; 0 means 1.
	RopeAttnFactor float64
	// Window is the sliding window of the local layers, which are all but
	// every LocalEvery-th (Gemma 2: 2, Gemma 3: 6); 0 means none.
	Window, LocalEvery int
	// LocalBase is the rope base of the local layers, which are never
	// scaled; 0 means RopeBase.
	LocalBase float64
	// AttnScale scales the attention scores; 0 means 1/sqrt(HeadDim).
	AttnScale float64
	// AttnSoftcap and FinalSoftcap cap the attention scores and the logits
	// at ±cap with cap·tanh(x/cap), as Gemma 2 does; 0 means none.
	AttnSoftcap, FinalSoftcap float64
}

// Local reports whether block l attends to a sliding window only.
func (c Config) Local(l int) bool {
	return c.Window > 0 && c.LocalEvery > 0 && l%c.LocalEvery < c.LocalEvery-1
}

// gemma reports whether c is of the Gemma 2 or 3 architecture.
func (c Config) gemma() bool { return c.Arch == "gemma2" || c.Arch == "gemma3" }

// headNorms reports whether c normalizes each query and key head.
func (c Config) headNorms() bool { return c.Arch == "qwen3" || c.Arch == "gemma3" }

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
// from its metadata md, as mim-import's graph builder reads them.
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
		Arch:     arch,
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
	// llama.cpp reads a missing scaling type as linear, by a factor of 1
	// unless the file gives one; Phi 3's LongRoPE is in its factors.
	scaling := ""
	if v, ok := md[arch+".rope.scaling.type"]; ok {
		scaling, _ = v.(string)
	}
	switch scaling {
	case "", "linear":
		c.RopeScale = float("rope.scaling.factor", float("rope.scale_linear", 1))
	case "none":
	case "longrope":
		if arch != "phi3" {
			errs = append(errs, fmt.Errorf("rope scaling %q", scaling))
		}
	default:
		errs = append(errs, fmt.Errorf("rope scaling %q", scaling))
	}
	if af := float("rope.scaling.attn_factor", 1); af != 1 {
		c.RopeAttnFactor = af
	}
	if c.gemma() {
		// As llama.cpp: 1/sqrt(head size), but 1/sqrt(embedding/heads) for
		// the 27B models, which it knows by their layer count.
		c.AttnScale = 1 / math.Sqrt(float64(c.HeadDim))
		if arch == "gemma2" && c.Layers == 46 || arch == "gemma3" && c.Layers == 62 {
			c.AttnScale = 1 / math.Sqrt(float64(c.Embed/max(c.Heads, 1)))
		}
		if arch == "gemma2" {
			c.LocalEvery, c.Window, c.LocalBase = 2, 4096, c.RopeBase
			c.AttnSoftcap = float("attn_logit_softcapping", 0)
		} else {
			c.LocalEvery, c.LocalBase = 6, 10000
		}
		c.FinalSoftcap = float("final_logit_softcapping", 0)
		c.Window = integer("attention.sliding_window", c.Window, false)
		c.LocalEvery = integer("attention.sliding_window_pattern", c.LocalEvery, false)
		c.LocalBase = float("rope.freq_base_swa", c.LocalBase)
		if c.Window == 0 {
			c.LocalEvery, c.LocalBase = 0, 0
		}
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
		return c, fmt.Errorf("llm: model hyperparameters: %w: %w", err, tensor.ErrShape)
	}
	return c, nil
}

// Block is one decoder block. Each projection is an *nn.Linear, or after
// AddLoRA an *nn.LoRA around one. The norms after AttnNorm and MLPNorm are
// nil for an architecture without them: QNorm and KNorm normalize each
// query and key head (Qwen 3, Gemma 3), PostAttnNorm and PostMLPNorm the
// outputs of attention and of the MLP before they are added (Gemma).
type Block struct {
	AttnNorm, MLPNorm         *nn.RMSNorm
	Q, K, V, O                nn.Module
	Gate, Up, Down            nn.Module
	QNorm, KNorm              *nn.RMSNorm
	PostAttnNorm, PostMLPNorm *nn.RMSNorm
}

// Projections names a block's projections as llama.cpp names their
// weights, blk.N.<name>.weight, in the order of a block's fields.
var Projections = []string{"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}

// projections returns the fields of b's projections, in the order of
// Projections.
func (b *Block) projections() []*nn.Module {
	return []*nn.Module{&b.Q, &b.K, &b.V, &b.O, &b.Gate, &b.Up, &b.Down}
}

// norms returns b's norms, those it lacks left out.
func (b *Block) norms() []*nn.RMSNorm {
	var ns []*nn.RMSNorm
	for _, n := range []*nn.RMSNorm{b.AttnNorm, b.MLPNorm, b.QNorm, b.KNorm, b.PostAttnNorm, b.PostMLPNorm} {
		if n != nil {
			ns = append(ns, n)
		}
	}
	return ns
}

// linearOf returns the Linear layer m is or wraps.
func linearOf(m nn.Module) *nn.Linear {
	switch m := m.(type) {
	case *nn.Linear:
		return m
	case *nn.LoRA:
		return m.Base
	}
	return nil
}

// Model is a decoder of one of Architectures.
type Model struct {
	Config Config `nn:"-"`
	Embed  *nn.Embedding
	Blocks []*Block
	Norm   *nn.RMSNorm
	Output *nn.Linear // shares Embed's weight when the file ties them
	Rope   *nn.RotaryEmbedding
	// LocalRope rotates the local blocks' queries and keys when they turn
	// otherwise than the global ones' (Gemma 3); nil means Rope.
	LocalRope *nn.RotaryEmbedding
	// alpha is that of the LoRA pairs, for WriteAdapter.
	alpha float64
	// decoding holds the recorded decoding steps (see step).
	decoding decoder
}

// Load reads the model in f, whose metadata is md, to dev. Its weights
// keep the file's types, quantized or not; on the CPU they read the
// file's mapping in place.
func Load(f *mim.Model, md mimfile.Metadata, dev tensor.Device) (*Model, error) {
	arch, _ := md["general.architecture"].(string)
	if !slices.Contains(Architectures, arch) {
		return nil, fmt.Errorf("llm: architecture %q, want one of %v: %w", arch, Architectures, errors.ErrUnsupported)
	}
	c, err := readConfig(md, arch)
	if err != nil {
		return nil, err
	}
	names := f.Names() // sorted
	has := func(name string) bool {
		_, ok := slices.BinarySearch(names, name)
		return ok
	}
	if arch == "phi3" && has("blk.0.attn_qkv.weight") {
		return nil, fmt.Errorf("llm: Phi 3 weights with fused projections; convert the GGUF file with mim-import -graph, which splits them: %w", errors.ErrUnsupported)
	}
	load := func(name string, shape ...int) (*tensor.Tensor, error) {
		w, err := f.Tensor(name)
		if err != nil {
			return nil, fmt.Errorf("llm: %w", err)
		}
		if got := w.Shape(); !slices.Equal(got, shape) {
			return nil, fmt.Errorf("llm: %s of shape %v, want %v: %w", name, got, shape, tensor.ErrShape)
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
	// biased is linear with the bias qwen2 gives its queries, keys and
	// values.
	biased := func(name string, out, in int) (nn.Module, error) {
		l, err := linear(name+".weight", out, in)
		if err != nil || c.Arch != "qwen2" {
			return l, err
		}
		l.Bias, err = buffer(name+".bias", out)
		return l, err
	}
	module := func(name string, out, in int) (nn.Module, error) { return linear(name, out, in) }
	normOf := func(name string, dim int) (*nn.RMSNorm, error) {
		w, err := buffer(name, dim)
		return &nn.RMSNorm{Weight: w, Eps: c.Eps}, err
	}
	norm := func(name string) (*nn.RMSNorm, error) { return normOf(name, c.Embed) }

	emb, err := f.Tensor("token_embd.weight")
	if err != nil {
		return nil, fmt.Errorf("llm: %w", err)
	}
	if emb.Rank() != 2 {
		return nil, fmt.Errorf("llm: token_embd.weight of shape %v: %w", emb.Shape(), tensor.ErrShape)
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
		var e [13]error
		b.AttnNorm, e[0] = norm(p + "attn_norm.weight")
		b.Q, e[1] = biased(p+"attn_q", q, c.Embed)
		b.K, e[2] = biased(p+"attn_k", kv, c.Embed)
		b.V, e[3] = biased(p+"attn_v", kv, c.Embed)
		b.O, e[4] = module(p+"attn_output.weight", c.Embed, q)
		b.MLPNorm, e[5] = norm(p + "ffn_norm.weight")
		b.Gate, e[6] = module(p+"ffn_gate.weight", c.Hidden, c.Embed)
		b.Up, e[7] = module(p+"ffn_up.weight", c.Hidden, c.Embed)
		b.Down, e[8] = module(p+"ffn_down.weight", c.Embed, c.Hidden)
		if c.headNorms() {
			b.QNorm, e[9] = normOf(p+"attn_q_norm.weight", c.HeadDim)
			b.KNorm, e[10] = normOf(p+"attn_k_norm.weight", c.HeadDim)
		}
		if c.gemma() {
			b.PostAttnNorm, e[11] = norm(p + "post_attention_norm.weight")
			b.PostMLPNorm, e[12] = norm(p + "post_ffw_norm.weight")
		}
		errs = append(errs, e[:]...)
		m.Blocks = append(m.Blocks, b)
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	if m.Rope, err = rope(f, has("rope_freqs.weight"), c); err != nil {
		return nil, err
	}
	// The local blocks turn at their own base, unscaled.
	if c.Window > 0 && (c.LocalBase != c.RopeBase || m.Rope.Scaling != nil || m.Rope.Mscale != 0) {
		m.LocalRope = &nn.RotaryEmbedding{Dim: c.RopeDim, Base: c.LocalBase, Interleaved: m.Rope.Interleaved}
	}
	return m, nil
}

// rope returns the rotary embedding of c's global blocks. llama.cpp
// rotates the "llama" architecture's interleaved pairs of features, which
// its converter permutes Hugging Face's weights for, and the halves of
// each head of the others, which it doesn't. Llama 3.1 and later store a
// divisor of each pair's frequency in rope_freqs.weight, which f holds if
// freqs is set, as do Phi 3 files that mim-import converted, from their
// LongRoPE factors, along with a scale of the rotated features.
func rope(f *mim.Model, freqs bool, c Config) (*nn.RotaryEmbedding, error) {
	r := &nn.RotaryEmbedding{Dim: c.RopeDim, Base: c.RopeBase, Interleaved: c.Arch == "llama", Mscale: c.RopeAttnFactor}
	var factors []float32
	if freqs {
		w, err := f.Tensor("rope_freqs.weight")
		if err != nil {
			return nil, fmt.Errorf("llm: %w", err)
		}
		if !slices.Equal(w.Shape(), []int{c.RopeDim / 2}) || w.DType() != tensor.Float32 {
			return nil, fmt.Errorf("llm: rope_freqs.weight of %v %v, want %d float32: %w", w.Shape(), w.DType(), c.RopeDim/2, tensor.ErrShape)
		}
		if factors, err = w.Data[float32](context.Background()); err != nil {
			return nil, err
		}
		for _, x := range factors {
			if !(x > 0) || math.IsInf(float64(x), 0) {
				return nil, fmt.Errorf("llm: rope frequency factor %v: %w", x, tensor.ErrShape)
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

// ropeOf returns the rotary embedding of block i.
func (m *Model) ropeOf(i int) *nn.RotaryEmbedding {
	if m.LocalRope != nil && m.Config.Local(i) {
		return m.LocalRope
	}
	return m.Rope
}

// Params returns the weights and biases of the base model, each once,
// without the pairs of AddLoRA (see Adapters).
func (m *Model) Params() []*tensor.Param {
	ps := []*tensor.Param{m.Embed.Weight, m.Norm.Weight}
	if m.Output.Weight != m.Embed.Weight {
		ps = append(ps, m.Output.Weight)
	}
	for _, b := range m.Blocks {
		for _, n := range b.norms() {
			ps = append(ps, n.Weight)
		}
		for _, p := range b.projections() {
			l := linearOf(*p)
			ps = append(ps, l.Weight)
			if l.Bias != nil {
				ps = append(ps, l.Bias)
			}
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

// embed returns the embeddings of tokens, Int32 [B, S], as [B, S, Embed],
// scaled by the square root of Embed for Gemma.
func (m *Model) embed(tokens *tensor.Tensor) *tensor.Tensor {
	x := m.Embed.Forward(tokens.To(m.Embed.Weight.Value().Device()))
	if m.Config.gemma() {
		x = x.Mul(tensor.Scalar(float32(math.Sqrt(float64(m.Config.Embed)))))
	}
	return x
}

// Logits runs tokens, which follow the positions the caches hold,
// appends their keys and values to the caches, and returns the logits of
// the last token, [Vocab], or of every token, [len(tokens), Vocab], if
// all is set.
func (m *Model) Logits(ctx context.Context, tokens []int32, caches []*nn.KVCache, all bool) (*tensor.Tensor, error) {
	c := m.Config
	if len(tokens) == 0 || len(caches) != len(m.Blocks) {
		return nil, fmt.Errorf("llm: %d tokens with %d caches for %d blocks: %w", len(tokens), len(caches), len(m.Blocks), tensor.ErrShape)
	}
	for _, t := range tokens {
		if t < 0 || int(t) >= c.Vocab {
			return nil, fmt.Errorf("llm: token %d of a vocabulary of %d: %w", t, c.Vocab, tensor.ErrIndex)
		}
	}
	if len(tokens) == 1 && !all {
		if logits, ok, err := m.step(ctx, tokens[0], caches); err != nil || ok {
			return logits, err
		}
	}
	s := len(tokens)
	x := m.embed(tensor.FromSlice(tokens, 1, s)) // [1, S, embed]
	for i, b := range m.Blocks {
		past := caches[i].Len()
		q, k, v := b.project(b.AttnNorm.Forward(x), c)
		r := m.ropeOf(i)
		q, k = r.Apply(q, past), r.Apply(k, past)
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
		a := m.attend(i, q, keys, values, past)
		x = b.rest(x, a, c)
	}
	if !all {
		x = x.Shrink([][2]int{{0, 1}, {s - 1, s}, {0, c.Embed}})
		s = 1
	}
	logits := m.head(m.Norm.Forward(x)).Reshape(s, c.Vocab)
	if !all {
		logits = logits.Reshape(c.Vocab)
	}
	return logits, nil
}

// head returns the logits of h, the final norm's output, soft-capped if
// the model caps them.
func (m *Model) head(h *tensor.Tensor) *tensor.Tensor {
	return softcap(m.Output.Forward(h), m.Config.FinalSoftcap)
}

// softcap returns cap·tanh(x/cap), or x if cap is 0.
func softcap(x *tensor.Tensor, cap float64) *tensor.Tensor {
	if cap == 0 {
		return x
	}
	return x.Mul(tensor.Scalar(float32(1 / cap))).Tanh().Mul(tensor.Scalar(float32(cap)))
}

// attend returns block i's attention of queries q, [B, Heads, S, HeadDim],
// at positions past to past+S-1, over keys and values k and v, [B,
// KVHeads, K, HeadDim], at positions 0 to K-1: each query sees the keys
// at its position and before, within the block's window if it is local.
func (m *Model) attend(i int, q, k, v *tensor.Tensor, past int) *tensor.Tensor {
	c := m.Config
	scale := c.AttnScale
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(c.HeadDim))
	}
	s, n := q.Shape()[2], k.Shape()[2]
	local := c.Local(i) && c.Window < past+s
	if c.AttnSoftcap == 0 && !local {
		return tensor.ScaledDotProductAttention(q, k, v, tensor.CausalOffset(past), tensor.AttentionScale(scale))
	}
	// A mask of which keys each query sees: key j from the query at
	// position p when j <= p, and, in a local block, p - j < Window.
	mask := make([]float32, s*n)
	for a := range s {
		p := past + a
		for j := range n {
			if j > p || local && p-j >= c.Window {
				mask[a*n+j] = float32(math.Inf(-1))
			}
		}
	}
	mt := tensor.FromSlice(mask, 1, 1, s, n).To(q.Device())
	if c.AttnSoftcap == 0 {
		return tensor.ScaledDotProductAttention(q, k, v, tensor.AdditiveMask(mt), tensor.AttentionScale(scale))
	}
	// The soft cap goes between the scores and the softmax, which the
	// fused kernel doesn't do, so the scores are made whole here.
	b, h, hkv := q.Shape()[0], q.Shape()[1], k.Shape()[1]
	rep := func(t *tensor.Tensor) *tensor.Tensor {
		dv := t.Shape()[3]
		return t.Reshape(b, hkv, 1, n, dv).Expand(b, hkv, h/hkv, n, dv).Reshape(b, h, n, dv)
	}
	scores := q.MatMul(rep(k).Transpose(2, 3)).Mul(tensor.Scalar(float32(scale)))
	scores = softcap(scores, c.AttnSoftcap).Add(mt)
	return scores.Softmax(3).MatMul(rep(v))
}

// rest finishes a block from its input x, [B, S, Embed], and its
// attention a, [B, Heads, S, HeadDim]: the output projection, the MLP and
// the residual sums, and the norms after attention and the MLP for Gemma.
func (b *Block) rest(x, a *tensor.Tensor, c Config) *tensor.Tensor {
	sh := x.Shape()
	o := b.O.Forward(a.Transpose(1, 2).Reshape(sh[0], sh[1], c.Heads*c.HeadDim))
	if b.PostAttnNorm != nil {
		o = b.PostAttnNorm.Forward(o)
	}
	x = x.Add(o)
	f := b.mlp(b.MLPNorm.Forward(x), c)
	if b.PostMLPNorm != nil {
		f = b.PostMLPNorm.Forward(f)
	}
	return x.Add(f)
}

// project returns the queries, keys and values of h, [B, S, Embed], each
// [B, heads, S, HeadDim], the queries and keys normalized head by head
// if the block does that.
func (b *Block) project(h *tensor.Tensor, c Config) (q, k, v *tensor.Tensor) {
	q, k = b.Q.Forward(h), b.K.Forward(h)
	if b.QNorm != nil {
		q = headsNorm(q, c.Heads, c.HeadDim, b.QNorm)
		k = headsNorm(k, c.KVHeads, c.HeadDim, b.KNorm)
	}
	return heads(q, c.Heads, c.HeadDim), heads(k, c.KVHeads, c.HeadDim), heads(b.V.Forward(h), c.KVHeads, c.HeadDim)
}

// headsNorm normalizes each of the n heads of dim features of x, [B, S,
// n·dim], with norm.
func headsNorm(x *tensor.Tensor, n, dim int, norm *nn.RMSNorm) *tensor.Tensor {
	s := x.Shape()
	return norm.Forward(x.Reshape(s[0], s[1], n, dim)).Reshape(s...)
}

// mlp returns the gated feed-forward layer of h: Down(act(Gate(h))·Up(h)),
// with act SiLU, or for Gemma GELU in its tanh form.
func (b *Block) mlp(h *tensor.Tensor, c Config) *tensor.Tensor {
	g := b.Gate.Forward(h)
	if c.gemma() {
		g = g.GELU()
	} else {
		g = g.SiLU()
	}
	return b.Down.Forward(g.Mul(b.Up.Forward(h)))
}

// heads maps [B, S, n·dim] to [B, n, S, dim].
func heads(x *tensor.Tensor, n, dim int) *tensor.Tensor {
	s := x.Shape()
	return x.Reshape(s[0], s[1], n, dim).Transpose(1, 2)
}
