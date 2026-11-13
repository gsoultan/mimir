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

// Package whisper turns speech into text with OpenAI's Whisper models,
// read from Mimir Model (.mim) files that mim-import converts from a
// Hugging Face checkpoint (WhisperForConditionalGeneration): tiny, base,
// small, medium and large, English-only or multilingual.
//
// Load builds a Model from a file's weights and metadata. Transcribe
// takes 16 kHz mono samples, cuts them into windows of 30 seconds and
// writes down each one: it computes the window's log mel spectrogram as
// Whisper's feature extractor does (LogMel), runs the encoder over it
// (Encode), detects the language when it isn't given and decodes the
// text greedily, one token at a time with a key-value cache. Windows
// with no speech are skipped. Translate asks for an English translation
// instead.
package whisper

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Config holds a Whisper model's hyperparameters, as the metadata of a
// file mim-import converted names them under "whisper.".
type Config struct {
	Vocab       int
	Embed       int // d_model
	Mels        int // mel bands of the input: 80, or 128 for large-v3
	AudioCtx    int // encoder positions: 1500, for 30 s
	TextCtx     int // decoder positions: 448
	EncLayers   int
	EncHeads    int
	EncHidden   int
	DecLayers   int
	DecHeads    int
	DecHidden   int
	SampleRate  int // 16,000
	FFT         int // 400
	Hop         int // 160
	ChunkSecond int // 30
}

// Limits on what a file may ask for, which keep a corrupt or hostile
// file from building a model too large to describe.
const (
	maxLayers = 1 << 8
	maxDim    = 1 << 16
)

func metaInt(md mimfile.Metadata, key string) (int64, bool) {
	switch v := md[key].(type) {
	case int64:
		return v, true
	case uint64:
		return int64(v), v <= math.MaxInt64
	}
	return 0, false
}

func readConfig(md mimfile.Metadata) (Config, error) {
	var errs []error
	get := func(key string, def int) int {
		key = "whisper." + key
		v, ok := md[key]
		if !ok {
			if def == 0 {
				errs = append(errs, fmt.Errorf("no %s", key))
			}
			return def
		}
		n, ok := metaInt(md, key)
		if !ok || n <= 0 || n > 1<<24 {
			errs = append(errs, fmt.Errorf("%s = %v", key, v))
			return def
		}
		return int(n)
	}
	c := Config{
		Vocab:       get("vocab_size", 0),
		Embed:       get("embedding_length", 0),
		Mels:        get("mel_bins", 80),
		AudioCtx:    get("encoder.context_length", 1500),
		TextCtx:     get("decoder.context_length", 448),
		EncLayers:   get("encoder.block_count", 0),
		EncHeads:    get("encoder.head_count", 0),
		EncHidden:   get("encoder.feed_forward_length", 0),
		DecLayers:   get("decoder.block_count", 0),
		DecHeads:    get("decoder.head_count", 0),
		DecHidden:   get("decoder.feed_forward_length", 0),
		SampleRate:  get("sample_rate", 16000),
		FFT:         get("fft_length", 400),
		Hop:         get("hop_length", 160),
		ChunkSecond: get("chunk_length", 30),
	}
	if err := errors.Join(errs...); err != nil {
		return c, fmt.Errorf("whisper: hyperparameters: %w: %w", err, tensor.ErrShape)
	}
	switch {
	case c.EncLayers > maxLayers || c.DecLayers > maxLayers || c.Embed > maxDim || c.EncHidden > 4*maxDim || c.DecHidden > 4*maxDim || c.Mels > 1024:
		errs = append(errs, fmt.Errorf("a model of %d and %d layers of %d features", c.EncLayers, c.DecLayers, c.Embed))
	case c.Embed%c.EncHeads != 0 || c.Embed%c.DecHeads != 0:
		errs = append(errs, fmt.Errorf("%d features in %d and %d heads", c.Embed, c.EncHeads, c.DecHeads))
	case c.FFT < 2 || c.Hop > c.FFT || c.ChunkSecond > 600:
		errs = append(errs, fmt.Errorf("frames of %d samples every %d in windows of %d s", c.FFT, c.Hop, c.ChunkSecond))
	case 2*c.AudioCtx != c.Frames():
		errs = append(errs, fmt.Errorf("%d encoder positions for %d frames", c.AudioCtx, c.Frames()))
	}
	if err := errors.Join(errs...); err != nil {
		return c, fmt.Errorf("whisper: hyperparameters: %w: %w", err, tensor.ErrShape)
	}
	return c, nil
}

// Samples returns the samples in a window: 480,000, 30 s at 16 kHz.
func (c Config) Samples() int { return c.ChunkSecond * c.SampleRate }

// Frames returns the frames of a window's spectrogram: 3,000.
func (c Config) Frames() int { return c.Samples() / c.Hop }

// attention is a block's attention: queries, keys (without a bias, as
// Whisper's), values and output.
type attention struct {
	Q, K, V, O *nn.Linear
	heads      int
}

// heads maps [1, S, heads·dim] to [1, heads, S, dim].
func (a *attention) split(x *tensor.Tensor) *tensor.Tensor {
	s := x.Shape()
	return x.Reshape(s[0], s[1], a.heads, s[2]/a.heads).Transpose(1, 2)
}

// merge maps the heads' results, [1, heads, S, dim], to [1, S, embed] and
// projects them.
func (a *attention) merge(x *tensor.Tensor) *tensor.Tensor {
	s := x.Shape()
	return a.O.Forward(x.Transpose(1, 2).Reshape(s[0], s[2], s[1]*s[3]))
}

// block is an encoder or decoder layer: pre-norm self-attention, for the
// decoder pre-norm cross-attention to the audio, and a GELU MLP.
type block struct {
	AttnNorm  *nn.LayerNorm
	Attn      *attention
	CrossNorm *nn.LayerNorm // decoder only
	Cross     *attention    // decoder only
	MLPNorm   *nn.LayerNorm
	Up, Down  *nn.Linear
}

func (b *block) mlp(x *tensor.Tensor) *tensor.Tensor {
	return b.Down.Forward(gelu(b.Up.Forward(b.MLPNorm.Forward(x))))
}

// gelu is the exact GELU Whisper uses, x/2 · (1 + erf(x/√2)), rather
// than tensor.GELU's tanh approximation, which drifts by a few hundredths
// over the encoder's layers.
func gelu(x *tensor.Tensor) *tensor.Tensor {
	half := tensor.Scalar[float32](0.5)
	return x.Mul(half).Mul(x.Mul(tensor.Scalar[float32](math.Sqrt2 / 2)).Erf().Add(tensor.Scalar[float32](1)))
}

// Model is a Whisper encoder-decoder.
type Model struct {
	Config Config

	conv1, conv2 *nn.Conv1d
	audioPos     *tensor.Tensor // [AudioCtx, Embed]
	encoder      []*block
	encNorm      *nn.LayerNorm

	tokens       *nn.Embedding  // [Vocab, Embed], also the output projection
	textPos      *tensor.Tensor // [TextCtx, Embed]
	decoder      []*block
	decNorm      *nn.LayerNorm
	filters      []float32 // [Mels, FFT/2+1]
	vocab        vocabulary
	multilingual bool
	suppress     []int32 // never written
	begin        []int32 // not written first
}

// Load reads the Whisper model in f, whose metadata is md, to dev.
// Weights stored in another float type than float32 are converted to it.
func Load(f *mim.Model, md mimfile.Metadata, dev tensor.Device) (*Model, error) {
	if arch, _ := md["general.architecture"].(string); arch != "whisper" {
		return nil, fmt.Errorf("whisper: architecture %q, want whisper: %w", arch, errors.ErrUnsupported)
	}
	c, err := readConfig(md)
	if err != nil {
		return nil, err
	}
	voc, err := newVocabulary(md, c.Vocab)
	if err != nil {
		return nil, err
	}
	var errs []error
	load := func(name string, shape ...int) *tensor.Tensor {
		w, err := f.Tensor(name)
		if err != nil {
			errs = append(errs, fmt.Errorf("whisper: %w", err))
			return nil
		}
		if got := w.Shape(); !slices.Equal(got, shape) {
			errs = append(errs, fmt.Errorf("whisper: %s of shape %v, want %v: %w", name, got, shape, tensor.ErrShape))
			return nil
		}
		w = w.To(dev)
		if w.DType() != tensor.Float32 {
			w = w.Cast(tensor.Float32)
			if err := tensor.Realize(context.Background(), w); err != nil {
				errs = append(errs, fmt.Errorf("whisper: %s: %w", name, err))
				return nil
			}
		}
		return w
	}
	param := func(name string, shape ...int) *tensor.Param {
		w := load(name, shape...)
		if w == nil {
			return nil
		}
		return tensor.NewBuffer(w)
	}
	linear := func(name string, out, in int, bias bool) *nn.Linear {
		l := &nn.Linear{Weight: param(name+".weight", out, in)}
		if bias {
			l.Bias = param(name+".bias", out)
		}
		return l
	}
	norm := func(name string) *nn.LayerNorm {
		return &nn.LayerNorm{Weight: param(name+".weight", c.Embed), Bias: param(name+".bias", c.Embed), Eps: 1e-5}
	}
	attn := func(name string, heads int) *attention {
		return &attention{
			Q:     linear(name+".q_proj", c.Embed, c.Embed, true),
			K:     linear(name+".k_proj", c.Embed, c.Embed, false),
			V:     linear(name+".v_proj", c.Embed, c.Embed, true),
			O:     linear(name+".out_proj", c.Embed, c.Embed, true),
			heads: heads,
		}
	}
	layer := func(p string, heads, hidden int, cross bool) *block {
		b := &block{
			AttnNorm: norm(p + "self_attn_layer_norm"),
			Attn:     attn(p+"self_attn", heads),
			MLPNorm:  norm(p + "final_layer_norm"),
			Up:       linear(p+"fc1", hidden, c.Embed, true),
			Down:     linear(p+"fc2", c.Embed, hidden, true),
		}
		if cross {
			b.CrossNorm = norm(p + "encoder_attn_layer_norm")
			b.Cross = attn(p+"encoder_attn", heads)
		}
		return b
	}
	m := &Model{Config: c, vocab: voc}
	m.conv1 = &nn.Conv1d{Weight: param("model.encoder.conv1.weight", c.Embed, c.Mels, 3), Bias: param("model.encoder.conv1.bias", c.Embed), Options: tensor.ConvOptions{Stride: 1, Padding: 1}}
	m.conv2 = &nn.Conv1d{Weight: param("model.encoder.conv2.weight", c.Embed, c.Embed, 3), Bias: param("model.encoder.conv2.bias", c.Embed), Options: tensor.ConvOptions{Stride: 2, Padding: 1}}
	m.audioPos = load("model.encoder.embed_positions.weight", c.AudioCtx, c.Embed)
	for i := range c.EncLayers {
		m.encoder = append(m.encoder, layer(fmt.Sprintf("model.encoder.layers.%d.", i), c.EncHeads, c.EncHidden, false))
	}
	m.encNorm = norm("model.encoder.layer_norm")
	m.tokens = &nn.Embedding{Weight: param("model.decoder.embed_tokens.weight", c.Vocab, c.Embed)}
	m.textPos = load("model.decoder.embed_positions.weight", c.TextCtx, c.Embed)
	for i := range c.DecLayers {
		m.decoder = append(m.decoder, layer(fmt.Sprintf("model.decoder.layers.%d.", i), c.DecHeads, c.DecHidden, true))
	}
	m.decNorm = norm("model.decoder.layer_norm")
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	m.filters = MelFilters(c.SampleRate, c.FFT, c.Mels)
	m.suppress = tokenList(md, "whisper.suppress_tokens", c.Vocab)
	m.begin = tokenList(md, "whisper.begin_suppress_tokens", c.Vocab)
	var ok bool
	if m.multilingual, ok = md["whisper.multilingual"].(bool); !ok {
		m.multilingual = c.Vocab >= 51865
	}
	return m, nil
}

// tokenList reads a list of token ids, keeping those in the vocabulary.
func tokenList(md mimfile.Metadata, key string, vocab int) []int32 {
	ids, _ := md[key].([]int64)
	out := make([]int32, 0, len(ids))
	for _, id := range ids {
		if id >= 0 && id < int64(vocab) {
			out = append(out, int32(id))
		}
	}
	return out
}

// Multilingual reports whether the model knows languages other than
// English and can translate: whisper.multilingual, or when the file
// doesn't say, a vocabulary of 51,865 tokens or more, as OpenAI tells.
func (m *Model) Multilingual() bool { return m.multilingual && len(m.vocab.languages) > 0 }

// Languages returns the codes of the languages the model knows, such as
// "en" and "de", in the order of their tokens; only "en" for an
// English-only model.
func (m *Model) Languages() []string {
	if !m.Multilingual() {
		return []string{"en"}
	}
	out := make([]string, len(m.vocab.languages))
	for i, l := range m.vocab.languages {
		out[i] = l.code
	}
	return out
}

// Encode runs the encoder over a log mel spectrogram, [Mels, Frames], and
// returns the audio's features, [1, AudioCtx, Embed].
func (m *Model) Encode(ctx context.Context, mel *tensor.Tensor) (*tensor.Tensor, error) {
	c := m.Config
	if !slices.Equal(mel.Shape(), []int{c.Mels, c.Frames()}) {
		return nil, fmt.Errorf("whisper: a spectrogram of %v, want [%d %d]: %w", mel.Shape(), c.Mels, c.Frames(), tensor.ErrShape)
	}
	dev := m.audioPos.Device()
	x := mel.To(dev).Reshape(1, c.Mels, c.Frames())
	x = gelu(m.conv1.Forward(x))
	x = gelu(m.conv2.Forward(x))          // [1, Embed, AudioCtx]
	x = x.Transpose(1, 2).Add(m.audioPos) // [1, AudioCtx, Embed]
	for _, b := range m.encoder {
		h := b.AttnNorm.Forward(x)
		a := b.Attn
		y := tensor.ScaledDotProductAttention(a.split(a.Q.Forward(h)), a.split(a.K.Forward(h)), a.split(a.V.Forward(h)))
		x = x.Add(a.merge(y))
		x = x.Add(b.mlp(x))
		// Each layer's output is computed before the next is built, so
		// a schedule holds one layer at a time.
		if err := tensor.Realize(ctx, x); err != nil {
			return nil, err
		}
	}
	x = m.encNorm.Forward(x)
	if err := tensor.Realize(ctx, x); err != nil {
		return nil, err
	}
	return x, nil
}

// state is the decoder's state while it writes one window: the keys and
// values of the tokens so far and of the audio.
type state struct {
	self   []*nn.KVCache
	crossK []*tensor.Tensor // [1, heads, AudioCtx, dim] per layer
	crossV []*tensor.Tensor
	past   int
}

// newState computes each decoder layer's keys and values of the audio
// features, once for the window.
func (m *Model) newState(ctx context.Context, audio *tensor.Tensor) (*state, error) {
	c := m.Config
	s := &state{}
	dim := c.Embed / c.DecHeads
	for _, b := range m.decoder {
		k, v := b.Cross.split(b.Cross.K.Forward(audio)), b.Cross.split(b.Cross.V.Forward(audio))
		if err := tensor.Realize(ctx, k, v); err != nil {
			return nil, err
		}
		s.crossK, s.crossV = append(s.crossK, k), append(s.crossV, v)
		s.self = append(s.self, nn.NewKVCache(1, c.DecHeads, c.TextCtx, dim, tensor.Float32))
	}
	return s, nil
}

// logits runs tokens, which follow the s.past tokens before them, and
// returns the logits of the last, [Vocab].
func (m *Model) logits(ctx context.Context, s *state, tokens []int32) ([]float32, error) {
	c := m.Config
	n := len(tokens)
	if s.past+n > c.TextCtx {
		return nil, fmt.Errorf("whisper: %d tokens past %d of a context of %d: %w", n, s.past, c.TextCtx, tensor.ErrShape)
	}
	dev := m.textPos.Device()
	x := m.tokens.Forward(tensor.FromSlice(tokens, 1, n).To(dev))
	x = x.Add(m.textPos.Shrink([][2]int{{s.past, s.past + n}, {0, c.Embed}}))
	for i, b := range m.decoder {
		h := b.AttnNorm.Forward(x)
		a := b.Attn
		q, k, v := a.split(a.Q.Forward(h)), a.split(a.K.Forward(h)), a.split(a.V.Forward(h))
		if err := tensor.Realize(ctx, x, q, k, v); err != nil {
			return nil, err
		}
		keys, values, err := s.self[i].Append(ctx, k, v)
		if err != nil {
			return nil, err
		}
		x = x.Add(a.merge(tensor.ScaledDotProductAttention(q, keys, values, tensor.CausalOffset(s.past))))
		h = b.CrossNorm.Forward(x)
		q = b.Cross.split(b.Cross.Q.Forward(h))
		x = x.Add(b.Cross.merge(tensor.ScaledDotProductAttention(q, s.crossK[i], s.crossV[i])))
		x = x.Add(b.mlp(x))
	}
	s.past += n
	last := x.Shrink([][2]int{{0, 1}, {n - 1, n}, {0, c.Embed}})
	out := m.decNorm.Forward(last).Reshape(1, c.Embed).MatMulT(m.tokens.Weight.Value()).Reshape(c.Vocab)
	return out.Data[float32](ctx)
}

// Logits runs the decoder over tokens, from the start of a window, with
// audio, the encoder's output, and returns the logits of the last token,
// [Vocab]. It is for tests and tools; Transcribe decodes with a cache.
func (m *Model) Logits(ctx context.Context, audio *tensor.Tensor, tokens []int32) ([]float32, error) {
	s, err := m.newState(ctx, audio)
	if err != nil {
		return nil, err
	}
	return m.logits(ctx, s, tokens)
}

// vocabulary holds the token texts and the ids of Whisper's special
// tokens.
type vocabulary struct {
	tokens                                                  []string
	types                                                   []int64
	eos, sot, transcribe, translate, noTimestamps, noSpeech int32
	languages                                               []language
	firstSpecial                                            int32 // tokens from here on are not text
}

type language struct {
	code string
	id   int32
}

// Token types of tokenizer.ggml.token_type.
const typeControl = 3

func newVocabulary(md mimfile.Metadata, size int) (vocabulary, error) {
	v := vocabulary{}
	if m, _ := md["tokenizer.ggml.model"].(string); m != "gpt2" {
		return v, fmt.Errorf("whisper: tokenizer %q, want gpt2: %w", m, errors.ErrUnsupported)
	}
	v.tokens, _ = md["tokenizer.ggml.tokens"].([]string)
	v.types, _ = md["tokenizer.ggml.token_type"].([]int64)
	if len(v.tokens) != size || len(v.types) != size {
		return v, fmt.Errorf("whisper: %d tokens and %d types for a vocabulary of %d: %w", len(v.tokens), len(v.types), size, tensor.ErrShape)
	}
	ids := map[string]int32{}
	for i, t := range v.tokens {
		if v.types[i] == typeControl {
			if _, dup := ids[t]; !dup {
				ids[t] = int32(i)
			}
		}
	}
	id := func(s string) int32 {
		if i, ok := ids[s]; ok {
			return i
		}
		return -1
	}
	v.eos, v.sot = id("<|endoftext|>"), id("<|startoftranscript|>")
	v.transcribe, v.translate = id("<|transcribe|>"), id("<|translate|>")
	v.noTimestamps = id("<|notimestamps|>")
	v.noSpeech = id("<|nospeech|>")
	if v.noSpeech < 0 {
		v.noSpeech = id("<|nocaptions|>")
	}
	if v.eos < 0 || v.sot < 0 || v.noTimestamps < 0 {
		return v, fmt.Errorf("whisper: the tokenizer lacks <|endoftext|>, <|startoftranscript|> or <|notimestamps|>: %w", tensor.ErrShape)
	}
	// The languages are the tokens <|xx|> between <|startoftranscript|>
	// and <|translate|>.
	if v.translate > v.sot {
		for i := v.sot + 1; i < v.translate; i++ {
			t := v.tokens[i]
			if strings.HasPrefix(t, "<|") && strings.HasSuffix(t, "|>") && v.types[i] == typeControl {
				v.languages = append(v.languages, language{code: t[2 : len(t)-2], id: i})
			}
		}
	}
	v.firstSpecial = v.eos
	return v, nil
}

// text decodes text tokens, skipping special ones.
func (v *vocabulary) text(ids []int32) string {
	var b []byte
	for _, id := range ids {
		if id < 0 || id >= v.firstSpecial || int(id) >= len(v.tokens) || v.types[id] == typeControl {
			continue
		}
		for _, r := range v.tokens[id] {
			if c, ok := runeByte[r]; ok {
				b = append(b, c)
			} else {
				b = append(b, string(r)...)
			}
		}
	}
	return strings.ToValidUTF8(string(b), "�")
}

// byteRune and runeByte are GPT-2's map between bytes and the printable
// characters that stand for them in the vocabulary.
var runeByte = map[rune]byte{}

func init() {
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !(b >= '!' && b <= '~' || b >= 0xA1 && b <= 0xAC || b >= 0xAE) {
			r = rune(256 + n)
			n++
		}
		runeByte[r] = byte(b)
	}
}
