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

// Package clip puts images and text in one space with OpenAI's CLIP, read
// from Mimir Model (.mim) files that mim-import converts from a Hugging
// Face checkpoint (CLIPModel), so that photos can be searched by words
// and labelled from a list of words without training.
//
// Load builds a Model from a file's weights and metadata. EmbedImages and
// EmbedTexts return unit-length vectors whose dot product says how well
// an image and a text match; Scores turns them into each image's
// chances over a list of texts, as CLIP's zero-shot classification does.
// Images are prepared as CLIP's image processor prepares them: the
// shorter side resized to the model's size with a bicubic filter, the
// middle square cut out, and the colours scaled by the model's mean and
// deviation.
package clip

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Config holds a CLIP model's hyperparameters, as the metadata of a file
// mim-import converted names them under "clip.".
type Config struct {
	Projection int // features of the shared space
	Text       Tower
	Vision     Tower
	Vocab      int
	Context    int // text positions: 77
	ImageSize  int // 224
	PatchSize  int // 32, 16 or 14
	Resize     int // the shorter side is resized to this before the crop
	Mean, Std  [3]float64
}

// Tower holds one encoder's hyperparameters.
type Tower struct {
	Embed, Layers, Heads, Hidden int
	Eps                          float64
	QuickGELU                    bool // x·sigmoid(1.702x), CLIP's own; else exact GELU
}

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
		key = "clip." + key
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
	tower := func(name string) Tower {
		t := Tower{
			Embed:  get(name+".embedding_length", 0),
			Layers: get(name+".block_count", 0),
			Heads:  get(name+".head_count", 0),
			Hidden: get(name+".feed_forward_length", 0),
			Eps:    1e-5,
		}
		if x, ok := md["clip."+name+".layer_norm_epsilon"].(float64); ok {
			t.Eps = x
		}
		switch act, _ := md["clip."+name+".activation"].(string); act {
		case "", "quick_gelu":
			t.QuickGELU = true
		case "gelu":
		default:
			errs = append(errs, fmt.Errorf("clip.%s.activation %q", name, act))
		}
		if t.Layers > maxLayers || t.Embed > maxDim || t.Hidden > 4*maxDim || t.Heads > t.Embed || t.Embed%max(t.Heads, 1) != 0 || !(t.Eps > 0 && t.Eps < 1) {
			errs = append(errs, fmt.Errorf("a %s encoder of %d layers of %d features in %d heads", name, t.Layers, t.Embed, t.Heads))
		}
		return t
	}
	c := Config{
		Projection: get("projection_dim", 0),
		Text:       tower("text"),
		Vision:     tower("vision"),
		Vocab:      get("text.vocab_size", 0),
		Context:    get("text.context_length", 77),
		ImageSize:  get("vision.image_size", 224),
		PatchSize:  get("vision.patch_size", 0),
	}
	c.Resize = get("image.resize", c.ImageSize)
	c.Mean, c.Std = [3]float64{0.48145466, 0.4578275, 0.40821073}, [3]float64{0.26862954, 0.26130258, 0.27577711}
	if v, ok := md["clip.image.mean"].([]float64); ok && len(v) == 3 {
		copy(c.Mean[:], v)
	}
	if v, ok := md["clip.image.std"].([]float64); ok && len(v) == 3 {
		copy(c.Std[:], v)
	}
	if c.PatchSize > 0 && (c.ImageSize%c.PatchSize != 0 || c.ImageSize > 4096 || c.Resize < c.ImageSize || c.Resize > 4096) {
		errs = append(errs, fmt.Errorf("images of %d in patches of %d, resized to %d", c.ImageSize, c.PatchSize, c.Resize))
	}
	if c.Context < 3 || c.Context > 1<<12 || c.Projection > maxDim {
		errs = append(errs, fmt.Errorf("a context of %d and %d shared features", c.Context, c.Projection))
	}
	for i := range 3 {
		if !(c.Std[i] > 0) {
			errs = append(errs, fmt.Errorf("deviation %v", c.Std))
			break
		}
	}
	if err := errors.Join(errs...); err != nil {
		return c, fmt.Errorf("clip: hyperparameters: %w: %w", err, tensor.ErrShape)
	}
	return c, nil
}

// layer is one encoder layer: pre-norm self-attention and a pre-norm MLP.
type layer struct {
	Norm1, Norm2 *nn.LayerNorm
	Q, K, V, O   *nn.Linear
	Up, Down     *nn.Linear
}

// encoder is a stack of layers.
type encoder struct {
	tower  Tower
	layers []*layer
}

func (e *encoder) act(x *tensor.Tensor) *tensor.Tensor {
	if e.tower.QuickGELU {
		return x.Mul(x.Mul(tensor.Scalar[float32](1.702)).Sigmoid())
	}
	return x.Mul(tensor.Scalar[float32](0.5)).Mul(x.Mul(tensor.Scalar[float32](math.Sqrt2 / 2)).Erf().Add(tensor.Scalar[float32](1)))
}

// forward runs x, [N, S, Embed], through the layers, with the attention
// options opts.
func (e *encoder) forward(ctx context.Context, x *tensor.Tensor, opts ...tensor.AttentionOption) (*tensor.Tensor, error) {
	heads := e.tower.Heads
	split := func(y *tensor.Tensor) *tensor.Tensor {
		s := y.Shape()
		return y.Reshape(s[0], s[1], heads, s[2]/heads).Transpose(1, 2)
	}
	for _, l := range e.layers {
		h := l.Norm1.Forward(x)
		a := tensor.ScaledDotProductAttention(split(l.Q.Forward(h)), split(l.K.Forward(h)), split(l.V.Forward(h)), opts...)
		s := a.Shape()
		x = x.Add(l.O.Forward(a.Transpose(1, 2).Reshape(s[0], s[2], s[1]*s[3])))
		x = x.Add(l.Down.Forward(e.act(l.Up.Forward(l.Norm2.Forward(x)))))
		if err := tensor.Realize(ctx, x); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// Model is a CLIP model: an image encoder and a text encoder projected
// into one space.
type Model struct {
	Config Config
	tok    *Tokenizer

	patch     *tensor.Tensor // [Embed, 3, Patch, Patch]
	class     *tensor.Tensor // [Embed]
	imagePos  *tensor.Tensor // [1+patches, Embed]
	preNorm   *nn.LayerNorm
	vision    *encoder
	postNorm  *nn.LayerNorm
	imageProj *nn.Linear

	tokens    *nn.Embedding
	textPos   *tensor.Tensor // [Context, Embed]
	text      *encoder
	finalNorm *nn.LayerNorm
	textProj  *nn.Linear

	scale float64 // exp(logit_scale)
}

// Load reads the CLIP model in f, whose metadata is md, to dev. Weights
// stored in another float type than float32 are converted to it.
func Load(f *mim.Model, md mimfile.Metadata, dev tensor.Device) (*Model, error) {
	if arch, _ := md["general.architecture"].(string); arch != "clip" {
		return nil, fmt.Errorf("clip: architecture %q, want clip: %w", arch, errors.ErrUnsupported)
	}
	c, err := readConfig(md)
	if err != nil {
		return nil, err
	}
	tok, err := newTokenizer(md, c.Vocab, c.Context)
	if err != nil {
		return nil, err
	}
	var errs []error
	load := func(name string, shape ...int) *tensor.Tensor {
		w, err := f.Tensor(name)
		if err != nil {
			errs = append(errs, fmt.Errorf("clip: %w", err))
			return nil
		}
		if got := w.Shape(); !slices.Equal(got, shape) {
			errs = append(errs, fmt.Errorf("clip: %s of shape %v, want %v: %w", name, got, shape, tensor.ErrShape))
			return nil
		}
		w = w.To(dev)
		if w.DType() != tensor.Float32 {
			w = w.Cast(tensor.Float32)
			if err := tensor.Realize(context.Background(), w); err != nil {
				errs = append(errs, fmt.Errorf("clip: %s: %w", name, err))
				return nil
			}
		}
		return w
	}
	param := func(name string, shape ...int) *tensor.Param {
		if w := load(name, shape...); w != nil {
			return tensor.NewBuffer(w)
		}
		return nil
	}
	linear := func(name string, out, in int, bias bool) *nn.Linear {
		l := &nn.Linear{Weight: param(name+".weight", out, in)}
		if bias {
			l.Bias = param(name+".bias", out)
		}
		return l
	}
	norm := func(name string, dim int, eps float64) *nn.LayerNorm {
		return &nn.LayerNorm{Weight: param(name+".weight", dim), Bias: param(name+".bias", dim), Eps: eps}
	}
	stack := func(prefix string, t Tower) *encoder {
		e := &encoder{tower: t}
		for i := range t.Layers {
			p := fmt.Sprintf("%s.encoder.layers.%d.", prefix, i)
			e.layers = append(e.layers, &layer{
				Norm1: norm(p+"layer_norm1", t.Embed, t.Eps),
				Norm2: norm(p+"layer_norm2", t.Embed, t.Eps),
				Q:     linear(p+"self_attn.q_proj", t.Embed, t.Embed, true),
				K:     linear(p+"self_attn.k_proj", t.Embed, t.Embed, true),
				V:     linear(p+"self_attn.v_proj", t.Embed, t.Embed, true),
				O:     linear(p+"self_attn.out_proj", t.Embed, t.Embed, true),
				Up:    linear(p+"mlp.fc1", t.Hidden, t.Embed, true),
				Down:  linear(p+"mlp.fc2", t.Embed, t.Hidden, true),
			})
		}
		return e
	}
	v, tt := c.Vision, c.Text
	grid := c.ImageSize / c.PatchSize
	m := &Model{Config: c, tok: tok}
	m.patch = load("vision_model.embeddings.patch_embedding.weight", v.Embed, 3, c.PatchSize, c.PatchSize)
	m.class = load("vision_model.embeddings.class_embedding", v.Embed)
	m.imagePos = load("vision_model.embeddings.position_embedding.weight", 1+grid*grid, v.Embed)
	m.preNorm = norm("vision_model.pre_layrnorm", v.Embed, v.Eps)
	m.vision = stack("vision_model", v)
	m.postNorm = norm("vision_model.post_layernorm", v.Embed, v.Eps)
	m.imageProj = linear("visual_projection", c.Projection, v.Embed, false)
	m.tokens = &nn.Embedding{Weight: param("text_model.embeddings.token_embedding.weight", c.Vocab, tt.Embed)}
	m.textPos = load("text_model.embeddings.position_embedding.weight", c.Context, tt.Embed)
	m.text = stack("text_model", tt)
	m.finalNorm = norm("text_model.final_layer_norm", tt.Embed, tt.Eps)
	m.textProj = linear("text_projection", c.Projection, tt.Embed, false)
	scale := load("logit_scale")
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	s, err := scale.Data[float32](context.Background())
	if err != nil {
		return nil, err
	}
	m.scale = math.Exp(min(float64(s[0]), math.Log(100))) // CLIP clamps it at 100
	return m, nil
}

// Tokenizer returns the model's tokenizer.
func (m *Model) Tokenizer() *Tokenizer { return m.tok }

// Scale returns the factor CLIP multiplies the similarity of an image
// and a text by before a softmax over texts: exp(logit_scale), at most
// 100.
func (m *Model) Scale() float64 { return m.scale }

// EmbedPixels returns the unit-length embeddings, [N][Projection], of
// prepared images, [N, 3, ImageSize, ImageSize] (see Prepare).
func (m *Model) EmbedPixels(ctx context.Context, pixels *tensor.Tensor) ([][]float32, error) {
	c := m.Config
	s := c.ImageSize
	if pixels.Rank() != 4 || !slices.Equal(pixels.Shape()[1:], []int{3, s, s}) {
		return nil, fmt.Errorf("clip: pixels of %v, want [N 3 %d %d]: %w", pixels.Shape(), s, s, tensor.ErrShape)
	}
	n, e, grid := pixels.Shape()[0], c.Vision.Embed, s/c.PatchSize
	dev := m.patch.Device()
	x := pixels.To(dev).Conv2d(m.patch, nil, tensor.ConvOptions{Stride: c.PatchSize}) // [N, E, g, g]
	x = x.Reshape(n, e, grid*grid).Transpose(1, 2)                                    // [N, g², E]
	x = tensor.Cat(1, m.class.Reshape(1, 1, e).Expand(n, 1, e), x).Add(m.imagePos)
	x, err := m.vision.forward(ctx, m.preNorm.Forward(x))
	if err != nil {
		return nil, err
	}
	first := x.Shrink([][2]int{{0, n}, {0, 1}, {0, e}}).Reshape(n, e)
	return unit(ctx, m.imageProj.Forward(m.postNorm.Forward(first)), n, c.Projection)
}

// EmbedImages prepares images and returns their unit-length embeddings.
func (m *Model) EmbedImages(ctx context.Context, imgs []image.Image) ([][]float32, error) {
	if len(imgs) == 0 {
		return nil, nil
	}
	s := m.Config.ImageSize
	px := make([]float32, 0, len(imgs)*3*s*s)
	for _, img := range imgs {
		p, err := m.Prepare(img)
		if err != nil {
			return nil, err
		}
		px = append(px, p...)
	}
	return m.EmbedPixels(ctx, tensor.FromSlice(px, len(imgs), 3, s, s))
}

// EmbedTexts returns the unit-length embeddings of texts,
// [len(texts)][Projection].
func (m *Model) EmbedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	c := m.Config
	ids := make([][]int32, len(texts))
	longest := 0
	for i, t := range texts {
		ids[i] = m.tok.Encode(t)
		longest = max(longest, len(ids[i]))
	}
	n, e := len(texts), c.Text.Embed
	flat := make([]int32, 0, n*longest)
	for _, row := range ids {
		flat = append(flat, row...)
		for range longest - len(row) {
			flat = append(flat, m.tok.eos)
		}
	}
	dev := m.textPos.Device()
	x := m.tokens.Forward(tensor.FromSlice(flat, n, longest).To(dev))
	x = x.Add(m.textPos.Shrink([][2]int{{0, longest}, {0, e}}))
	// The text encoder is causal, so the padding after a text's end token
	// doesn't change it; its embedding is the end token's.
	x, err := m.text.forward(ctx, x, tensor.Causal())
	if err != nil {
		return nil, err
	}
	x = m.finalNorm.Forward(x)
	rows := make([]*tensor.Tensor, n)
	for i, row := range ids {
		end := len(row) - 1
		rows[i] = x.Shrink([][2]int{{i, i + 1}, {end, end + 1}, {0, e}}).Reshape(1, e)
	}
	return unit(ctx, m.textProj.Forward(tensor.Cat(0, rows...)), n, c.Projection)
}

// unit returns the rows of x, [n, d], scaled to unit length.
func unit(ctx context.Context, x *tensor.Tensor, n, d int) ([][]float32, error) {
	v, err := x.Data[float32](ctx)
	if err != nil {
		return nil, err
	}
	out := make([][]float32, n)
	for i := range n {
		row := slices.Clone(v[i*d : (i+1)*d])
		var s float64
		for _, a := range row {
			s += float64(a) * float64(a)
		}
		if s = math.Sqrt(s); s > 0 {
			for j := range row {
				row[j] = float32(float64(row[j]) / s)
			}
		}
		out[i] = row
	}
	return out, nil
}

// Similarity returns the cosine similarity of two unit-length
// embeddings, their dot product.
func Similarity(a, b []float32) float64 {
	var s float64
	for i := range min(len(a), len(b)) {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// Scores returns, for an image's embedding, the chance of each text's,
// as CLIP's zero-shot classification computes it: a softmax of their
// similarities times Scale.
func (m *Model) Scores(image []float32, texts [][]float32) []float64 {
	out := make([]float64, len(texts))
	top := math.Inf(-1)
	for i, t := range texts {
		out[i] = m.scale * Similarity(image, t)
		top = max(top, out[i])
	}
	var sum float64
	for i := range out {
		out[i] = math.Exp(out[i] - top)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
