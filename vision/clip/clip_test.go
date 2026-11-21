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

package clip_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/vision/clip"
)

func load(t *testing.T, path string) *clip.Model {
	t.Helper()
	f, err := mim.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	md, err := f.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	m, err := clip.Load(f, md, tensor.DefaultDevice())
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func near(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	worst := 0.0
	for i := range got {
		worst = max(worst, math.Abs(float64(got[i]-want[i])))
	}
	if worst > tol {
		t.Errorf("%s: off by up to %g, want at most %g", what, worst, tol)
	}
}

// unit scales v to unit length.
func unit(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / math.Sqrt(s))
	}
	return out
}

// The tiny model in testdata is a randomly initialized CLIP of two layers
// of 16 features a tower, made with Hugging Face Transformers and
// converted by mim-import; clip_ref.json holds the image features
// Transformers computes for the pixels in clip_pixels.f32 and the text
// features of two token lists.
func TestMatchesTransformers(t *testing.T) {
	m := load(t, filepath.Join("testdata", "tiny.mim"))
	b, err := os.ReadFile(filepath.Join("testdata", "clip_ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs        [][]int32   `json:"ids"`
		Image      []float32   `json:"image"`
		Text       [][]float32 `json:"text"`
		LogitScale float64     `json:"logit_scale"`
	}
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join("testdata", "clip_pixels.f32"))
	if err != nil {
		t.Fatal(err)
	}
	px := make([]float32, len(raw)/4)
	for i := range px {
		px[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	ctx := context.Background()
	img, err := m.EmbedPixels(ctx, tensor.FromSlice(px, 1, 3, 32, 32))
	if err != nil {
		t.Fatal(err)
	}
	near(t, "image", img[0], unit(ref.Image), 1e-4)
	// The texts are "cat a" and "dog" in the tiny vocabulary.
	for i, want := range [][]int32{m.Tokenizer().Encode("Cat  a "), m.Tokenizer().Encode("DOG")} {
		if !slices.Equal(want, ref.IDs[i][:len(want)]) {
			t.Errorf("tokens %v, want %v", want, ref.IDs[i])
		}
	}
	texts, err := m.EmbedTexts(ctx, []string{"cat a", "dog"})
	if err != nil {
		t.Fatal(err)
	}
	for i := range texts {
		near(t, "text", texts[i], unit(ref.Text[i]), 1e-4)
	}
	if math.Abs(m.Scale()-math.Exp(ref.LogitScale)) > 1e-3 {
		t.Errorf("scale %v, want %v", m.Scale(), math.Exp(ref.LogitScale))
	}
	p := m.Scores(img[0], texts)
	if math.Abs(p[0]+p[1]-1) > 1e-9 {
		t.Errorf("scores %v", p)
	}
}

func TestPrepare(t *testing.T) {
	m := load(t, filepath.Join("testdata", "tiny.mim"))
	// A wide image is resized to 32 high and cut in the middle; a plain
	// colour stays itself.
	img := image.NewRGBA(image.Rect(0, 0, 80, 40))
	for i := range img.Pix {
		img.Pix[i] = 200
	}
	px, err := m.Prepare(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(px) != 3*32*32 {
		t.Fatalf("%d values", len(px))
	}
	c := m.Config
	for ch := range 3 {
		want := float32((200.0/255 - c.Mean[ch]) / c.Std[ch])
		if got := px[ch*1024+517]; math.Abs(float64(got-want)) > 1e-5 {
			t.Errorf("channel %d: %v, want %v", ch, got, want)
		}
	}
	embs, err := m.EmbedImages(context.Background(), []image.Image{img, img.SubImage(image.Rect(10, 0, 50, 40))})
	if err != nil {
		t.Fatal(err)
	}
	if len(embs) != 2 || len(embs[0]) != c.Projection {
		t.Errorf("%d embeddings", len(embs))
	}
}

func TestLoadRefuses(t *testing.T) {
	f, err := mim.Open(filepath.Join("testdata", "tiny.mim"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	md, err := f.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		key string
		val any
	}{
		{"general.architecture", "whisper"},
		{"clip.vision.patch_size", int64(5)},
		{"clip.text.embedding_length", int64(12)},
		{"clip.text.activation", "relu"},
		{"tokenizer.ggml.model", "gpt2"},
	} {
		bad := maps.Clone(md)
		bad[tc.key] = tc.val
		if _, err := clip.Load(f, bad, tensor.DefaultDevice()); err == nil {
			t.Errorf("%s = %v loaded", tc.key, tc.val)
		}
	}
}

// TestRealModel checks a real CLIP model, when MIMIR_CLIP names its .mim
// file and MIMIR_CLIP_IMAGE a photo: the texts in MIMIR_CLIP_TEXTS,
// separated by |, the first of which must be the best match.
func TestRealModel(t *testing.T) {
	path, photo := os.Getenv("MIMIR_CLIP"), os.Getenv("MIMIR_CLIP_IMAGE")
	if path == "" || photo == "" {
		t.Skip("MIMIR_CLIP and MIMIR_CLIP_IMAGE are not set")
	}
	m := load(t, path)
	r, err := os.Open(photo)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	img, _, err := image.Decode(r)
	if err != nil {
		t.Fatal(err)
	}
	texts := slices.Collect(func(yield func(string) bool) {
		for _, s := range splitBar(os.Getenv("MIMIR_CLIP_TEXTS")) {
			if !yield(s) {
				return
			}
		}
	})
	ctx := context.Background()
	ie, err := m.EmbedImages(ctx, []image.Image{img})
	if err != nil {
		t.Fatal(err)
	}
	te, err := m.EmbedTexts(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	if dump := os.Getenv("MIMIR_CLIP_DUMP"); dump != "" {
		px, _ := m.Prepare(img)
		b := make([]byte, 0, 4*len(px))
		for _, f := range px {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(f))
		}
		os.WriteFile(dump+".pix", b, 0o644)
		js, _ := json.Marshal(map[string]any{"image": ie[0], "texts": te})
		os.WriteFile(dump+".json", js, 0o644)
	}
	p := m.Scores(ie[0], te)
	t.Logf("chances %v", p)
	if len(p) > 1 && slices.Index(p, slices.Max(p)) != 0 {
		t.Errorf("%q isn't the best match", texts[0])
	}
}

func splitBar(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == '|' {
			if i > start {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	return out
}
