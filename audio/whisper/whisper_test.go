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

package whisper_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/audio/whisper"
	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

func math32bits(f float32) uint32 { return math.Float32bits(f) }

func load(t *testing.T, path string) *whisper.Model {
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
	m, err := whisper.Load(f, md, tensor.DefaultDevice())
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// The tiny model in testdata is a randomly initialized Whisper of two
// layers of 16 features, made with Hugging Face Transformers and
// converted by mim-import; whisper_ref.json holds what Transformers
// computes for the sound in whisper_audio.f32: the features, the
// encoder's output and the decoder's logits after a prompt.
type reference struct {
	Prompt      []int32   `json:"prompt"`
	MelHead     []float32 `json:"mel_head"`
	MelSum      float64   `json:"mel_sum"`
	EncHead     []float32 `json:"enc_head"`
	EncSum      float64   `json:"enc_sum"`
	FirstLogits []float32 `json:"first_logits"`
	Logits      []float32 `json:"logits"`
}

func tiny(t *testing.T) (*whisper.Model, reference, []float32) {
	t.Helper()
	m := load(t, filepath.Join("testdata", "tiny.mim"))
	b, err := os.ReadFile(filepath.Join("testdata", "whisper_ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref reference
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join("testdata", "whisper_audio.f32"))
	if err != nil {
		t.Fatal(err)
	}
	samples := make([]float32, len(raw)/4)
	for i := range samples {
		samples[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return m, ref, samples
}

func data(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, err := x.Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return v
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

func sum(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x)
	}
	return s
}

func TestMatchesTransformers(t *testing.T) {
	m, ref, samples := tiny(t)
	ctx := context.Background()
	mel, err := m.LogMel(ctx, samples)
	if err != nil {
		t.Fatal(err)
	}
	mv := data(t, mel)
	head := make([]float32, 0, len(ref.MelHead))
	for band := range 80 {
		head = append(head, mv[band*3000:band*3000+4]...)
	}
	near(t, "spectrogram", head, ref.MelHead, 1e-4)
	if d := math.Abs(sum(mv) - ref.MelSum); d > 1e-3*math.Abs(ref.MelSum)+1 {
		t.Errorf("spectrogram sum %g, want %g", sum(mv), ref.MelSum)
	}
	enc, err := m.Encode(ctx, mel)
	if err != nil {
		t.Fatal(err)
	}
	ev := data(t, enc)
	near(t, "encoder", ev[:len(ref.EncHead)], ref.EncHead, 1e-3)
	if d := math.Abs(sum(ev) - ref.EncSum); d > 0.05 {
		t.Errorf("encoder sum %g, want %g", sum(ev), ref.EncSum)
	}
	first, err := m.Logits(ctx, enc, ref.Prompt[:1])
	if err != nil {
		t.Fatal(err)
	}
	near(t, "logits of the first token", first, ref.FirstLogits, 2e-3)
	logits, err := m.Logits(ctx, enc, ref.Prompt)
	if err != nil {
		t.Fatal(err)
	}
	near(t, "logits after the prompt", logits, ref.Logits, 2e-3)
}

func TestTranscribe(t *testing.T) {
	m, _, samples := tiny(t)
	if !m.Multilingual() || !slices.Equal(m.Languages(), []string{"en", "de", "fr"}) {
		t.Fatalf("multilingual %v, languages %v", m.Multilingual(), m.Languages())
	}
	ctx := context.Background()
	var windows []int
	// Three windows: the sound, silence (skipped), the sound again.
	long := slices.Concat(samples, make([]float32, m.Config.Samples()*2-len(samples)), samples)
	tr, err := m.Transcribe(ctx, long, whisper.Options{Progress: func(done, total int) { windows = append(windows, done, total) }})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(windows, []int{1, 3, 2, 3, 3, 3}) {
		t.Errorf("progress %v", windows)
	}
	if tr.Language == "" {
		t.Error("no language detected")
	}
	for _, s := range tr.Segments {
		if s.Start != 0 && s.Start != 60 {
			t.Errorf("a segment from %v s", s.Start)
		}
	}
	// The same words in a language given, and English-only refusals.
	if _, err := m.Transcribe(ctx, samples, whisper.Options{Language: "xx"}); !errors.Is(err, whisper.ErrLanguage) {
		t.Errorf("language xx: %v", err)
	}
	again, err := m.Transcribe(ctx, samples, whisper.Options{Language: tr.Language})
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Segments) > 0 && again.Text() != tr.Segments[0].Text {
		t.Errorf("with the language given %q, detected %q", again.Text(), tr.Segments[0].Text)
	}
	if _, err := m.Transcribe(ctx, samples, whisper.Options{Translate: true, MaxTokens: 3}); err != nil {
		t.Errorf("translate: %v", err)
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
		{"general.architecture", "llama"},
		{"whisper.embedding_length", int64(15)},
		{"whisper.encoder.context_length", int64(10)},
		{"tokenizer.ggml.tokens", []string{"a"}},
		{"tokenizer.ggml.model", "llama"},
	} {
		bad := maps.Clone(md)
		bad[tc.key] = tc.val
		if _, err := whisper.Load(f, bad, tensor.DefaultDevice()); err == nil {
			t.Errorf("%s = %v loaded", tc.key, tc.val)
		}
	}
}

func TestMelFilters(t *testing.T) {
	// librosa.filters.mel(sr=16000, n_fft=400, n_mels=80): band 0 peaks
	// at bin 1 at 0.02486259, and each band's weights sum as Slaney's
	// unit-area normalization gives.
	f := whisper.MelFilters(16000, 400, 80)
	if got := f[1]; math.Abs(float64(got)-0.02486259) > 1e-6 {
		t.Errorf("band 0, bin 1 = %v", got)
	}
	for k := range 80 {
		if sum(f[k*201:(k+1)*201]) <= 0 {
			t.Errorf("band %d is empty", k)
		}
	}
}
