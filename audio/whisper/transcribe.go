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

package whisper

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/gsoultan/mimir/audio"
	"github.com/gsoultan/mimir/tensor"
)

// MelFilters returns the mel filter bank of Whisper's feature extractor,
// [mels, fft/2+1]: triangles on the Slaney mel scale from 0 Hz to half
// the rate, each scaled to unit area (Slaney normalization), as
// librosa.filters.mel and Hugging Face's mel_filter_bank build it.
func MelFilters(rate, fft, mels int) []float32 {
	toMel := func(hz float64) float64 {
		if hz < 1000 {
			return 3 * hz / 200
		}
		return 15 + math.Log(hz/1000)*27/math.Log(6.4)
	}
	toHz := func(m float64) float64 {
		if m < 15 {
			return 200 * m / 3
		}
		return 1000 * math.Exp(math.Log(6.4)/27*(m-15))
	}
	bins := fft/2 + 1
	lo, hi := toMel(0), toMel(float64(rate)/2)
	pts := make([]float64, mels+2)
	for i := range pts {
		pts[i] = toHz(lo + (hi-lo)*float64(i)/float64(mels+1))
	}
	out := make([]float32, mels*bins)
	for k := range mels {
		norm := 2 / (pts[k+2] - pts[k])
		for b := range bins {
			f := float64(rate/2) * float64(b) / float64(bins-1)
			down := (f - pts[k]) / (pts[k+1] - pts[k])
			up := (pts[k+2] - f) / (pts[k+2] - pts[k+1])
			out[k*bins+b] = float32(max(0, min(down, up)) * norm)
		}
	}
	return out
}

// LogMel returns the log mel spectrogram of one window of samples, at
// the model's rate and at most Samples() of them, padded with silence
// to a whole window: [Mels, Frames], as Whisper's feature extractor
// computes it. The power spectrum of Hann-windowed frames is mapped to
// the mel bands, its log10 clipped to 8 below its peak and scaled to
// about [-1, 1].
func (m *Model) LogMel(ctx context.Context, samples []float32) (*tensor.Tensor, error) {
	c := m.Config
	if len(samples) > c.Samples() {
		return nil, fmt.Errorf("whisper: %d samples for a window of %d: %w", len(samples), c.Samples(), tensor.ErrShape)
	}
	x := make([]float32, c.Samples())
	copy(x, samples)
	bins, frames := c.FFT/2+1, c.Frames()
	power := audio.Spectrogram(tensor.FromSlice(x, 1, len(x)), audio.Mel{Rate: c.SampleRate, FFT: c.FFT, Hop: c.Hop, Mels: c.Mels}) // [1, bins, frames+1]
	power = power.Reshape(bins, frames+1).Shrink([][2]int{{0, bins}, {0, frames}})
	mel := tensor.FromSlice(m.filters, c.Mels, bins).MatMul(power)
	v, err := mel.Data[float32](ctx)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(v))
	top := math.Inf(-1)
	for i, p := range v {
		l := math.Log10(max(float64(p), 1e-10))
		out[i] = float32(l)
		top = max(top, l)
	}
	for i, l := range out {
		out[i] = float32((max(float64(l), top-8) + 4) / 4)
	}
	return tensor.FromSlice(out, c.Mels, frames), nil
}

// Options configure Transcribe.
type Options struct {
	// Language is the code of the spoken language, such as "en" or "de";
	// empty to detect it from the first window with speech.
	Language string
	// Translate asks for an English translation instead of the words as
	// spoken. A multilingual model only.
	Translate bool
	// MaxTokens bounds the tokens written for each window; the default is
	// half the decoder's context, 224, as OpenAI's.
	MaxTokens int
	// Progress, if set, is called after each window with the windows
	// done and their total.
	Progress func(done, total int)
}

// Segment is the text of one window of the audio.
type Segment struct {
	Start, End float64 // seconds
	Text       string
}

// Transcript is what Transcribe heard.
type Transcript struct {
	Language string // the code of the language, given or detected
	Segments []Segment
}

// Text returns the transcript's segments joined by spaces.
func (t *Transcript) Text() string {
	parts := make([]string, 0, len(t.Segments))
	for _, s := range t.Segments {
		if s.Text != "" {
			parts = append(parts, s.Text)
		}
	}
	return strings.Join(parts, " ")
}

// ErrLanguage reports a language the model doesn't know.
var ErrLanguage = errors.New("whisper: unknown language")

// Thresholds of OpenAI's transcribe: a window whose first token is likely
// <|nospeech|> and whose text the model is unsure of holds no speech.
const (
	noSpeechProb  = 0.6
	unsureLogProb = -1.0
	silenceRMS    = 1e-4 // below this a window is skipped unheard
	repeatsToStop = 4    // a run of tokens repeated this often ends a window
	longestRepeat = 16   // the longest run of tokens checked for repeats
)

// Transcribe writes down the speech in samples, mono at the model's
// rate (16 kHz), window by window.
func (m *Model) Transcribe(ctx context.Context, samples []float32, o Options) (*Transcript, error) {
	c := m.Config
	v := &m.vocab
	t := &Transcript{Language: o.Language}
	lang := int32(-1)
	if o.Language != "" {
		if !m.Multilingual() {
			if o.Language != "en" {
				return nil, fmt.Errorf("%w: %q for an English-only model", ErrLanguage, o.Language)
			}
		} else {
			for _, l := range v.languages {
				if l.code == o.Language {
					lang = l.id
				}
			}
			if lang < 0 {
				return nil, fmt.Errorf("%w: %q", ErrLanguage, o.Language)
			}
		}
	}
	if o.Translate && (!m.Multilingual() || v.translate < 0) {
		return nil, fmt.Errorf("whisper: an English-only model can't translate: %w", errors.ErrUnsupported)
	}
	if !m.Multilingual() {
		t.Language = "en"
	}
	limit := o.MaxTokens
	if limit <= 0 {
		limit = c.TextCtx / 2
	}
	win := c.Samples()
	total := max((len(samples)+win-1)/win, 1)
	for w := range total {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		lo, hi := w*win, min((w+1)*win, len(samples))
		chunk := samples[lo:hi]
		if rms(chunk) >= silenceRMS {
			text, code, err := m.window(ctx, chunk, lang, o.Translate, limit)
			if err != nil {
				return nil, err
			}
			if t.Language == "" && code != "" {
				t.Language = code
				for _, l := range v.languages {
					if l.code == code {
						lang = l.id
					}
				}
			}
			if text != "" {
				rate := float64(c.SampleRate)
				t.Segments = append(t.Segments, Segment{Start: float64(lo) / rate, End: float64(hi) / rate, Text: text})
			}
		}
		if o.Progress != nil {
			o.Progress(w+1, total)
		}
	}
	return t, nil
}

func rms(x []float32) float64 {
	if len(x) == 0 {
		return 0
	}
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(len(x)))
}

// window writes down one window of samples in the language of token
// lang, or the one it detects when lang is -1, which it returns.
func (m *Model) window(ctx context.Context, samples []float32, lang int32, translate bool, limit int) (string, string, error) {
	v := &m.vocab
	mel, err := m.LogMel(ctx, samples)
	if err != nil {
		return "", "", err
	}
	feats, err := m.Encode(ctx, mel)
	if err != nil {
		return "", "", err
	}
	s, err := m.newState(ctx, feats)
	if err != nil {
		return "", "", err
	}
	first, err := m.logits(ctx, s, []int32{v.sot})
	if err != nil {
		return "", "", err
	}
	code := ""
	var prompt []int32
	if m.Multilingual() {
		if lang < 0 {
			best := float32(math.Inf(-1))
			for _, l := range v.languages {
				if first[l.id] > best {
					best, lang, code = first[l.id], l.id, l.code
				}
			}
		}
		task := v.transcribe
		if translate {
			task = v.translate
		}
		prompt = append(prompt, lang)
		if task >= 0 {
			prompt = append(prompt, task)
		}
	}
	prompt = append(prompt, v.noTimestamps)
	noSpeech := 0.0
	if v.noSpeech >= 0 {
		noSpeech = softmaxAt(first, v.noSpeech)
	}
	logits, err := m.logits(ctx, s, prompt)
	if err != nil {
		return "", "", err
	}
	limit = min(limit, m.Config.TextCtx-s.past)
	var out []int32
	var logProb float64
	for len(out) < limit {
		m.suppressIn(logits, len(out) == 0)
		next := argmax(logits)
		logProb += logSoftmaxAt(logits, next)
		if next == v.eos {
			break
		}
		out = append(out, next)
		if n := repeated(out); n > 0 {
			out = out[:len(out)-n]
			break
		}
		if len(out) == limit {
			break
		}
		if logits, err = m.logits(ctx, s, []int32{next}); err != nil {
			return "", "", err
		}
	}
	avg := logProb / float64(len(out)+1)
	if noSpeech > noSpeechProb && avg < unsureLogProb {
		return "", code, nil
	}
	return strings.TrimSpace(v.text(out)), code, nil
}

// suppressIn sets the logits of tokens the model may not write to -∞:
// the file's suppressed tokens, every special token but <|endoftext|>,
// and at the first position its tokens not to begin with.
func (m *Model) suppressIn(logits []float32, first bool) {
	inf := float32(math.Inf(-1))
	for _, id := range m.suppress {
		logits[id] = inf
	}
	for id := m.vocab.firstSpecial; int(id) < len(logits); id++ {
		if id != m.vocab.eos {
			logits[id] = inf
		}
	}
	if first {
		for _, id := range m.begin {
			logits[id] = inf
		}
	}
}

func argmax(x []float32) int32 {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return int32(best)
}

// logSoftmaxAt returns log softmax(x)[i].
func logSoftmaxAt(x []float32, i int32) float64 {
	top := float64(x[argmax(x)])
	var s float64
	for _, v := range x {
		s += math.Exp(float64(v) - top)
	}
	return float64(x[i]) - top - math.Log(s)
}

func softmaxAt(x []float32, i int32) float64 { return math.Exp(logSoftmaxAt(x, i)) }

// repeated returns how many tokens to drop from the end of out when it
// ends in a run of tokens repeated repeatsToStop times, which a greedy
// decoder caught in a loop writes: all but the first copy; or 0.
func repeated(out []int32) int {
	for n := 1; n <= longestRepeat && n*repeatsToStop <= len(out); n++ {
		end := len(out)
		run := out[end-n:]
		same := true
		for k := 1; k < repeatsToStop && same; k++ {
			prev := out[end-(k+1)*n : end-k*n]
			for j := range n {
				if prev[j] != run[j] {
					same = false
					break
				}
			}
		}
		if same {
			return n * (repeatsToStop - 1)
		}
	}
	return 0
}
