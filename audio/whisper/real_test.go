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
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gsoultan/mimir/audio"
	"github.com/gsoultan/mimir/audio/whisper"
	"github.com/gsoultan/mimir/tensor"
)

// TestRealModel transcribes a recording with a real Whisper model, when
// MIMIR_WHISPER names its .mim file and MIMIR_WHISPER_WAV a WAV file;
// MIMIR_WHISPER_TEXT, if set, is a phrase the transcript must hold.
func TestRealModel(t *testing.T) {
	path, wav := os.Getenv("MIMIR_WHISPER"), os.Getenv("MIMIR_WHISPER_WAV")
	if path == "" || wav == "" {
		t.Skip("MIMIR_WHISPER and MIMIR_WHISPER_WAV are not set")
	}
	m := load(t, path)
	r, err := os.Open(wav)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	clip, err := audio.ReadWAV(r)
	if err != nil {
		t.Fatal(err)
	}
	clip = clip.Resample(m.Config.SampleRate)
	if dump := os.Getenv("MIMIR_WHISPER_DUMP"); dump != "" {
		mel, err := m.LogMel(context.Background(), clip.Samples[:min(len(clip.Samples), m.Config.Samples())])
		if err != nil {
			t.Fatal(err)
		}
		writeFloats(t, dump+".mel", mel)
		enc, err := m.Encode(context.Background(), mel)
		if err != nil {
			t.Fatal(err)
		}
		writeFloats(t, dump+".enc", enc)
	}
	start := time.Now()
	tr, err := m.Transcribe(context.Background(), clip.Samples, whisper.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s in %v (%.1f s of audio): %q", tr.Language, time.Since(start).Round(time.Millisecond), clip.Seconds(), tr.Text())
	if want := os.Getenv("MIMIR_WHISPER_TEXT"); want != "" && !strings.Contains(strings.ToLower(tr.Text()), strings.ToLower(want)) {
		t.Errorf("the transcript lacks %q", want)
	}
}

func writeFloats(t *testing.T, path string, x *tensor.Tensor) {
	t.Helper()
	v, err := x.Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, 0, 4*len(v))
	for _, f := range v {
		u := math32bits(f)
		b = append(b, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
