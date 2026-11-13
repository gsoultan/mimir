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

package clip

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"unicode/utf8"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/tensor"
)

// Tokenizer is CLIP's BPE tokenizer: text is lowercased and its runs of
// white space made single spaces, cut into words, letters, digits and
// other symbols, each word's bytes mapped to printable characters as
// GPT-2 does and its last one marked with </w>, and the pairs of the
// merge list merged lowest rank first, as OpenAI's simple_tokenizer and
// Hugging Face's CLIPTokenizer do.
type Tokenizer struct {
	ids      map[string]int32
	ranks    map[[2]string]int
	bos, eos int32
	context  int

	mu    sync.Mutex
	words map[string][]int32
}

// words of CLIP's pre-tokenizer.
var wordPattern = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+`)

var spaces = regexp.MustCompile(`\s+`)

// byteRune is GPT-2's map from bytes to the printable characters that
// stand for them in the vocabulary.
var byteRune [256]rune

func init() {
	n := 0
	for b := 0; b < 256; b++ {
		if b >= '!' && b <= '~' || b >= 0xA1 && b <= 0xAC || b >= 0xAE {
			byteRune[b] = rune(b)
		} else {
			byteRune[b] = rune(256 + n)
			n++
		}
	}
}

// newTokenizer builds the tokenizer a file's metadata describes, for a
// text encoder of context positions.
func newTokenizer(md mimfile.Metadata, vocab, context int) (*Tokenizer, error) {
	if m, _ := md["tokenizer.ggml.model"].(string); m != "clip" {
		return nil, fmt.Errorf("clip: tokenizer %q, want clip: %w", m, errors.ErrUnsupported)
	}
	tokens, _ := md["tokenizer.ggml.tokens"].([]string)
	merges, _ := md["tokenizer.ggml.merges"].([]string)
	if len(tokens) != vocab || len(merges) == 0 {
		return nil, fmt.Errorf("clip: %d tokens and %d merges for a vocabulary of %d: %w", len(tokens), len(merges), vocab, tensor.ErrShape)
	}
	id := func(key string) int32 {
		n, ok := metaInt(md, key)
		if !ok || n < 0 || n >= int64(vocab) {
			return -1
		}
		return int32(n)
	}
	t := &Tokenizer{
		ids:     make(map[string]int32, len(tokens)),
		ranks:   make(map[[2]string]int, len(merges)),
		bos:     id("tokenizer.ggml.bos_token_id"),
		eos:     id("tokenizer.ggml.eos_token_id"),
		context: context,
	}
	if t.bos < 0 || t.eos < 0 {
		return nil, fmt.Errorf("clip: no start or end token: %w", tensor.ErrShape)
	}
	for i, s := range tokens {
		if _, dup := t.ids[s]; !dup {
			t.ids[s] = int32(i)
		}
	}
	for i, m := range merges {
		l, r, ok := strings.Cut(m, " ")
		if !ok {
			return nil, fmt.Errorf("clip: merge %d %q has no space: %w", i, m, tensor.ErrShape)
		}
		if _, dup := t.ranks[[2]string{l, r}]; !dup {
			t.ranks[[2]string{l, r}] = i
		}
	}
	return t, nil
}

// Encode returns the tokens of text between the start and end tokens,
// cut to the text encoder's context: at most its length minus two
// tokens of text.
func (t *Tokenizer) Encode(text string) []int32 {
	text = strings.ToLower(strings.TrimSpace(spaces.ReplaceAllString(strings.ToValidUTF8(text, ""), " ")))
	out := []int32{t.bos}
	for _, w := range wordPattern.FindAllString(text, -1) {
		out = append(out, t.word(w)...)
		if len(out) >= t.context-1 {
			out = out[:t.context-1]
			break
		}
	}
	return append(out, t.eos)
}

// cacheWords is the most words the tokenizer remembers the tokens of.
const cacheWords = 1 << 14

func (t *Tokenizer) word(w string) []int32 {
	t.mu.Lock()
	ids, ok := t.words[w]
	t.mu.Unlock()
	if ok {
		return ids
	}
	var syms []string
	for i := 0; i < len(w); i++ {
		syms = append(syms, string(byteRune[w[i]]))
	}
	syms[len(syms)-1] += "</w>"
	for len(syms) > 1 {
		best, at := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]; ok && (best < 0 || r < best) {
				best, at = r, i
			}
		}
		if at < 0 {
			break
		}
		l, r := syms[at], syms[at+1]
		merged := syms[:0:0]
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i] == l && syms[i+1] == r {
				merged = append(merged, l+r)
				i++
				continue
			}
			merged = append(merged, syms[i])
		}
		syms = merged
	}
	for _, s := range syms {
		if id, ok := t.ids[s]; ok {
			ids = append(ids, id)
			continue
		}
		// Not in the vocabulary: each character on its own, as far as
		// the vocabulary has them.
		for _, r := range s {
			if id, ok := t.ids[string(r)]; ok {
				ids = append(ids, id)
			}
		}
	}
	if utf8.RuneCountInString(w) <= 64 {
		t.mu.Lock()
		if t.words == nil || len(t.words) >= cacheWords {
			t.words = map[string][]int32{}
		}
		t.words[w] = ids
		t.mu.Unlock()
	}
	return ids
}
