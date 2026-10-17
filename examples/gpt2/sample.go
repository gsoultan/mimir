package main

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// The tokenizer is byte-level: each byte of UTF-8 text is a token, so the
// vocabulary has 256 tokens and any text encodes without unknown tokens.
const byteVocab = 256

// encode returns the tokens of text.
func encode(text []byte) []int32 {
	out := make([]int32, len(text))
	for i, b := range text {
		out[i] = int32(b)
	}
	return out
}

// Sampler picks the next token from logits.
type Sampler struct {
	// Temperature divides the logits; 0 picks the most likely token.
	Temperature float64
	// TopK, if positive, samples among the TopK most likely tokens only.
	TopK int
	Rand *rand.Rand
}

// Sample returns a token drawn from logits.
func (s *Sampler) Sample(logits []float32) int32 {
	if s.Temperature <= 0 {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		return int32(best)
	}
	idx := make([]int32, len(logits))
	for i := range idx {
		idx[i] = int32(i)
	}
	if s.TopK > 0 && s.TopK < len(idx) {
		slices.SortFunc(idx, func(a, b int32) int {
			switch {
			case logits[a] > logits[b]:
				return -1
			case logits[a] < logits[b]:
				return 1
			}
			return int(a - b)
		})
		idx = idx[:s.TopK]
	}
	top := float64(logits[idx[0]])
	for _, i := range idx {
		top = max(top, float64(logits[i]))
	}
	p := make([]float64, len(idx))
	var sum float64
	for j, i := range idx {
		p[j] = math.Exp((float64(logits[i]) - top) / s.Temperature)
		sum += p[j]
	}
	r := s.Rand.Float64() * sum
	for j, i := range idx {
		if r -= p[j]; r < 0 {
			return i
		}
	}
	return idx[len(idx)-1]
}

// Generate continues prompt by n tokens, sampled with s, calling emit
// with each. With cache it runs the prompt once and then one token per
// step against the KV caches, starting over from the last half of the
// context whenever the context fills up. Without, it runs the model over
// the last context's worth of tokens for every token.
func Generate(ctx context.Context, g *GPT, prompt []int32, n int, s *Sampler, cache bool, emit func(int32) error) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("gpt2: generate from an empty prompt: %w", tensor.ErrShape)
	}
	dev := g.Tok.Weight.Value().Device()
	seq := slices.Clone(prompt)
	ctxLen := g.Config.Context
	caches := g.NewCaches(1)
	var pending []int32 // tokens not in the caches yet
	if cache {
		pending = slices.Clone(seq[max(len(seq)-ctxLen, 0):])
	}
	for range n {
		var logits *tensor.Tensor
		if cache {
			if caches[0].Len()+len(pending) > ctxLen {
				for _, c := range caches {
					c.Reset()
				}
				pending = slices.Clone(seq[len(seq)-ctxLen/2:])
			}
			var err error
			logits, err = g.Decode(ctx, tensor.FromSlice(pending, 1, len(pending)).To(dev), caches)
			if err != nil {
				return seq, err
			}
		} else {
			window := seq[max(len(seq)-ctxLen, 0):]
			all := g.Forward(tensor.FromSlice(window, 1, len(window)).To(dev)).Detach()
			logits = all.Shrink([][2]int{{0, 1}, {len(window) - 1, len(window)}, {0, g.Config.Vocab}})
		}
		v, err := logits.Data[float32](ctx)
		if err != nil {
			return seq, err
		}
		tok := s.Sample(v)
		seq = append(seq, tok)
		pending = []int32{tok}
		if emit != nil {
			if err := emit(tok); err != nil {
				return seq, err
			}
		}
	}
	return seq, nil
}
