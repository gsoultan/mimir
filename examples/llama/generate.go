package main

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"time"

	"github.com/gsoultan/mimir/tensor"
)

// Sampler picks the next token from logits.
type Sampler struct {
	// Temperature divides the logits; 0 picks the likeliest token.
	Temperature float64
	// TopK, if positive, keeps the TopK likeliest tokens.
	TopK int
	// TopP, if below 1, keeps the likeliest tokens whose probabilities
	// add up to at least TopP: nucleus sampling.
	TopP float64
	Rand *rand.Rand

	idx   []int
	probs []float64
}

// Sample returns a token drawn from logits, [Vocab].
func (s *Sampler) Sample(logits []float32) int32 {
	if s.Temperature <= 0 || s.TopK == 1 {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		return int32(best)
	}
	idx := s.candidates(logits)
	top := float64(logits[idx[0]])
	s.probs = s.probs[:0]
	var sum float64
	for _, i := range idx {
		p := math.Exp((float64(logits[i]) - top) / s.Temperature)
		s.probs = append(s.probs, p)
		sum += p
	}
	if s.TopP > 0 && s.TopP < 1 {
		var acc float64
		for j, p := range s.probs {
			if acc += p; acc >= s.TopP*sum {
				idx, s.probs, sum = idx[:j+1], s.probs[:j+1], acc
				break
			}
		}
	}
	r := s.Rand.Float64() * sum
	for j, p := range s.probs {
		if r -= p; r < 0 {
			return int32(idx[j])
		}
	}
	return int32(idx[len(idx)-1])
}

// candidates returns the indices of the TopK largest logits, or of all,
// from the largest down.
func (s *Sampler) candidates(logits []float32) []int {
	// before reports whether logit a ranks before logit b.
	before := func(a, b int) bool { return logits[a] > logits[b] || logits[a] == logits[b] && a < b }
	idx := s.idx[:0]
	k := s.TopK
	if k <= 0 || k >= len(logits) {
		for i := range logits {
			idx = append(idx, i)
		}
	} else {
		// Keep the k best seen in a heap whose root is the worst of them.
		for i := range logits {
			if len(idx) < k {
				idx = append(idx, i)
				for j := len(idx) - 1; j > 0 && before(idx[(j-1)/2], idx[j]); j = (j - 1) / 2 {
					idx[j], idx[(j-1)/2] = idx[(j-1)/2], idx[j]
				}
				continue
			}
			if !before(i, idx[0]) {
				continue
			}
			idx[0] = i
			for j := 0; ; {
				w, l, r := j, 2*j+1, 2*j+2
				if l < k && before(idx[w], idx[l]) {
					w = l
				}
				if r < k && before(idx[w], idx[r]) {
					w = r
				}
				if w == j {
					break
				}
				idx[j], idx[w] = idx[w], idx[j]
				j = w
			}
		}
	}
	slices.SortFunc(idx, func(a, b int) int {
		switch {
		case a == b:
			return 0
		case before(a, b):
			return -1
		}
		return 1
	})
	s.idx = idx
	return idx
}

// Stats are the timings of a generation.
type Stats struct {
	PromptTokens int
	PromptTime   time.Duration // the batched pass over the prompt, to its logits
	Tokens       int           // tokens generated after the first
	Time         time.Duration // their decoding steps
}

func (s Stats) String() string {
	return fmt.Sprintf("prompt: %d tokens in %v, %.1f tokens/s; generation: %d tokens in %v, %.1f tokens/s",
		s.PromptTokens, s.PromptTime.Round(time.Millisecond), float64(s.PromptTokens)/s.PromptTime.Seconds(),
		s.Tokens, s.Time.Round(time.Millisecond), float64(s.Tokens)/s.Time.Seconds())
}

// Generate runs prompt through m in one batched pass, then generates up
// to n tokens one at a time through the KV caches, calling emit with
// each, until a token for which stop is true. It returns the generated
// tokens and the timings of the two phases.
func Generate(ctx context.Context, m *Model, prompt []int32, n int, s *Sampler, stop func(int32) bool, emit func(int32) error) ([]int32, Stats, error) {
	st := Stats{PromptTokens: len(prompt)}
	if len(prompt) == 0 {
		return nil, st, fmt.Errorf("llama: generate from an empty prompt: %w", tensor.ErrShape)
	}
	capacity := min(len(prompt)+n, max(m.Config.Context, len(prompt)+1))
	caches := m.NewCaches(capacity)
	var out []int32
	next := prompt
	start := time.Now()
	for len(out) < n {
		if caches[0].Len()+len(next) > capacity {
			break // the context is full
		}
		logits, err := m.Logits(ctx, next, caches, false)
		if err != nil {
			return out, st, err
		}
		v, err := logits.Data[float32](ctx)
		if err != nil {
			return out, st, err
		}
		tok := s.Sample(v)
		if len(out) == 0 {
			st.PromptTime, start = time.Since(start), time.Now()
		} else {
			st.Tokens, st.Time = st.Tokens+1, time.Since(start)
		}
		if stop != nil && stop(tok) {
			break
		}
		out = append(out, tok)
		if emit != nil {
			if err := emit(tok); err != nil {
				return out, st, err
			}
		}
		next = []int32{tok}
	}
	return out, st, nil
}
