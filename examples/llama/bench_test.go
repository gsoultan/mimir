package main

import (
	"os"
	"testing"

	"github.com/gsoultan/mimir/io/gguf"
	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkLlama times the model at MIMIR_LLAMA_MODEL, a GGUF file, or
// the tiny testdata/bpe.gguf: a batched pass over a prompt of 64 tokens
// to the last token's logits, and a decoding step with 64 positions in
// the caches, as llama-bench's pp64 and tg tests time llama.cpp.
func BenchmarkLlama(b *testing.B) {
	path := os.Getenv("MIMIR_LLAMA_MODEL")
	if path == "" {
		path = "testdata/bpe.gguf"
	}
	f, err := gguf.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	m, err := Load(f, tensor.CPU())
	if err != nil {
		b.Fatal(err)
	}
	const n = 64
	prompt := make([]int32, n)
	for i := range prompt {
		prompt[i] = int32(i*7919) % int32(m.Config.Vocab)
	}
	b.Run("prompt64", func(b *testing.B) {
		caches := m.NewCaches(n + 1)
		for b.Loop() {
			for _, c := range caches {
				c.Reset()
			}
			logits, err := m.Logits(b.Context(), prompt, caches, false)
			if err != nil {
				b.Fatal(err)
			}
			if err := logits.Realize(b.Context()); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(n*b.N)/b.Elapsed().Seconds(), "tokens/s")
	})
	b.Run("decode", func(b *testing.B) {
		caches := m.NewCaches(n + 1)
		if _, err := m.Logits(b.Context(), prompt, caches, false); err != nil {
			b.Fatal(err)
		}
		for b.Loop() {
			for _, c := range caches {
				c.Truncate(n)
			}
			logits, err := m.Logits(b.Context(), prompt[:1], caches, false)
			if err != nil {
				b.Fatal(err)
			}
			if _, err := logits.Data[float32](b.Context()); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
	})
}
