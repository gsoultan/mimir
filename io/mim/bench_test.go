package mim

import (
	"context"
	"path/filepath"
	"testing"
)

// BenchmarkSessionRun runs the small models of testdata, a run after
// another in one session, as a server answering requests does: the tiny
// CNN on a batch of 8, and the tiny llama decoding a token at a time
// from its key-value cache. Each is run replaying the first run's program
// (see replay.go) and building its graph every run.
func BenchmarkSessionRun(b *testing.B) {
	ctx := context.Background()
	for _, bm := range []struct {
		name, file string
		size       int
	}{
		{"tiny-cnn/batch8", filepath.Join("testdata", "onnx", "tiny-cnn.mim"), 8},
		{"tiny-llama/decode", filepath.Join("testdata", "llama", "tiny-q4km-exact.mim"), 1},
	} {
		for _, replay := range []bool{true, false} {
			name := bm.name + "/replay"
			if !replay {
				name = bm.name + "/rebuilt"
			}
			b.Run(name, func(b *testing.B) {
				replaying = replay
				defer func() { replaying = true }()
				m, err := Open(bm.file)
				if err != nil {
					b.Fatal(err)
				}
				defer m.Close()
				s := m.NewSession()
				in := testInputs(b, m, bm.size, 1)
				for b.Loop() {
					if _, err := s.Run(ctx, in); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
