package data_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/data/datasets"
)

// BenchmarkStream streams an epoch of 2^18 synthetic examples of 32
// features in batches of 4096, in order and shuffled, with one shard
// read ahead and with four. The stream keeps its chunks and buffers from
// one epoch to the next, so after the first epoch the only allocations
// per batch are its two tensors.
func BenchmarkStream(b *testing.B) {
	const rows, features, batch = 1 << 18, 32, 4096
	for _, bm := range []struct {
		name string
		opts []data.Option
	}{
		{"inorder/prefetch1", []data.Option{data.Prefetch(1)}},
		{"inorder/prefetch4", []data.Option{data.Prefetch(4)}},
		{"shuffle/prefetch4", []data.Option{data.Prefetch(4), data.Shuffle(1)}},
		{"shuffle+normalize+classes/prefetch4", []data.Option{data.Prefetch(4), data.Shuffle(1), data.ClassTargets(), data.Normalize(&data.Stats{Mean: make([]float32, features), Std: ones(features)})}},
	} {
		b.Run(bm.name, func(b *testing.B) {
			src := datasets.Synthetic(rows, features, 8, 1)
			s, err := data.NewStream(src, append([]data.Option{data.BatchSize(batch)}, bm.opts...)...)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			for b.Loop() {
				for bt, err := range s.All(b.Context()) {
					if err != nil {
						b.Fatal(err)
					}
					if err := bt.X.Err(); err != nil {
						b.Fatal(err)
					}
				}
			}
			b.ReportMetric(float64(rows)*float64(b.N)/b.Elapsed().Seconds(), "rows/s")
		})
	}
}

func ones(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func ExampleStatistics() {
	src := datasets.Synthetic(1000, 4, 2, 7)
	stats, err := data.Statistics(context.Background(), src, data.Prefetch(2))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(stats.Rows, len(stats.Mean), len(stats.Std))
	// Output: 1000 4 4
}
