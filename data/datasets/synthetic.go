package datasets

import (
	"context"
	"fmt"
	"io"

	"github.com/gsoultan/mimir/data"
)

// Synthetic returns a deterministic data.Source of rows examples of
// features values in shards shards, generated as they are read from
// (seed, shard, row) by a hash, so it holds nothing in memory and reads
// cost a few nanoseconds per value. The values are uniform in [-1, 1)
// and the target is 1 when w·x plus a little noise is positive, else 0,
// for a weight vector w drawn from seed. It is for scale tests and
// benchmarks. rows below 0 counts as 0, and features or shards below 1
// as 1.
func Synthetic(rows int64, features, shards int, seed uint64) data.Source {
	rows, features, shards = max(rows, 0), max(features, 1), max(shards, 1)
	w := make([]float32, features)
	s := seed ^ 0x9E3779B97F4A7C15
	for j := range w {
		w[j] = unit(mix(&s))
	}
	return &synthetic{rows: rows, features: features, shards: shards, seed: seed, w: w}
}

type synthetic struct {
	rows     int64
	features int
	shards   int
	seed     uint64
	w        []float32
}

func (s *synthetic) Shards() int   { return s.shards }
func (s *synthetic) Features() int { return s.features }

func (s *synthetic) Open(ctx context.Context, shard int) (data.Rows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if shard < 0 || shard >= s.shards {
		return nil, fmt.Errorf("datasets: synthetic shard %d of %d: %w", shard, s.shards, data.ErrSource)
	}
	// Shard i holds rows i*rows/shards to (i+1)*rows/shards.
	lo := s.rows * int64(shard) / int64(s.shards)
	hi := s.rows * int64(shard+1) / int64(s.shards)
	return &syntheticRows{src: s, shard: uint64(shard), row: lo, end: hi}, nil
}

type syntheticRows struct {
	src      *synthetic
	shard    uint64
	row, end int64
}

func (r *syntheticRows) Read(x, y []float32) (int, error) {
	f := r.src.features
	room := min(len(x)/f, len(y))
	if room == 0 {
		return 0, io.ErrShortBuffer
	}
	n := int(min(int64(room), r.end-r.row))
	if n == 0 {
		return 0, io.EOF
	}
	w := r.src.w[:f]
	for i := range n {
		s := r.src.seed ^ (r.shard+1)*0xA0761D6478BD642F ^ uint64(r.row+int64(i))*0xE7037ED1A0B428DB
		row := x[i*f : (i+1)*f : (i+1)*f]
		var dot float32
		for j := range row {
			v := unit(mix(&s))
			row[j] = v
			dot += w[j] * v
		}
		y[i] = 0
		if dot+0.1*unit(mix(&s)) > 0 {
			y[i] = 1
		}
	}
	r.row += int64(n)
	return n, nil
}

func (r *syntheticRows) Close() error { return nil }

// mix advances s and returns a well-mixed 64-bit value (SplitMix64).
func mix(s *uint64) uint64 {
	*s += 0x9E3779B97F4A7C15
	z := *s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// unit maps a random 64-bit value to [-1, 1).
func unit(z uint64) float32 { return float32(int32(z>>32)) * (1.0 / (1 << 31)) }
