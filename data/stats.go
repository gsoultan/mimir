package data

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"

	"github.com/gsoultan/mimir/tensor"
)

// Stats are one-pass per-feature statistics of a Source, as Statistics
// computes them and Normalize applies them.
type Stats struct {
	Rows                int64
	Mean, Std, Min, Max []float32 // per feature; Std is the population standard deviation
}

// affine returns the per-feature scale and shift that standardize an
// example of features values: x*scale+shift.
func (s *Stats) affine(features int) (scale, shift []float32, err error) {
	if s == nil {
		return nil, nil, fmt.Errorf("data: Normalize with nil statistics: %w", ErrOption)
	}
	if len(s.Mean) != features || len(s.Std) != features {
		return nil, nil, fmt.Errorf("data: statistics of %d means and %d deviations for %d features: %w", len(s.Mean), len(s.Std), features, tensor.ErrShape)
	}
	scale, shift = make([]float32, features), make([]float32, features)
	for j := range features {
		scale[j] = 1
		if s.Std[j] > 0 {
			scale[j] = 1 / s.Std[j]
		}
		shift[j] = -s.Mean[j] * scale[j]
	}
	return scale, shift, nil
}

// Statistics reads every shard of src once, Prefetch shards at a time,
// and returns the per-feature statistics of its examples, accumulated in
// float64. A cancelled ctx returns its error; a shard that fails returns
// its error, which names the shard; a source without examples is
// ErrSource.
func Statistics(ctx context.Context, src Source, opts ...Option) (*Stats, error) {
	if src == nil {
		return nil, fmt.Errorf("data: nil source: %w", ErrSource)
	}
	cfg := config{batch: 1, prefetch: defaultPrefetch()}
	for _, o := range opts {
		o(&cfg)
	}
	shards, features := src.Shards(), src.Features()
	switch {
	case shards < 1:
		return nil, fmt.Errorf("data: source has %d shards: %w", shards, ErrSource)
	case features < 1:
		return nil, fmt.Errorf("data: source has %d features: %w", features, ErrSource)
	case cfg.prefetch < 1:
		return nil, fmt.Errorf("data: prefetch %d shards: %w", cfg.prefetch, ErrOption)
	}

	parent := ctx
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	accs := make([]moments, shards)
	var (
		next     atomic.Int64
		mu       sync.Mutex
		firstErr error
		readers  sync.WaitGroup
	)
	for range min(cfg.prefetch, shards) {
		readers.Go(func() {
			x := make([]float32, minChunk*features)
			y := make([]float32, minChunk)
			for ctx.Err() == nil {
				i := int(next.Add(1) - 1)
				if i >= shards {
					return
				}
				if err := accumulate(ctx, src, i, &accs[i], x, y); err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("data: shard %d: %w", i, err)
					}
					mu.Unlock()
					cancel()
					return
				}
			}
		})
	}
	readers.Wait()
	if err := parent.Err(); err != nil {
		return nil, err
	}
	if firstErr != nil {
		return nil, firstErr
	}
	// Merging in shard order makes the result independent of timing.
	total := newMoments(features)
	for i := range accs {
		total.merge(&accs[i])
	}
	if total.n == 0 {
		return nil, fmt.Errorf("data: source has no examples: %w", ErrSource)
	}
	st := &Stats{
		Rows: total.n,
		Mean: make([]float32, features),
		Std:  make([]float32, features),
		Min:  make([]float32, features),
		Max:  make([]float32, features),
	}
	for j := range features {
		st.Mean[j] = float32(total.mean[j])
		st.Std[j] = float32(math.Sqrt(total.m2[j] / float64(total.n)))
		st.Min[j] = float32(total.lo[j])
		st.Max[j] = float32(total.hi[j])
	}
	return st, nil
}

// accumulate reads shard i of src into acc, through the buffers x and y.
func accumulate(ctx context.Context, src Source, i int, acc *moments, x, y []float32) (err error) {
	rows, err := src.Open(ctx, i)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := rows.Close(); err == nil {
			err = cerr
		}
	}()
	features := len(x) / len(y)
	for ctx.Err() == nil {
		n, rerr := rows.Read(x, y)
		if n < 0 || n > len(y) {
			return fmt.Errorf("read %d examples into room for %d: %w", n, len(y), ErrSource)
		}
		if n > 0 {
			acc.add(x[:n*features], n)
		}
		if rerr != nil {
			if errors.Is(rerr, io.EOF) {
				return nil
			}
			return rerr
		}
	}
	return ctx.Err()
}

// moments accumulates per-feature count, mean, sum of squared deviations
// and range in float64. Each chunk's moments are computed in two passes
// over it and merged with Chan's parallel formula, so no value costs a
// division and the merge of shards is exact in the same way.
type moments struct {
	n                int64
	mean, m2, lo, hi []float64
	cmean, cm2       []float64 // the chunk being added
}

func newMoments(features int) *moments {
	m := &moments{
		mean: make([]float64, features),
		m2:   make([]float64, features),
		lo:   make([]float64, features),
		hi:   make([]float64, features),
	}
	for j := range features {
		m.lo[j], m.hi[j] = math.Inf(1), math.Inf(-1)
	}
	return m
}

// add merges the n examples of x, row-major, into m.
func (m *moments) add(x []float32, n int) {
	f := len(x) / n
	if m.mean == nil {
		*m = *newMoments(f)
	}
	if m.cmean == nil {
		m.cmean, m.cm2 = make([]float64, f), make([]float64, f)
	}
	cmean, cm2, lo, hi := m.cmean[:f], m.cm2[:f], m.lo[:f], m.hi[:f]
	clear(cmean)
	clear(cm2)
	for i := range n {
		row := x[i*f : (i+1)*f : (i+1)*f]
		for j, v := range row {
			d := float64(v)
			cmean[j] += d
			lo[j] = min(lo[j], d)
			hi[j] = max(hi[j], d)
		}
	}
	for j := range cmean {
		cmean[j] /= float64(n)
	}
	for i := range n {
		row := x[i*f : (i+1)*f : (i+1)*f]
		for j, v := range row {
			d := float64(v) - cmean[j]
			cm2[j] += d * d
		}
	}
	m.combine(int64(n), cmean, cm2)
}

// merge adds the moments of o to m.
func (m *moments) merge(o *moments) {
	if o.n == 0 {
		return
	}
	if m.mean == nil {
		*m = *newMoments(len(o.mean))
	}
	for j := range m.lo {
		m.lo[j] = min(m.lo[j], o.lo[j])
		m.hi[j] = max(m.hi[j], o.hi[j])
	}
	m.combine(o.n, o.mean, o.m2)
}

// combine adds n examples of the given per-feature mean and sum of
// squared deviations to m, with Chan's formula.
func (m *moments) combine(n int64, mean, m2 []float64) {
	if m.n == 0 {
		m.n = n
		copy(m.mean, mean)
		copy(m.m2, m2)
		return
	}
	na, nb := float64(m.n), float64(n)
	total := na + nb
	for j := range m.mean {
		delta := mean[j] - m.mean[j]
		m.mean[j] += delta * nb / total
		m.m2[j] += m2[j] + delta*delta*na*nb/total
	}
	m.n += n
}
