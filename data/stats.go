package data

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
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

// standardizer returns the per-feature mean and scale that standardize
// an example of features values: (x-mean)*scale. The mean is subtracted
// before scaling, which is exact when x is near it, rather than folded
// into a shift, which would round x*scale at the magnitude of mean/std.
func (s *Stats) standardizer(features int) (mean, scale []float32, err error) {
	if s == nil {
		return nil, nil, fmt.Errorf("data: Normalize with nil statistics: %w", ErrOption)
	}
	if len(s.Mean) != features || len(s.Std) != features {
		return nil, nil, fmt.Errorf("data: statistics of %d means and %d deviations for %d features: %w", len(s.Mean), len(s.Std), features, tensor.ErrShape)
	}
	mean, scale = slices.Clone(s.Mean), make([]float32, features)
	for j := range features {
		scale[j] = 1
		if s.Std[j] > 0 {
			scale[j] = 1 / s.Std[j]
		}
	}
	return mean, scale, nil
}

// Statistics reads every shard of src once, Prefetch shards at a time,
// through a chunk of about 4 MiB per shard being read, and returns the
// per-feature statistics of its examples, accumulated in float64. It
// holds the moments of at most Prefetch shards at a time, however many
// shards there are. A cancelled ctx returns its error; a shard that fails
// returns its error, which names the shard, the lowest when several fail;
// a source without examples is ErrSource.
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
	// The readers' shards are merged in shard order, so the result does
	// not depend on timing, and only a bounded number wait to be merged:
	// a reader takes a slot before claiming a shard, and the merge of the
	// shard returns the slot. The moments of shard i wait in the ring at
	// i mod workers, whose earlier occupant, shard i-workers, was merged
	// before the slot that claimed i came free, so the send never blocks.
	// A reader takes the moments to accumulate into from free, which
	// holds one per slot, so that never blocks either.
	workers := min(cfg.prefetch, shards)
	slots := make(chan struct{}, workers)
	free := make(chan *moments, workers)
	ring := make([]chan *moments, workers)
	for i := range workers {
		slots <- struct{}{}
		free <- newMoments(features)
		ring[i] = make(chan *moments, 1)
	}
	errs := make([]error, shards) // each set by the reader of its shard, before Wait returns
	var (
		next    atomic.Int64
		readers sync.WaitGroup
	)
	rows := chunkRows(features)
	for range workers {
		readers.Go(func() {
			x := make([]float32, rows*features)
			y := make([]float32, rows)
			sums := newChunkSums(features)
			for ctx.Err() == nil {
				select {
				case <-slots:
				case <-ctx.Done():
					return
				}
				i := int(next.Add(1) - 1)
				if i >= shards {
					return
				}
				acc := <-free
				if err := accumulate(ctx, src, i, acc, sums, x, y); err != nil {
					errs[i] = fmt.Errorf("data: shard %d: %w", i, err)
					cancel()
					return
				}
				ring[i%workers] <- acc
			}
		})
	}
	total := newMoments(features)
	for i := range shards {
		var acc *moments
		select {
		case acc = <-ring[i%workers]:
		case <-ctx.Done():
		}
		if acc == nil {
			break
		}
		total.merge(acc)
		acc.reset()
		free <- acc
		slots <- struct{}{}
	}
	readers.Wait()
	if err := parent.Err(); err != nil {
		return nil, err
	}
	if err := firstFailure(errs); err != nil {
		return nil, err
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

// firstFailure returns the error of the lowest failed shard, so what
// Statistics reports does not depend on timing. A shard cut short by the
// cancellation that another shard's failure triggered reports that
// cancellation, which the failure outranks.
func firstFailure(errs []error) error {
	var failed error
	for _, err := range errs {
		if err != nil && (failed == nil || cancelled(failed) && !cancelled(err)) {
			failed = err
		}
	}
	return failed
}

// cancelled reports whether err is a context's end.
func cancelled(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// accumulate reads shard i of src into acc, through the buffers x and y
// and the chunk sums.
func accumulate(ctx context.Context, src Source, i int, acc *moments, sums *chunkSums, x, y []float32) (err error) {
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
	idle := 0 // consecutive reads of no examples
	for ctx.Err() == nil {
		n, rerr := rows.Read(x, y)
		if n < 0 || n > len(y) {
			return fmt.Errorf("read %d examples into room for %d: %w", n, len(y), ErrSource)
		}
		if n > 0 {
			acc.add(x[:n*features], n, sums)
			idle = 0
		} else if rerr == nil {
			if idle++; idle >= noProgressLimit {
				return fmt.Errorf("%w after %d reads of no examples in a row: %w", io.ErrNoProgress, idle, ErrSource)
			}
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
}

func newMoments(features int) *moments {
	m := &moments{
		mean: make([]float64, features),
		m2:   make([]float64, features),
		lo:   make([]float64, features),
		hi:   make([]float64, features),
	}
	m.reset()
	return m
}

// reset empties m, to accumulate another shard.
func (m *moments) reset() {
	m.n = 0
	for j := range m.lo {
		m.lo[j], m.hi[j] = math.Inf(1), math.Inf(-1)
	}
}

// chunkSums is a reader's scratch for the mean and squared deviations of
// the chunk being added.
type chunkSums struct {
	mean, m2 []float64
}

func newChunkSums(features int) *chunkSums {
	return &chunkSums{mean: make([]float64, features), m2: make([]float64, features)}
}

// add merges the n examples of x, row-major, into m, through sums.
func (m *moments) add(x []float32, n int, sums *chunkSums) {
	f := len(m.mean)
	cmean, cm2, lo, hi := sums.mean[:f], sums.m2[:f], m.lo[:f], m.hi[:f]
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
