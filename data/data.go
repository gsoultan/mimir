// Package data feeds batches of examples to training, like
// torch.utils.data.
//
//	trainSet, err := data.FromTensors(x, y, data.BatchSize(128), data.Shuffle(42))
//	for batch, err := range trainSet.All(ctx) {
//		if err != nil {
//			return err
//		}
//		pred := model.Forward(batch.X)
//		...
//	}
//
// A training set larger than memory streams from a Source of shards, such
// as the CSV files of datasets.CSVShards or the row groups of the Parquet
// files of datasets.ParquetShards, through a Stream, which reads a
// few shards ahead of the batches it yields and shuffles the shard order
// and a buffer of examples every epoch, so resident memory stays flat
// however many rows there are: a Stream holds up to 2×Prefetch+1 chunks
// of at most 4 MiB each, the shuffle buffer of ShuffleBuffer×Features×4 bytes
// and one batch, and keeps them from one epoch to the next. Statistics
// reads the source once for the per-feature mean and deviation that
// Normalize applies on the way:
//
//	stats, err := data.Statistics(ctx, src)
//	trainSet, err := data.NewStream(src, data.BatchSize(4096), data.Shuffle(42), data.Normalize(stats))
package data

import (
	"context"
	"fmt"
	"iter"
	"math/rand/v2"
	"slices"
	"sync/atomic"

	"github.com/gsoultan/mimir/tensor"
)

// Batch is a batch of examples: inputs X and targets Y, both with the
// examples along their first dimension.
type Batch struct {
	X, Y *tensor.Tensor
}

// Dataset yields batches. Each call to All is one pass, an epoch, over the
// data.
type Dataset interface {
	All(ctx context.Context) iter.Seq2[Batch, error]
}

// Option configures a dataset.
type Option func(*config)

type config struct {
	batch    int
	shuffle  bool
	seed     uint64
	dropLast bool
	// Stream and Statistics options, which FromTensors rejects.
	prefetch   int
	shuffleBuf int
	classes    bool
	normalize  bool
	stats      *Stats
}

// streamOnly names the first option set that only a Stream takes, or
// returns "" when none is: the zero values are unset, since Prefetch
// must be at least 1 for a Stream and ShuffleBuffer(0) means nothing to
// tensors.
func (c *config) streamOnly() string {
	switch {
	case c.prefetch != 0:
		return "Prefetch"
	case c.shuffleBuf != 0:
		return "ShuffleBuffer"
	case c.classes:
		return "ClassTargets"
	case c.normalize:
		return "Normalize"
	}
	return ""
}

// BatchSize sets the number of examples per batch. The default is 1.
func BatchSize(n int) Option { return func(c *config) { c.batch = n } }

// Shuffle visits the examples in a new random order every epoch. The
// orders depend only on seed and the epoch number, so runs repeat.
func Shuffle(seed uint64) Option { return func(c *config) { c.shuffle, c.seed = true, seed } }

// InOrder visits the examples in order, undoing an earlier Shuffle.
func InOrder() Option { return func(c *config) { c.shuffle = false } }

// DropLast skips a final batch smaller than the batch size.
func DropLast() Option { return func(c *config) { c.dropLast = true } }

// Tensors is a dataset held in two tensors whose first dimension counts
// the examples. It is safe for concurrent use.
type Tensors struct {
	x, y  *tensor.Tensor
	n     int
	cfg   config
	epoch atomic.Int64
}

// FromTensors returns a dataset of the examples along the first dimension
// of x and y, which must have the same length. Batches are views of x and
// y, or gathers when shuffling, so x and y should already be computed. It
// takes BatchSize, Shuffle, InOrder and DropLast; the options of a Stream
// are ErrOption.
func FromTensors(x, y *tensor.Tensor, opts ...Option) (*Tensors, error) {
	if err := x.Err(); err != nil {
		return nil, err
	}
	if err := y.Err(); err != nil {
		return nil, err
	}
	cfg := config{batch: 1}
	for _, o := range opts {
		o(&cfg)
	}
	if name := cfg.streamOnly(); name != "" {
		return nil, fmt.Errorf("data: %s is an option of NewStream, not of FromTensors: %w", name, ErrOption)
	}
	if cfg.batch < 1 {
		return nil, fmt.Errorf("data: batch size %d: %w", cfg.batch, tensor.ErrShape)
	}
	if x.Rank() == 0 || y.Rank() == 0 || x.Shape()[0] != y.Shape()[0] {
		return nil, fmt.Errorf("data: inputs %v and targets %v, want the same first dimension: %w", x.Shape(), y.Shape(), tensor.ErrShape)
	}
	return &Tensors{x: x, y: y, n: x.Shape()[0], cfg: cfg}, nil
}

// Len returns the number of examples.
func (d *Tensors) Len() int { return d.n }

// Batches returns the number of batches in an epoch.
func (d *Tensors) Batches() int {
	if d.cfg.dropLast {
		return d.n / d.cfg.batch
	}
	return (d.n + d.cfg.batch - 1) / d.cfg.batch
}

// SetEpoch sets the epoch that the next call to All visits, which decides
// its shuffled order. Training resumed from a checkpoint calls it so the
// order matches an uninterrupted run.
func (d *Tensors) SetEpoch(epoch int) { d.epoch.Store(int64(epoch)) }

// All yields the batches of the next epoch. A cancelled ctx ends the
// epoch with ctx's error.
func (d *Tensors) All(ctx context.Context) iter.Seq2[Batch, error] {
	epoch := d.epoch.Add(1) - 1
	return func(yield func(Batch, error) bool) {
		var order []int32
		if d.cfg.shuffle {
			r := rand.New(rand.NewPCG(d.cfg.seed, uint64(epoch)))
			order = make([]int32, d.n)
			for i, j := range r.Perm(d.n) {
				order[i] = int32(j)
			}
		}
		for b := range d.Batches() {
			if err := ctx.Err(); err != nil {
				yield(Batch{}, err)
				return
			}
			lo, hi := b*d.cfg.batch, min((b+1)*d.cfg.batch, d.n)
			var batch Batch
			if order == nil {
				batch = Batch{X: rows(d.x, lo, hi), Y: rows(d.y, lo, hi)}
			} else {
				idx := tensor.FromSlice(order[lo:hi])
				batch = Batch{X: pick(d.x, idx), Y: pick(d.y, idx)}
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

// rows returns examples [lo, hi) of t as a view.
func rows(t *tensor.Tensor, lo, hi int) *tensor.Tensor {
	ranges := make([][2]int, t.Rank())
	for i, d := range t.Shape() {
		ranges[i] = [2]int{0, d}
	}
	ranges[0] = [2]int{lo, hi}
	return t.Shrink(ranges)
}

// pick returns the examples of t at the positions in idx.
func pick(t *tensor.Tensor, idx *tensor.Tensor) *tensor.Tensor {
	shape := t.Shape()
	at := slices.Repeat([]int{1}, len(shape))
	at[0] = idx.Size()
	full := slices.Clone(shape)
	full[0] = idx.Size()
	return t.Gather(0, idx.Reshape(at...).Expand(full...))
}
