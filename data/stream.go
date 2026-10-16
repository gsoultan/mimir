package data

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/gsoultan/mimir/tensor"
)

// minChunk is the fewest examples a chunk, the unit of work between a
// shard reader and the batches, holds.
const minChunk = 8192

// defaultShuffleBuffer is the number of examples ShuffleBuffer holds by
// default.
const defaultShuffleBuffer = 10000

// defaultPrefetch is the number of shards read at once by default:
// min(4, GOMAXPROCS).
func defaultPrefetch() int { return max(1, min(4, runtime.GOMAXPROCS(0))) }

// Prefetch sets how many shards a Stream, or Statistics, reads at once,
// ahead of the batches it yields; it is at least 1. The default is
// min(4, GOMAXPROCS). Memory grows with it: a Stream holds up to
// 2×shards+1 chunks of max(batch size, 8192) examples.
func Prefetch(shards int) Option { return func(c *config) { c.prefetch = shards } }

// ShuffleBuffer sets how many examples a shuffling Stream holds to
// reorder them as they stream by: each example yielded is drawn at random
// from the buffer and the next example read takes its place. The default
// is 10000; 0 shuffles the order of the shards only. A larger buffer
// mixes examples from more shards, at the cost of rows×Features floats
// of memory.
func ShuffleBuffer(rows int) Option { return func(c *config) { c.shuffleBuf = rows } }

// ClassTargets makes a Stream's Y Int32 class indices: each target is
// rounded to the nearest integer, and a target further than 1e-3 from a
// non-negative integer ends the epoch with ErrTarget.
func ClassTargets() Option { return func(c *config) { c.classes = true } }

// Normalize standardizes each feature a Stream yields to (x-Mean)/Std
// with s, as Statistics computes it; a feature whose Std is 0 is only
// centered.
func Normalize(s *Stats) Option { return func(c *config) { c.normalize, c.stats = true, s } }

// Stream is a dataset read from a Source, one epoch per call to All, with
// Prefetch shards read ahead and, when shuffling, the shard order and a
// buffer of ShuffleBuffer examples reordered every epoch. It is safe for
// concurrent use.
//
// Batches are X [b, Features] Float32 and Y [b] Float32, or Int32 class
// indices with ClassTargets. An epoch's order depends only on the Shuffle
// seed and the epoch number, so it repeats across runs, after SetEpoch
// and whatever the Prefetch. Without Shuffle, shards are read in order
// and examples in order.
//
// Memory is bounded by the chunks in flight (see Prefetch), the shuffle
// buffer and one batch, however many examples the source holds.
type Stream struct {
	src          Source
	shards       int
	features     int
	cfg          config
	chunkRows    int
	scale, shift []float32 // x*scale+shift standardizes, per feature; nil without Normalize
	epoch        atomic.Int64
}

// NewStream returns a dataset streaming from src. It applies BatchSize,
// Shuffle, InOrder and DropLast as FromTensors does, and Prefetch,
// ShuffleBuffer, ClassTargets and Normalize.
func NewStream(src Source, opts ...Option) (*Stream, error) {
	if src == nil {
		return nil, fmt.Errorf("data: nil source: %w", ErrSource)
	}
	cfg := config{batch: 1, prefetch: defaultPrefetch(), shuffleBuf: defaultShuffleBuffer}
	for _, o := range opts {
		o(&cfg)
	}
	shards, features := src.Shards(), src.Features()
	switch {
	case shards < 1:
		return nil, fmt.Errorf("data: source has %d shards: %w", shards, ErrSource)
	case features < 1:
		return nil, fmt.Errorf("data: source has %d features: %w", features, ErrSource)
	case cfg.batch < 1:
		return nil, fmt.Errorf("data: batch size %d: %w", cfg.batch, tensor.ErrShape)
	case cfg.prefetch < 1:
		return nil, fmt.Errorf("data: prefetch %d shards: %w", cfg.prefetch, ErrOption)
	case cfg.shuffleBuf < 0:
		return nil, fmt.Errorf("data: shuffle buffer of %d examples: %w", cfg.shuffleBuf, ErrOption)
	}
	s := &Stream{src: src, shards: shards, features: features, cfg: cfg, chunkRows: max(cfg.batch, minChunk)}
	if cfg.normalize {
		var err error
		if s.scale, s.shift, err = cfg.stats.affine(features); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// Features returns the number of values per example in X.
func (s *Stream) Features() int { return s.features }

// SetEpoch sets the epoch that the next call to All visits, which decides
// its shuffled order, as Tensors.SetEpoch does for resumed training.
func (s *Stream) SetEpoch(epoch int) { s.epoch.Store(int64(epoch)) }

// All yields the batches of the next epoch. A cancelled ctx ends the epoch
// with ctx's error; a shard that fails to open or read ends it with that
// error, which names the shard. Every reader started for the epoch has
// finished, and every Rows is closed, by the time the iteration ends.
func (s *Stream) All(ctx context.Context) iter.Seq2[Batch, error] {
	epoch := s.epoch.Add(1) - 1
	return func(yield func(Batch, error) bool) {
		ctx, cancel := context.WithCancel(ctx)
		var readers sync.WaitGroup
		defer func() {
			cancel()
			readers.Wait()
		}()

		var rng *rand.Rand
		var order []int
		if s.cfg.shuffle {
			rng = rand.New(rand.NewPCG(s.cfg.seed, uint64(epoch)))
			order = rng.Perm(s.shards)
		} else {
			order = make([]int, s.shards)
			for i := range order {
				order[i] = i
			}
		}
		p := s.startReaders(ctx, &readers, order)
		c := s.newCollector(ctx, rng, yield)
		for pos, shard := range order {
			sh := &p.shards[pos]
			for {
				var ch *chunk
				var ok bool
				select {
				case ch, ok = <-sh.out:
				case <-ctx.Done():
					yield(Batch{}, ctx.Err())
					return
				}
				if !ok {
					break
				}
				if !c.add(ch) {
					return
				}
				p.free <- ch
			}
			if sh.err != nil {
				if err := ctx.Err(); err != nil {
					yield(Batch{}, err)
					return
				}
				yield(Batch{}, fmt.Errorf("data: shard %d: %w", shard, sh.err))
				return
			}
			p.slots <- struct{}{}
		}
		c.finish()
	}
}

// chunk is a run of n examples read from one shard: x holds n×features
// values and y the n targets.
type chunk struct {
	x, y []float32
	n    int
}

// shardStream carries the chunks of one shard, in order, from its reader
// to the consumer. err, if any, is set before out is closed.
type shardStream struct {
	out chan *chunk
	err error
}

// pipeline is the reader side of an epoch: the readers decode the shards
// of the planned order into chunks, prefetch of them at a time.
//
// Chunks come from free, which holds every chunk of the epoch (a nil
// entry stands for one not allocated yet), so their number is fixed. A
// reader takes a slot before claiming the next shard and the consumer
// returns the slot once it has drained that shard, so at most prefetch
// shards are open ahead of the consumer, each holding at most one chunk
// in flight and one in its channel. Readers claim shards in the planned
// order, so with slots taken first a reader never waits on a slot for a
// shard the consumer is waiting for.
type pipeline struct {
	shards []shardStream // by position in the planned order
	free   chan *chunk
	slots  chan struct{}
}

func (s *Stream) startReaders(ctx context.Context, wg *sync.WaitGroup, order []int) *pipeline {
	workers := min(s.cfg.prefetch, len(order))
	p := &pipeline{
		shards: make([]shardStream, len(order)),
		free:   make(chan *chunk, 2*workers+1),
		slots:  make(chan struct{}, workers),
	}
	for i := range p.shards {
		p.shards[i].out = make(chan *chunk, 1)
	}
	for range cap(p.free) {
		p.free <- nil
	}
	for range workers {
		p.slots <- struct{}{}
	}
	var next atomic.Int64
	for range workers {
		wg.Go(func() {
			for {
				select {
				case <-p.slots:
				case <-ctx.Done():
					return
				}
				pos := int(next.Add(1) - 1)
				if pos >= len(order) {
					return
				}
				sh := &p.shards[pos]
				sh.err = s.readShard(ctx, order[pos], sh.out, p.free)
				close(sh.out)
			}
		})
	}
	return p
}

// readShard decodes one shard into chunks sent on out, in order. It
// returns once the shard is exhausted, fails or ctx ends, with the Rows
// closed.
func (s *Stream) readShard(ctx context.Context, shard int, out chan<- *chunk, free chan *chunk) (err error) {
	rows, err := s.src.Open(ctx, shard)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := rows.Close(); err == nil {
			err = cerr
		}
	}()
	f := s.features
	row := 0
	for {
		var c *chunk
		select {
		case c = <-free:
		case <-ctx.Done():
			return ctx.Err()
		}
		if c == nil {
			c = &chunk{x: make([]float32, s.chunkRows*f), y: make([]float32, s.chunkRows)}
		}
		n, rerr := rows.Read(c.x, c.y)
		if n < 0 || n > s.chunkRows {
			return fmt.Errorf("read %d examples into room for %d: %w", n, s.chunkRows, ErrSource)
		}
		c.n = n
		if n == 0 {
			free <- c
		} else {
			if err := s.prepare(c, row); err != nil {
				return err
			}
			row += n
			select {
			case out <- c:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		if rerr != nil {
			if errors.Is(rerr, io.EOF) {
				return nil
			}
			return rerr
		}
	}
}

// prepare standardizes the examples of c and checks its targets, which
// are the rows from row on of the shard.
func (s *Stream) prepare(c *chunk, row int) error {
	f := s.features
	if s.scale != nil {
		scale, shift := s.scale[:f], s.shift[:f]
		for i := range c.n {
			x := c.x[i*f : (i+1)*f : (i+1)*f]
			for j, v := range x {
				x[j] = v*scale[j] + shift[j]
			}
		}
	}
	if s.cfg.classes {
		for i, y := range c.y[:c.n] {
			r := math.Round(float64(y))
			if r < 0 || r >= 1<<31 || math.Abs(float64(y)-r) > 1e-3 {
				return fmt.Errorf("row %d: target %v is not a class index: %w", row+i, y, ErrTarget)
			}
			c.y[i] = float32(r)
		}
	}
	return nil
}

// collector turns chunks into batches: it feeds examples through the
// shuffle buffer, if any, into the batch buffer and yields each batch as
// tensors, which copy the buffer, so one buffer serves the epoch.
type collector struct {
	ctx      context.Context
	yield    func(Batch, error) bool
	features int
	batch    int
	classes  bool
	dropLast bool
	x        []float32 // batch×features
	y        []float32 // batch
	yi       []int32   // batch, under ClassTargets
	n        int       // examples in the batch buffer

	// The shuffle buffer: size slots of one example each, filled slots
	// of which hold examples. Once full, every example in exchanges with
	// a random slot. The draining order is drawn from live.
	rng    *rand.Rand
	size   int
	filled int
	bufX   []float32
	bufY   []float32
}

func (s *Stream) newCollector(ctx context.Context, rng *rand.Rand, yield func(Batch, error) bool) *collector {
	c := &collector{
		ctx:      ctx,
		yield:    yield,
		features: s.features,
		batch:    s.cfg.batch,
		classes:  s.cfg.classes,
		dropLast: s.cfg.dropLast,
		x:        make([]float32, s.cfg.batch*s.features),
		rng:      rng,
	}
	if c.classes {
		c.yi = make([]int32, s.cfg.batch)
	} else {
		c.y = make([]float32, s.cfg.batch)
	}
	if rng != nil && s.cfg.shuffleBuf > 0 {
		c.size = s.cfg.shuffleBuf
		c.bufX = make([]float32, c.size*s.features)
		c.bufY = make([]float32, c.size)
	}
	return c
}

// add feeds the examples of ch through the pipeline, yielding the batches
// they complete. It returns false once the iteration should end.
func (c *collector) add(ch *chunk) bool {
	f := c.features
	for i := range ch.n {
		row, y := ch.x[i*f:(i+1)*f], ch.y[i]
		if c.size == 0 {
			if !c.push(row, y) {
				return false
			}
			continue
		}
		slot := c.filled
		if c.filled < c.size {
			c.filled++
		} else {
			slot = c.rng.IntN(c.size)
			if !c.push(c.bufX[slot*f:(slot+1)*f], c.bufY[slot]) {
				return false
			}
		}
		copy(c.bufX[slot*f:], row)
		c.bufY[slot] = y
	}
	return true
}

// finish drains the shuffle buffer and yields the last batch.
func (c *collector) finish() {
	f := c.features
	// Each draw removes a random live slot; a slot drawn swaps places
	// with the last live one, so the live slots stay a prefix.
	live := make([]int32, c.filled)
	for i := range live {
		live[i] = int32(i)
	}
	for len(live) > 0 {
		j := c.rng.IntN(len(live))
		slot := int(live[j])
		if !c.push(c.bufX[slot*f:(slot+1)*f], c.bufY[slot]) {
			return
		}
		live[j] = live[len(live)-1]
		live = live[:len(live)-1]
	}
	c.filled = 0
	if c.n > 0 && !c.dropLast {
		c.flush()
	}
}

// push adds one example to the batch buffer and yields the batch when it
// is full.
func (c *collector) push(row []float32, y float32) bool {
	copy(c.x[c.n*c.features:], row)
	if c.classes {
		c.yi[c.n] = int32(y)
	} else {
		c.y[c.n] = y
	}
	c.n++
	if c.n < c.batch {
		return true
	}
	return c.flush()
}

// flush yields the batch buffer as tensors and empties it.
func (c *collector) flush() bool {
	n := c.n
	c.n = 0
	if err := c.ctx.Err(); err != nil {
		c.yield(Batch{}, err)
		return false
	}
	b := Batch{X: tensor.FromSlice(c.x[:n*c.features], n, c.features)}
	if c.classes {
		b.Y = tensor.FromSlice(c.yi[:n])
	} else {
		b.Y = tensor.FromSlice(c.y[:n])
	}
	return c.yield(b, nil)
}
