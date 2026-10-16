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

// chunkBytes is the memory budget of a chunk, the unit of work between a
// shard reader and the batches: its examples and targets as float32.
const chunkBytes = 4 << 20

// maxChunkRows is the most examples a chunk holds, however narrow they
// are.
const maxChunkRows = 8192

// chunkRows returns the examples per chunk of features values each: as
// many as fit chunkBytes, at least 1 and at most maxChunkRows. It does
// not depend on the batch size; the collector spans chunks.
func chunkRows(features int) int {
	return max(1, min(maxChunkRows, chunkBytes/(4*(features+1))))
}

// noProgressLimit is how many consecutive reads of no examples and no
// error fail a shard, as bufio gives up on a reader that makes no
// progress.
const noProgressLimit = 100

// defaultShuffleBuffer is the number of examples ShuffleBuffer holds by
// default.
const defaultShuffleBuffer = 10000

// defaultPrefetch is the number of shards read at once by default:
// min(4, GOMAXPROCS).
func defaultPrefetch() int { return max(1, min(4, runtime.GOMAXPROCS(0))) }

// Prefetch sets how many shards a Stream, or Statistics, reads at once,
// ahead of the batches it yields; it is at least 1. The default is
// min(4, GOMAXPROCS). Memory grows with it: a Stream holds up to
// 2×shards+1 chunks of about 4 MiB each (at most 8192 examples), which
// it keeps from one epoch to the next, and Statistics one such chunk per
// shard being read.
func Prefetch(shards int) Option { return func(c *config) { c.prefetch = shards } }

// ShuffleBuffer sets how many examples a shuffling Stream holds to
// reorder them as they stream by: each example yielded is drawn at random
// from the buffer and the next example read takes its place. The default
// is 10000; 0 shuffles the order of the shards only. A larger buffer
// mixes examples from more shards, at the cost of rows×Features×4 bytes
// of memory.
func ShuffleBuffer(rows int) Option { return func(c *config) { c.shuffleBuf = rows } }

// ClassTargets makes a Stream's Y Int32 class indices: each target is
// rounded to the nearest integer, and a target further than 1e-3 from a
// non-negative integer, or not a number, ends the epoch with ErrTarget.
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
// Memory is bounded however many examples the source holds: 2×Prefetch+1
// chunks of about 4 MiB each, the shuffle buffer of
// ShuffleBuffer×Features×4 bytes and one batch of BatchSize×Features×4
// bytes, which the Stream keeps from one epoch to the next, plus the
// tensors of the batch being yielded.
type Stream struct {
	src          Source
	shards       int
	features     int
	cfg          config
	chunkRows    int
	scale, shift []float32 // x*scale+shift standardizes, per feature; nil without Normalize
	epoch        atomic.Int64

	// The buffers of past epochs, for the next: an epoch takes chunks
	// and a collector at its start and puts them back at its end, so a
	// Stream allocates them once however many epochs it runs, and
	// concurrent epochs each have their own.
	mu         sync.Mutex
	chunks     []*chunk
	collectors []*collector
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
	s := &Stream{src: src, shards: shards, features: features, cfg: cfg, chunkRows: chunkRows(features)}
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
		c := s.takeCollector(ctx, rng, yield)
		defer func() {
			cancel()
			// Once every reader has finished, every chunk of the epoch is
			// in free or in an out channel, so the next epoch gets them all.
			readers.Wait()
			s.keep(p.collect(), c)
		}()
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
				ok = c.add(ch)
				p.free <- ch
				if !ok {
					return
				}
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
// Chunks come from free, which holds a token for every chunk of the
// epoch (a nil stands for one not allocated yet), so their number is
// fixed. Every chunk is in exactly one place, free, an out channel, a
// reader's hands or the consumer's, so whoever holds one can return it
// to free without blocking. A reader takes a slot before claiming the
// next shard and the consumer returns the slot once it has drained that
// shard, so at most prefetch shards are open ahead of the consumer, each
// holding at most one chunk in flight and one in its channel. Readers
// claim shards in the planned order, so with slots taken first a reader
// never waits on a slot for a shard the consumer is waiting for.
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
	// The chunks kept from earlier epochs come first; a reader allocates
	// a chunk when it draws a nil.
	s.mu.Lock()
	for range cap(p.free) {
		var c *chunk
		if n := len(s.chunks); n > 0 {
			c, s.chunks = s.chunks[n-1], s.chunks[:n-1]
		}
		p.free <- c
	}
	s.mu.Unlock()
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

// collect returns the chunks of the epoch, to call once every reader has
// finished: those in free and those sent on an out channel that the
// consumer did not drain. The nils of chunks never allocated are dropped.
func (p *pipeline) collect() []*chunk {
	var chunks []*chunk
	drain := func(ch <-chan *chunk) {
		for {
			select {
			case c, ok := <-ch:
				if !ok {
					return
				}
				if c != nil {
					chunks = append(chunks, c)
				}
			default:
				return
			}
		}
	}
	drain(p.free)
	for i := range p.shards {
		drain(p.shards[i].out)
	}
	return chunks
}

// keep stores the chunks and the collector of a finished epoch for the
// next.
func (s *Stream) keep(chunks []*chunk, c *collector) {
	c.ctx, c.yield, c.rng = nil, nil, nil
	s.mu.Lock()
	s.chunks = append(s.chunks, chunks...)
	s.collectors = append(s.collectors, c)
	s.mu.Unlock()
}

// readShard decodes one shard into chunks sent on out, in order. It
// returns once the shard is exhausted, fails or ctx ends, with the Rows
// closed and every chunk it took from free sent on out or returned to
// free.
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
	row, idle := 0, 0
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
		// From here on c goes back to free, or on to out, before any
		// return. The reader holds it, so free has room and the send
		// never blocks.
		if n < 0 || n > s.chunkRows {
			free <- c
			return fmt.Errorf("read %d examples into room for %d: %w", n, s.chunkRows, ErrSource)
		}
		c.n = n
		if n == 0 {
			free <- c
			if rerr == nil {
				if idle++; idle >= noProgressLimit {
					return fmt.Errorf("%w after %d reads of no examples in a row: %w", io.ErrNoProgress, idle, ErrSource)
				}
			}
		} else {
			idle = 0
			if err := s.prepare(c, row); err != nil {
				free <- c
				return err
			}
			row += n
			select {
			case out <- c:
			case <-ctx.Done():
				free <- c
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
			r, ok := classIndex(y)
			if !ok {
				return fmt.Errorf("row %d: target %v is not a class index: %w", row+i, y, ErrTarget)
			}
			c.y[i] = r
		}
	}
	return nil
}

// classIndex returns the class index y rounds to and whether it is one:
// a non-negative integer below 2^31 within 1e-3 of y. The test is
// written as what passes, so a NaN, which fails every comparison, is
// rejected rather than converted.
func classIndex(y float32) (float32, bool) {
	r := math.Round(float64(y))
	if r >= 0 && r < 1<<31 && math.Abs(float64(y)-r) <= 1e-3 {
		return float32(r), true
	}
	return 0, false
}

// collector turns chunks into batches: it feeds examples through the
// shuffle buffer, if any, into the batch buffer and yields each batch as
// tensors, which copy the buffer, so one buffer serves the epoch. A
// Stream keeps its collectors across epochs; ctx, yield and rng are those
// of the epoch using it.
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

// takeCollector returns a collector for an epoch, one kept from an
// earlier epoch when there is one, emptied.
func (s *Stream) takeCollector(ctx context.Context, rng *rand.Rand, yield func(Batch, error) bool) *collector {
	var c *collector
	s.mu.Lock()
	if n := len(s.collectors); n > 0 {
		c, s.collectors = s.collectors[n-1], s.collectors[:n-1]
	}
	s.mu.Unlock()
	if c == nil {
		c = s.newCollector()
	}
	c.ctx, c.yield, c.rng, c.n, c.filled = ctx, yield, rng, 0, 0
	return c
}

// newCollector returns a collector with the buffers an epoch of s needs.
func (s *Stream) newCollector() *collector {
	c := &collector{
		features: s.features,
		batch:    s.cfg.batch,
		classes:  s.cfg.classes,
		dropLast: s.cfg.dropLast,
		x:        make([]float32, s.cfg.batch*s.features),
	}
	if c.classes {
		c.yi = make([]int32, s.cfg.batch)
	} else {
		c.y = make([]float32, s.cfg.batch)
	}
	if s.cfg.shuffle && s.cfg.shuffleBuf > 0 {
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
