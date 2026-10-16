package data

import (
	"context"
	"errors"
	"fmt"
	"io"
)

// Errors of streaming datasets. Test for them with errors.Is.
var (
	// ErrOption reports an invalid Stream or Statistics option.
	ErrOption = errors.New("data: invalid option")
	// ErrSource reports a Source without shards, features or examples, or
	// whose Rows read more examples than fit.
	ErrSource = errors.New("data: invalid source")
	// ErrTarget reports a target that is not a class index under
	// ClassTargets.
	ErrTarget = errors.New("data: invalid target")
)

// Source is a sharded source of examples, such as a directory of files.
// Shards are opened independently, possibly several at once from
// different goroutines.
type Source interface {
	// Shards returns the number of shards, at least 1.
	Shards() int
	// Features returns the number of values per example in X.
	Features() int
	// Open starts reading shard i. The Rows must be closed.
	Open(ctx context.Context, shard int) (Rows, error)
}

// Rows reads the examples of one shard in order, like an io.Reader: Read
// fills x with up to len(x)/features examples, row-major, and y with
// their targets, and returns how many it read. It returns io.EOF once the
// shard is exhausted, and n > 0 with a nil error until then; a final short
// read may come with nil and the next call returns 0, io.EOF, or with
// io.EOF itself.
type Rows interface {
	Read(x, y []float32) (n int, err error)
	Close() error
}

// Shard is one shard of a Slices source: X holds len(Y) examples,
// row-major, and Y their targets.
type Shard struct {
	X, Y []float32
}

// Slices is a Source held in memory, for tests and small data.
type Slices struct {
	features int
	shards   []Shard
}

// FromSlices returns a Source of shards, each holding len(Shard.Y)
// examples of features values in Shard.X. The slices are kept, not
// copied.
func FromSlices(features int, shards ...Shard) (*Slices, error) {
	if features < 1 {
		return nil, fmt.Errorf("data: %d features: %w", features, ErrSource)
	}
	if len(shards) == 0 {
		return nil, fmt.Errorf("data: no shards: %w", ErrSource)
	}
	for i, sh := range shards {
		if len(sh.X) != features*len(sh.Y) {
			return nil, fmt.Errorf("data: shard %d holds %d values for %d targets of %d features: %w", i, len(sh.X), len(sh.Y), features, ErrSource)
		}
	}
	return &Slices{features: features, shards: shards}, nil
}

// Shards returns the number of shards.
func (s *Slices) Shards() int { return len(s.shards) }

// Features returns the number of values per example.
func (s *Slices) Features() int { return s.features }

// Open starts reading shard i.
func (s *Slices) Open(ctx context.Context, shard int) (Rows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if shard < 0 || shard >= len(s.shards) {
		return nil, fmt.Errorf("data: shard %d of %d: %w", shard, len(s.shards), ErrSource)
	}
	return &sliceRows{shard: s.shards[shard], features: s.features}, nil
}

type sliceRows struct {
	shard    Shard
	features int
	pos      int
}

func (r *sliceRows) Read(x, y []float32) (int, error) {
	room := min(len(x)/r.features, len(y))
	if room == 0 {
		return 0, io.ErrShortBuffer
	}
	n := min(room, len(r.shard.Y)-r.pos)
	if n == 0 {
		return 0, io.EOF
	}
	f := r.features
	copy(x, r.shard.X[r.pos*f:(r.pos+n)*f])
	copy(y, r.shard.Y[r.pos:r.pos+n])
	r.pos += n
	return n, nil
}

func (r *sliceRows) Close() error { return nil }
