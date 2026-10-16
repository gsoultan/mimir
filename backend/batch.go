package backend

import (
	"context"
	"math"
)

// Batch is the kernels of one schedule, to run in order as one Run each
// would: kernel i reads Ins[i] and writes Outs[i].
//
// The scheduler compiles each schedule once and replays it for every
// later graph of the same structure, such as the next step of a training
// loop, with the same Batch. A Batch run again has the same kernels, the
// same *Kernel values in the same order, with the same views, reading
// and writing buffers of the same dtypes and lengths. Only the buffers
// themselves and the kernels' arguments change between runs: Attrs.Value,
// Attrs.Seed and the Value of each constant of Attrs.Program (see
// Kernel.AppendArgs). So a backend may keep what it derives from the
// kernels in State, such as compiled launches or a CUDA graph, and reuse
// it for as long as the arguments it derived it from are unchanged.
type Batch struct {
	Kernels   []*Kernel
	Outs, Ins [][]Buffer

	// State belongs to the backend that runs the batch, which keeps there
	// what it may reuse in the next run. The scheduler never reads it.
	State any
}

// Batcher is a Backend that runs a whole Batch at once, more cheaply than
// one Run per kernel.
type Batcher interface {
	Backend
	// RunBatch runs the kernels of b in order, as Run would run each, and
	// returns the first error. Like Run, it may return before the kernels
	// finish and report their errors at Sync.
	RunBatch(ctx context.Context, b *Batch) error
}

// AppendArgs appends the arguments of k to dst and returns the result:
// the values that may change while a Batch keeps k (see Batch), as bits.
// Two kernels of one Batch run alike exactly when their arguments are
// equal.
func (k *Kernel) AppendArgs(dst []uint64) []uint64 {
	dst = append(dst, math.Float64bits(k.Attrs.Value), k.Attrs.Seed)
	for _, in := range k.Attrs.Program {
		if in.Op == OpConst {
			dst = append(dst, math.Float64bits(in.Value))
		}
	}
	return dst
}
