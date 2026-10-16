//go:build darwin && arm64

package metal

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// A batch encodes into the open command buffer as the kernels it holds
// would one by one, which the host waits for once, at the schedule's
// Sync. The backend plans each kernel once and keeps the plan in the
// batch's state, planning it again only when its arguments or buffers'
// kinds change, so a replayed schedule generates no kernel source and
// looks up no pipeline.

var _ backend.Batcher = (*Backend)(nil)

// batchState is what the backend keeps of a batch between runs.
type batchState struct {
	calls   []batchCall
	bufs    []*buffer
	scratch []uint64
}

// batchCall is one kernel of a batch and the call that encodes it.
type batchCall struct {
	call
	k    *backend.Kernel
	args []uint64     // k's arguments when it was planned
	bufs []bufferKind // its buffers when it was planned, outputs first
}

type bufferKind struct {
	dt backend.DType
	n  int
}

// RunBatch encodes the kernels of bt into the open command buffer.
func (b *Backend) RunBatch(ctx context.Context, bt *backend.Batch) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	st, _ := bt.State.(*batchState)
	if st == nil || len(st.calls) != len(bt.Kernels) {
		st = &batchState{calls: make([]batchCall, len(bt.Kernels))}
		bt.State = st
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	for i, k := range bt.Kernels {
		c, outs, ins := &st.calls[i], bt.Outs[i], bt.Ins[i]
		st.scratch = k.AppendArgs(st.scratch[:0])
		if c.k != k || !c.planned(st.scratch, outs, ins) {
			if err := backend.CheckRun(k, outs, ins); err != nil {
				return fmt.Errorf("metal: %w", err)
			}
			if err := b.plan(&c.call, k, outs, ins); err != nil {
				return err
			}
			c.k, c.args, c.bufs = k, append(c.args[:0], st.scratch...), c.bufs[:0]
			for _, buf := range slices.Concat(outs, ins) {
				c.bufs = append(c.bufs, bufferKind{buf.DType(), buf.Len()})
			}
		}
		st.bufs = st.bufs[:0]
		for _, bufs := range [2][]backend.Buffer{outs, ins} {
			for _, buf := range bufs {
				mb, err := b.own(buf)
				if err != nil {
					return err
				}
				st.bufs = append(st.bufs, mb)
			}
		}
		b.issue(&c.call, st.bufs)
	}
	clear(st.bufs)
	return nil
}

// planned reports whether the call was planned for a kernel with
// arguments args, over buffers like outs and ins.
func (c *batchCall) planned(args []uint64, outs, ins []backend.Buffer) bool {
	if len(c.bufs) != len(outs)+len(ins) {
		return false
	}
	for j, buf := range outs {
		if c.bufs[j] != (bufferKind{buf.DType(), buf.Len()}) {
			return false
		}
	}
	for j, buf := range ins {
		if c.bufs[len(outs)+j] != (bufferKind{buf.DType(), buf.Len()}) {
			return false
		}
	}
	return slices.Equal(c.args, args)
}
