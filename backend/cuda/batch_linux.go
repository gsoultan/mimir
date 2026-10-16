//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"slices"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// A batch runs as calls the backend plans once and keeps in the batch's
// state, planning a kernel again only when its arguments change, so a
// replayed schedule generates no kernel source and looks up no compiled
// kernel. From its second run on, a batch launches as one CUDA graph:
// the backend captures the calls on its stream once, instantiates the
// graph, and launches it as a whole. Before each launch it updates the
// kernel nodes whose parameters or buffers changed in the executable
// graph, and captures the graph again when a cuBLAS call's operands
// changed, since cuBLAS decides itself how a product runs. When the
// driver lacks the graph API, or capture or instantiation fail, the
// batch's calls launch one by one, as they do with MIMIR_CUDA_GRAPHS=0.

// graphsOff turns CUDA graphs off.
var graphsOff = os.Getenv("MIMIR_CUDA_GRAPHS") == "0"

// graphsReady loads the CUDA graph API, or reports why it cannot.
var graphsReady = loadGraphs

var _ backend.Batcher = (*Backend)(nil)

// batchState is what the backend keeps of a batch between runs.
type batchState struct {
	calls   []batchCall
	runs    int
	graph   *graphHandles // the captured graph, or nil
	noGraph bool          // capture failed: launch the calls one by one
	args    []unsafe.Pointer
	scratch []uint64
}

// batchCall is one kernel of a batch and the call that runs it.
type batchCall struct {
	call
	k     *backend.Kernel
	args  []uint64     // k's arguments when it was planned
	bufs  []bufferKind // its buffers when it was planned, outputs first
	ptrs  []uint64     // its buffers' device pointers in this run
	node  uintptr      // the graph's node of its kernel launch, or 0
	dirty bool         // the launch or pointers changed since the graph took them
}

type bufferKind struct {
	dt backend.DType
	n  int
}

// graphHandles are an instantiated CUDA graph and the graph it came from.
type graphHandles struct{ graph, exec uintptr }

// RunBatch queues the kernels of bt on the stream, as one CUDA graph from
// the batch's second run on.
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
	defer b.enter()()
	recapture := false
	for i, k := range bt.Kernels {
		c, outs, ins := &st.calls[i], bt.Outs[i], bt.Ins[i]
		st.scratch = k.AppendArgs(st.scratch[:0])
		if c.k != k || !c.planned(st.scratch, outs, ins) {
			if err := backend.CheckRun(k, outs, ins); err != nil {
				return fmt.Errorf("cuda: %w", err)
			}
			old := c.call
			if err := b.plan(&c.call, k, outs, ins); err != nil {
				return err
			}
			c.k, c.args, c.dirty = k, append(c.args[:0], st.scratch...), true
			c.bufs = c.bufs[:0]
			for _, buf := range slices.Concat(outs, ins) {
				c.bufs = append(c.bufs, bufferKind{buf.DType(), buf.Len()})
			}
			c.ptrs = slices.Grow(c.ptrs[:0], len(c.bufs))[:len(c.bufs)]
			st.args = slices.Grow(st.args, len(c.bufs)+2)
			// A kernel node takes any launch, but a different set of
			// nodes, or another cuBLAS call, needs a new graph.
			recapture = recapture || (old.l == nil) != (c.l == nil) || (old.gemm == nil) != (c.gemm == nil) ||
				old.gemm != nil && *old.gemm != *c.gemm
		}
		j := 0
		for _, bufs := range [2][]backend.Buffer{outs, ins} {
			for _, buf := range bufs {
				cb, err := b.own(buf)
				if err != nil {
					return err
				}
				if c.ptrs[j] != cb.ptr {
					c.ptrs[j], c.dirty = cb.ptr, true
					recapture = recapture || c.gemm != nil
				}
				j++
			}
		}
	}
	st.runs++
	if !graphsOff && !st.noGraph && st.runs >= 2 && graphsReady() == nil {
		if err := b.graph(st, recapture); err != nil {
			st.noGraph = true
		} else {
			return check(cuGraphLaunch(st.graph.exec, b.stream), "launch graph")
		}
	}
	for i := range st.calls {
		c := &st.calls[i]
		if err := b.issue(&c.call, c.ptrs, st.args); err != nil {
			return fmt.Errorf("cuda: %s: %w", c.k.Op, err)
		}
	}
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

// graph brings the batch's CUDA graph up to date with its calls: it
// updates the kernel nodes that changed, or captures the graph anew when
// there is none, recapture is set or an update fails. b.mu is held and
// the context is current.
func (b *Backend) graph(st *batchState, recapture bool) error {
	if st.graph != nil && !recapture {
		var params kernelNodeParams
		for i := range st.calls {
			c := &st.calls[i]
			if !c.dirty || c.node == 0 {
				continue
			}
			st.args = kernelArgs(st.args, c.ptrs, c.l, &b.errDev)
			params = kernelNodeParams{
				fn:    c.fn,
				gridX: uint32(c.l.Groups), gridY: 1, gridZ: 1,
				blockX: uint32(c.l.Threads), blockY: 1, blockZ: 1,
				params: &st.args[0],
			}
			r := cuGraphExecKernelNodeSetParams(st.graph.exec, c.node, &params)
			runtime.KeepAlive(c)
			if r != cudaSuccess {
				recapture = true
				break
			}
			c.dirty = false
		}
		if !recapture {
			return nil
		}
	}
	return b.capture(st)
}

// capture captures the batch's calls on the stream as a CUDA graph and
// instantiates it, in place of the graph it had. b.mu is held and the
// context is current.
func (b *Backend) capture(st *batchState) error {
	if st.graph == nil {
		st.graph = &graphHandles{}
		runtime.AddCleanup(st, b.dropGraph, st.graph)
	}
	st.graph.destroy()
	if r := cuStreamBeginCapture(b.stream, captureModeThreadLocal); r != cudaSuccess {
		return check(r, "begin capture")
	}
	var err error
	for i := range st.calls {
		c := &st.calls[i]
		if err = b.issue(&c.call, c.ptrs, st.args); err != nil {
			break
		}
		c.node, c.dirty = 0, false
		if c.l == nil {
			continue
		}
		// The node the launch added is what the stream's next work
		// would depend on.
		var status int32
		var id uint64
		var g, n uintptr
		var deps *uintptr
		if r := cuStreamGetCaptureInfo(b.stream, &status, &id, &g, &deps, &n); r != cudaSuccess || status != captureStatusActive || n != 1 || deps == nil {
			err = fmt.Errorf("cuda: capture: no node for a launch: %w", backend.ErrInternal)
			break
		}
		c.node = *deps
	}
	var graph uintptr
	r := cuStreamEndCapture(b.stream, &graph)
	if err == nil {
		err = check(r, "end capture")
	}
	if err == nil {
		st.graph.graph = graph
		err = check(cuGraphInstantiate(&st.graph.exec, graph, 0), "instantiate graph")
	} else if graph != 0 {
		cuGraphDestroy(graph)
	}
	if err != nil {
		st.graph.destroy()
		for i := range st.calls {
			st.calls[i].node = 0
		}
	}
	return err
}

// destroy destroys the graph, if there is one. The backend's mutex is
// held and its context current.
func (g *graphHandles) destroy() {
	if g.exec != 0 {
		cuGraphExecDestroy(g.exec)
	}
	if g.graph != 0 {
		cuGraphDestroy(g.graph)
	}
	g.graph, g.exec = 0, 0
}

// dropGraph destroys the graph of a batch that is gone, if it has one.
// Nothing else holds g then, so it reads g unlocked.
func (b *Backend) dropGraph(g *graphHandles) {
	if g.graph == 0 && g.exec == 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.enter()()
	g.destroy()
}
