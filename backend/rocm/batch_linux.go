//go:build linux && (amd64 || arm64) && !norocm

package rocm

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"slices"
	"sync/atomic"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// A batch runs as calls the backend plans once and keeps in the batch's
// state. A replayed schedule generates no kernel source and looks up no
// compiled kernel: when only a kernel's arguments changed, such as an
// optimizer's constants from one step to the next, the backend writes
// them into the parameters of the launch it planned (see
// gen.Launch.SetArgs), and it plans a kernel again only when its buffers
// change shape.
//
// With MIMIR_ROCM_GRAPHS=1, a batch whose runs repeat launches as one
// HIP graph: the backend captures the calls on its stream once,
// instantiates the graph, and from then on launches it as a whole, after
// updating the kernel nodes whose parameters or buffers changed in the
// executable graph. A hipBLAS call cannot be updated so: hipBLAS chooses
// its own kernels and bakes the operands into them, so a product over
// other buffers needs the graph captured anew. Capturing, instantiating
// and uploading a graph costs much more than launching its calls one by
// one, so a batch whose products move to other buffers from run to run
// must not capture it every time. A batch therefore captures a graph only
// once a run has repeated the last one's products, and each time the
// products move under a graph it waits twice as many repeated runs
// before capturing again; meanwhile it launches its calls one by one.
// The scheduler gives a replayed schedule the buffers of its last run
// whenever it can, so the products of a training step keep theirs and
// its graph is captured once. When the runtime lacks the graph API, or
// capture, instantiation or an update fail, the batch's calls launch one
// by one, or are captured anew.
//
// Graphs are off by default, as on CUDA, until an AMD GPU shows them
// faster than launching the calls one by one.

// graphsOn turns HIP graphs on.
var graphsOn = os.Getenv("MIMIR_ROCM_GRAPHS") == "1"

// graphsReady loads the HIP graph API, or reports why it cannot.
var graphsReady = loadGraphs

// maxBackoff bounds how long a batch whose products moved waits before it
// captures a graph again: 1<<maxBackoff repeated runs.
const maxBackoff = 6

var _ backend.Batcher = (*Backend)(nil)

// batchState is what the backend keeps of a batch between runs.
type batchState struct {
	calls   []batchCall
	runs    int
	graph   *graphHandles // the captured graph, or nil
	noGraph bool          // capture failed: launch the calls one by one
	// repeats counts the runs in a row that repeated the last one's
	// products, and moved the times the products moved under a graph.
	repeats, moved int
	args           []unsafe.Pointer
	scratch        []uint64
}

// batchCall is one kernel of a batch and the call that runs it.
type batchCall struct {
	call
	k     *backend.Kernel
	args  []uint64     // k's arguments in the last run
	bufs  []bufferKind // its buffers when it was planned, outputs first
	ptrs  []uint64     // its buffers' device pointers in this run
	node  uintptr      // the graph's node of its kernel launch, or 0
	dirty bool         // the launch or pointers changed since the graph took them
}

type bufferKind struct {
	dt backend.DType
	n  int
}

// graphHandles are an instantiated HIP graph and the graph it came from.
type graphHandles struct{ graph, exec uintptr }

// live reports whether g holds an executable graph.
func (g *graphHandles) live() bool { return g != nil && g.exec != 0 }

// graphCounts counts what batches did with HIP graphs, for tests.
var graphCounts struct {
	captures, launches, direct atomic.Int64
}

// RunBatch queues the kernels of bt on the stream, as one HIP graph when
// graphs are on and its runs repeat.
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
	moved := false // a product, or the set of nodes, differs from the last run's
	for i, k := range bt.Kernels {
		c, outs, ins := &st.calls[i], bt.Outs[i], bt.Ins[i]
		st.scratch = k.AppendArgs(st.scratch[:0])
		switch {
		case c.k != k || !c.planned(outs, ins):
			if err := backend.CheckRun(k, outs, ins); err != nil {
				return fmt.Errorf("rocm: %w", err)
			}
			old := c.call
			if err := b.plan(&c.call, k, outs, ins); err != nil {
				return err
			}
			c.k, c.dirty = k, true
			c.bufs = c.bufs[:0]
			for _, buf := range slices.Concat(outs, ins) {
				c.bufs = append(c.bufs, bufferKind{buf.DType(), buf.Len()})
			}
			c.ptrs = slices.Grow(c.ptrs[:0], len(c.bufs))[:len(c.bufs)]
			st.args = slices.Grow(st.args, len(c.bufs)+2)
			// A kernel node takes any launch, but a different set of
			// nodes, or another hipBLAS call, needs a new graph.
			moved = moved || (old.l == nil) != (c.l == nil) || (old.gemm == nil) != (c.gemm == nil) ||
				old.gemm != nil && *old.gemm != *c.gemm
		case !slices.Equal(c.args, st.scratch):
			// Kernels that differ in their arguments alone share their
			// source: the launch takes the new ones as it is.
			if c.l != nil {
				c.l.SetArgs(st.scratch)
			}
			c.dirty = true
		}
		c.args = append(c.args[:0], st.scratch...)
		j := 0
		for _, bufs := range [2][]backend.Buffer{outs, ins} {
			for _, buf := range bufs {
				cb, err := b.own(buf)
				if err != nil {
					return err
				}
				if c.ptrs[j] != cb.ptr {
					c.ptrs[j], c.dirty = cb.ptr, true
					moved = moved || c.gemm != nil
				}
				j++
			}
		}
	}
	st.runs++
	if moved {
		st.repeats = 0
	} else {
		st.repeats++
	}
	if graphsOn && !st.noGraph && graphsReady() == nil {
		if st.graph.live() && moved {
			st.graph.destroy()
			st.moved++
		}
		if st.graph.live() || st.repeats >= 1<<min(st.moved, maxBackoff) {
			if err := b.graph(st); err != nil {
				st.noGraph = true
			} else {
				graphCounts.launches.Add(1)
				return check(hipGraphLaunch(st.graph.exec, b.stream), "launch graph")
			}
		}
	}
	graphCounts.direct.Add(1)
	for i := range st.calls {
		c := &st.calls[i]
		if err := b.issue(&c.call, c.ptrs, st.args); err != nil {
			return fmt.Errorf("rocm: %s: %w", c.k.Op, err)
		}
	}
	return nil
}

// planned reports whether the call was planned for buffers like outs and
// ins.
func (c *batchCall) planned(outs, ins []backend.Buffer) bool {
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
	return true
}

// graph brings the batch's HIP graph up to date with its calls: it
// updates the kernel nodes that changed in the live graph, or captures
// the graph anew when there is none or an update fails. b.mu is held and
// the device is current.
func (b *Backend) graph(st *batchState) error {
	if st.graph.live() {
		var params kernelNodeParams
		updated := true
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
			r := hipGraphExecKernelNodeSetParams(st.graph.exec, c.node, &params)
			runtime.KeepAlive(c)
			if r != hipSuccess {
				updated = false
				break
			}
			c.dirty = false
		}
		if updated {
			return nil
		}
	}
	return b.capture(st)
}

// capture captures the batch's calls on the stream as a HIP graph and
// instantiates it, in place of the graph it had. b.mu is held and the
// device is current.
func (b *Backend) capture(st *batchState) error {
	if st.graph == nil {
		st.graph = &graphHandles{}
		runtime.AddCleanup(st, b.dropGraph, st.graph)
	}
	st.graph.destroy()
	if r := hipStreamBeginCapture(b.stream, captureModeThreadLocal); r != hipSuccess {
		return check(r, "begin capture")
	}
	graphCounts.captures.Add(1)
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
		if r := hipStreamGetCaptureInfo(b.stream, &status, &id, &g, &deps, &n); r != hipSuccess || status != captureStatusActive || n != 1 || deps == nil {
			err = fmt.Errorf("rocm: capture: no node for a launch: %w", backend.ErrInternal)
			break
		}
		c.node = *deps
	}
	var graph uintptr
	r := hipStreamEndCapture(b.stream, &graph)
	if err == nil {
		err = check(r, "end capture")
	}
	if err == nil {
		st.graph.graph = graph
		err = check(hipGraphInstantiate(&st.graph.exec, graph, 0), "instantiate graph")
	} else if graph != 0 {
		hipGraphDestroy(graph)
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
		hipGraphExecDestroy(g.exec)
	}
	if g.graph != 0 {
		hipGraphDestroy(g.graph)
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
