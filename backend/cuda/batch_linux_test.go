//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"math"
	"slices"
	"testing"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// fakeDriver stands in for the CUDA driver and cuBLAS, counting what the
// backend asks of them, so that the batch's graph logic runs without a
// GPU.
type fakeDriver struct {
	counts
	capturing, failBegin bool
	lastNode             uintptr
	nodes                int
	updated              []kernelNodeParams // the parameters of each node update
}

// counts are the calls the fake driver saw.
type counts struct {
	launches, products              int // queued on the stream or captured
	captures, instantiated          int
	graphLaunches, updates          int
	destroyedExecs, destroyedGraphs int
}

// install replaces the driver's functions with f's until the test ends.
func (f *fakeDriver) install(t *testing.T) {
	t.Helper()
	if loadDriver() == nil {
		t.Skip("a CUDA driver is loaded: the GPU tests cover batches")
	}
	saved := []any{cuCtxSetCurrent, cuLaunchKernel, cublasSgemmStridedBatched, cuGetErrorName,
		cuStreamBeginCapture, cuStreamEndCapture, cuStreamGetCaptureInfo, cuGraphInstantiate,
		cuGraphExecKernelNodeSetParams, cuGraphLaunch, cuGraphExecDestroy, cuGraphDestroy, graphsReady, graphsOn}
	t.Cleanup(func() {
		cuCtxSetCurrent = saved[0].(func(uintptr) result)
		cuLaunchKernel = saved[1].(func(uintptr, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, *unsafe.Pointer, uintptr) result)
		cublasSgemmStridedBatched = saved[2].(func(uintptr, int32, int32, int32, int32, int32, *float32, uint64, int32, int64, uint64, int32, int64, *float32, uint64, int32, int64, int32) blasStatus)
		cuGetErrorName = saved[3].(func(result, **byte) result)
		cuStreamBeginCapture = saved[4].(func(uintptr, int32) result)
		cuStreamEndCapture = saved[5].(func(uintptr, *uintptr) result)
		cuStreamGetCaptureInfo = saved[6].(func(uintptr, *int32, *uint64, *uintptr, **uintptr, *uintptr) result)
		cuGraphInstantiate = saved[7].(func(*uintptr, uintptr, uint64) result)
		cuGraphExecKernelNodeSetParams = saved[8].(func(uintptr, uintptr, *kernelNodeParams) result)
		cuGraphLaunch = saved[9].(func(uintptr, uintptr) result)
		cuGraphExecDestroy = saved[10].(func(uintptr) result)
		cuGraphDestroy = saved[11].(func(uintptr) result)
		graphsReady = saved[12].(func() error)
		graphsOn = saved[13].(bool)
	})
	cuCtxSetCurrent = func(uintptr) result { return cudaSuccess }
	cuGetErrorName = func(result, **byte) result { return 1 }
	cuLaunchKernel = func(fn uintptr, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, params *unsafe.Pointer, extra uintptr) result {
		f.launches++
		if f.capturing {
			f.nodes++
			f.lastNode = uintptr(f.nodes)
		}
		return cudaSuccess
	}
	cublasSgemmStridedBatched = func(uintptr, int32, int32, int32, int32, int32, *float32, uint64, int32, int64, uint64, int32, int64, *float32, uint64, int32, int64, int32) blasStatus {
		f.products++
		if f.capturing {
			f.nodes++
			f.lastNode = uintptr(f.nodes)
		}
		return blasSuccess
	}
	cuStreamBeginCapture = func(uintptr, int32) result {
		if f.failBegin {
			return 900 // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
		}
		f.capturing = true
		f.captures++
		return cudaSuccess
	}
	cuStreamEndCapture = func(_ uintptr, g *uintptr) result {
		f.capturing = false
		*g = 100 + uintptr(f.captures)
		return cudaSuccess
	}
	cuStreamGetCaptureInfo = func(_ uintptr, status *int32, _ *uint64, _ *uintptr, deps **uintptr, n *uintptr) result {
		*status, *deps, *n = captureStatusActive, &f.lastNode, 1
		return cudaSuccess
	}
	cuGraphInstantiate = func(exec *uintptr, graph uintptr, _ uint64) result {
		f.instantiated++
		*exec = graph + 1000
		return cudaSuccess
	}
	cuGraphExecKernelNodeSetParams = func(_, node uintptr, p *kernelNodeParams) result {
		f.updates++
		f.updated = append(f.updated, *p)
		return cudaSuccess
	}
	cuGraphLaunch = func(uintptr, uintptr) result { f.graphLaunches++; return cudaSuccess }
	cuGraphExecDestroy = func(uintptr) result { f.destroyedExecs++; return cudaSuccess }
	cuGraphDestroy = func(uintptr) result { f.destroyedGraphs++; return cudaSuccess }
	graphsReady = func() error { return nil }
	graphsOn = true
}

// fakeBatch returns a backend over the fake driver and a batch of a fused
// kernel with a constant and a product that runs on cuBLAS.
func fakeBatch(t *testing.T) (*Backend, *backend.Batch) {
	t.Helper()
	b := &Backend{funcs: map[string]uintptr{}, stream: 7, blasMin: 0}
	buf := func(ptr uint64, n int) *buffer { return &buffer{ptr: ptr, dt: backend.Float32, n: n} }
	fused, product := batchKernels()
	bt := &backend.Batch{
		Kernels: []*backend.Kernel{fused, product},
		Outs:    [][]backend.Buffer{{buf(0x100, 8)}, {buf(0x300, 16)}},
		Ins:     [][]backend.Buffer{{buf(0x200, 8)}, {buf(0x400, 16), buf(0x500, 16)}},
	}
	// The kernels are compiled already: the fake driver cannot.
	l, err := gen.Plan(gen.CUDA, fused, bt.Outs[0], bt.Ins[0])
	if err != nil {
		t.Fatal(err)
	}
	b.funcs[l.Src] = 1
	// The graph goes before the fake driver does, rather than when the
	// collector finds the batch gone.
	t.Cleanup(func() {
		if st, ok := bt.State.(*batchState); ok && st.graph != nil {
			st.graph.destroy()
		}
	})
	return b, bt
}

// batchKernels returns the kernels of a test batch: a fused kernel that
// multiplies 8 numbers by a constant, 2, and a 4 by 4 product.
func batchKernels() (fused, product *backend.Kernel) {
	fused = &backend.Kernel{Op: backend.OpFused, DType: backend.Float32, Shape: []int{8},
		In: []backend.View{backend.Contiguous([]int{8})},
		Attrs: backend.Attrs{Program: []backend.Instr{
			{Op: backend.OpLoad, DType: backend.Float32, Args: []int{0}},
			{Op: backend.OpConst, DType: backend.Float32, Value: 2},
			{Op: backend.OpMul, DType: backend.Float32, Args: []int{0, 1}},
		}}}
	product = &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: []int{4, 4},
		In: []backend.View{backend.Contiguous([]int{4, 4}), backend.Contiguous([]int{4, 4})}}
	return fused, product
}

func TestBatchesLaunchAsCUDAGraphs(t *testing.T) {
	f := &fakeDriver{}
	f.install(t)
	b, bt := fakeBatch(t)
	run := func(what string) {
		t.Helper()
		if err := b.RunBatch(t.Context(), bt); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	want := func(what string, w counts) {
		t.Helper()
		if f.counts != w {
			t.Errorf("%s: driver saw %+v, want %+v", what, f.counts, w)
		}
	}

	run("first run")
	want("the first run launches its kernels one by one", counts{launches: 1, products: 1})
	run("second run")
	want("the second run captures a graph and launches it", counts{launches: 2, products: 2, captures: 1, instantiated: 1, graphLaunches: 1})
	run("third run")
	want("a run like the last launches the graph alone", counts{launches: 2, products: 2, captures: 1, instantiated: 1, graphLaunches: 2})

	// A new constant, as an optimizer's step changes, updates its node.
	bt.Kernels[0].Attrs.Program[1].Value = 3
	run("new constant")
	want("a new constant updates the kernel's node", counts{launches: 2, products: 2, captures: 1, instantiated: 1, graphLaunches: 3, updates: 1})
	// The node's parameters, after the buffers, hold the new constant.
	l := bt.State.(*batchState).calls[0].l
	if params := unsafe.Slice(f.updated[0].params, 4); params[2] != unsafe.Pointer(&l.Params[0]) || !slices.Contains(l.Params, int32(math.Float32bits(3))) {
		t.Error("updated node does not take the new constant")
	}

	// A new buffer for the generated kernel updates its node too.
	bt.Outs[0][0] = &buffer{ptr: 0x600, dt: backend.Float32, n: 8}
	run("new output")
	want("a new buffer updates the kernel's node", counts{launches: 2, products: 2, captures: 1, instantiated: 1, graphLaunches: 4, updates: 2})
	if ptr := *(*uint64)(unsafe.Slice(f.updated[1].params, 4)[0]); ptr != 0x600 {
		t.Errorf("updated node writes %#x, want %#x", ptr, 0x600)
	}

	// A product's new operand needs a new capture, which waits for the
	// batch's runs to repeat twice.
	bt.Ins[1][0] = &buffer{ptr: 0x700, dt: backend.Float32, n: 16}
	run("new operand")
	want("a product's new operand drops the graph", counts{launches: 3, products: 3, captures: 1, instantiated: 1, graphLaunches: 4, updates: 2, destroyedExecs: 1, destroyedGraphs: 1})
	run("first repeat")
	want("a run that repeats once launches one by one", counts{launches: 4, products: 4, captures: 1, instantiated: 1, graphLaunches: 4, updates: 2, destroyedExecs: 1, destroyedGraphs: 1})
	run("second repeat")
	want("a run that repeats twice captures the graph again", counts{launches: 5, products: 5, captures: 2, instantiated: 2, graphLaunches: 5, updates: 2, destroyedExecs: 1, destroyedGraphs: 1})
}

// TestBatchesWhoseProductsMoveLaunchOneByOne checks that a batch whose
// products take other buffers from run to run does not capture its graph
// again and again, which costs more than its launches.
func TestBatchesWhoseProductsMoveLaunchOneByOne(t *testing.T) {
	for _, c := range []struct {
		name     string
		every    int // runs between moves of a product
		captures int
	}{
		{"every run", 1, 0},
		{"every other run", 2, 1}, // the second run repeats the first
		{"every eighth run", 8, 3},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := &fakeDriver{}
			f.install(t)
			b, bt := fakeBatch(t)
			const runs = 64
			for i := range runs {
				if i > 0 && i%c.every == 0 {
					bt.Ins[1][0] = &buffer{ptr: 0x1000 + uint64(i), dt: backend.Float32, n: 16}
				}
				if err := b.RunBatch(t.Context(), bt); err != nil {
					t.Fatal(err)
				}
			}
			// A capture issues the calls too, without running them.
			if f.captures != c.captures || f.products-f.captures+f.graphLaunches != runs {
				t.Errorf("driver saw %+v, want %d runs and %d captures", f.counts, runs, c.captures)
			}
		})
	}
}

func TestBatchesFallBackWhenCaptureFails(t *testing.T) {
	f := &fakeDriver{failBegin: true}
	f.install(t)
	b, bt := fakeBatch(t)
	for range 3 {
		if err := b.RunBatch(t.Context(), bt); err != nil {
			t.Fatal(err)
		}
	}
	if f.launches != 3 || f.products != 3 || f.graphLaunches != 0 {
		t.Errorf("driver saw %+v, want three runs launched one by one", f.counts)
	}
	if st := bt.State.(*batchState); !st.noGraph {
		t.Error("a batch whose capture failed tries graphs again")
	}
}

func TestBatchesWithoutGraphsLaunchOneByOne(t *testing.T) {
	f := &fakeDriver{}
	f.install(t)
	graphsOn = false
	b, bt := fakeBatch(t)
	for range 3 {
		if err := b.RunBatch(t.Context(), bt); err != nil {
			t.Fatal(err)
		}
	}
	if f.launches != 3 || f.captures != 0 {
		t.Errorf("driver saw %+v, want three runs launched one by one", f.counts)
	}
	// A run plans each kernel once and keeps it while its arguments stay.
	st := bt.State.(*batchState)
	if !slices.Equal(st.calls[0].args, bt.Kernels[0].AppendArgs(nil)) {
		t.Error("the call keeps arguments other than the kernel's")
	}
}

// TestKernelNodeParamsMatchCUDA checks the layout of kernelNodeParams
// against the CUDA_KERNEL_NODE_PARAMS_v2 of cuda.h.
func TestKernelNodeParamsMatchCUDA(t *testing.T) {
	var p kernelNodeParams
	if size, at := unsafe.Sizeof(p), unsafe.Offsetof(p.params); size != 72 || at != 40 {
		t.Errorf("kernelNodeParams is %d bytes with parameters at %d, want 72 and 40", size, at)
	}
}
