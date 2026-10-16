//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

var (
	gpu     *Backend
	gpuErr  error
	gpuOnce sync.Once
	gpuMu   sync.Mutex
)

// gpuTarget runs kernels on the CUDA device, running products of at least
// blasMin multiply-adds on cuBLAS, or none if it is -1.
type gpuTarget struct {
	b       *Backend
	blasMin int
}

// newGPU returns the CUDA device as a target, or nil if there is none.
func newGPU(t *testing.T, blasMin int) kerneltest.Target {
	t.Helper()
	gpuOnce.Do(func() { gpu, gpuErr = New() })
	if gpuErr != nil {
		if !errors.Is(gpuErr, ErrUnavailable) {
			t.Fatal(gpuErr)
		}
		t.Logf("no CUDA device: %v", gpuErr)
		return nil
	}
	return gpuTarget{gpu, blasMin}
}

func (g gpuTarget) Run(t *testing.T, k *backend.Kernel, outs, ins []backend.Buffer) ([][]byte, bool) {
	t.Helper()
	// The GPU is shared by parallel tests, and a Sync reports the errors
	// of every kernel run since the last one, so tests take turns.
	gpuMu.Lock()
	defer gpuMu.Unlock()
	g.b.mu.Lock()
	defer func(blasMin int) {
		g.b.mu.Lock()
		g.b.blasMin = blasMin
		g.b.mu.Unlock()
	}(g.b.blasMin)
	g.b.blasMin = g.blasMin
	g.b.mu.Unlock()
	copyTo := func(src backend.Buffer) backend.Buffer {
		b, err := g.b.Alloc(src.DType(), src.Len())
		if err != nil {
			t.Fatal(err)
		}
		if err := g.b.Upload(t.Context(), b, kerneltest.Contents(t, src)); err != nil {
			t.Fatal(err)
		}
		return b
	}
	var mo, mi []backend.Buffer
	for _, b := range outs {
		mo = append(mo, copyTo(b))
	}
	for _, b := range ins {
		mi = append(mi, copyTo(b))
	}
	if err := g.b.Run(t.Context(), k, mo, mi); err != nil {
		t.Fatal(err)
	}
	failed := false
	if err := g.b.Sync(t.Context()); err != nil {
		if !errors.Is(err, backend.ErrIndex) && !errors.Is(err, backend.ErrDivideByZero) {
			t.Fatal(err)
		}
		failed = true
	}
	res := make([][]byte, len(mo))
	for i, b := range mo {
		res[i] = make([]byte, b.Len()*b.DType().Size())
		if err := g.b.Download(t.Context(), res[i], b); err != nil {
			t.Fatal(err)
		}
	}
	for _, b := range mo {
		g.b.Free(b)
	}
	for _, b := range mi {
		g.b.Free(b)
	}
	return res, failed
}

// TestKernelsCompileWithNVRTC compiles the kernel of every case with
// NVRTC, which needs no device, so the generated CUDA C++ is checked by
// the real compiler wherever libnvrtc is installed.
func TestKernelsCompileWithNVRTC(t *testing.T) {
	if err := loadNVRTC(); err != nil {
		t.Skipf("no NVRTC: %v", err)
	}
	t.Log(compilerVersion())
	arch, cubin, err := architecture(75)
	if err != nil {
		t.Fatal(err)
	}
	if !cubin {
		t.Fatalf("NVRTC does not compile for sm_75: %s", arch)
	}
	options := []string{arch, "--fmad=false"}
	sources := map[string]bool{}
	var cases []kerneltest.Case
	for _, f := range []func(*testing.T) []kerneltest.Case{
		kerneltest.Elementwise, kerneltest.Random, kerneltest.Reduce,
		kerneltest.Movement, kerneltest.Window, kerneltest.MatMul,
		kerneltest.Scan,
		kerneltest.Quant,
		kerneltest.Attention,
		kerneltest.Conv,
	} {
		cases = append(cases, f(t)...)
	}
	for _, c := range cases {
		n, _ := backend.NumElements(c.K.Shape)
		outs := make([]backend.Buffer, 1+len(c.K.Attrs.Stores))
		for i := range outs {
			outs[i] = kerneltest.HostBuffer(t, nil, c.K.DType, n, 0, 0)
		}
		var launches []*gen.Launch
		l, err := gen.Plan(gen.CUDA, &c.K, outs, c.Ins)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		launches = append(launches, l)
		if c.K.Op == backend.OpMatMul && c.K.Attrs.Program != nil {
			if l, err = gen.PlanEpilogue(gen.CUDA, &c.K, outs, c.Ins); err != nil {
				t.Fatalf("%s: %v", c.Name, err)
			}
			launches = append(launches, l)
		}
		for _, l := range launches {
			if l == nil || sources[l.Src] {
				continue
			}
			sources[l.Src] = true
			if _, err := compileNVRTC(l.Src, options, cubin); err != nil {
				t.Errorf("%s: %v", c.Name, err)
			}
		}
	}
	t.Logf("compiled %d kernels", len(sources))
}

// TestDiskCacheRoundTrips checks that a compiled kernel stored in the
// cache comes back as it was.
func TestDiskCacheRoundTrips(t *testing.T) {
	t.Setenv("MIMIR_CACHE_DIR", t.TempDir())
	c := openDiskCache()
	if c == nil {
		t.Fatal("no cache")
	}
	key := c.key("nvrtc 12.9", []string{"--gpu-architecture=sm_75"}, "kernel")
	if _, ok := c.get(key); ok {
		t.Fatal("entry before put")
	}
	c.put(key, []byte("image"))
	if got, ok := c.get(key); !ok || string(got) != "image" {
		t.Fatalf("got %q, %t", got, ok)
	}
	if other := c.key("nvrtc 12.9", []string{"--gpu-architecture=sm_86"}, "kernel"); other == key {
		t.Error("the key ignores the options")
	}
}

// BenchmarkMatMul times square float32 products on the CPU, with the
// generated kernel and on cuBLAS, each waiting for the result.
func BenchmarkMatMul(b *testing.B) {
	be, err := New()
	if err != nil {
		b.Skip(err)
	}
	for _, n := range []int{128, 512, 1024, 2048, 4096} {
		k := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: []int{n, n},
			In: []backend.View{backend.Contiguous([]int{n, n}), backend.Contiguous([]int{n, n})}}
		for _, c := range []struct {
			name    string
			be      backend.Backend
			blasMin int
		}{
			{"cpu", kerneltest.Host, -1},
			{"kernel", be, -1},
			{"cublas", be, 0},
		} {
			if c.name == "cpu" && n > 2048 || c.name == "kernel" && n > 2048 {
				continue
			}
			b.Run(fmt.Sprintf("%d/%s", n, c.name), func(b *testing.B) {
				be.mu.Lock()
				be.blasMin = c.blasMin
				be.mu.Unlock()
				bufs := make([]backend.Buffer, 3)
				for i := range bufs {
					if bufs[i], err = c.be.Alloc(backend.Float32, n*n); err != nil {
						b.Fatal(err)
					}
				}
				defer func() {
					for _, buf := range bufs {
						c.be.Free(buf)
					}
				}()
				for b.Loop() {
					if err := c.be.Run(b.Context(), k, bufs[:1], bufs[1:]); err != nil {
						b.Fatal(err)
					}
					if err := c.be.Sync(b.Context()); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(2*float64(n)*float64(n)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}

// TestDeviceReportsItself logs the device the tests ran on.
func TestDeviceReportsItself(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Skip(err)
	}
	t.Logf("%s: compute capability %d.%d, %s, %s", be.DeviceName(), be.cc/10, be.cc%10, be.arch, compilerVersion())
	if !strings.Contains(be.arch, "--gpu-architecture=") {
		t.Errorf("architecture option %q", be.arch)
	}
}

// TestBatchesMatchTheirKernelsOnTheGPU runs a batch again and again, as a
// CUDA graph from its second run on, with a new constant or buffers
// between runs, and checks each run's results against the host's.
func TestBatchesMatchTheirKernelsOnTheGPU(t *testing.T) {
	if newGPU(t, 0) == nil {
		t.Skip("no CUDA device")
	}
	gpuMu.Lock()
	defer gpuMu.Unlock()
	b := gpu
	b.mu.Lock()
	defer func(blasMin int) {
		b.mu.Lock()
		b.blasMin = blasMin
		b.mu.Unlock()
	}(b.blasMin)
	b.blasMin = 0 // the product runs on cuBLAS
	b.mu.Unlock()

	upload := func(vs []float32) backend.Buffer {
		t.Helper()
		buf, err := b.Alloc(backend.Float32, len(vs))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { b.Free(buf) })
		var data []byte
		for _, v := range vs {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		if err := b.Upload(t.Context(), buf, data); err != nil {
			t.Fatal(err)
		}
		return buf
	}
	download := func(buf backend.Buffer) []float32 {
		t.Helper()
		data := make([]byte, 4*buf.Len())
		if err := b.Download(t.Context(), data, buf); err != nil {
			t.Fatal(err)
		}
		vs := make([]float32, buf.Len())
		for i := range vs {
			vs[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
		return vs
	}

	xs := []float32{1, -2, 3, -4, 5, -6, 7, -8}
	as, bs := make([]float32, 16), make([]float32, 16)
	for i := range as {
		as[i], bs[i] = float32(i)/4, float32(15-i)/8
	}
	fused, product := batchKernels()
	bt := &backend.Batch{
		Kernels: []*backend.Kernel{fused, product},
		Outs:    [][]backend.Buffer{{upload(make([]float32, 8))}, {upload(make([]float32, 16))}},
		Ins:     [][]backend.Buffer{{upload(xs)}, {upload(as), upload(bs)}},
	}
	defer func() {
		if st, ok := bt.State.(*batchState); ok && st.graph != nil {
			b.mu.Lock()
			defer b.mu.Unlock()
			defer b.enter()()
			st.graph.destroy()
		}
	}()
	for run, c := range []float64{2, 2, 3, 3, 5, 5} {
		fused.Attrs.Program[1].Value = c
		switch run {
		case 3: // a new output for the generated kernel
			bt.Outs[0][0] = upload(make([]float32, 8))
		case 5: // a new operand for cuBLAS
			bt.Ins[1][0] = upload(as)
		}
		if err := b.RunBatch(t.Context(), bt); err != nil {
			t.Fatal(err)
		}
		if err := b.Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
		for i, v := range download(bt.Outs[0][0]) {
			if want := xs[i] * float32(c); v != want {
				t.Fatalf("run %d: element %d is %v, want %v", run, i, v, want)
			}
		}
		got := download(bt.Outs[1][0])
		for i := range 4 {
			for j := range 4 {
				var want float32
				for k := range 4 {
					want += as[i*4+k] * bs[k*4+j]
				}
				if d := got[i*4+j] - want; d > 1e-4 || d < -1e-4 {
					t.Fatalf("run %d: product element %d,%d is %v, want %v", run, i, j, got[i*4+j], want)
				}
			}
		}
	}
	if st := bt.State.(*batchState); !graphsOff && (st.noGraph || st.graph == nil || st.graph.exec == 0) {
		t.Error("the batch did not run as a CUDA graph")
	}
}
