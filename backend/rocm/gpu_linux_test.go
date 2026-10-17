//go:build linux && (amd64 || arm64) && !norocm

package rocm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/ebitengine/purego"
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

// gpuTarget runs kernels on the ROCm device, running products of at least
// blasMin multiply-adds on hipBLAS, or none if it is -1.
type gpuTarget struct {
	b       *Backend
	blasMin int
}

// newGPU returns the ROCm device as a target, or nil if there is none,
// or if blasMin asks for hipBLAS and there is no hipBLAS.
func newGPU(t *testing.T, blasMin int) kerneltest.Target {
	t.Helper()
	gpuOnce.Do(func() { gpu, gpuErr = New() })
	if gpuErr != nil {
		if !errors.Is(gpuErr, ErrUnavailable) {
			t.Fatal(gpuErr)
		}
		t.Logf("no ROCm device: %v", gpuErr)
		return nil
	}
	if blasMin >= 0 && gpu.blas == 0 {
		t.Logf("no hipBLAS: %v", blasErr)
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

// hiprtcArchs are the architectures TestKernelsCompileWithHIPRTC compiles
// for, unless MIMIR_HIPRTC_ARCHS lists others: CDNA 2 (MI200), CDNA 3
// (MI300) and its first ISA, which hiprtc versions before gfx942 know,
// with wavefronts of 64 threads, and RDNA 3 (Radeon RX 7900), with 32.
var hiprtcArchs = []string{"gfx90a", "gfx940", "gfx942", "gfx1100"}

// archWave returns the wavefront size the backend uses on architecture
// arch: 32 on RDNA, gfx10 and later, and 64 on GCN and CDNA.
func archWave(arch string) int {
	if strings.HasPrefix(arch, "gfx9") || strings.HasPrefix(arch, "gfx8") {
		return 64
	}
	return 32
}

// TestKernelsCompileWithHIPRTC compiles the kernel of every case with
// hiprtc, which needs no device, for the architectures of hiprtcArchs, in
// the dialect of their wavefront size, so the generated HIP C++ is
// checked by the real compiler wherever hiprtc is installed or
// MIMIR_HIPRTC names it. An architecture this hiprtc does not know is
// skipped, but each wavefront size must compile for one.
func TestKernelsCompileWithHIPRTC(t *testing.T) {
	if err := loadRTC(); err != nil {
		t.Skipf("no hiprtc: %v", err)
	}
	t.Log(compilerVersion())
	archs := hiprtcArchs
	if s := os.Getenv("MIMIR_HIPRTC_ARCHS"); s != "" {
		archs = strings.Split(s, ",")
	}
	known := isas(t)
	var cases []kerneltest.Case
	for _, f := range []func(*testing.T) []kerneltest.Case{
		kerneltest.Elementwise, kerneltest.Random, kerneltest.Reduce,
		kerneltest.Movement, kerneltest.Window, kerneltest.MatMul,
		kerneltest.Scan,
		kerneltest.Quant,
		kerneltest.Attention,
		kerneltest.Conv,
		kerneltest.Sort,
		kerneltest.FFT,
		kerneltest.Segment,
	} {
		cases = append(cases, f(t)...)
	}
	compiled := map[int]bool{}
	for _, arch := range archs {
		if known != nil && !known(arch) {
			t.Logf("%s: hiprtc does not know it", arch)
			continue
		}
		d := gen.HIP(archWave(arch))
		sources := map[string]string{} // case names by source
		for _, c := range cases {
			n, _ := backend.NumElements(c.K.Shape)
			outs := make([]backend.Buffer, 1+len(c.K.Attrs.Stores))
			for i := range outs {
				outs[i] = kerneltest.HostBuffer(t, nil, c.K.DType, n, 0, 0)
			}
			var launches []*gen.Launch
			l, err := gen.Plan(d, &c.K, outs, c.Ins)
			if err != nil {
				t.Fatalf("%s: %v", c.Name, err)
			}
			launches = append(launches, l)
			if c.K.Op == backend.OpMatMul && c.K.Attrs.Program != nil {
				if l, err = gen.PlanEpilogue(d, &c.K, outs, c.Ins); err != nil {
					t.Fatalf("%s: %v", c.Name, err)
				}
				launches = append(launches, l)
			}
			for _, l := range launches {
				if l != nil && sources[l.Src] == "" {
					sources[l.Src] = c.Name
				}
			}
		}
		options := compileOptions(arch)
		failed := 0
		for src, name := range sources {
			if _, err := compileHIPRTC(src, options); err != nil {
				if failed++; failed <= 5 {
					t.Errorf("%s %s: %v", arch, name, err)
				}
			}
		}
		if failed == 0 {
			compiled[d.Wave()] = true
		}
		t.Logf("%s: compiled %d %s kernels, %d failed", arch, len(sources), d.Name, failed)
	}
	for _, wave := range []int{64, 32} {
		if !compiled[wave] {
			t.Errorf("no architecture with wavefronts of %d compiled", wave)
		}
	}
}

// isas returns whether hiprtc's code object manager knows an
// architecture, or nil if the manager cannot be asked. hiprtc fails, or
// crashes, compiling for an architecture newer than it is.
func isas(t *testing.T) func(arch string) bool {
	var lib uintptr
	for _, name := range []string{"libamd_comgr.so.3", "libamd_comgr.so.2", "libamd_comgr.so"} {
		var err error
		if lib, err = purego.Dlopen(name, purego.RTLD_GLOBAL|purego.RTLD_NOW); err == nil {
			break
		}
	}
	if lib == 0 {
		t.Log("no code object manager to list architectures")
		return nil
	}
	var count func(n *uintptr) uint32
	var name func(i uintptr, s **byte) uint32
	if register(lib, []symbol{{&count, "amd_comgr_get_isa_count"}, {&name, "amd_comgr_get_isa_name"}}) != nil {
		return nil
	}
	var n uintptr
	if count(&n) != 0 {
		return nil
	}
	names := map[string]bool{}
	for i := range n {
		var s *byte
		if name(i, &s) == 0 && s != nil {
			// Such as amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-.
			_, isa, _ := strings.Cut(goString(s), "--")
			isa, _, _ = strings.Cut(isa, ":")
			names[isa] = true
		}
	}
	return func(arch string) bool { return names[arch] }
}

// BenchmarkMatMul times square float32 products on the CPU, with the
// generated kernel and on hipBLAS, each waiting for the result.
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
			{"hipblas", be, 0},
		} {
			if c.name == "cpu" && n > 2048 || c.name == "kernel" && n > 2048 || c.name == "hipblas" && be.blas == 0 {
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
				product := func() {
					if err := c.be.Run(b.Context(), k, bufs[:1], bufs[1:]); err != nil {
						b.Fatal(err)
					}
					if err := c.be.Sync(b.Context()); err != nil {
						b.Fatal(err)
					}
				}
				product() // compiles the kernel, or lets hipBLAS load its own
				for b.Loop() {
					product()
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
	t.Logf("%s: %s, wavefronts of %d, %d compute units, hipBLAS %t, %s", be.DeviceName(), be.arch, be.dialect.Wave(), be.sms, be.blas != 0, compilerVersion())
	if !strings.HasPrefix(be.arch, "gfx") {
		t.Errorf("architecture %q", be.arch)
	}
}

// TestBatchesMatchTheirKernelsOnTheGPU runs a batch again and again, with
// graphs on, as a HIP graph from its second run on, with a new constant
// or buffers between runs, and checks each run's results against the
// host's and that the runs it should launch as a graph did.
func TestBatchesMatchTheirKernelsOnTheGPU(t *testing.T) {
	if newGPU(t, 0) == nil {
		t.Skip("no ROCm device with hipBLAS")
	}
	if err := graphsReady(); err != nil {
		t.Skipf("no HIP graphs: %v", err)
	}
	gpuMu.Lock()
	defer gpuMu.Unlock()
	defer func(on bool) { graphsOn = on }(graphsOn)
	graphsOn = true
	b := gpu
	b.mu.Lock()
	defer func(blasMin int) {
		b.mu.Lock()
		b.blasMin = blasMin
		b.mu.Unlock()
	}(b.blasMin)
	b.blasMin = 0 // the product runs on hipBLAS
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
	launches := graphCounts.launches.Load()
	for run, c := range []float64{2, 2, 3, 3, 5, 5, 7, 7} {
		fused.Attrs.Program[1].Value = c
		switch run {
		case 3: // a new output for the generated kernel
			bt.Outs[0][0] = upload(make([]float32, 8))
		case 5: // a new operand for hipBLAS: one by one until runs repeat twice
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
	if st := bt.State.(*batchState); st.noGraph || !st.graph.live() {
		t.Error("the batch did not run as a HIP graph")
	}
	// Runs 1 to 4 and 7.
	if n := graphCounts.launches.Load() - launches; n != 5 {
		t.Errorf("%d runs launched as a graph, want 5", n)
	}
}
