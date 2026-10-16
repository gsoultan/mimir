//go:build darwin && arm64

package metal

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

var (
	gpu     *Backend
	gpuErr  error
	gpuOnce sync.Once
)

// gpuTarget runs kernels on the Metal device, running products of at
// least tensorMin multiply-adds on Metal 4 tensor operations and of at
// least mpsMin through Metal Performance Shaders, or none if either is -1.
type gpuTarget struct {
	b                 *Backend
	mpsMin, tensorMin int
}

// newGPU returns the Metal device as a target, or nil if there is none or
// it cannot run the products asked of it.
func newGPU(t *testing.T, mpsMin, tensorMin int) target {
	t.Helper()
	gpuOnce.Do(func() { gpu, gpuErr = New() })
	if gpuErr != nil {
		if !errors.Is(gpuErr, ErrUnavailable) {
			t.Fatal(gpuErr)
		}
		t.Logf("no Metal device: %v", gpuErr)
		return nil
	}
	if mpsMin >= 0 && !gpu.hasMPS {
		t.Log("no Metal Performance Shaders on this device")
		return nil
	}
	if tensorMin >= 0 && !gpu.hasTensor {
		t.Log("no hardware Metal 4 tensor operations on this device")
		return nil
	}
	return gpuTarget{gpu, mpsMin, tensorMin}
}

func (g gpuTarget) run(t *testing.T, k *backend.Kernel, outs, ins []backend.Buffer) ([][]byte, bool) {
	t.Helper()
	// The GPU is shared by parallel tests, and a Sync reports the errors
	// of every kernel run since the last one, so tests take turns.
	gpuMu.Lock()
	defer gpuMu.Unlock()
	g.b.mu.Lock()
	defer func(mps, tensor int) {
		g.b.mu.Lock()
		g.b.mpsMin, g.b.tensorMin = mps, tensor
		g.b.mu.Unlock()
	}(g.b.mpsMin, g.b.tensorMin)
	g.b.mpsMin, g.b.tensorMin = g.mpsMin, g.tensorMin
	g.b.mu.Unlock()
	copyTo := func(src backend.Buffer) backend.Buffer {
		b, err := g.b.Alloc(src.DType(), src.Len())
		if err != nil {
			t.Fatal(err)
		}
		if err := g.b.Upload(t.Context(), b, contents(t, src)); err != nil {
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
		res[i] = make([]byte, 4*b.Len())
		if err := g.b.Download(t.Context(), res[i], b); err != nil {
			t.Fatal(err)
		}
	}
	return res, failed
}

var gpuMu sync.Mutex

// BenchmarkMatMul times square float32 products on the CPU, with the
// generated kernel, with Metal Performance Shaders and with Metal 4
// tensor operations, each waiting for the result.
func BenchmarkMatMul(b *testing.B) {
	be, err := New()
	if err != nil {
		b.Skip(err)
	}
	for _, n := range []int{128, 512, 1000, 1024, 2048, 4096} {
		k := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: []int{n, n},
			In: []backend.View{backend.Contiguous([]int{n, n}), backend.Contiguous([]int{n, n})}}
		for _, c := range []struct {
			name              string
			be                backend.Backend
			mpsMin, tensorMin int
		}{
			{"cpu", host, -1, -1},
			{"kernel", be, -1, -1},
			{"mps", be, 0, -1},
			{"tensor", be, -1, 0},
		} {
			if c.name == "mps" && !be.hasMPS || c.name == "tensor" && !be.hasTensor || c.name == "cpu" && n > 2048 {
				continue
			}
			b.Run(fmt.Sprintf("%d/%s", n, c.name), func(b *testing.B) {
				be.mu.Lock()
				be.mpsMin, be.tensorMin = c.mpsMin, c.tensorMin
				be.mu.Unlock()
				bufs := make([]backend.Buffer, 3)
				for i := range bufs {
					if bufs[i], err = c.be.Alloc(backend.Float32, n*n); err != nil {
						b.Fatal(err)
					}
				}
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

// TestM5RunsTensorOperations checks that an M5 GPU runs float32 products
// on Metal 4 tensor operations, as it can from macOS 26.
func TestM5RunsTensorOperations(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Skip(err)
	}
	t.Logf("%s: tensor operations %t", be.DeviceName(), be.hasTensor)
	if strings.Contains(be.DeviceName(), "M5") && !be.hasTensor {
		t.Errorf("%s does not run products on tensor operations", be.DeviceName())
	}
}
