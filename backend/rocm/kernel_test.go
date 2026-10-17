package rocm

import (
	"os"
	"testing"

	"github.com/gsoultan/mimir/backend/internal/gen"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// The kernel tests run the cases of kerneltest on the ROCm device, if
// there is one. With MIMIR_EMULATE=1 the products also run as the backend
// asks hipBLAS to compute them, on the host, with their epilogues
// emulated in both HIP dialects; the generated kernels themselves are
// emulated by the tests of the generator.

func TestMain(m *testing.M) {
	code := m.Run()
	kerneltest.Cleanup()
	os.Exit(code)
}

// targets returns the ROCm device as a target, if there is one, and, if
// products is set, the device running every product it can on hipBLAS,
// and the emulated hipBLAS.
func targets(t *testing.T, products bool) map[string]kerneltest.Target {
	out := map[string]kerneltest.Target{}
	if g := newGPU(t, -1); g != nil {
		out["gpu"] = g
	}
	if !products {
		return out
	}
	for _, d := range []*gen.Dialect{gen.HIP64, gen.HIP32} {
		if e := kerneltest.NewEmulator(t, d); e != nil {
			out["emulator-hipblas-"+d.Name] = kerneltest.EmulatedGEMM{Emulator: e, Plan: gen.PlanGEMM, Compute: emulatedSgemm}
		}
	}
	if g := newGPU(t, 0); g != nil {
		out["gpu-hipblas"] = g
	}
	return out
}

// emulatedSgemm computes the product g as the backend asks hipBLAS to.
func emulatedSgemm(g gen.GEMM, x, y []float32) []float32 {
	return kerneltest.ColumnMajor(g.ColumnMajor(), [2][]float32{x, y})
}

func TestElementwiseKernels(t *testing.T) {
	kerneltest.Run(t, kerneltest.Elementwise(t), targets(t, false))
}
func TestRandomKernels(t *testing.T) { kerneltest.Run(t, kerneltest.Random(t), targets(t, false)) }
func TestReduceKernels(t *testing.T) { kerneltest.Run(t, kerneltest.Reduce(t), targets(t, false)) }
func TestMovementKernels(t *testing.T) {
	kerneltest.Run(t, kerneltest.Movement(t), targets(t, false))
}
func TestWindowKernels(t *testing.T) { kerneltest.Run(t, kerneltest.Window(t), targets(t, false)) }
func TestMatMulKernels(t *testing.T) { kerneltest.Run(t, kerneltest.MatMul(t), targets(t, true)) }
func TestScanKernels(t *testing.T)   { kerneltest.Run(t, kerneltest.Scan(t), targets(t, false)) }
func TestQuantKernels(t *testing.T)  { kerneltest.Run(t, kerneltest.Quant(t), targets(t, false)) }
func TestAttentionKernels(t *testing.T) {
	kerneltest.Run(t, kerneltest.Attention(t), targets(t, false))
}
func TestConvKernels(t *testing.T) { kerneltest.Run(t, kerneltest.Conv(t), targets(t, false)) }
func TestSortKernels(t *testing.T) { kerneltest.Run(t, kerneltest.Sort(t), targets(t, false)) }
func TestFFTKernels(t *testing.T)  { kerneltest.Run(t, kerneltest.FFT(t), targets(t, false)) }
func TestSegmentKernels(t *testing.T) {
	kerneltest.Run(t, kerneltest.Segment(t), targets(t, false))
}
