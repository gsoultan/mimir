package metal

import (
	"os"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// The kernel tests run the cases of kerneltest on the Metal device, if
// there is one. With MIMIR_EMULATE=1 the products also run as Metal
// Performance Shaders would read them, with their epilogues emulated.

func TestMain(m *testing.M) {
	code := m.Run()
	kerneltest.Cleanup()
	os.Exit(code)
}

// targets returns the Metal device as a target, if there is one, and, if
// products is set, the device running every product it can through Metal
// Performance Shaders and on Metal 4 tensor operations, and the emulated
// MPS.
func targets(t *testing.T, products bool) map[string]kerneltest.Target {
	out := map[string]kerneltest.Target{}
	if g := newGPU(t, -1, -1); g != nil {
		out["gpu"] = g
	}
	if !products {
		return out
	}
	if e := kerneltest.NewEmulator(t, gen.Metal); e != nil {
		out["emulator-mps"] = kerneltest.EmulatedGEMM{Emulator: e, Plan: planMPS, Compute: kerneltest.GEMM}
	}
	if g := newGPU(t, 0, -1); g != nil {
		out["gpu-mps"] = g
	}
	if g := newGPU(t, -1, 0); g != nil {
		out["gpu-tensor"] = g
	}
	return out
}

// planMPS returns the products Metal Performance Shaders runs.
func planMPS(k *backend.Kernel, lens []int) (gen.GEMM, bool) {
	g, ok := gen.PlanGEMM(k, lens)
	return g, ok && packed(g)
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
