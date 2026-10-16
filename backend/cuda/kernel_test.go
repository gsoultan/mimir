package cuda

import (
	"os"
	"testing"

	"github.com/gsoultan/mimir/backend/internal/gen"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// The kernel tests run the cases of kerneltest on the CUDA device, if
// there is one. With MIMIR_EMULATE=1 the products also run as the backend
// asks cuBLAS to compute them, on the host, with their epilogues
// emulated; the generated kernels themselves are emulated by the tests
// of the generator.

func TestMain(m *testing.M) {
	code := m.Run()
	kerneltest.Cleanup()
	os.Exit(code)
}

// targets returns the CUDA device as a target, if there is one, and, if
// products is set, the device running every product it can on cuBLAS,
// and the emulated cuBLAS.
func targets(t *testing.T, products bool) map[string]kerneltest.Target {
	out := map[string]kerneltest.Target{}
	if g := newGPU(t, -1); g != nil {
		out["gpu"] = g
	}
	if !products {
		return out
	}
	if e := kerneltest.NewEmulator(t, gen.CUDA); e != nil {
		out["emulator-cublas"] = kerneltest.EmulatedGEMM{Emulator: e, Plan: gen.PlanGEMM, Compute: emulatedSgemm}
	}
	if g := newGPU(t, 0); g != nil {
		out["gpu-cublas"] = g
	}
	return out
}

// emulatedSgemm computes the product g as the backend asks cuBLAS to.
func emulatedSgemm(g gen.GEMM, x, y []float32) []float32 {
	return cublas(planSgemm(g), [2][]float32{x, y})
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
