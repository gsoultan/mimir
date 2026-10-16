package gen_test

import (
	"os"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// With MIMIR_EMULATE=1 the kernel tests run the generated kernels of every
// dialect on the CPU and compare them with the CPU backend. The GPU
// backends run the same cases on their devices.

var dialects = []*gen.Dialect{gen.Metal, gen.CUDA}

func TestMain(m *testing.M) {
	code := m.Run()
	kerneltest.Cleanup()
	os.Exit(code)
}

// emulators returns the emulator of each dialect, if there is one.
func emulators(t *testing.T) map[string]kerneltest.Target {
	targets := map[string]kerneltest.Target{}
	for _, d := range dialects {
		if e := kerneltest.NewEmulator(t, d); e != nil {
			targets[d.Name] = e
		}
	}
	return targets
}

func TestElementwiseKernels(t *testing.T) { kerneltest.Run(t, kerneltest.Elementwise(t), emulators(t)) }
func TestRandomKernels(t *testing.T)      { kerneltest.Run(t, kerneltest.Random(t), emulators(t)) }
func TestReduceKernels(t *testing.T)      { kerneltest.Run(t, kerneltest.Reduce(t), emulators(t)) }
func TestMovementKernels(t *testing.T)    { kerneltest.Run(t, kerneltest.Movement(t), emulators(t)) }
func TestWindowKernels(t *testing.T)      { kerneltest.Run(t, kerneltest.Window(t), emulators(t)) }
func TestMatMulKernels(t *testing.T)      { kerneltest.Run(t, kerneltest.MatMul(t), emulators(t)) }

// TestConstantsAreParameters checks that kernels differing only in their
// constants, as an optimizer's update does from one step to the next,
// share their source and so one compiled program.
func TestConstantsAreParameters(t *testing.T) {
	f := backend.Float32
	shape := []int{4, 5}
	in := backend.Contiguous(shape)
	kernels := func(c float64) []*backend.Kernel {
		return []*backend.Kernel{
			{Op: backend.OpConst, DType: f, Shape: shape, Attrs: backend.Attrs{Value: c}},
			{Op: backend.OpFused, DType: f, Shape: shape, In: []backend.View{in}, Attrs: backend.Attrs{Program: []backend.Instr{
				{Op: backend.OpLoad, DType: f, Args: []int{0}},
				{Op: backend.OpConst, DType: f, Value: c},
				{Op: backend.OpMul, DType: f, Args: []int{0, 1}},
			}}},
			{Op: backend.OpPad, DType: f, Shape: []int{6, 5}, In: []backend.View{in}, Attrs: backend.Attrs{Value: c, Pads: [][2]int{{1, 1}, {0, 0}}}},
		}
	}
	a, b := kernels(0.9), kernels(0.999)
	for _, d := range dialects {
		for i := range a {
			var srcs [2]string
			for j, k := range []*backend.Kernel{a[i], b[i]} {
				n, _ := backend.NumElements(k.Shape)
				outs := []backend.Buffer{kerneltest.HostBuffer(t, nil, f, n, 0, 0)}
				var ins []backend.Buffer
				for range k.In {
					ins = append(ins, kerneltest.HostBuffer(t, nil, f, 20, 0, 0))
				}
				l, err := gen.Plan(d, k, outs, ins)
				if err != nil {
					t.Fatal(err)
				}
				srcs[j] = l.Src
			}
			if srcs[0] != srcs[1] {
				t.Errorf("%s %s: sources differ with the constant:\n%s\n%s", d.Name, a[i].Op, srcs[0], srcs[1])
			}
		}
	}
}
