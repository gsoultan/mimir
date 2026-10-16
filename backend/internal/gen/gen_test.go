package gen_test

import (
	"errors"
	"math/rand/v2"
	"os"
	"strings"
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
func TestScanKernels(t *testing.T)        { kerneltest.Run(t, kerneltest.Scan(t), emulators(t)) }
func TestQuantKernels(t *testing.T)       { kerneltest.Run(t, kerneltest.Quant(t), emulators(t)) }
func TestAttentionKernels(t *testing.T)   { kerneltest.Run(t, kerneltest.Attention(t), emulators(t)) }

// TestSliceScatterInPlaceWritesTheBlock checks that a slice scatter
// written over its input writes the block and nothing else.
func TestSliceScatterInPlaceWritesTheBlock(t *testing.T) {
	f := backend.Float32
	rng := rand.New(rand.NewPCG(3, 4))
	shape := []int{4, 6}
	k := &backend.Kernel{Op: backend.OpSliceScatter, DType: f, Shape: shape,
		In:    []backend.View{backend.Contiguous(shape), backend.Contiguous([]int{3, 2}).Permute([]int{1, 0})},
		Attrs: backend.Attrs{Ranges: [][2]int{{1, 3}, {2, 5}}}}
	x, src := kerneltest.HostBuffer(t, rng, f, 24, 0, 0), kerneltest.HostBuffer(t, rng, f, 6, 0, 0)
	want := kerneltest.HostBuffer(t, nil, f, 24, 0, 0)
	if err := kerneltest.Host.Run(t.Context(), k, []backend.Buffer{want}, []backend.Buffer{x, src}); err != nil {
		t.Fatal(err)
	}
	wv, xv := kerneltest.Floats(t, want), kerneltest.Floats(t, x)
	for _, d := range dialects {
		l, err := gen.Plan(d, k, []backend.Buffer{x}, []backend.Buffer{x, src})
		if err != nil {
			t.Fatal(err)
		}
		e := kerneltest.NewEmulator(t, d)
		if e == nil {
			continue
		}
		// The launch writes into a buffer of NaNs: outside the block they
		// must stay, as x does when the output is x.
		got, _ := e.Exec(t, l, []backend.Buffer{kerneltest.HostBuffer(t, nil, f, 24, 0, 0)}, []backend.Buffer{x, src})
		for i, v := range kerneltest.FromBytes(got[0]) {
			row, col := i/6, i%6
			inside := row >= 1 && row < 3 && col >= 2 && col < 5
			if inside && v != wv[i] || !inside && v == v {
				t.Fatalf("%s: element %d = %v, want %v (x has %v)", d.Name, i, v, wv[i], xv[i])
			}
		}
	}
}

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

// TestHalfProductsAreRejected checks that no dialect generates a product
// of half elements, which the tensor package computes in float32.
func TestHalfProductsAreRejected(t *testing.T) {
	square := backend.Contiguous([]int{2, 2})
	for _, dt := range []backend.DType{backend.BFloat16, backend.Float16} {
		k := &backend.Kernel{Op: backend.OpMatMul, DType: dt, Shape: []int{2, 2}, In: []backend.View{square, square}}
		out := kerneltest.HostBuffer(t, nil, dt, 4, 0, 0)
		x := kerneltest.HostBuffer(t, nil, dt, 4, 0, 0)
		for _, d := range dialects {
			if _, err := gen.Plan(d, k, []backend.Buffer{out}, []backend.Buffer{x, x}); !errors.Is(err, backend.ErrDType) {
				t.Errorf("%s %s product: %v, want ErrDType", d.Name, dt, err)
			}
		}
		if _, ok := gen.PlanGEMM(k, []int{4, 4}); ok {
			t.Errorf("%s product planned as a GEMM", dt)
		}
	}
}

// TestHalfKernelsConvertOnLoadAndStore checks the shape of the generated
// code for half elements: ushort buffers, float values and the prelude's
// conversions around loads and stores.
func TestHalfKernelsConvertOnLoadAndStore(t *testing.T) {
	shape := []int{4, 5}
	in := backend.Contiguous(shape)
	k := &backend.Kernel{Op: backend.OpFused, DType: backend.BFloat16, Shape: shape, In: []backend.View{in, in}, Attrs: backend.Attrs{Program: []backend.Instr{
		{Op: backend.OpLoad, DType: backend.BFloat16, Args: []int{0}},
		{Op: backend.OpLoad, DType: backend.Float16, Args: []int{1}},
		{Op: backend.OpCast, DType: backend.BFloat16, Args: []int{1}},
		{Op: backend.OpAdd, DType: backend.BFloat16, Args: []int{0, 2}},
	}}}
	out := kerneltest.HostBuffer(t, nil, backend.BFloat16, 20, 0, 0)
	ins := []backend.Buffer{kerneltest.HostBuffer(t, nil, backend.BFloat16, 20, 0, 0), kerneltest.HostBuffer(t, nil, backend.Float16, 20, 0, 0)}
	for _, d := range dialects {
		l, err := gen.Plan(d, k, []backend.Buffer{out}, ins)
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{"ushort* o0", "ushort* i0", "ushort* i1", "float v3 = bf2f(i0[", "float v4 = h2f(i1[", "float v5 = bf2f(f2bf(float(v4)));", "float v6 = bf2f(f2bf(v3 + v5));", "o0[i] = f2bf(v6);"} {
			if !strings.Contains(l.Src, want) {
				t.Errorf("%s: no %q in\n%s", d.Name, want, l.Src)
			}
		}
	}
}
