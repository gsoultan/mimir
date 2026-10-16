package tensor_test

import (
	"context"
	"math"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/tensor"
)

// convolver is the CPU backend asking the scheduler for convolution
// kernels, as the GPU backends do, so that the scheduler's rewriting of
// convolutions is checked against the products it replaces. It counts
// the kernels it runs of each op.
type convolver struct {
	*cpu.Backend
	parts int

	mu  sync.Mutex
	ops map[backend.Op]int
}

var _ backend.Convolver = (*convolver)(nil)

func newConvolver(parts int) *convolver {
	return &convolver{Backend: cpu.New(), parts: parts, ops: map[backend.Op]int{}}
}

func (c *convolver) ConvParts(*backend.Kernel) int { return c.parts }

func (c *convolver) Run(ctx context.Context, k *backend.Kernel, outs, ins []backend.Buffer) error {
	c.mu.Lock()
	c.ops[k.Op]++
	c.mu.Unlock()
	return c.Backend.Run(ctx, k, outs, ins)
}

// TestConvolutionKernelsMatchFiniteDifferences checks the gradients of
// convolutions computed by convolution kernels against central
// differences, and that they ran as convolution kernels, not as windows.
func TestConvolutionKernelsMatchFiniteDifferences(t *testing.T) {
	for _, parts := range []int{1, 3} {
		be := newConvolver(parts)
		restore := tensor.SetDevice(be)
		for _, c := range gradCases() {
			if strings.HasPrefix(c.name, "conv2d") {
				t.Run(c.name, func(t *testing.T) { checkFiniteDifferences(t, c) })
			}
		}
		restore()
		if tensor.Eager() {
			continue // every unfold is computed as it is created
		}
		for _, op := range []backend.Op{backend.OpConv, backend.OpConvWeight, backend.OpConvInput} {
			if be.ops[op] == 0 {
				t.Errorf("%d parts: no %s kernel ran", parts, op)
			}
		}
		for _, op := range []backend.Op{backend.OpUnfold, backend.OpFold} {
			if be.ops[op] != 0 {
				t.Errorf("%d parts: %d %s kernels ran", parts, be.ops[op], op)
			}
		}
	}
}

// TestConvolutionKernelsSecondDerivatives checks Hessian-vector products
// through convolutions on a backend that runs convolution kernels: a
// product of the second order, whose epilogue reads more inputs than a
// weight gradient kernel takes, must run as a product.
func TestConvolutionKernelsSecondDerivatives(t *testing.T) {
	for _, parts := range []int{1, 3} {
		restore := tensor.SetDevice(newConvolver(parts))
		for _, c := range gradCases() {
			if strings.HasPrefix(c.name, "conv2d") {
				t.Run(c.name, func(t *testing.T) { checkHessianVectorProducts(t, c) })
			}
		}
		restore()
	}
}

// TestConvolutionKernelsMatchProducts compares a convolution layer's
// output and gradients computed by convolution kernels with those of
// the products of windows.
func TestConvolutionKernelsMatchProducts(t *testing.T) {
	ctx := t.Context()
	rng := rand.New(rand.NewPCG(3, 4))
	values := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	for _, c := range []struct {
		name          string
		n, cin, cout  int
		size, k       int
		opts          tensor.ConvOptions
		bias, weights bool // a bias, and the weights' gradient
	}{
		{"padded with bias", 4, 3, 8, 10, 3, tensor.ConvOptions{Padding: 1}, true, true},
		{"strided grouped", 3, 6, 4, 11, 3, tensor.ConvOptions{Stride: 2, Padding: 1, Groups: 2}, false, true},
		{"dilated", 2, 2, 5, 12, 3, tensor.ConvOptions{Dilation: 2, Padding: 2}, true, false},
		{"7x7 stem", 2, 3, 6, 20, 7, tensor.ConvOptions{Stride: 2, Padding: 3}, false, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			x := values(c.n * c.cin * c.size * c.size)
			w := values(c.cout * c.cin / max(c.opts.Groups, 1) * c.k * c.k)
			b := values(c.cout)
			// run returns the output and the gradients of the image, the
			// weights and the bias on be.
			run := func(be backend.Backend) [][]float32 {
				defer tensor.SetDevice(be)()
				xt := tensor.FromSlice(x, c.n, c.cin, c.size, c.size).RequiresGrad()
				wt := tensor.FromSlice(w, c.cout, c.cin/max(c.opts.Groups, 1), c.k, c.k)
				if c.weights {
					wt = wt.RequiresGrad()
				}
				var bt *tensor.Tensor
				if c.bias {
					bt = tensor.FromSlice(b, c.cout).RequiresGrad()
				}
				out := xt.Conv2d(wt, bt, c.opts).ReLU()
				loss := out.Mul(tensor.Rand(7, out.Shape()...)).Sum()
				if err := tensor.Realize(ctx, out); err != nil {
					t.Fatal(err)
				}
				if err := loss.Backward(ctx); err != nil {
					t.Fatal(err)
				}
				res := [][]float32{data[float32](t, out), data[float32](t, xt.Grad())}
				if c.weights {
					res = append(res, data[float32](t, wt.Grad()))
				}
				if c.bias {
					res = append(res, data[float32](t, bt.Grad()))
				}
				return res
			}
			want := run(cpu.New())
			for _, parts := range []int{1, 4} {
				got := run(newConvolver(parts))
				for i := range want {
					for j, v := range want[i] {
						if d := math.Abs(float64(got[i][j] - v)); d > 1e-4*math.Max(1, math.Abs(float64(v))) {
							t.Fatalf("%d parts: result %d element %d: got %v, want %v", parts, i, j, got[i][j], v)
						}
					}
				}
			}
		})
	}
}

// windowChooser is a convolver that asks for the products of windows
// instead of every forward and input-gradient kernel, and records the
// kernels it was asked about.
type windowChooser struct {
	*convolver
	asked []backend.Kernel
}

var _ backend.WindowChooser = (*windowChooser)(nil)

func (w *windowChooser) Windowed(k *backend.Kernel) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.asked = append(w.asked, *k)
	return true
}

// TestConvolutionsAWindowChooserDeclinesRunAsProducts checks that a
// backend that chooses the products of windows for a convolution's
// forward pass and input gradient gets them, with the weight gradient
// still a kernel, and that they compute what the CPU computes.
func TestConvolutionsAWindowChooserDeclinesRunAsProducts(t *testing.T) {
	if tensor.Eager() {
		t.Skip("every unfold is computed as it is created")
	}
	ctx := t.Context()
	x := tensor.Randn(1, 2, 4, 9, 9)
	w := tensor.Randn(2, 6, 4, 3, 3)
	run := func(be backend.Backend) [][]float32 {
		defer tensor.SetDevice(be)()
		xt, wt := x.To(tensor.DefaultDevice()).Detach().RequiresGrad(), w.To(tensor.DefaultDevice()).Detach().RequiresGrad()
		out := xt.Conv2d(wt, nil, tensor.ConvOptions{Padding: 1})
		if err := out.Mul(out).Sum().Backward(ctx); err != nil {
			t.Fatal(err)
		}
		return [][]float32{data[float32](t, out), data[float32](t, xt.Grad()), data[float32](t, wt.Grad())}
	}
	want := run(cpu.New())
	be := &windowChooser{convolver: newConvolver(1)}
	got := run(be)
	for i := range want {
		for j, v := range want[i] {
			if d := math.Abs(float64(got[i][j] - v)); d > 1e-4*math.Max(1, math.Abs(float64(v))) {
				t.Fatalf("result %d element %d: got %v, want %v", i, j, got[i][j], v)
			}
		}
	}
	for _, op := range []backend.Op{backend.OpUnfold, backend.OpFold, backend.OpConvWeight} {
		if be.ops[op] == 0 {
			t.Errorf("no %s kernel ran", op)
		}
	}
	for _, op := range []backend.Op{backend.OpConv, backend.OpConvInput} {
		if be.ops[op] != 0 {
			t.Errorf("%d %s kernels ran", be.ops[op], op)
		}
	}
	asked := map[backend.Op]bool{}
	for _, k := range be.asked {
		if _, err := backend.ConvOf(&k); err != nil {
			t.Errorf("asked about a %s kernel that is not one: %v", k.Op, err)
		}
		asked[k.Op] = true
	}
	if !asked[backend.OpConv] || !asked[backend.OpConvInput] || asked[backend.OpConvWeight] {
		t.Errorf("asked about %v, want the forward pass and the input gradient", asked)
	}
}
