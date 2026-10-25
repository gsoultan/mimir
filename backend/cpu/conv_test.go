package cpu_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// TestConvKernels checks the convolution kernels, those of the GPU
// backends' tests and others that run as Winograd or direct
// convolutions, against sums in float64 over the windows of their
// definitions: with one worker and several, and with several workers
// again with forward convolutions of few positions run as Winograd
// convolutions rather than direct ones.
func TestConvKernels(t *testing.T) {
	cases := slices.Concat(kerneltest.Conv(t), strideOneCases(t))
	// The sums take longer than the kernels, so each case's are computed
	// once.
	wants := make([]convWant, len(cases))
	for i := range cases {
		wants[i] = convWants(t, &cases[i])
	}
	for _, run := range []struct {
		name     string
		workers  int
		winograd bool
	}{{"1 worker", 1, false}, {"3 workers", 3, false}, {"3 workers, winograd", 3, true}} {
		t.Run(run.name, func(t *testing.T) {
			if run.winograd {
				defer cpu.SetDirectPositions(0)()
			}
			be := cpu.NewWorkers(run.workers)
			for i, c := range cases {
				t.Run(c.Name, func(t *testing.T) {
					checkConv(t, be, c, wants[i])
				})
			}
		})
	}
}

// strideOneCases returns convolutions of stride 1 and their gradients: of
// 3×3 windows of dilation 1, with enough channels to run as Winograd
// convolutions, with odd and even outputs, every padding up to 2, grouped,
// in blocks of whole items and of tile rows; and of few channels, which
// run as direct convolutions, with other windows and dilations, and
// outputs that fill no whole tile.
func strideOneCases(t *testing.T) []kerneltest.Case {
	rng := rand.New(rand.NewPCG(5, 6))
	f := backend.Float32
	var cases []kerneltest.Case
	for _, w := range []struct {
		name             string
		n, g, c, m       int
		image            []int
		pads             [][2]int
		kernel, dilation []int
	}{
		{"same", 3, 1, 8, 8, []int{6, 7}, [][2]int{{1, 1}, {1, 1}}, nil, nil},
		{"valid", 2, 1, 9, 12, []int{7, 8}, [][2]int{{0, 0}, {0, 0}}, nil, nil},
		{"uneven pads", 2, 2, 10, 9, []int{5, 9}, [][2]int{{2, 0}, {1, 2}}, nil, nil},
		{"full", 1, 1, 33, 17, []int{4, 4}, [][2]int{{2, 2}, {2, 2}}, nil, nil},
		{"mnist", 2, 1, 32, 64, []int{14, 14}, [][2]int{{1, 1}, {1, 1}}, nil, nil},
		{"tile rows", 2, 1, 8, 16, []int{30, 70}, [][2]int{{1, 1}, {1, 1}}, nil, nil},
		{"deep", 2, 1, 300, 8, []int{4, 5}, [][2]int{{1, 1}, {1, 1}}, nil, nil},
		{"direct mnist", 3, 1, 1, 32, []int{28, 28}, [][2]int{{1, 1}, {1, 1}}, nil, nil},
		{"direct one item", 1, 1, 2, 13, []int{9, 11}, [][2]int{{1, 1}, {1, 1}}, nil, nil},
		{"direct dilated", 2, 2, 3, 9, []int{9, 11}, [][2]int{{2, 0}, {1, 2}}, []int{3, 2}, []int{2, 3}},
		{"direct 5x5", 3, 1, 4, 10, []int{12, 10}, [][2]int{{2, 2}, {0, 1}}, []int{5, 5}, []int{1, 1}},
	} {
		if w.kernel == nil {
			w.kernel, w.dilation = []int{3, 3}, []int{1, 1}
		}
		attrs := backend.Attrs{Kernel: w.kernel, Stride: []int{1, 1}, Dilation: w.dilation, Pads: w.pads}
		image := slices.Concat([]int{w.n, w.g * w.c}, w.image)
		cols := 1
		for d := range 2 {
			cols *= w.image[d] + w.pads[d][0] + w.pads[d][1] - (w.kernel[d]-1)*w.dilation[d]
		}
		rows := w.c * w.kernel[0] * w.kernel[1]
		shared := func(shape []int) (backend.View, int) {
			one := slices.Clone(shape)
			one[0] = 1
			n, _ := backend.NumElements(one)
			return backend.Contiguous(one).Expand(shape), n
		}
		for name, x := range kerneltest.Views(image) {
			if name == "broadcast" || name == "flipped" {
				continue
			}
			a, na := shared([]int{w.n, w.g, w.m, rows})
			cases = append(cases, kerneltest.Case{
				Name: fmt.Sprintf("conv/%s/%s", w.name, name),
				K:    backend.Kernel{Op: backend.OpConv, DType: f, Shape: []int{w.n, w.g, w.m, cols}, In: []backend.View{a, x.V}, Attrs: attrs},
				Ins:  []backend.Buffer{kerneltest.HostBuffer(t, rng, f, na, 0, 0), kerneltest.HostBuffer(t, rng, f, x.N, 0, 0)},
			})
			wt, nw := shared([]int{w.n, w.g, rows, w.m})
			dy := kerneltest.Views([]int{w.n, w.g, w.m, cols})[name]
			cases = append(cases, kerneltest.Case{
				Name: fmt.Sprintf("conv input/%s/%s", w.name, name),
				K:    backend.Kernel{Op: backend.OpConvInput, DType: f, Shape: image, In: []backend.View{wt, dy.V}, Attrs: attrs},
				Ins:  []backend.Buffer{kerneltest.HostBuffer(t, rng, f, nw, 0, 0), kerneltest.HostBuffer(t, rng, f, dy.N, 0, 0)},
			})
			cases = append(cases, kerneltest.Case{
				Name: fmt.Sprintf("conv weight/%s/%s", w.name, name),
				K:    backend.Kernel{Op: backend.OpConvWeight, DType: f, Shape: []int{1, w.g, w.m, rows}, In: []backend.View{dy.V, x.V}, Attrs: attrs},
				Ins:  []backend.Buffer{kerneltest.HostBuffer(t, rng, f, dy.N, 0, 0), kerneltest.HostBuffer(t, rng, f, x.N, 0, 0)},
			})
		}
	}
	return cases
}

// convWant holds the outputs of a case's convolution kernel, summed in
// float64, and the sums of the magnitudes of their products.
type convWant struct{ sums, mags []float64 }

// convWants returns the outputs that the convolution kernel of c should
// compute, with its epilogue, if any.
func convWants(t *testing.T, c *kerneltest.Case) convWant {
	k := &c.K
	ins := make([][]float32, len(c.Ins))
	for i, b := range c.Ins {
		ins[i] = kerneltest.Floats(t, b)
	}
	want, mag := convSums(t, k, ins)
	if k.Attrs.Program != nil {
		// The epilogue of kerneltest's conv/epilogue: a bias, then a ReLU.
		bias := k.In[2]
		for i := range want {
			idx := unravel(bias.Shape, i)
			want[i] = math.Max(0, want[i]+float64(ins[2][at(bias, idx)]))
		}
	}
	return convWant{want, mag}
}

// checkConv runs the convolution kernel of c on be and compares its
// output with w, allowing a relative error of 1e-5 of the sum of the
// products' magnitudes.
func checkConv(t *testing.T, be *cpu.Backend, c kerneltest.Case, w convWant) {
	k := &c.K
	n, _ := backend.NumElements(k.Shape)
	out := kerneltest.HostBuffer(t, nil, backend.Float32, n, 0, 0)
	if err := be.Run(t.Context(), k, []backend.Buffer{out}, c.Ins); err != nil {
		t.Fatal(err)
	}
	got := kerneltest.Floats(t, out)
	for i, want := range w.sums {
		if d := math.Abs(float64(got[i]) - want); d > 1e-5*max(1, w.mags[i]) || math.IsNaN(float64(got[i])) {
			t.Fatalf("element %d: got %v, want %v", i, got[i], want)
		}
	}
}

// convSums returns the outputs of the convolution kernel k from its
// inputs, summed in float64 over the windows, and the sums of the
// magnitudes of their products.
func convSums(t *testing.T, k *backend.Kernel, ins [][]float32) (sums, mags []float64) {
	c, err := backend.ConvOf(k)
	if err != nil {
		t.Fatal(err)
	}
	a := k.Attrs
	s := len(a.Kernel)
	size, _ := backend.NumElements(k.Shape)
	sums, mags = make([]float64, size), make([]float64, size)
	img := k.In[1] // [N, G*C, D...]
	if k.Op == backend.OpConvInput {
		img = backend.Contiguous(k.Shape)
	}
	rows, cols := c.Rows(), c.Cols()
	// pixel returns the index of the element of image img that window
	// element j of group g reads at position o, or -1 in the padding.
	kidx, pos := make([][]int, rows), make([][]int, cols)
	for j := range rows {
		kidx[j] = unravel(a.Kernel, j%c.Window)
	}
	for o := range cols {
		pos[o] = unravel(c.Positions, o)
	}
	idx := make([]int, 2+s)
	pixel := func(n, g, j, o int) int {
		idx[0], idx[1] = n, g*c.C+j/c.Window
		for d := range s {
			y := pos[o][d]*a.Stride[d] + kidx[j][d]*a.Dilation[d] - a.Pads[d][0]
			if y < 0 || y >= c.Image[d] {
				return -1
			}
			idx[2+d] = y
		}
		return at(img, idx)
	}
	add := func(i int, v float64) { sums[i] += v; mags[i] += math.Abs(v) }
	for n := range c.N {
		for g := range c.G {
			for m := range c.M {
				for j := range rows {
					for o := range cols {
						p := pixel(n, g, j, o)
						switch k.Op {
						case backend.OpConv:
							if p >= 0 {
								add(((n*c.G+g)*c.M+m)*cols+o, float64(ins[0][at(k.In[0], []int{n, g, m, j})])*float64(ins[1][p]))
							}
						case backend.OpConvWeight:
							if part := (n*cols + o) / ((c.N*cols + c.Parts - 1) / c.Parts); p >= 0 {
								add(((part*c.G+g)*c.M+m)*rows+j, float64(ins[0][at(k.In[0], []int{n, g, m, o})])*float64(ins[1][p]))
							}
						case backend.OpConvInput:
							if p >= 0 {
								add(p, float64(ins[0][at(k.In[0], []int{n, g, j, m})])*float64(ins[1][at(k.In[1], []int{n, g, m, o})]))
							}
						}
					}
				}
			}
		}
	}
	return sums, mags
}

// at returns the buffer index of element idx of the view v.
func at(v backend.View, idx []int) int {
	off := v.Offset
	for d, i := range idx {
		off += i * v.Strides[d]
	}
	return off
}

// unravel returns the row-major index into shape of element i.
func unravel(shape []int, i int) []int {
	idx := make([]int, len(shape))
	for d := len(shape) - 1; d >= 0; d-- {
		idx[d] = i % shape[d]
		i /= shape[d]
	}
	return idx
}

// BenchmarkConvKernels times the convolution kernels of the second layer
// of examples/mnist's CNN on a batch of 128, and its forward pass on one
// item.
func BenchmarkConvKernels(b *testing.B) {
	for _, n := range []int{1, 128} {
		b.Run(fmt.Sprint(n), func(b *testing.B) { benchmarkConvKernels(b, n) })
	}
}

func benchmarkConvKernels(b *testing.B, n int) {
	rng := rand.New(rand.NewPCG(1, 2))
	f := backend.Float32
	be := cpu.New()
	c, m, size := 32, 64, 14
	attrs := backend.Attrs{Kernel: []int{3, 3}, Stride: []int{1, 1}, Dilation: []int{1, 1}, Pads: [][2]int{{1, 1}, {1, 1}}}
	image := []int{n, c, size, size}
	rows, cols := c*9, size*size
	buf := func(n int) backend.Buffer {
		buf, err := be.Alloc(f, n)
		if err != nil {
			b.Fatal(err)
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()
		}
		if err := be.Upload(b.Context(), buf, kerneltest.Bytes(v)); err != nil {
			b.Fatal(err)
		}
		return buf
	}
	shared := func(shape ...int) backend.View {
		return backend.Contiguous(slices.Concat([]int{1}, shape[1:])).Expand(shape)
	}
	x, dy, w := buf(n*c*size*size), buf(n*m*cols), buf(m*rows)
	flops := 2 * float64(n*m*rows*cols)
	for _, k := range []backend.Kernel{
		{Op: backend.OpConv, DType: f, Shape: []int{n, 1, m, cols}, In: []backend.View{shared(n, 1, m, rows), backend.Contiguous(image)}, Attrs: attrs},
		{Op: backend.OpConvInput, DType: f, Shape: image, In: []backend.View{shared(n, 1, rows, m), backend.Contiguous([]int{n, 1, m, cols})}, Attrs: attrs},
		{Op: backend.OpConvWeight, DType: f, Shape: []int{1, 1, m, rows}, In: []backend.View{backend.Contiguous([]int{n, 1, m, cols}), backend.Contiguous(image)}, Attrs: attrs},
	} {
		ins := []backend.Buffer{w, x}
		switch k.Op {
		case backend.OpConvInput:
			ins = []backend.Buffer{w, dy}
		case backend.OpConvWeight:
			ins = []backend.Buffer{dy, x}
		}
		if n == 1 && k.Op != backend.OpConv {
			continue
		}
		size, _ := backend.NumElements(k.Shape)
		out, err := be.Alloc(f, size)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(k.Op.String(), func(b *testing.B) {
			for b.Loop() {
				if err := be.Run(b.Context(), &k, []backend.Buffer{out}, ins); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}
