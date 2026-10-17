package kerneltest

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// Conv returns cases of the convolution kernels: products of windows
// that a GPU kernel reads from the image where they lie. Each geometry
// runs with the image read through every way Views reads it, weights
// shared by the batch's items, as a layer's are, and weights of their
// own, and weight gradients in one part and in several.
func Conv(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(13, 14))
	f := backend.Float32
	var cases []Case
	for _, w := range []struct {
		name                     string
		n, g, c, m               int
		image                    []int
		kernel, stride, dilation []int
		pads                     [][2]int
	}{
		{"3x3 padded", 2, 1, 3, 5, []int{6, 7}, []int{3, 3}, []int{1, 1}, []int{1, 1}, [][2]int{{1, 1}, {1, 1}}},
		{"strided dilated grouped", 2, 2, 3, 4, []int{9, 8}, []int{3, 2}, []int{2, 3}, []int{2, 1}, [][2]int{{2, 0}, {1, 2}}},
		{"1d", 3, 1, 4, 6, []int{11}, []int{4}, []int{2}, []int{1}, [][2]int{{0, 3}}},
		{"several tiles", 3, 1, 8, 70, []int{9, 9}, []int{3, 3}, []int{1, 1}, []int{1, 1}, [][2]int{{1, 1}, {1, 1}}},
	} {
		attrs := backend.Attrs{Kernel: w.kernel, Stride: w.stride, Dilation: w.dilation, Pads: w.pads}
		image := slices.Concat([]int{w.n, w.g * w.c}, w.image)
		window, cols := 1, 1
		for d, k := range w.kernel {
			window *= k
			cols *= (w.image[d]+w.pads[d][0]+w.pads[d][1]-(k-1)*w.dilation[d]-1)/w.stride[d] + 1
		}
		rows := w.c * window
		// The weights or the gradient, for every item or shared by them.
		matrix := func(shape []int, share bool) View {
			if !share {
				return Views(shape)["contiguous"]
			}
			one := slices.Clone(shape)
			one[0] = 1
			n, _ := backend.NumElements(one)
			return View{backend.Contiguous(one).Expand(shape), n}
		}
		for name, x := range Views(image) {
			for _, share := range []bool{true, false} {
				a := matrix([]int{w.n, w.g, w.m, rows}, share)
				cases = append(cases, Case{
					Name: fmt.Sprintf("conv/%s/%s/shared %v", w.name, name, share),
					K:    backend.Kernel{Op: backend.OpConv, DType: f, Shape: []int{w.n, w.g, w.m, cols}, In: []backend.View{a.V, x.V}, Attrs: attrs},
					Ins:  []backend.Buffer{HostBuffer(t, rng, f, a.N, 0, 0), HostBuffer(t, rng, f, x.N, 0, 0)},
					Tol:  1e-4,
				})
				wt := matrix([]int{w.n, w.g, rows, w.m}, share)
				dy := Views([]int{w.n, w.g, w.m, cols})[name]
				cases = append(cases, Case{
					Name: fmt.Sprintf("conv input/%s/%s/shared %v", w.name, name, share),
					K:    backend.Kernel{Op: backend.OpConvInput, DType: f, Shape: image, In: []backend.View{wt.V, dy.V}, Attrs: attrs},
					Ins:  []backend.Buffer{HostBuffer(t, rng, f, wt.N, 0, 0), HostBuffer(t, rng, f, dy.N, 0, 0)},
					Tol:  1e-4,
				})
			}
			for _, parts := range []int{1, 3} {
				dy := Views([]int{w.n, w.g, w.m, cols})["transposed"]
				cases = append(cases, Case{
					Name: fmt.Sprintf("conv weight/%s/%s/%d parts", w.name, name, parts),
					K:    backend.Kernel{Op: backend.OpConvWeight, DType: f, Shape: []int{parts, w.g, w.m, rows}, In: []backend.View{dy.V, x.V}, Attrs: attrs},
					Ins:  []backend.Buffer{HostBuffer(t, rng, f, dy.N, 0, 0), HostBuffer(t, rng, f, x.N, 0, 0)},
					Tol:  1e-4,
				})
			}
		}
	}
	// A bias and a ReLU as the epilogue of a convolution, read as the
	// [N, M, O] output of one group.
	x := backend.Contiguous([]int{2, 3, 5, 5})
	bias := backend.Contiguous([]int{1, 4, 1}).Expand([]int{2, 4, 25})
	cases = append(cases, Case{
		Name: "conv/epilogue",
		K: backend.Kernel{Op: backend.OpConv, DType: f, Shape: []int{2, 1, 4, 25},
			In: []backend.View{backend.Contiguous([]int{1, 1, 4, 27}).Expand([]int{2, 1, 4, 27}), x, bias},
			Attrs: backend.Attrs{Kernel: []int{3, 3}, Stride: []int{1, 1}, Dilation: []int{1, 1}, Pads: [][2]int{{1, 1}, {1, 1}},
				Program: []backend.Instr{
					{Op: backend.OpLoad, DType: f, Args: []int{0}},
					{Op: backend.OpLoad, DType: f, Args: []int{1}},
					{Op: backend.OpAdd, DType: f, Args: []int{0, 1}},
					{Op: backend.OpConst, DType: f},
					{Op: backend.OpMax, DType: f, Args: []int{2, 3}},
				}}},
		Ins: []backend.Buffer{HostBuffer(t, rng, f, 108, 0, 0), HostBuffer(t, rng, f, 150, 0, 0), HostBuffer(t, rng, f, 4, 0, 0)},
		Tol: 1e-4,
	})
	return cases
}
