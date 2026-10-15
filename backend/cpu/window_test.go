package cpu

import (
	"fmt"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// stridedView is a view of a tensor, and the size of the buffer it reads.
type stridedView struct {
	name string
	view backend.View
	size int
}

// stridedViews returns views of a tensor of the given shape laid out in
// different ways: row-major, transposed, inside a larger buffer, reversed
// and broadcast along the first dimension.
func stridedViews(shape []int) []stridedView {
	r := len(shape)
	n, _ := backend.NumElements(shape)
	rev, reverse, all := slices.Clone(shape), make([]int, r), make([]int, r)
	slices.Reverse(rev)
	grown, inner := make([]int, r), make([][2]int, r)
	for d, s := range shape {
		reverse[d], all[d] = r-1-d, d
		grown[d], inner[d] = s+2, [2]int{1, s + 1}
	}
	g, _ := backend.NumElements(grown)
	views := []stridedView{
		{"contiguous", backend.Contiguous(shape), n},
		{"transposed", backend.Contiguous(rev).Permute(reverse), n},
		{"padded", backend.Contiguous(grown).Shrink(inner), g},
		{"flipped", backend.Contiguous(shape).Flip(all), n},
	}
	if r > 0 && shape[0] > 1 {
		one := slices.Clone(shape)
		one[0] = 1
		m, _ := backend.NumElements(one)
		views = append(views, stridedView{"broadcast", backend.Contiguous(one).Expand(shape), m})
	}
	return views
}

// eachIndex calls fn with every index of shape in row-major order.
func eachIndex(shape []int, fn func(idx []int)) {
	n, _ := backend.NumElements(shape)
	idx := make([]int, len(shape))
	for range n {
		fn(idx)
		for d := len(idx) - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
}

// offsetOf returns the buffer offset of element idx of v.
func offsetOf(v backend.View, idx []int) int {
	off := v.Offset
	for d, i := range idx {
		off += i * v.Strides[d]
	}
	return off
}

// runOn runs k on a backend with the given number of workers.
func runOn[T elem](t *testing.T, workers int, k *backend.Kernel, ins ...backend.Buffer) []T {
	t.Helper()
	b := New()
	b.workers = workers
	got, err := runWith[T](t, b, k, ins...)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// windowCase is a window over the trailing dimensions of an input.
type windowCase struct {
	in    []int
	attrs backend.Attrs
}

// windows returns the shape of the windows of c's input.
func (c windowCase) windows() []int {
	a := c.attrs
	s := len(a.Kernel)
	lead, spatial := c.in[:len(c.in)-s], c.in[len(c.in)-s:]
	pos := make([]int, s)
	for d := range s {
		span := (a.Kernel[d]-1)*a.Dilation[d] + 1
		pos[d] = (spatial[d]+a.Pads[d][0]+a.Pads[d][1]-span)/a.Stride[d] + 1
	}
	return slices.Concat(lead, a.Kernel, pos)
}

// eachWindowElement calls fn with the contiguous offsets of every element
// of c's windows and of the input element it reads, or -1 for padding.
func (c windowCase) eachWindowElement(fn func(win, in int)) {
	a := c.attrs
	s := len(a.Kernel)
	lead := len(c.in) - s
	inStrides := backend.ContiguousStrides(c.in)
	w := c.windows()
	win := 0
	eachIndex(w, func(idx []int) {
		in := 0
		for d := range lead {
			in += idx[d] * inStrides[d]
		}
		for d := range s {
			i := idx[lead+s+d]*a.Stride[d] - a.Pads[d][0] + idx[lead+d]*a.Dilation[d]
			if in >= 0 && i >= 0 && i < c.in[lead+d] {
				in += i * inStrides[lead+d]
			} else {
				in = -1
			}
		}
		fn(win, in)
		win++
	})
}

func window(kernel, stride, dilation []int, pads ...[2]int) backend.Attrs {
	return backend.Attrs{Kernel: kernel, Stride: stride, Dilation: dilation, Pads: pads}
}

var windowCases = []windowCase{
	{[]int{2, 9}, window([]int{3}, []int{2}, []int{1}, [2]int{1, 2})},
	{[]int{6, 5}, window([]int{2, 3}, []int{1, 1}, []int{1, 1}, [2]int{0, 0}, [2]int{1, 1})},
	{[]int{2, 3, 7, 6}, window([]int{3, 2}, []int{2, 1}, []int{1, 2}, [2]int{1, 1}, [2]int{0, 2})},
	{[]int{1, 5, 4, 6}, window([]int{2, 2, 3}, []int{1, 2, 2}, []int{2, 1, 1}, [2]int{0, 1}, [2]int{1, 0}, [2]int{2, 2})},
	// Padding wider than the reach of some kernel offsets, so that whole
	// rows of windows fall outside the input.
	{[]int{2, 4, 4}, window([]int{3, 3}, []int{1, 1}, []int{1, 1}, [2]int{2, 2}, [2]int{2, 2})},
	{[]int{3, 3}, window([]int{5}, []int{1}, []int{2}, [2]int{4, 4})},
}

func TestUnfoldMatchesReference(t *testing.T) {
	for _, c := range windowCases {
		w := c.windows()
		for _, sv := range stridedViews(c.in) {
			x := make([]int32, sv.size)
			for i := range x {
				x[i] = int32(i)
			}
			// The value of each input element in row-major order.
			ordered := make([]int32, 0, sv.view.Size())
			eachIndex(c.in, func(idx []int) { ordered = append(ordered, x[offsetOf(sv.view, idx)]) })
			want := make([]int32, 0, len(ordered))
			c.eachWindowElement(func(_, in int) {
				if in < 0 {
					want = append(want, -7)
					return
				}
				want = append(want, ordered[in])
			})
			for _, workers := range []int{1, 3} {
				attrs := c.attrs
				attrs.Value = -7
				k := &backend.Kernel{Op: backend.OpUnfold, DType: backend.Int32, Shape: w, In: []backend.View{sv.view}, Attrs: attrs}
				if got := runOn[int32](t, workers, k, i32buf(x...)); !slices.Equal(got, want) {
					t.Fatalf("%v %s %+v, %d workers:\ngot  %v\nwant %v", c.in, sv.name, c.attrs, workers, got, want)
				}
			}
		}
	}
}

func TestFoldMatchesReference(t *testing.T) {
	for _, c := range windowCases {
		w := c.windows()
		for _, sv := range stridedViews(w) {
			y := make([]float32, sv.size)
			for i := range y {
				y[i] = float32(i%17 - 8)
			}
			n, _ := backend.NumElements(c.in)
			want := make([]float32, n)
			idxs := make([][]int, 0, sv.view.Size())
			eachIndex(w, func(idx []int) { idxs = append(idxs, slices.Clone(idx)) })
			c.eachWindowElement(func(win, in int) {
				if in >= 0 {
					want[in] += y[offsetOf(sv.view, idxs[win])]
				}
			})
			for _, workers := range []int{1, 3} {
				name := fmt.Sprintf("%v %s %+v, %d workers", c.in, sv.name, c.attrs, workers)
				k := &backend.Kernel{Op: backend.OpFold, DType: backend.Float32, Shape: c.in, In: []backend.View{sv.view}, Attrs: c.attrs}
				if got := runOn[float32](t, workers, k, f32buf(y...)); !slices.Equal(got, want) {
					t.Fatalf("%s:\ngot  %v\nwant %v", name, got, want)
				}
			}
		}
	}
}
