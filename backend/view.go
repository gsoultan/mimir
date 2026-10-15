package backend

import "slices"

// View describes how a logical tensor of Shape is laid out in a buffer:
// element (i0, i1, ...) lives at Offset + i0*Strides[0] + i1*Strides[1] + ...
// Strides are counted in elements. A stride of 0 repeats an element
// (broadcast) and a negative stride walks backwards (flip).
type View struct {
	Shape   []int
	Strides []int
	Offset  int
}

// Contiguous returns the row-major view of shape starting at offset 0.
func Contiguous(shape []int) View {
	return View{Shape: slices.Clone(shape), Strides: ContiguousStrides(shape)}
}

// ContiguousStrides returns row-major strides for shape.
func ContiguousStrides(shape []int) []int {
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return strides
}

// Size returns the number of elements in the view.
func (v View) Size() int {
	n, _ := NumElements(v.Shape)
	return n
}

// IsContiguous reports whether the view's elements are laid out row-major
// without gaps, starting at Offset. Dimensions of size 1 may have any stride.
func (v View) IsContiguous() bool {
	want := 1
	for i := len(v.Shape) - 1; i >= 0; i-- {
		if v.Shape[i] == 1 {
			continue
		}
		if v.Shape[i] == 0 {
			return true
		}
		if v.Strides[i] != want {
			return false
		}
		want *= v.Shape[i]
	}
	return true
}

// Permute reorders the dimensions: output dimension i is input dimension perm[i].
func (v View) Permute(perm []int) View {
	out := View{Shape: make([]int, len(perm)), Strides: make([]int, len(perm)), Offset: v.Offset}
	for i, p := range perm {
		out.Shape[i] = v.Shape[p]
		out.Strides[i] = v.Strides[p]
	}
	return out
}

// Expand broadcasts dimensions of size 1 to the sizes in shape, which must
// have the same rank.
func (v View) Expand(shape []int) View {
	out := View{Shape: slices.Clone(shape), Strides: slices.Clone(v.Strides), Offset: v.Offset}
	for i := range shape {
		if v.Shape[i] == 1 && shape[i] != 1 {
			out.Strides[i] = 0
		}
	}
	return out
}

// Shrink keeps the half-open range [r[0], r[1]) of each dimension.
func (v View) Shrink(ranges [][2]int) View {
	out := View{Shape: make([]int, len(ranges)), Strides: slices.Clone(v.Strides), Offset: v.Offset}
	for i, r := range ranges {
		out.Shape[i] = r[1] - r[0]
		if out.Shape[i] > 0 {
			out.Offset += r[0] * v.Strides[i]
		}
	}
	return out
}

// Flip reverses the order of elements along axes.
func (v View) Flip(axes []int) View {
	out := View{Shape: slices.Clone(v.Shape), Strides: slices.Clone(v.Strides), Offset: v.Offset}
	for _, a := range axes {
		if out.Shape[a] > 0 {
			out.Offset += (out.Shape[a] - 1) * out.Strides[a]
		}
		out.Strides[a] = -out.Strides[a]
	}
	return out
}

// Reshape returns the view with a new shape of the same size. It reports
// false when the layout cannot be expressed without copying.
func (v View) Reshape(shape []int) (View, bool) {
	if !v.IsContiguous() {
		return View{}, false
	}
	out := Contiguous(shape)
	out.Offset = v.Offset
	return out, true
}
