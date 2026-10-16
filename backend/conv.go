package backend

import (
	"fmt"
	"slices"
)

// Convolver is a Backend that runs the convolution kernels OpConv,
// OpConvWeight and OpConvInput. The scheduler runs a convolution on such
// a backend as one of them instead of as a product of the windows that
// OpUnfold copies out, or a product that OpFold folds.
type Convolver interface {
	Backend
	// ConvParts returns the number of parts into which a weight
	// gradient, the OpConvWeight kernel k of one part, should be split
	// so that the device sums them in parallel; the scheduler adds the
	// parts together after. It returns 1 to sum it in one kernel.
	ConvParts(k *Kernel) int
}

// Conv is the geometry of a convolution kernel: an image of N items of
// G groups of C channels over the windowed dimensions Image, read
// through windows of Window elements at Positions along each dimension,
// and the other dimension M of the kernel's product (see OpConv).
type Conv struct {
	N, G, C, M int
	Image      []int
	Positions  []int
	Window     int // elements of a window: the product of Attrs.Kernel
	Parts      int // OpConvWeight: the parts S its output sums separately
}

// Rows returns C times Window, the elements of a group's windows at one
// position: the inner dimension of OpConv's product.
func (c Conv) Rows() int { return c.C * c.Window }

// Cols returns the number of window positions.
func (c Conv) Cols() int {
	n, _ := NumElements(c.Positions)
	return n
}

// Part returns the pairs (n, o), numbered in row-major order, that part s
// of an OpConvWeight kernel sums: [lo, hi). The parts split the pairs
// into Parts runs of equal length, but for a shorter last one.
func (c Conv) Part(s int) (lo, hi int) {
	pairs := c.N * c.Cols()
	size := (pairs + c.Parts - 1) / c.Parts
	lo = min(s*size, pairs)
	return lo, min(lo+size, pairs)
}

// ConvOf returns the geometry of the convolution kernel k, or an error
// wrapping ErrInternal if its shapes and window do not fit together.
func ConvOf(k *Kernel) (Conv, error) {
	bad := func(format string, args ...any) (Conv, error) {
		return Conv{}, fmt.Errorf("%s: %s: %w", k.Op, fmt.Sprintf(format, args...), ErrInternal)
	}
	if k.DType != Float32 {
		return Conv{}, fmt.Errorf("%s of %s: %w", k.Op, k.DType, ErrDType)
	}
	a := k.Attrs
	s := len(a.Kernel)
	if s == 0 || len(a.Stride) != s || len(a.Dilation) != s || len(a.Pads) != s || len(k.In) < 2 {
		return bad("window %+v over %d inputs", a, len(k.In))
	}
	if k.Op != OpConv && len(k.In) != 2 {
		return bad("%d inputs", len(k.In))
	}
	var image []int // [N, G*C, D...]
	x, y := k.In[0].Shape, k.In[1].Shape
	switch k.Op {
	case OpConv, OpConvWeight:
		image = y
	case OpConvInput:
		image = k.Shape
	default:
		return bad("not a convolution")
	}
	if len(image) != 2+s || len(x) != 4 || len(k.Shape) != 4 && k.Op != OpConvInput {
		return bad("shapes %v and %v for %v", x, y, k.Shape)
	}
	c := Conv{N: image[0], G: x[1], Image: image[2:], Window: 1, Parts: 1}
	if c.G == 0 || image[1]%c.G != 0 {
		return bad("%d channels in %d groups", image[1], c.G)
	}
	c.C = image[1] / c.G
	for d := range s {
		kd, st, dil, p := a.Kernel[d], a.Stride[d], a.Dilation[d], a.Pads[d]
		if kd < 1 || st < 1 || dil < 1 || p[0] < 0 || p[1] < 0 {
			return bad("window %+v", a)
		}
		span := (kd-1)*dil + 1
		if c.Image[d]+p[0]+p[1] < span {
			return bad("window %+v larger than %v", a, c.Image)
		}
		c.Positions = append(c.Positions, (c.Image[d]+p[0]+p[1]-span)/st+1)
		c.Window *= kd
	}
	rows, cols := c.Rows(), c.Cols()
	eq := func(got []int, want ...int) bool { return slices.Equal(got, want) }
	switch k.Op {
	case OpConv:
		c.M = x[2]
		if !eq(x, c.N, c.G, c.M, rows) || !eq(k.Shape, c.N, c.G, c.M, cols) {
			return bad("%v times the windows of %v into %v", x, y, k.Shape)
		}
	case OpConvWeight:
		c.M, c.Parts = x[2], k.Shape[0]
		if c.Parts < 1 || !eq(x, c.N, c.G, c.M, cols) || !eq(k.Shape, c.Parts, c.G, c.M, rows) {
			return bad("%v times the windows of %v into %v", x, y, k.Shape)
		}
	case OpConvInput:
		c.M = x[3]
		if !eq(x, c.N, c.G, rows, c.M) || !eq(y, c.N, c.G, c.M, cols) {
			return bad("%v times %v folded into %v", x, y, k.Shape)
		}
	}
	return c, nil
}
