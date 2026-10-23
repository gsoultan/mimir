package cpu

import (
	"context"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// Unfold and Fold walk the same geometry: for each plane, the elements of
// the leading dimensions, the windows split into rows along the last
// windowed dimension. A row holds the window positions along that dimension
// for one kernel offset and one position along every other dimension, and
// reads a strided run of one row of the input.

// windowRow is one row of the windows of a plane. Positions [lo, hi) of
// the row fall inside the input; position lo reads input element in, and
// position o sits at windows element win + o*winStep.
type windowRow struct {
	win, in, lo, hi int
}

// windowRows describes every row of a plane's windows.
type windowRows struct {
	rows    []windowRow
	n       int // positions per row
	inStep  int // input elements between successive positions
	winStep int // windows elements between successive positions
}

// planWindows lists the rows of the windows of one plane. spatial is the
// size of the windowed input dimensions and pos the window positions along
// each; inStrides are the strides of the windowed input dimensions, and
// winStrides of the [K..., O...] dimensions of the windows.
func planWindows(a backend.Attrs, spatial, pos, inStrides, winStrides []int) windowRows {
	s := len(a.Kernel)
	last := s - 1
	w := windowRows{n: pos[last], inStep: a.Stride[last] * inStrides[last], winStep: winStrides[2*s-1]}
	// The rows in the order of the windows: kernel offsets outermost, then
	// the positions along all but the last dimension.
	outer := slices.Concat(a.Kernel, pos[:last])
	count, _ := backend.NumElements(outer)
	w.rows = make([]windowRow, 0, count)
	idx := make([]int, len(outer))
	for range count {
		var r windowRow
		inside := true
		for d := range last {
			k, o := idx[d], idx[s+d]
			r.win += k*winStrides[d] + o*winStrides[s+d]
			i := o*a.Stride[d] - a.Pads[d][0] + k*a.Dilation[d]
			inside = inside && i >= 0 && i < spatial[d]
			r.in += i * inStrides[d]
		}
		k := idx[last]
		r.win += k * winStrides[last]
		// Position o reads input element o*stride + off of the row, which
		// must lie in [0, spatial).
		st, off := a.Stride[last], k*a.Dilation[last]-a.Pads[last][0]
		if inside {
			r.lo = min(max(ceilDiv(-off, st), 0), w.n)
			r.hi = min(max(ceilDiv(spatial[last]-off, st), r.lo), w.n)
			r.in += (r.lo*st + off) * inStrides[last]
		}
		w.rows = append(w.rows, r)
		for d := len(idx) - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < outer[d] {
				break
			}
			idx[d] = 0
		}
	}
	return w
}

// ceilDiv returns a/b rounded up, for b > 0.
func ceilDiv(a, b int) int {
	q := a / b
	if a%b != 0 && a > 0 {
		q++
	}
	return q
}

// unfold copies the sliding windows of the input into out; see
// backend.OpUnfold.
func unfold[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, x []T) error {
	v := k.In[0]
	s, r := len(k.Attrs.Kernel), len(v.Shape)
	lead, spatial, win := v.Shape[:r-s], v.Shape[r-s:], k.Shape[r-s:]
	w := planWindows(k.Attrs, spatial, win[s:], v.Strides[r-s:], backend.ContiguousStrides(win))
	per, _ := backend.NumElements(win)
	bases := batchOffsets(lead, v)
	fill := valueOf[T](k.Attrs.Value)
	return b.parallelFor(ctx, len(bases), max(1, elementGrain/max(per, 1)), func(lo, hi int) error {
		for p := lo; p < hi; p++ {
			plane := out[p*per:][:per]
			for _, row := range w.rows {
				dst := plane[row.win:][:w.n]
				for i := range dst[:row.lo] {
					dst[i] = fill
				}
				for i := range dst[row.hi:] {
					dst[row.hi+i] = fill
				}
				if row.lo == row.hi {
					continue
				}
				src := bases[p] + row.in
				if w.inStep == 1 {
					copy(dst[row.lo:row.hi], x[src:])
					continue
				}
				for i := row.lo; i < row.hi; i++ {
					dst[i] = x[src]
					src += w.inStep
				}
			}
		}
		return nil
	})
}

// fold sums the windows of the input into out; see backend.OpFold. Half
// elements are summed in float32 and rounded once.
func fold[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, x []T) error {
	v := k.In[0]
	s, r := len(k.Attrs.Kernel), len(k.Shape)
	lead, spatial := k.Shape[:r-s], k.Shape[r-s:]
	w := planWindows(k.Attrs, spatial, v.Shape[r:], backend.ContiguousStrides(spatial), v.Strides[r-s:])
	per, _ := backend.NumElements(spatial)
	bases := batchOffsets(lead, v)
	half := dtypeOf[T]().Size() == 2
	return b.parallelFor(ctx, len(bases), max(1, elementGrain/max(len(w.rows)*w.n, 1)), func(lo, hi int) error {
		var sums []float32
		if half {
			sums = make([]float32, per)
		}
		for p := lo; p < hi; p++ {
			switch plane := any(out[p*per:][:per]).(type) {
			case []float32:
				foldRows(plane, x, bases[p], w)
			case []int32:
				foldRows(plane, x, bases[p], w)
			default:
				foldRows(sums, x, bases[p], w)
				for i, v := range sums {
					out[p*per+i] = ofF32[T](v)
				}
			}
		}
		return nil
	})
}

// foldRows sums the rows of the windows of x at base into plane.
func foldRows[A number, S elem](plane []A, x []S, base int, w windowRows) {
	clear(plane)
	if p, ok := any(plane).([]float32); ok {
		if f, ok := any(x).([]float32); ok {
			foldRowsF32(p, f, base, w)
			return
		}
	}
	for _, row := range w.rows {
		src, dst := base+row.win+row.lo*w.winStep, row.in
		for range row.hi - row.lo {
			plane[dst] += conv[A](x[src])
			src += w.winStep
			dst += w.inStep
		}
	}
}

// foldRowsF32 is foldRows for float32 windows and plane, whose rows add
// as slices when both run contiguously, as the windows of an unstrided
// convolution do, without a conversion call per element.
func foldRowsF32(plane, x []float32, base int, w windowRows) {
	for _, row := range w.rows {
		n := row.hi - row.lo
		if n <= 0 {
			continue
		}
		src, dst := base+row.win+row.lo*w.winStep, row.in
		if w.winStep == 1 && w.inStep == 1 {
			d := plane[dst:][:n]
			for i, v := range x[src:][:n] {
				d[i] += v
			}
			continue
		}
		for range n {
			plane[dst] += x[src]
			src += w.winStep
			dst += w.inStep
		}
	}
}
