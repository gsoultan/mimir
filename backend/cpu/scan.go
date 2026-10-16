package cpu

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// A scan runs along lines: the elements that differ only in their
// position along the axis. When the axis is the output's innermost, each
// line is a contiguous run of the output and a worker scans whole lines.
// Otherwise the elements of a line lie inner apart and a worker scans
// rows: up to rowChunk lines adjacent in the output, one position along
// the axis at a time, so that input and output stream in order however
// long the axis is.
//
// When there are too few lines or rows to share between the workers, a
// long axis is cut into segments. A first pass combines every segment
// but the last; a second scans each segment from the combination of the
// segments before it. The segments depend only on the shape, so a scan's
// result does not depend on the number of workers.
//
// Float elements combine in float64, rounded to float32 as they are
// stored, like PyTorch's scans on the CPU; Int32 sums wrap.

// scanSegment is the number of elements a segment of a split scan covers.
const scanSegment = 1 << 14

// scanUnits is the number of lines or row chunks below which a long axis
// is split into segments.
const scanUnits = 16

// scanPlan is the plan of a scan.
type scanPlan struct {
	lines   backend.View // the first element of each line or row, in the input
	outs    []int        // the strides of lines in the output; offset 0
	length  int          // positions along the axis
	inStep  int          // input stride along the axis
	outStep int          // output stride along the axis
	rowLen  int          // lines a row scans together; 1 when the axis is innermost
	rowIn   int          // input stride between the lines of a row; theirs in the output is 1
	chunks  int          // chunks of at most rowChunk lines per row
	segLen  int          // positions per segment
	segs    int          // segments per line
}

func planScan(shape []int, v backend.View, axis int) scanPlan {
	out := backend.ContiguousStrides(shape)
	rest := slices.Delete(slices.Clone(shape), axis, axis+1)
	in := backend.View{Strides: slices.Delete(slices.Clone(v.Strides), axis, axis+1), Offset: v.Offset}
	o := backend.View{Strides: slices.Delete(slices.Clone(out), axis, axis+1)}
	rest, views := backend.Coalesce(rest, []backend.View{in, o})
	p := scanPlan{
		lines:   views[0],
		outs:    views[1].Strides,
		length:  shape[axis],
		inStep:  v.Strides[axis],
		outStep: out[axis],
		rowLen:  1,
		chunks:  1,
	}
	if r := len(rest) - 1; r >= 0 && p.outStep != 1 {
		// The innermost output dimension, after the axis, has stride 1
		// and coalesces into the last dimension of the rest.
		p.rowLen, p.rowIn = rest[r], p.lines.Strides[r]
		p.lines.Shape, p.lines.Strides, p.outs = rest[:r], p.lines.Strides[:r], p.outs[:r]
		p.chunks = (p.rowLen + rowChunk - 1) / rowChunk
	}
	p.segLen, p.segs = p.length, 1
	width := min(p.rowLen, rowChunk)
	if seg := max(scanSegment/width, 1); p.units() < scanUnits && p.length >= 2*seg {
		p.segLen = seg
		p.segs = (p.length + seg - 1) / seg
	}
	return p
}

// units returns the number of lines, or of row chunks, the scan runs.
func (p scanPlan) units() int {
	n, _ := backend.NumElements(p.lines.Shape)
	return n * p.chunks
}

// runScan runs a scan of in into out. Half elements are scanned as
// float32 values and rounded once.
func runScan(ctx context.Context, b *Backend, k *backend.Kernel, out, in backend.Buffer) error {
	if k.Attrs.Combine < backend.CombineSum || k.Attrs.Combine > backend.CombineLogSumExp {
		return fmt.Errorf("no combine %s: %w", k.Attrs.Combine, backend.ErrInternal)
	}
	axis := k.Attrs.Axis
	if axis < 0 || axis >= len(k.Shape) || len(k.In[0].Shape) != len(k.Shape) {
		return fmt.Errorf("scan along axis %d of %v: %w", axis, k.Shape, backend.ErrInternal)
	}
	switch in := in.(type) {
	case *buffer[float32]:
		o, err := data[float32](out)
		if err != nil {
			return err
		}
		return scan[float32, float64](ctx, b, k, o, in.data, k.In[0])
	case *buffer[int32]:
		if k.Attrs.Combine == backend.CombineLogSumExp {
			return fmt.Errorf("%s of int32: %w", k.Attrs.Combine, backend.ErrDType)
		}
		o, err := data[int32](out)
		if err != nil {
			return err
		}
		return scan[int32, int32](ctx, b, k, o, in.data, k.In[0])
	case *buffer[bf16]:
		return scanHalf(ctx, b, k, out, in.data)
	case *buffer[f16]:
		return scanHalf(ctx, b, k, out, in.data)
	default:
		return fmt.Errorf("foreign buffer %T: %w", in, backend.ErrDType)
	}
}

// scanHalf scans half elements: it reads them as float32 values into a
// contiguous buffer, scans that in place and rounds the results into out.
func scanHalf[S bf16 | f16](ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, in []S) error {
	out, err := data[S](outBuf)
	if err != nil {
		return err
	}
	vals := make([]float32, len(out))
	if err := mapStrided(ctx, b, k.Shape, k.In[0], len(out), func(i, off int) { vals[i] = conv[float32](in[off]) }); err != nil {
		return err
	}
	if err := scan[float32, float64](ctx, b, k, vals, vals, backend.Contiguous(k.Shape)); err != nil {
		return err
	}
	return b.parallelFor(ctx, len(out), elementGrain, func(lo, hi int) error {
		for i := lo; i < hi; i++ {
			out[i] = ofF32[S](vals[i])
		}
		return nil
	})
}

// scan computes the scan of k, reading in through v, into out with
// accumulators of type A. out may be in itself when v is contiguous:
// each element is read before the one at its index is written.
func scan[T number, A float64 | int32](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T, v backend.View) error {
	if len(out) == 0 {
		return nil
	}
	p := planScan(k.Shape, v, k.Attrs.Axis)
	c := k.Attrs.Combine
	width := min(p.rowLen, rowChunk)

	// The first pass combines each segment but the last, and then the
	// segments of each unit up to each one, the carry into the next.
	var carries []A
	if p.segs > 1 {
		units, per := p.units(), (p.segs-1)*width
		carries = make([]A, units*per)
		err := forScan(ctx, b, p, p.segs-1, func(u, s, src, _ int, acc, aux []A) {
			t := carries[u*per+s*width:][:len(acc)]
			if p.rowLen == 1 {
				t[0] = combineLine[T, A](c, in, src, p.inStep, p.segLen)
				return
			}
			combineRows(c, t, aux, in, src, p.inStep, p.rowIn, p.segLen)
		})
		if err != nil {
			return err
		}
		err = b.parallelFor(ctx, units, max(1, elementGrain/per), func(lo, hi int) error {
			for u := lo; u < hi; u++ {
				t := carries[u*per : (u+1)*per]
				for j := width; j < len(t); j++ {
					t[j] = combine(c, t[j-width], t[j])
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
	}

	// The second scans every segment from its carry.
	return forScan(ctx, b, p, p.segs, func(u, s, src, dst int, acc, aux []A) {
		positions := min(p.segLen, p.length-s*p.segLen)
		if s > 0 {
			copy(acc, carries[(u*(p.segs-1)+s-1)*width:])
		}
		if p.rowLen == 1 {
			scanLine(c, out, in, src, p.inStep, dst, positions, acc[0], s > 0)
			return
		}
		scanRows(c, out, in, src, p.inStep, p.rowIn, dst, p.outStep, positions, acc, aux, s > 0)
	})
}

// forScan calls fn(u, s, src, dst, acc, aux) for the segments s < segs of
// every unit u of the plan p, in parallel: src and dst are the input and
// output offsets of the segment's first element, and acc and aux hold
// two accumulators for each line the unit scans, for fn to use.
func forScan[A float64 | int32](ctx context.Context, b *Backend, p scanPlan, segs int, fn func(u, s, src, dst int, acc, aux []A)) error {
	width := min(p.rowLen, rowChunk)
	outs := backend.View{Shape: p.lines.Shape, Strides: p.outs}
	// Lines along one dimension, the common case, are found without a
	// cursor, which costs as much as scanning a short line.
	var inLine, outLine int
	linear := len(p.lines.Shape) <= 1
	if len(p.lines.Shape) == 1 {
		inLine, outLine = p.lines.Strides[0], p.outs[0]
	}
	return b.parallelFor(ctx, p.units()*segs, max(1, elementGrain/max(p.segLen*width, 1)), func(lo, hi int) error {
		scratch := make([]A, 2*width)
		at := lo / segs / p.chunks
		var c *cursor
		if !linear {
			c = newCursor(p.lines.Shape, []backend.View{p.lines, outs}, at)
		}
		for item := lo; item < hi; item++ {
			u := item / segs
			line := u / p.chunks
			var src, dst int
			if linear {
				src, dst = p.lines.Offset+line*inLine, line*outLine
			} else {
				for ; at < line; at++ {
					c.next()
				}
				src, dst = c.offs[0], c.offs[1]
			}
			s := item % segs
			j := u % p.chunks * rowChunk
			n := min(rowChunk, p.rowLen-j)
			src += j*p.rowIn + s*p.segLen*p.inStep
			dst += j + s*p.segLen*p.outStep
			fn(u, s, src, dst, scratch[:n], scratch[width:width+n])
		}
		return nil
	})
}

// combine returns a and b combined by c.
func combine[A float64 | int32](c backend.Combine, a, b A) A {
	switch c {
	case backend.CombineSum:
		return a + b
	case backend.CombineMax:
		return max(a, b)
	default:
		return A(logAddExp(float64(a), float64(b)))
	}
}

// logAddExp returns log(exp(a) + exp(b)): exactly the larger when the
// other is -Inf or the larger is +Inf, and NaN if either is.
func logAddExp(a, b float64) float64 {
	hi, lo := max(a, b), min(a, b)
	if math.IsInf(lo, -1) || math.IsInf(hi, 1) {
		return hi
	}
	return hi + math.Log1p(math.Exp(lo-hi))
}

// A log-sum-exp scan keeps, for each line, the largest element so far,
// m, and the sum s of exp(x - m) over the elements so far, which is at
// least 1; the scan's value is m + log(s). Each element adds to s, so
// that the exponentials and logarithms of successive elements do not wait
// for each other, and a new maximum rescales s.

// lseStep returns m and s after the element x.
func lseStep(m, s, x float64) (float64, float64) {
	switch {
	case x < m:
		return m, s + math.Exp(x-m)
	case x > m:
		return x, s*math.Exp(m-x) + 1
	case x == m: // also equal infinities, whose difference is NaN
		return m, s + 1
	default:
		return math.NaN(), s
	}
}

// combineLine returns the combination of the n >= 1 elements of in from
// src on, step apart.
func combineLine[T number, A float64 | int32](c backend.Combine, in []T, src, step, n int) A {
	switch {
	case c == backend.CombineSum && step == 1:
		// Four running sums let the additions overlap.
		var a0, a1, a2, a3 A
		x := in[src : src+n]
		for len(x) >= 4 {
			a0 += A(x[0])
			a1 += A(x[1])
			a2 += A(x[2])
			a3 += A(x[3])
			x = x[4:]
		}
		for _, v := range x {
			a0 += A(v)
		}
		return (a0 + a1) + (a2 + a3)
	case c == backend.CombineLogSumExp:
		m, s := float64(in[src]), 1.0
		for range n - 1 {
			src += step
			m, s = lseStep(m, s, float64(in[src]))
		}
		return A(m + math.Log(s))
	}
	acc := A(in[src])
	for range n - 1 {
		src += step
		acc = combine(c, acc, A(in[src]))
	}
	return acc
}

// combineRows sets acc[j] to the combination of the n >= 1 elements of
// line j of a row, the lines rowIn apart and their elements step apart
// from src on. aux is scratch of len(acc).
func combineRows[T number, A float64 | int32](c backend.Combine, acc, aux []A, in []T, src, step, rowIn, n int) {
	for j := range acc {
		acc[j], aux[j] = A(in[src+j*rowIn]), 1
	}
	for range n - 1 {
		src += step
		switch {
		case c == backend.CombineSum && rowIn == 1:
			for j, x := range in[src : src+len(acc)] {
				acc[j] += A(x)
			}
		case c == backend.CombineMax && rowIn == 1:
			for j, x := range in[src : src+len(acc)] {
				acc[j] = max(acc[j], A(x))
			}
		case c == backend.CombineLogSumExp:
			for j := range acc {
				m, s := lseStep(float64(acc[j]), float64(aux[j]), float64(in[src+j*rowIn]))
				acc[j], aux[j] = A(m), A(s)
			}
		default:
			for j := range acc {
				acc[j] = combine(c, acc[j], A(in[src+j*rowIn]))
			}
		}
	}
	if c == backend.CombineLogSumExp {
		for j := range acc {
			acc[j] = A(float64(acc[j]) + math.Log(float64(aux[j])))
		}
	}
}

// scanLine scans the n elements of in from src on, step apart, into out
// from dst on. With carry, acc combines before the first element;
// without, the first element starts the scan.
func scanLine[T number, A float64 | int32](c backend.Combine, out, in []T, src, step, dst, n int, acc A, carry bool) {
	if n == 0 {
		return
	}
	if !carry {
		acc = A(in[src])
		out[dst] = T(acc)
		src += step
		dst++
		n--
	}
	o := out[dst : dst+n]
	switch c {
	case backend.CombineSum:
		if step == 1 {
			sumLine(o, in[src:src+n], acc)
			return
		}
		for i := range o {
			acc += A(in[src])
			o[i] = T(acc)
			src += step
		}
	case backend.CombineMax:
		if step == 1 {
			maxLine(o, in[src:src+n], acc)
			return
		}
		for i := range o {
			acc = max(acc, A(in[src]))
			o[i] = T(acc)
			src += step
		}
	default:
		m, s := float64(acc), 1.0
		for i := range o {
			m, s = lseStep(m, s, float64(in[src]))
			o[i] = T(m + math.Log(s))
			src += step
		}
	}
}

// sumLine writes the running sums of x, after acc, into o. It sums four
// elements apart from acc and then adds acc to each partial sum, so that
// each element waits for one addition instead of the whole line's.
func sumLine[T number, A float64 | int32](o, x []T, acc A) {
	o = o[:len(x)]
	i := 0
	for ; i+4 <= len(x); i += 4 {
		p0 := A(x[i])
		p1 := p0 + A(x[i+1])
		p2 := p1 + A(x[i+2])
		p3 := p2 + A(x[i+3])
		o[i] = T(acc + p0)
		o[i+1] = T(acc + p1)
		o[i+2] = T(acc + p2)
		o[i+3] = T(acc + p3)
		acc += p3
	}
	for ; i < len(x); i++ {
		acc += A(x[i])
		o[i] = T(acc)
	}
}

// maxLine writes the running maxima of x, after acc, into o, four
// elements at a time like sumLine.
func maxLine[T number, A float64 | int32](o, x []T, acc A) {
	o = o[:len(x)]
	i := 0
	for ; i+4 <= len(x); i += 4 {
		p0 := A(x[i])
		p1 := max(p0, A(x[i+1]))
		p2 := max(p1, A(x[i+2]))
		p3 := max(p2, A(x[i+3]))
		o[i] = T(max(acc, p0))
		o[i+1] = T(max(acc, p1))
		o[i+2] = T(max(acc, p2))
		o[i+3] = T(max(acc, p3))
		acc = max(acc, p3)
	}
	for ; i < len(x); i++ {
		acc = max(acc, A(x[i]))
		o[i] = T(acc)
	}
}

// scanRows scans n positions of the len(acc) lines of a row: the lines
// lie rowIn apart in the input from src on and 1 apart in the output
// from dst on, and their elements step and outStep apart. With carry,
// acc[j] combines before the first element of line j; without, the first
// elements start the scan. aux is scratch of len(acc).
func scanRows[T number, A float64 | int32](c backend.Combine, out, in []T, src, step, rowIn, dst, outStep, n int, acc, aux []A, carry bool) {
	if n == 0 {
		return
	}
	for j := range aux {
		aux[j] = 1
	}
	if !carry {
		for j := range acc {
			acc[j] = A(in[src+j*rowIn])
			out[dst+j] = T(acc[j])
		}
		src += step
		dst += outStep
		n--
	}
	for range n {
		o := out[dst : dst+len(acc)]
		switch {
		case c == backend.CombineSum && rowIn == 1:
			for j, x := range in[src : src+len(acc)] {
				acc[j] += A(x)
				o[j] = T(acc[j])
			}
		case c == backend.CombineMax && rowIn == 1:
			for j, x := range in[src : src+len(acc)] {
				acc[j] = max(acc[j], A(x))
				o[j] = T(acc[j])
			}
		case c == backend.CombineLogSumExp:
			for j := range acc {
				m, s := lseStep(float64(acc[j]), float64(aux[j]), float64(in[src+j*rowIn]))
				acc[j], aux[j] = A(m), A(s)
				o[j] = T(m + math.Log(s))
			}
		default:
			for j := range acc {
				acc[j] = combine(c, acc[j], A(in[src+j*rowIn]))
				o[j] = T(acc[j])
			}
		}
		src += step
		dst += outStep
	}
}
