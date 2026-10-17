package cpu

import (
	"context"
	"fmt"
	"slices"
	"sort"

	"github.com/gsoultan/mimir/backend"
)

// segments describes an OpSegment: the offsets of its segments into the
// row list and the rows' inner elements.
type segments struct {
	n             int // segments
	offs, rows    []int32
	offAt, offStp int // offs[offAt + s*offStp] is offset s
	rowAt, rowStp int // rows[rowAt + j*rowStp] is entry j of the row list
	entries       int // entries of the row list
	limit         int // rows of the source
	srcAt, srcStp int // row r of the source starts at srcAt + r*srcStp
	width         int // elements of a row
	// inner holds the offset of each element of a row from its start, or
	// is nil when the elements lie next to each other.
	inner []int
}

// offset returns offset s, the start of segment s in the row list.
func (g *segments) offset(s int) int { return int(g.offs[g.offAt+s*g.offStp]) }

// row returns entry j of the row list.
func (g *segments) row(j int) int { return int(g.rows[g.rowAt+j*g.rowStp]) }

// runSegment combines rows of In[0] by segment; see backend.OpSegment.
// A segment combines its rows in the order the row list holds them, as
// every device does, so results do not depend on the number of workers.
func runSegment(ctx context.Context, b *Backend, k *backend.Kernel, out backend.Buffer, ins []backend.Buffer) error {
	vx, vo, vr := k.In[0], k.In[1], k.In[2]
	r := len(k.Shape)
	if r == 0 || len(vx.Shape) != r || len(vo.Shape) != 1 || len(vr.Shape) != 1 || vo.Shape[0] != k.Shape[0]+1 || !slices.Equal(vx.Shape[1:], k.Shape[1:]) {
		return fmt.Errorf("segments %v of rows %v into %v: %w", vo.Shape, vx.Shape, k.Shape, backend.ErrInternal)
	}
	offs, err := data[int32](ins[1])
	if err != nil {
		return err
	}
	rows, err := data[int32](ins[2])
	if err != nil {
		return err
	}
	g := &segments{
		n: k.Shape[0], offs: offs, rows: rows,
		offAt: vo.Offset, offStp: vo.Strides[0],
		rowAt: vr.Offset, rowStp: vr.Strides[0], entries: vr.Shape[0],
		limit: vx.Shape[0], srcAt: vx.Offset, srcStp: vx.Strides[0],
	}
	g.width, _ = backend.NumElements(k.Shape[1:])
	shape, views := backend.Coalesce(vx.Shape[1:], []backend.View{{Strides: vx.Strides[1:]}})
	if len(shape) > 1 || len(shape) == 1 && views[0].Strides[0] != 1 {
		g.inner = make([]int, g.width)
		c := newCursor(shape, views, 0)
		for d := range g.inner {
			g.inner[d] = c.offs[0]
			c.next()
		}
	}
	// Offsets rise from 0 to at most the length of the row list: then
	// offset s plus s rises with s, and splits the work of the segments,
	// their rows and the segments themselves, by binary search.
	prev := 0
	for s := range g.n + 1 {
		o := g.offset(s)
		if o < prev || o > g.entries {
			return fmt.Errorf("offset %d of segment %d after %d, in a row list of %d: %w", o, s, prev, g.entries, backend.ErrIndex)
		}
		prev = o
	}
	switch k.DType {
	case backend.Float32:
		return segmentInto[float32](ctx, b, k, g, out, ins[0], segmentNumber[float32])
	case backend.Int32:
		return segmentInto[int32](ctx, b, k, g, out, ins[0], segmentNumber[int32])
	case backend.BFloat16:
		return segmentInto[bf16](ctx, b, k, g, out, ins[0], segmentHalf[bf16])
	case backend.Float16:
		return segmentInto[f16](ctx, b, k, g, out, ins[0], segmentHalf[f16])
	}
	return fmt.Errorf("%s segments: %w", k.DType, backend.ErrDType)
}

// segmentInto writes the segments of src into out by combine, which
// writes one, splitting them between workers in pieces of about equal
// work. acc is a row of float32 combine may use.
func segmentInto[T elem](ctx context.Context, b *Backend, k *backend.Kernel, g *segments, outBuf, srcBuf backend.Buffer,
	combine func(g *segments, s int, out, src []T, acc []float32, c backend.Combine) error) error {
	out, err := data[T](outBuf)
	if err != nil {
		return err
	}
	src, err := data[T](srcBuf)
	if err != nil {
		return err
	}
	half := k.DType.Size() == 2
	work := g.offset(g.n) + g.n
	pieces := min(b.workers*chunksPerWorker, max(1, work*max(g.width, 1)/elementGrain))
	start := func(p int) int {
		at := p * work / pieces
		return sort.Search(g.n, func(s int) bool { return g.offset(s)+s >= at })
	}
	return b.parallelFor(ctx, pieces, 1, func(lo, hi int) error {
		var acc []float32
		if half {
			acc = make([]float32, g.width)
		}
		for s, end := start(lo), start(hi); s < end; s++ {
			if err := combine(g, s, out, src, acc, k.Attrs.Combine); err != nil {
				return err
			}
		}
		return nil
	})
}

// source returns the start of row j of the row list in the source,
// checking the row.
func (g *segments) source(j int) (int, error) {
	r := g.row(j)
	if r < 0 || r >= g.limit {
		return 0, fmt.Errorf("row %d of %d: %w", r, g.limit, backend.ErrIndex)
	}
	return g.srcAt + r*g.srcStp, nil
}

// segmentNumber combines segment s of src into its row of out.
func segmentNumber[T number](g *segments, s int, out, src []T, _ []float32, c backend.Combine) error {
	dst := out[s*g.width : (s+1)*g.width]
	lo, hi := g.offset(s), g.offset(s+1)
	if lo == hi {
		clear(dst)
		return nil
	}
	for j := lo; j < hi; j++ {
		at, err := g.source(j)
		if err != nil {
			return err
		}
		if g.inner != nil {
			for d, off := range g.inner {
				v := src[at+off]
				switch {
				case j == lo:
					dst[d] = v
				case c == backend.CombineSum:
					dst[d] += v
				default:
					dst[d] = max(dst[d], v)
				}
			}
			continue
		}
		row := src[at : at+g.width]
		switch {
		case j == lo:
			copy(dst, row)
		case c == backend.CombineSum:
			addInto(dst, row)
		default:
			maxInto(dst, row)
		}
	}
	return nil
}

// addInto adds row to dst, on the vector routines for float32.
func addInto[T number](dst, row []T) {
	if d, ok := any(dst).([]float32); ok {
		if n := vectorPart(len(d)); n > 0 {
			x := any(row).([]float32)
			rowVector.addVV(&d[0], &d[0], &x[0], n)
			dst, row = dst[n:], row[n:]
		}
	}
	row = row[:len(dst)]
	for d, v := range row {
		dst[d] += v
	}
}

// maxInto sets dst to the elementwise maximum of dst and row.
func maxInto[T number](dst, row []T) {
	row = row[:len(dst)]
	for d, v := range row {
		dst[d] = max(dst[d], v)
	}
}

// segmentHalf combines segment s of half src in float32, in acc, and
// rounds it into its row of out.
func segmentHalf[T elem](g *segments, s int, out, src []T, acc []float32, c backend.Combine) error {
	lo, hi := g.offset(s), g.offset(s+1)
	clear(acc)
	for j := lo; j < hi; j++ {
		at, err := g.source(j)
		if err != nil {
			return err
		}
		for d := range acc {
			off := d
			if g.inner != nil {
				off = g.inner[d]
			}
			v := conv[float32](src[at+off])
			switch {
			case j == lo:
				acc[d] = v
			case c == backend.CombineSum:
				acc[d] += v
			default:
				acc[d] = max(acc[d], v)
			}
		}
	}
	dst := out[s*g.width : (s+1)*g.width]
	for d, v := range acc {
		dst[d] = ofF32[T](v)
	}
	return nil
}
