package cpu

import (
	"context"
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// Pools reports that the backend runs OpMaxPool and OpMaxPoolGrad; see
// backend.Pooler.
func (b *Backend) Pools() bool { return true }

// poolPlan is the geometry of a pooling kernel: the rows of its windows
// over the input, as planWindows lists them, with row.win the offset of
// the row's first window position in a plane of the output, whose
// positions are contiguous.
type poolPlan struct {
	lead, spatial, pos []int
	per                int // output elements per plane
	rows               windowRows
}

func planPool(a backend.Attrs, lead, spatial, pos, inStrides []int) poolPlan {
	s := len(a.Kernel)
	per, _ := backend.NumElements(pos)
	// The kernel offsets take stride 0: a row's win offset is then its
	// first position's offset in the output plane.
	winStrides := slices.Concat(make([]int, s), backend.ContiguousStrides(pos))
	return poolPlan{lead: lead, spatial: spatial, pos: pos, per: per, rows: planWindows(a, spatial, pos, inStrides, winStrides)}
}

// A window's maximum is found on the float32 bit patterns mapped to
// int32 keys that order as the floats do: the compiler selects the larger
// key without a branch, where a branch on random data mispredicts half
// the time, and where the max builtin also orders the zeros. A positive
// NaN maps above +Inf, so it propagates as OpReduceMax's does. -0 takes
// the key of +0, so that the two tie as they compare, and the gradient of
// a window with both goes to whichever comes first.
func poolKey(f float32) int32 {
	b := math.Float32bits(f)
	if b == 1<<31 {
		b = 0
	}
	i := int32(b)
	return i ^ (i>>31)&math.MaxInt32
}

// poolValue inverts poolKey.
func poolValue(k int32) float32 {
	return math.Float32frombits(uint32(k ^ (k>>31)&math.MaxInt32))
}

// poolMaxRows sets keys, the keys of one plane of an OpMaxPool output, to
// those of the largest element of each window of x at base.
func poolMaxRows(keys []int32, x []float32, base int, w windowRows) {
	for i := range keys {
		keys[i] = math.MinInt32
	}
	for _, row := range w.rows {
		if row.lo == row.hi {
			continue
		}
		dst := keys[row.win:][:w.n][row.lo:row.hi]
		src := base + row.in
		switch w.inStep {
		case 1:
			for i, v := range x[src : src+len(dst)] {
				dst[i] = max(dst[i], poolKey(v))
			}
		case 2:
			src := x[src : src+2*len(dst)-1]
			for i := range dst {
				dst[i] = max(dst[i], poolKey(src[2*i]))
			}
		default:
			for i := range dst {
				dst[i] = max(dst[i], poolKey(x[src]))
				src += w.inStep
			}
		}
	}
}

// runPool runs OpMaxPool and OpMaxPoolGrad.
func runPool(ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, ins []backend.Buffer) error {
	out, err := data[float32](outBuf)
	if err != nil {
		return err
	}
	x, err := data[float32](ins[0])
	if err != nil {
		return err
	}
	v := k.In[0]
	s, r := len(k.Attrs.Kernel), len(v.Shape)
	lead, spatial := v.Shape[:r-s], v.Shape[r-s:]
	if pairs2x2(k) {
		var g []float32
		if k.Op == backend.OpMaxPoolGrad {
			if g, err = data[float32](ins[1]); err != nil {
				return err
			}
		}
		return pool2x2(ctx, b, k, out, x, g)
	}
	bases := batchOffsets(lead, v)
	if k.Op == backend.OpMaxPool {
		p := planPool(k.Attrs, lead, spatial, k.Shape[r-s:], v.Strides[r-s:])
		return b.parallelFor(ctx, len(bases), max(1, elementGrain/max(len(p.rows.rows)*p.rows.n, 1)), func(lo, hi int) error {
			keys := make([]int32, p.per)
			for i := lo; i < hi; i++ {
				poolMaxRows(keys, x, bases[i], p.rows)
				for j, k := range keys {
					out[i*p.per+j] = poolValue(k)
				}
			}
			return nil
		})
	}
	g, err := data[float32](ins[1])
	if err != nil {
		return err
	}
	vg := k.In[1]
	pos := vg.Shape[len(lead):]
	// The windows over x, read through its view, and over the gradient,
	// written contiguously: the same rows, offset differently.
	px := planPool(k.Attrs, lead, spatial, pos, v.Strides[r-s:])
	pd := planPool(k.Attrs, lead, spatial, pos, backend.ContiguousStrides(spatial))
	inPer, _ := backend.NumElements(spatial)
	gBases := batchOffsets(lead, vg)
	gView := backend.View{Shape: pos, Strides: vg.Strides[len(lead):]}
	gContiguous := slices.Equal(gView.Strides, backend.ContiguousStrides(pos))
	return b.parallelFor(ctx, len(bases), max(1, elementGrain/max(len(px.rows.rows)*px.rows.n, 1)), func(lo, hi int) error {
		maxes := make([]int32, px.per)
		done := make([]bool, px.per)
		var gp []float32
		if !gContiguous {
			gp = make([]float32, px.per)
		}
		for i := lo; i < hi; i++ {
			poolMaxRows(maxes, x, bases[i], px.rows)
			clear(done)
			plane := out[i*inPer:][:inPer]
			clear(plane)
			grad := gp
			if gContiguous {
				grad = g[gBases[i]:][:px.per]
			} else {
				c := newCursor(pos, []backend.View{{Shape: pos, Strides: gView.Strides, Offset: gBases[i]}}, 0)
				for j := range gp {
					gp[j] = g[c.offs[0]]
					c.next()
				}
			}
			for j, row := range px.rows.rows {
				if row.lo == row.hi {
					continue
				}
				src, dst := bases[i]+row.in, pd.rows.rows[j].in
				m, d, gr := maxes[row.win+row.lo:row.win+row.hi], done[row.win+row.lo:row.win+row.hi], grad[row.win+row.lo:row.win+row.hi]
				// One element in a window matches, so a branch on the
				// match would mispredict; the gradient is multiplied by
				// whether it matched instead.
				for o := range m {
					hit := poolKey(x[src]) == m[o] && !d[o]
					plane[dst] += gr[o] * boolTo[float32](hit)
					d[o] = d[o] || hit
					src += px.rows.inStep
					dst += pd.rows.inStep
				}
			}
		}
		return nil
	})
}

// pairs2x2 reports whether k pools 2×2 windows two apart over contiguous
// planes of even height and width, as most pooling layers do, which
// pool2x2 runs in long loops over whole chunks of planes rather than a
// loop per short row of windows.
func pairs2x2(k *backend.Kernel) bool {
	a := k.Attrs
	if len(a.Kernel) != 2 || a.Kernel[0] != 2 || a.Kernel[1] != 2 || a.Stride[0] != 2 || a.Stride[1] != 2 || a.Dilation[0] != 1 || a.Dilation[1] != 1 || a.Pads[0] != [2]int{} || a.Pads[1] != [2]int{} {
		return false
	}
	for _, v := range k.In {
		if !slices.Equal(v.Strides, backend.ContiguousStrides(v.Shape)) {
			return false
		}
	}
	r := len(k.In[0].Shape)
	return k.In[0].Shape[r-2]%2 == 0 && k.In[0].Shape[r-1]%2 == 0
}

// poolMax2x2 and poolGrad2x2 are the vector routines of pool2x2 for pairs
// of rows w wide, w even, the rows of a pair w elements apart and the
// pairs 2w apart, with w/2 outputs per pair; nil where the CPU has none,
// so that the Go loops run instead.
var (
	poolMax2x2  func(out, x *float32, w, pairs int)
	poolGrad2x2 func(dx, x, g *float32, w, pairs int)
)

// pool2x2 runs OpMaxPool or OpMaxPoolGrad over 2×2 windows two apart. The
// planes are contiguous, so a block of them is one [rows, W] matrix
// whose rows pair up within their planes, and a block small enough for
// its keys to stay in cache goes through in turn.
func pool2x2(ctx context.Context, b *Backend, k *backend.Kernel, out, x, g []float32) error {
	v := k.In[0]
	r := len(v.Shape)
	h, w := v.Shape[r-2], v.Shape[r-1]
	oh, ow := h/2, w/2
	planes, _ := backend.NumElements(v.Shape[:r-2])
	x = x[v.Offset:][:planes*h*w]
	if g != nil {
		g = g[k.In[1].Offset:][:planes*oh*ow]
	}
	block := max(1, poolBlock/(h*w))
	return b.parallelFor(ctx, planes, max(1, elementGrain/(h*w)), func(lo, hi int) error {
		if n := hi - lo; poolMax2x2 != nil {
			if g == nil {
				poolMax2x2(&out[lo*oh*ow], &x[lo*h*w], w, n*oh)
			} else {
				poolGrad2x2(&out[lo*h*w], &x[lo*h*w], &g[lo*oh*ow], w, n*oh)
			}
			return nil
		}
		n := min(block, hi-lo)
		keys := make([]int32, n*oh*ow)
		cols := make([]int32, n*h*ow)
		for p := lo; p < hi; p += block {
			n := min(block, hi-p)
			keys, cols := keys[:n*oh*ow], cols[:n*h*ow]
			xs := x[p*h*w:][:n*h*w]
			// The larger key of each pair along a row, [n*h, ow], then
			// of each pair of rows, [n*oh, ow].
			for i := range cols {
				cols[i] = max(poolKey(xs[2*i]), poolKey(xs[2*i+1]))
			}
			for r := range n * oh {
				kr, c0, c1 := keys[r*ow:][:ow], cols[2*r*ow:][:ow], cols[(2*r+1)*ow:][:ow]
				for j := range kr {
					kr[j] = max(c0[j], c1[j])
				}
			}
			if g == nil {
				o := out[p*oh*ow:][:n*oh*ow]
				for i, key := range keys {
					o[i] = poolValue(key)
				}
				continue
			}
			// The gradient goes to the first element of the window, in
			// row-major order, with the largest key: multiplying by the
			// match avoids a branch that mispredicts.
			gs := g[p*oh*ow:][:n*oh*ow]
			dx := out[p*h*w:][:n*h*w]
			for r := range n * oh {
				kr, gr := keys[r*ow:][:ow], gs[r*ow:][:ow]
				xa, xb := xs[2*r*w:][:w], xs[(2*r+1)*w:][:w]
				da, db := dx[2*r*w:][:w], dx[(2*r+1)*w:][:w]
				for j, key := range kr {
					c := 2 * j
					h0 := poolKey(xa[c]) == key
					h1 := !h0 && poolKey(xa[c+1]) == key
					h2 := !h0 && !h1 && poolKey(xb[c]) == key
					h3 := !h0 && !h1 && !h2 && poolKey(xb[c+1]) == key
					gv := gr[j]
					da[c] = gv * boolTo[float32](h0)
					da[c+1] = gv * boolTo[float32](h1)
					db[c] = gv * boolTo[float32](h2)
					db[c+1] = gv * boolTo[float32](h3)
				}
			}
		}
		return nil
	})
}

// poolBlock is how many input elements pool2x2 pools at a time, so that
// the keys of their windows stay in cache between its passes.
const poolBlock = 1 << 14
