package cpu

import (
	"context"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// A direct convolution reads its windows from shifted planes of each item:
// plane (c, kw) holds the rows of channel c, padded, from column kw on, as
// wide as the output, so the windows' row of tap (c, kh, kw) for the
// output positions is a contiguous run of the plane, kh rows down. The
// forward convolution packs those runs as the micro-kernel's panels and
// writes its tiles straight into the output; the weight gradient takes the
// dot products of the output's gradient with them. Neither copies out the
// windows of more than the item at hand, which stay in cache, and so they
// suit the convolutions of few channels that Winograd's leaves alone.

// directMaxTaps bounds the window elements, C·KH·KW, of a direct
// convolution, so that a packed panel stays in L1.
const directMaxTaps = 320

// directPositions bounds the output positions, over every item, of a
// forward convolution of many channels that runs as a direct one rather
// than Winograd's, whose transforms of the weights and the tiles' edges
// cost more than they save on few positions.
var directPositions = 2048

// directConv is a 2-D convolution of stride 1 of N items of G groups of C
// channels, H×W, padded by pt rows above and pl columns to the left, with
// KH×KW windows of dilation dh×dw, into M channels per group of OH×OW.
type directConv struct {
	n, g, c, m     int
	h, w, oh, ow   int
	kh, kw, dh, dw int
	pt, pl         int
}

// direct reports whether the convolution kernel k, of geometry c, runs as
// a direct convolution, and returns it: a forward convolution or a weight
// gradient of stride 1 over a 2-D image, with few enough window elements,
// of fewer than 8 channels, or a forward convolution of few positions.
func direct(k *backend.Kernel, c backend.Conv) (directConv, bool) {
	a := k.Attrs
	if k.Op == backend.OpConvInput || len(c.Image) != 2 || !slices.Equal(a.Stride, []int{1, 1}) || c.Rows() > directMaxTaps {
		return directConv{}, false
	}
	if k.Op == backend.OpConv && c.N > 1 && k.In[0].Strides[0] != 0 {
		return directConv{}, false // weights of every item's own
	}
	if c.C >= 8 && (k.Op != backend.OpConv || c.N*c.Cols() > directPositions) {
		return directConv{}, false
	}
	return directConv{n: c.N, g: c.G, c: c.C, m: c.M, h: c.Image[0], w: c.Image[1], oh: c.Positions[0], ow: c.Positions[1],
		kh: a.Kernel[0], kw: a.Kernel[1], dh: a.Dilation[0], dw: a.Dilation[1], pt: a.Pads[0][0], pl: a.Pads[1][0]}, true
}

// taps returns the elements of a window, C·KH·KW.
func (c directConv) taps() int { return c.c * c.kh * c.kw }

// planeSize returns the elements of one shifted plane: the rows that the
// windows of the output read, and below them zeros enough for dots4 to
// read four rows of runs at once and a panel to read past the last
// position.
func (c directConv) planeSize(nr int) int {
	return (c.oh+(c.kh-1)*c.dh)*c.ow + 4*c.dh*c.ow + nr
}

// directWorkspace holds a worker's shifted planes, panel, edge tile and
// row of the output's gradient.
type directWorkspace struct {
	q, b, dy []float32
	tile     [maxTile]float32
}

var directWorkspaces = sync.Pool{New: func() any { return new(directWorkspace) }}

// planes sets q to the shifted planes of item n, group g, of x, each qs
// elements apart, for the rows of output positions [r0, r1): plane
// (c, kw) holds, at row i and column j, the element of channel c at row
// r0+i-pt and column j+kw·dw-pl, or 0 in the padding.
func (c directConv) planes(q []float32, qs int, x winoImage, n, g, r0, r1 int) {
	rows := r1 - r0 + (c.kh-1)*c.dh
	for ci := range c.c {
		src := x.off + n*x.sn + g*x.sg + ci*x.sc
		for kw := range c.kw {
			p := q[(ci*c.kw+kw)*qs:][:qs]
			x0 := kw*c.dw - c.pl
			j0, j1 := max(0, -x0), min(c.ow, c.w-x0)
			for i := range rows {
				row := p[i*c.ow:][:c.ow]
				y := r0 + i - c.pt
				if y < 0 || y >= c.h || j0 >= j1 {
					clear(row)
					continue
				}
				clear(row[:j0])
				clear(row[max(j0, j1):])
				at := src + y*x.sh + (x0+j0)*x.sw
				if x.sw == 1 {
					copy(row[j0:j1], x.data[at:])
					continue
				}
				for j := j0; j < j1; j++ {
					row[j] = x.data[at]
					at += x.sw
				}
			}
			clear(p[rows*c.ow:])
		}
	}
}

// directForward computes the convolution c of x with the weights w, a
// view [N, G, M, C·KH·KW] of a that every item shares, into out
// [N, G, M, OH·OW], and applies the epilogue, if any, to each part of out
// once it is written.
func (be *Backend) directForward(ctx context.Context, c directConv, x winoImage, wv backend.View, a, out []float32, epi *evaluator) error {
	g := gemmKernels[0]
	mr, nr := g.mr, g.nr
	taps, o := c.taps(), c.oh*c.ow
	mp := (c.m + mr - 1) / mr
	// The weights, packed in panels of mr output channels.
	packed := convScratch.Get().(*[]float32)
	defer convScratch.Put(packed)
	*packed = grow(*packed, c.g*mp*taps*mr)
	wp := *packed
	err := be.parallelFor(ctx, c.g*mp, 1, func(lo, hi int) error {
		for u := lo; u < hi; u++ {
			gi, m0 := u/mp, u%mp*mr
			p := wp[u*taps*mr:][:taps*mr]
			for r := range mr {
				m := m0 + r
				if m >= c.m {
					for k := range taps {
						p[k*mr+r] = 0
					}
					continue
				}
				at := wv.Offset + gi*wv.Strides[1] + m*wv.Strides[2]
				for k := range taps {
					p[k*mr+r] = a[at+k*wv.Strides[3]]
				}
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	// Each item splits into runs of output rows, when there are too few
	// items to keep every worker busy.
	items := c.n * c.g
	runs := min(c.oh, max(1, (2*be.workers+items-1)/items))
	rows := (c.oh + runs - 1) / runs
	runs = (c.oh + rows - 1) / rows
	return be.parallelFor(ctx, items*runs, 1, func(lo, hi int) error {
		ws := directWorkspaces.Get().(*directWorkspace)
		defer directWorkspaces.Put(ws)
		qs := c.planeSize(nr)
		ws.q = grow(ws.q, c.c*c.kw*qs)
		ws.b = grow(ws.b, taps*nr)
		for u := lo; u < hi; u++ {
			item, r0 := u/runs, u%runs*rows
			r1 := min(c.oh, r0+rows)
			n, gi := item/c.g, item%c.g
			c.planes(ws.q, qs, x, n, gi, r0, r1)
			base := item * c.m * o
			v0, v1 := r0*c.ow, r1*c.ow
			for v := v0; v < v1; v += nr {
				for ci := range c.c {
					for kh := range c.kh {
						for kw := range c.kw {
							k := (ci*c.kh+kh)*c.kw + kw
							copy(ws.b[k*nr:][:nr], ws.q[(ci*c.kw+kw)*qs+kh*c.dh*c.ow+v-v0:])
						}
					}
				}
				cols := min(nr, v1-v)
				for i := range mp {
					ap := &wp[(gi*mp+i)*taps*mr]
					m0 := i * mr
					if cols == nr && m0+mr <= c.m {
						g.tile(taps, ap, &ws.b[0], &out[base+m0*o+v], o, false)
						continue
					}
					g.tile(taps, ap, &ws.b[0], &ws.tile[0], nr, false)
					for r := range min(mr, c.m-m0) {
						copy(out[base+(m0+r)*o+v:][:cols], ws.tile[r*nr:])
					}
				}
			}
			if r0 == 0 && r1 == c.oh {
				if err := epi.apply(base, base+c.m*o); err != nil {
					return err
				}
				continue
			}
			for m := range c.m {
				if err := epi.apply(base+m*o+v0, base+m*o+v1); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

// directParts is the number of parts that a direct weight gradient sums
// apart, in order, over runs of items, whatever the workers.
const directParts = 8

// directWeightGrad computes the weight gradient of the convolution c, the
// sum over the items of the products of the output's gradient dy, a view
// [N, G, M, OH·OW] of a, with the windows of x, into out [1, G, M, C·KH·KW].
func (be *Backend) directWeightGrad(ctx context.Context, c directConv, x winoImage, dv backend.View, a, out []float32) error {
	taps, o := c.taps(), c.oh*c.ow
	parts := min(c.n, directParts)
	per := (c.n + parts - 1) / parts
	parts = (c.n + per - 1) / per
	size := c.g * c.m * taps
	sums := convScratch.Get().(*[]float32)
	defer convScratch.Put(sums)
	*sums = grow(*sums, parts*size)
	acc := *sums
	err := be.parallelFor(ctx, parts*c.g, 1, func(lo, hi int) error {
		ws := directWorkspaces.Get().(*directWorkspace)
		defer directWorkspaces.Put(ws)
		qs := c.planeSize(0)
		ws.q = grow(ws.q, c.c*c.kw*qs)
		ws.dy = grow(ws.dy, o)
		for u := lo; u < hi; u++ {
			part, gi := u/c.g, u%c.g
			dst := acc[(part*c.g+gi)*c.m*taps:][:c.m*taps]
			clear(dst)
			for n := part * per; n < min(c.n, (part+1)*per); n++ {
				c.planes(ws.q, qs, x, n, gi, 0, c.oh)
				for m := range c.m {
					at := dv.Offset + n*dv.Strides[0] + gi*dv.Strides[1] + m*dv.Strides[2]
					var dy []float32
					if dv.Strides[3] == 1 {
						dy = a[at:][:o]
					} else {
						dy = ws.dy
						for j := range dy {
							dy[j] = a[at+j*dv.Strides[3]]
						}
					}
					row := dst[m*taps:][:taps]
					var d [4]float32
					for ci := range c.c {
						for kw := range c.kw {
							p := ws.q[(ci*c.kw+kw)*qs:]
							for kh := 0; kh < c.kh; kh += 4 {
								dots4(&d, dy, p[kh*c.dh*c.ow:], c.dh*c.ow)
								for r := range min(4, c.kh-kh) {
									row[(ci*c.kh+kh+r)*c.kw+kw] += d[r]
								}
							}
						}
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	return be.parallelFor(ctx, size, max(1024, size/(4*be.workers)), func(lo, hi int) error {
		dst := out[lo:hi]
		copy(dst, acc[lo:hi])
		for p := 1; p < parts; p++ {
			for i, v := range acc[p*size+lo : p*size+hi] {
				dst[i] += v
			}
		}
		return nil
	})
}
