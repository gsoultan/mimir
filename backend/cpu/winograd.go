package cpu

import (
	"context"
	"sync"
	"unsafe"
)

// Convolutions with 3×3 windows of stride and dilation 1, the bulk of a
// CNN's work, run as Winograd's minimal filtering F(2×2, 3×3) (Lavin and
// Gray, "Fast Algorithms for Convolutional Neural Networks", 2015): each
// 2×2 tile of the output is A^T [(G g G^T) ⊙ (B^T d B)] A, from the 4×4
// tile d of the input that covers its windows and the 3×3 window g of the
// weights. The 16 elementwise products of each tile, summed over the
// input channels, are 16 matrix products, one per element ξ of the 4×4
// transformed tiles, of the transformed weights, Cout×Cin, and the
// transformed input tiles, Cin×tiles: 16·Cout·Cin multiply-adds per tile
// of 4 outputs instead of 36·Cout·Cin. The GEMM micro-kernel computes
// them from panels that the transforms write in its packed layout.
//
// The gradient of the input is the convolution of the output's gradient
// with the weights flipped and transposed, which runs the same way. The
// gradient of the weights is G^T [Σ (A dY A^T) ⊙ (B^T d B)] G, summed
// over the tiles: the transpose of the forward pass, 16 products of the
// transformed output gradients, Cout×tiles, and the transformed input
// tiles, tiles×Cin.
//
// The transforms run over a row of tiles at a time, with tiles in the
// lanes of vector registers where the CPU has them. The work is split
// into blocks of tiles, whole items or runs of tile rows of one, small
// enough that a block's transformed tiles and products stay in cache,
// and each block is transformed, multiplied and transformed back by one
// worker. Each output is computed the same way whatever the number of
// workers, and so is each sum of a weight gradient, which sums its
// blocks in a fixed number of parts that are added together in order.

// The transforms of a row of tiles; see winoInputGo, winoGradGo and
// winoOutputGo. The architecture-specific files replace them with vector
// routines.
var (
	winoInputRow  = winoInputGo
	winoGradRow   = winoGradGo
	winoOutputRow = winoOutputGo
)

// winoInputChannels and winoGradChannels, where the CPU has them, are the
// transforms of a row of tiles with 16 channels in the lanes of vector
// registers, from planes with channels innermost, which the weight
// gradient runs on (see winoWeightGradChannels):
//
//	winoInputChannels(dst, xs, src, rs, cs, tiles, step)
//	winoGradChannels(dst, xs, src, rs, cs, tiles, step, half)
//
// store at dst+ξ·xs, for each tile q of tiles, element ξ of its transform
// for 16 channels, from element (i, j) of tile q at src+i·rs+(2q+j)·cs,
// dst moving step on from tile to tile. winoGradChannels stores the
// first 8 channels there and the others half on.
var (
	winoInputChannels func(dst *float32, xs int, src *float32, rs, cs, tiles, step int)
	winoGradChannels  func(dst *float32, xs int, src *float32, rs, cs, tiles, step, half int)
)

// winoImage is a 4-D image [N, G·C, H, W] read through strides: element
// (n, g, c, y, x) is data[off+n·sn+g·sg+c·sc+y·sh+x·sw].
type winoImage struct {
	data                    []float32
	off, sn, sg, sc, sh, sw int
}

// winoConv is a convolution of an image of N items of G groups of Cin
// channels, H×W, with 3×3 windows of stride and dilation 1, padded by pt
// rows above and pl columns to the left, into Cout channels per group of
// OH×OW outputs.
type winoConv struct {
	n, g, cin, cout int
	h, w, oh, ow    int
	pt, pl          int
}

// tiles returns the number of rows and columns of 2×2 tiles that cover
// the output.
func (c winoConv) tiles() (rows, cols int) { return (c.oh + 1) / 2, (c.ow + 1) / 2 }

// winoBlock is the tiles of items [n0, n1) in tile rows [r0, r1).
type winoBlock struct{ n0, n1, r0, r1 int }

// rows returns the number of rows of tiles of b.
func (b winoBlock) rows() int { return (b.n1 - b.n0) * (b.r1 - b.r0) }

// winoBlockTiles bounds the tiles of a block of the forward pass: enough
// that the products keep the micro-kernel busy, few enough that a
// block's panels stay in cache.
const winoBlockTiles = 256

// blocks splits the tiles of c, laid out sw to a row of tiles, into
// blocks of at most most tiles, or of one row of tiles if a row holds
// more: whole items, several to a block, or runs of tile rows of an item.
// A block of items is padded to a multiple of pad tiles, and holds the
// number of items that pads it least.
func (c winoConv) blocks(sw, most, pad int) []winoBlock {
	th, _ := c.tiles()
	var out []winoBlock
	if per := th * sw; per <= most {
		step, waste := 1, 1.0
		for s := 1; s <= min(most/per, c.n); s++ {
			if w := 1 - float64(s*per)/float64(roundUp(s*per, pad)); w <= waste {
				step, waste = s, w
			}
		}
		for n := 0; n < c.n; n += step {
			out = append(out, winoBlock{n, min(n+step, c.n), 0, th})
		}
		return out
	}
	step := max(1, most/sw)
	for n := range c.n {
		for r := 0; r < th; r += step {
			out = append(out, winoBlock{n, n + 1, r, min(r+step, th)})
		}
	}
	return out
}

// slotWidth returns the number of slots that a row of tiles takes in
// the B panels of the forward pass, nr slots wide, for tw slots or more:
// the smallest power of two no less than tw that divides nr, or a
// multiple of nr, so that no row but those of several panels crosses
// one. The slots past a row's tiles hold garbage, which only their own
// products read; the forward pass leaves one at least, which reads the
// columns past the row (see superRows).
func slotWidth(tw, nr int) int {
	if tw >= nr {
		return roundUp(tw, nr)
	}
	p := 1
	for p < tw {
		p *= 2
	}
	if nr%p != 0 {
		return nr
	}
	return p
}

// superRows sizes the plane of ws for the input tiles of a run of rows of
// tiles, as two halves, those of its even rows and its odd rows, of
// regions rows of width elements: row 2r of the plane at region r of the
// first half, and row 2r+1 of the second. Side by side, the regions hold
// rows of tiles one after the other, and the region past a row holds its
// last two rows of the plane: the tile in slot s of a row, which is
// width/2 slots wide, starts at column 2s. Each half ends in a few
// zeros, which the tiles of the last row's last slot read past the last
// region. It returns the offset of the second half.
func superRows(ws *winoWorkspace, regions, width int) int {
	odd := regions*width + 8
	ws.plane = grow(ws.plane, 2*odd)
	clear(ws.plane[odd-8 : odd])
	clear(ws.plane[2*odd-8:])
	return odd
}

// winoSkew pads the distance between the transforms ξ of a buffer, so
// that the 16 elements of a tile, stored or loaded together, do not fall
// in the same sets of the cache.
const winoSkew = 16

// winoWorkspace is a worker's buffers: a padded plane of the input, the
// transformed tiles and the products.
type winoWorkspace struct {
	plane, v, p, sv, sp, m []float32
}

var winoWorkspaces = sync.Pool{New: func() any { return new(winoWorkspace) }}

// plane fills rows rows of cols elements with channel (g, c) of item n
// of x from row y0 and column x0 on, zero outside the image, which is
// h×w: the even rows from dst on, step apart, and the odd ones from
// dst[odd] on.
func (x winoImage) plane(dst []float32, odd, step, n, g, c, y0, x0, rows, cols, h, w int) {
	base := x.off + n*x.sn + g*x.sg + c*x.sc
	lo, hi := max(0, -x0), min(cols, w-x0) // columns inside the image
	for i := range rows {
		row := dst[(i&1)*odd+(i>>1)*step:][:cols]
		y := y0 + i
		if y < 0 || y >= h || lo >= hi {
			for j := range row {
				row[j] = 0
			}
			continue
		}
		for j := range lo {
			row[j] = 0
		}
		for j := hi; j < cols; j++ {
			row[j] = 0
		}
		src := base + y*x.sh + (x0+lo)*x.sw
		if x.sw == 1 {
			copy(row[lo:hi], x.data[src:])
			continue
		}
		for j := lo; j < hi; j++ {
			row[j] = x.data[src]
			src += x.sw
		}
	}
}

// winoInputGo stores at dst+ξ·xs, for each of tiles tiles q of a row of
// 4×4 tiles d two apart, at lane q, element ξ of the transform B^T d B,
// from the rows at src, src+pw, src+step and src+pw+step: tile q's first
// column is column 2q.
func winoInputGo(dst *float32, xs int, src *float32, pw, step, tiles int) {
	if tiles <= 0 {
		return
	}
	out := unsafe.Slice(dst, 15*xs+tiles)
	in := unsafe.Slice(src, pw+step+2*tiles+2)
	for q := range tiles {
		r0, r1, r2, r3 := in[2*q:][:4], in[pw+2*q:][:4], in[step+2*q:][:4], in[pw+step+2*q:][:4]
		// B^T d: rows d0 - d2, d1 + d2, d2 - d1, d1 - d3.
		var e [4][4]float32
		for j := range 4 {
			d0, d1, d2, d3 := r0[j], r1[j], r2[j], r3[j]
			e[0][j], e[1][j], e[2][j], e[3][j] = d0-d2, d1+d2, d2-d1, d1-d3
		}
		for i, r := range e {
			o := 4*i*xs + q
			out[o], out[o+xs], out[o+2*xs], out[o+3*xs] = r[0]-r[2], r[1]+r[2], r[2]-r[1], r[1]-r[3]
		}
	}
}

// winoGradGo stores at dst+ξ·xs, for each of tiles tiles q of a row of
// 2×2 tiles y, at lane q, element ξ of the transform A y A^T, from the
// two rows pw apart at src.
func winoGradGo(dst *float32, xs int, src *float32, pw, tiles int) {
	if tiles <= 0 {
		return
	}
	out := unsafe.Slice(dst, 15*xs+tiles)
	in := unsafe.Slice(src, pw+2*tiles)
	for q := range tiles {
		a0, a1, b0, b1 := in[2*q], in[2*q+1], in[pw+2*q], in[pw+2*q+1]
		// A y: rows y0, y0 + y1, y0 - y1, -y1.
		e := [4][2]float32{{a0, a1}, {a0 + b0, a1 + b1}, {a0 - b0, a1 - b1}, {-b0, -b1}}
		for i, r := range e {
			o := 4*i*xs + q
			out[o], out[o+xs], out[o+2*xs], out[o+3*xs] = r[0], r[0]+r[1], r[0]-r[1], -r[1]
		}
	}
}

// winoOutputGo writes the 2×2 output tiles A^T m A of tiles tiles q of a
// row, from element ξ of each at lane q of src+ξ·xs, to the first rows
// rows, 1 or 2, of the rows ow apart at dst, stopping each row after ow
// elements.
func winoOutputGo(dst *float32, ow int, src *float32, xs int, tiles, rows int) {
	if tiles <= 0 {
		return
	}
	out := unsafe.Slice(dst, (rows-1)*ow+ow)
	in := unsafe.Slice(src, 15*xs+tiles)
	for q := range tiles {
		// A^T m: rows m0 + m1 + m2, m1 - m2 - m3.
		var e [2][4]float32
		for j := range 4 {
			a0, a1, a2, a3 := in[j*xs+q], in[(4+j)*xs+q], in[(8+j)*xs+q], in[(12+j)*xs+q]
			e[0][j], e[1][j] = a0+a1+a2, a1-a2-a3
		}
		for i := range rows {
			r := out[i*ow:][:ow]
			a := e[i]
			r[2*q] = a[0] + a[1] + a[2]
			if 2*q+1 < ow {
				r[2*q+1] = a[1] - a[2] - a[3]
			}
		}
	}
}

// winoFilters is the 3×3 windows of a convolution's weights read
// through strides: element (kh, kw) of the window of output channel o and
// input channel i of group g is data[off+g·sg+o·so+i·si+kh·sh+kw·sw].
type winoFilters struct {
	data                    []float32
	off, sg, so, si, sh, sw int
}

// winoWeightsPanel, where the CPU has it, is the transform of a panel of
// 8 output channels, one input channel at a time:
//
//	winoWeightsPanel(dst, xs, src, so, si, sh, sw, cin)
//
// stores at dst+ξ·xs+8i+o, for the 8 output channels o and cin input
// channels i whose windows are at src+o·so+i·si, element (kh, kw) at
// kh·sh+kw·sw on, element ξ of the transform of each window.
var winoWeightsPanel func(dst *float32, xs int, src *float32, so, si, sh, sw, cin int)

// winoWeightBuffers holds the transformed weights between convolutions,
// which transform the same weights on every call.
var winoWeightBuffers sync.Pool

// winoWeights returns the transformed weights G g G^T of each group, as
// the micro-kernel's packed A panels, mr output channels each: element
// (o, i) of transform ξ of group g at (g·16+ξ)·xs + (o/mr·cin+i)·mr + o%mr,
// zero past the last channel, and xs. The caller gives the buffer back
// with winoWeightBuffers.Put once its products are done.
func (be *Backend) winoWeights(ctx context.Context, c winoConv, mr int, w winoFilters) (*[]float32, int, error) {
	panels := (c.cout + mr - 1) / mr
	xs := panels*c.cin*mr + winoSkew
	buf, _ := winoWeightBuffers.Get().(*[]float32)
	if buf == nil {
		buf = new([]float32)
	}
	*buf = grow(*buf, c.g*16*xs)
	u := *buf
	if mr == 8 && winoWeightsPanel != nil && c.cin > 0 {
		// Whole panels in vector registers; a ragged last panel below.
		whole := c.cout / mr
		err := be.parallelFor(ctx, c.g*whole, max(1, elementGrain/max(32*mr*c.cin, 1)), func(lo, hi int) error {
			for r := lo; r < hi; r++ {
				g, p := r/whole, r%whole
				winoWeightsPanel(&u[g*16*xs+p*c.cin*mr], xs, &w.data[w.off+g*w.sg+p*mr*w.so], w.so, w.si, w.sh, w.sw, c.cin)
			}
			return nil
		})
		if err != nil {
			return buf, xs, err
		}
		if whole*mr == c.cout {
			return buf, xs, nil
		}
		err = be.parallelFor(ctx, c.g*(c.cout-whole*mr), max(1, elementGrain/max(32*c.cin, 1)), func(lo, hi int) error {
			per := c.cout - whole*mr
			for r := lo; r < hi; r++ {
				winoWeightsGo(u, xs, c, mr, w, r/per, whole*mr+r%per)
			}
			return nil
		})
		return buf, xs, err
	}
	err := be.parallelFor(ctx, c.g*c.cout, max(1, elementGrain/max(32*c.cin, 1)), func(lo, hi int) error {
		for r := lo; r < hi; r++ {
			winoWeightsGo(u, xs, c, mr, w, r/c.cout, r%c.cout)
		}
		return nil
	})
	return buf, xs, err
}

// winoWeightsGo transforms the windows of output channel o of group g
// into u (see winoWeights).
func winoWeightsGo(u []float32, xs int, c winoConv, mr int, w winoFilters, g, o int) {
	{
		{
			for i := range c.cin {
				at := w.off + g*w.sg + o*w.so + i*w.si
				// G g: rows g0, (g0+g1+g2)/2, (g0-g1+g2)/2, g2; then the
				// same over the columns.
				var t [4][3]float32
				for kw := range 3 {
					col := at + kw*w.sw
					g0, g1, g2 := w.data[col], w.data[col+w.sh], w.data[col+2*w.sh]
					t[0][kw], t[1][kw], t[2][kw], t[3][kw] = g0, (g0+g1+g2)/2, (g0-g1+g2)/2, g2
				}
				base := g*16*xs + (o/mr*c.cin+i)*mr + o%mr
				for r := range 4 {
					g0, g1, g2 := t[r][0], t[r][1], t[r][2]
					s := u[base+4*r*xs:]
					s[0], s[xs], s[2*xs], s[3*xs] = g0, (g0+g1+g2)/2, (g0-g1+g2)/2, g2
				}
			}
		}
	}
}

// winoKernel is the micro-kernel of the transforms' products.
func winoKernel() *gemmKernel { return gemmKernels[0] }

// winoForward sets out, [N, G, Cout, OH, OW], to the convolution c of
// the image x with the weights w (see winoWeights), and runs the
// epilogue over each part of it once it is done.
func (be *Backend) winoForward(ctx context.Context, c winoConv, x winoImage, w winoFilters, out []float32, epi *evaluator) error {
	g := winoKernel()
	mr, nr := g.mr, g.nr
	ub, us, err := be.winoWeights(ctx, c, mr, w)
	if err != nil {
		return err
	}
	defer winoWeightBuffers.Put(ub)
	u := *ub
	th, tw := c.tiles()
	sw := slotWidth(tw+1, nr)
	panels := (c.cout + mr - 1) / mr
	// Blocks of fewer tiles when there are too few to share between the
	// workers, down to a row of tiles, and then parts of the output
	// channels, which repeat the input transform.
	want := 2 * be.workers
	blocks := c.blocks(sw, max(nr, sw, min(winoBlockTiles, c.n*c.g*th*sw/want)), nr)
	chunks := min(panels, max(1, (want+len(blocks)*c.g-1)/(len(blocks)*c.g)))
	step := (panels + chunks - 1) / chunks
	chunks = (panels + step - 1) / step
	units := len(blocks) * c.g * chunks
	work := 2 * 16 * c.cin * c.cout * th * tw * c.n / max(units, 1)
	return be.parallelFor(ctx, units, max(1, minUnitFlops/max(work, 1)), func(lo, hi int) error {
		ws := winoWorkspaces.Get().(*winoWorkspace)
		defer winoWorkspaces.Put(ws)
		for unit := lo; unit < hi; unit++ {
			b, grp, ch := blocks[unit/(c.g*chunks)], unit/chunks%c.g, unit%chunks
			p0, p1 := ch*step, min((ch+1)*step, panels)
			m0, m1 := p0*mr, min(p1*mr, c.cout)
			mc := (p1 - p0) * mr
			rows := b.r1 - b.r0
			pad := roundUp(b.rows()*sw, nr)
			pw, odd := 2*sw, superRows(ws, rows+1, 2*sw)
			// The products: element (o, s) of transform ξ, for output
			// channel m0+o and the tile in slot s, at ξ·ms+o·pad+s.
			ms := mc*pad + winoSkew
			ws.m = grow(ws.m, 16*ms)
			for i0 := 0; i0 < c.cin; i0 += g.kc {
				kc := min(g.kc, c.cin-i0)
				// The transformed tiles as B panels, nr slots each:
				// element (i, s) of transform ξ at ξ·xs + (s/nr·kc+i)·nr + s%nr.
				// Slots past a row's tiles hold garbage, which only the
				// products of those slots read.
				xs := pad*kc + winoSkew
				ws.v = grow(ws.v, 16*xs)
				v := ws.v
				// Item by item, so that the planes read follow each other.
				for n := b.n0; n < b.n1; n++ {
					for i := range kc {
						x.plane(ws.plane, odd, pw, n, grp, i0+i, 2*b.r0-c.pt, -c.pl, 2*rows+2, pw, c.h, c.w)
						// The item's slots, in runs that end at the panels'.
						s0, s1 := (n-b.n0)*rows*sw, (n-b.n0+1)*rows*sw
						for s := s0; s < s1; {
							e := min(s1, (s/nr+1)*nr)
							winoInputRow(&v[(s/nr*kc+i)*nr+s%nr], xs, &ws.plane[2*(s-s0)], odd, pw, e-s)
							s = e
						}
					}
				}
				for xi := range 16 {
					for tp := range pad / nr {
						bp := &v[xi*xs+tp*kc*nr]
						for p := p0; p < p1; p++ {
							ap := &u[(grp*16+xi)*us+(p*c.cin+i0)*mr]
							g.tile(kc, ap, bp, &ws.m[xi*ms+(p-p0)*mr*pad+tp*nr], pad, i0 > 0)
						}
					}
				}
			}
			plane := c.oh * c.ow
			for n := b.n0; n < b.n1; n++ {
				for o := m0; o < m1; o++ {
					dst := out[((n*c.g+grp)*c.cout+o)*plane:][:plane]
					for r := range rows {
						y := 2 * (b.r0 + r)
						s := ((n-b.n0)*rows + r) * sw
						winoOutputRow(&dst[y*c.ow], c.ow, &ws.m[(o-m0)*pad+s], ms, tw, min(2, c.oh-y))
					}
				}
			}
			if epi == nil {
				continue
			}
			y0, y1 := 2*b.r0*c.ow, min(2*b.r1, c.oh)*c.ow
			for n := b.n0; n < b.n1; n++ {
				base := (n*c.g + grp) * c.cout * plane
				if y0 == 0 && y1 == plane {
					if err := epi.apply(base+m0*plane, base+m1*plane); err != nil {
						return err
					}
					continue
				}
				for o := m0; o < m1; o++ {
					if err := epi.apply(base+o*plane+y0, base+o*plane+y1); err != nil {
						return err
					}
				}
			}
		}
		return nil
	})
}

// winoWeightParts bounds the parts that a weight gradient sums its
// blocks in.
const winoWeightParts = 8

// winoWeightGrad sets out, [G, Cout, Cin·9], to the gradient of the
// weights of the convolution c of the image x whose output has the
// gradient dy, [N, G·Cout, OH, OW].
func (be *Backend) winoWeightGrad(ctx context.Context, c winoConv, x, dy winoImage, out []float32) error {
	g := winoKernel()
	mr, nr := g.mr, g.nr
	if winoInputChannels != nil && mr == 8 && nr%16 == 0 {
		return be.winoWeightGradChannels(ctx, c, x, dy, out)
	}
	mp, cp := (c.cout+mr-1)/mr, (c.cin+nr-1)/nr
	mpad, cpad := mp*mr, cp*nr
	th, tw := c.tiles()
	sw := tw + 1
	blocks := c.blocks(sw, max(sw, 32768/(mpad+cpad)), 1)
	// Each part sums a run of blocks into transformed gradients of its
	// own, [G, 16, Mpad, Cpad].
	size := c.g * 16 * mpad * cpad
	parts := max(1, min(len(blocks), winoWeightParts, (1<<23)/size))
	sums := make([]float32, parts*size)
	units := parts * c.g
	work := 2 * 16 * c.cin * c.cout * th * tw * c.n / units
	err := be.parallelFor(ctx, units, max(1, minUnitFlops/max(work, 1)), func(lo, hi int) error {
		ws := winoWorkspaces.Get().(*winoWorkspace)
		defer winoWorkspaces.Put(ws)
		for unit := lo; unit < hi; unit++ {
			part, grp := unit/c.g, unit%c.g
			acc := sums[(part*c.g+grp)*16*mpad*cpad:][:16*mpad*cpad]
			b0, b1 := part*len(blocks)/parts, (part+1)*len(blocks)/parts
			for bi := b0; bi < b1; bi++ {
				b := blocks[bi]
				rows := b.r1 - b.r0
				used := b.rows() * sw
				tiles := roundUp(used, 8)
				// The transforms with tiles in lanes, element (i, t) of
				// transform ξ at ξ·xs+i·tiles+t (ys for the gradient), zero past the
				// last tile and channel, and then transposed into panels:
				// of the input, B panels of nr channels, element (t, i) of
				// transform ξ at ξ·xs + (i/nr·tiles+t)·nr + i%nr, and of
				// the gradient, A panels of mr channels.
				xs, ys := cpad*tiles+winoSkew, mpad*tiles+winoSkew
				ws.sv = grow(ws.sv, 16*xs)
				ws.sp = grow(ws.sp, 16*ys)
				ws.v = grow(ws.v, 16*xs)
				ws.p = grow(ws.p, 16*ys)
				pw, odd := 2*sw, superRows(ws, rows+1, 2*sw)
				// Item by item, so that the planes read follow each other.
				for n := b.n0; n < b.n1; n++ {
					t := (n - b.n0) * rows * sw
					for i := range c.cin {
						x.plane(ws.plane, odd, pw, n, grp, i, 2*b.r0-c.pt, -c.pl, 2*rows+2, pw, c.h, c.w)
						winoInputRow(&ws.sv[i*tiles+t], xs, &ws.plane[0], odd, pw, rows*sw)
					}
					for o := range c.cout {
						// The rows of tiles side by side, sw tiles apart:
						// the slots past a row's tiles read zeros.
						dy.plane(ws.plane, rows*pw, pw, n, grp, o, 2*b.r0, 0, 2*rows, pw, c.oh, c.ow)
						winoGradRow(&ws.sp[o*tiles+t], ys, &ws.plane[0], rows*pw, rows*sw)
					}
				}
				for i := range cpad {
					if i >= c.cin {
						for xi := range 16 {
							clear(ws.sv[xi*xs+i*tiles:][:tiles])
						}
						continue
					}
					zeroTail(ws.sv[i*tiles:], xs, used, tiles)
				}
				for o := range mpad {
					if o >= c.cout {
						for xi := range 16 {
							clear(ws.sp[xi*ys+o*tiles:][:tiles])
						}
						continue
					}
					zeroTail(ws.sp[o*tiles:], ys, used, tiles)
				}
				for xi := range 16 {
					winoPanels(ws.v[xi*xs:], ws.sv[xi*xs:], cpad, tiles, nr)
					winoPanels(ws.p[xi*ys:], ws.sp[xi*ys:], mpad, tiles, mr)
				}
				for xi := range 16 {
					for q := range cp {
						bp := &ws.v[xi*xs+q*tiles*nr]
						for r := range mp {
							g.tile(tiles, &ws.p[xi*ys+r*tiles*mr], bp, &acc[(xi*mpad+r*mr)*cpad+q*nr], cpad, bi > b0)
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
	return be.winoWeightSums(ctx, c, sums, parts, mpad, cpad, out)
}

// winoWeightSums adds together, in order, the parts parts of the
// transformed weight gradient of the convolution c in sums, each
// [G, 16, mpad, cpad], and sets out, [G, Cout, Cin·9], to their transform
// G^T dU G.
func (be *Backend) winoWeightSums(ctx context.Context, c winoConv, sums []float32, parts, mpad, cpad int, out []float32) error {
	size := c.g * 16 * mpad * cpad
	err := be.parallelFor(ctx, size, max(1024, size/(4*be.workers)), func(lo, hi int) error {
		dst := sums[lo:hi]
		for part := 1; part < parts; part++ {
			for i, v := range sums[part*size+lo : part*size+hi] {
				dst[i] += v
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	// G^T dU G: rows d0 + (d1+d2)/2, (d1-d2)/2, (d1+d2)/2 + d3; then the
	// same over the columns.
	rows := c.cin * 9
	return be.parallelFor(ctx, c.g*c.cout, 1, func(lo, hi int) error {
		for r := lo; r < hi; r++ {
			grp, o := r/c.cout, r%c.cout
			for i := range c.cin {
				var d [4][4]float32
				for xi := range 16 {
					d[xi/4][xi%4] = sums[((grp*16+xi)*mpad+o)*cpad+i]
				}
				var t [3][4]float32
				for j := range 4 {
					d0, d1, d2, d3 := d[0][j], d[1][j], d[2][j], d[3][j]
					t[0][j], t[1][j], t[2][j] = d0+(d1+d2)/2, (d1-d2)/2, (d1+d2)/2+d3
				}
				dst := out[r*rows+i*9:][:9]
				for kh := range 3 {
					d0, d1, d2, d3 := t[kh][0], t[kh][1], t[kh][2], t[kh][3]
					dst[3*kh], dst[3*kh+1], dst[3*kh+2] = d0+(d1+d2)/2, (d1-d2)/2, (d1+d2)/2+d3
				}
			}
		}
		return nil
	})
}

// winoWeightGradChannels is winoWeightGrad with the transforms of
// winoInputChannels and winoGradChannels, for a micro-kernel of 8 rows
// and of columns a multiple of 16: each item's planes are copied out
// with channels innermost, and the transforms of a row of tiles go
// straight into the micro-kernel's panels, a tile at a time.
func (be *Backend) winoWeightGradChannels(ctx context.Context, c winoConv, x, dy winoImage, out []float32) error {
	g := winoKernel()
	mr, nr := g.mr, g.nr
	mp, cp := (c.cout+mr-1)/mr, (c.cin+nr-1)/nr
	mpad, cpad := mp*mr, cp*nr
	m16 := roundUp(c.cout, 16) // the gradient's channels, zero past the last
	th, tw := c.tiles()
	blocks := c.blocks(tw, max(tw, 8192/(mpad+cpad)), 1)
	size := c.g * 16 * mpad * cpad
	parts := max(1, min(len(blocks), winoWeightParts, (1<<23)/size))
	scratch := convScratch.Get().(*[]float32)
	defer convScratch.Put(scratch)
	*scratch = grow(*scratch, parts*size)
	sums := *scratch
	units := parts * c.g
	work := 2 * 16 * c.cin * c.cout * th * tw * c.n / units
	err := be.parallelFor(ctx, units, max(1, minUnitFlops/max(work, 1)), func(lo, hi int) error {
		ws := winoWorkspaces.Get().(*winoWorkspace)
		defer winoWorkspaces.Put(ws)
		for unit := lo; unit < hi; unit++ {
			part, grp := unit/c.g, unit%c.g
			acc := sums[(part*c.g+grp)*16*mpad*cpad:][:16*mpad*cpad]
			b0, b1 := part*len(blocks)/parts, (part+1)*len(blocks)/parts
			for bi := b0; bi < b1; bi++ {
				b := blocks[bi]
				rows := b.r1 - b.r0
				tiles := b.rows() * tw
				// The transformed input tiles as B panels, nr channels
				// each, element (t, i) of transform ξ at
				// ξ·xs + (i/nr·tiles+t)·nr + i%nr, and the transformed
				// gradients as A panels, mr channels each.
				xs, ys := cp*tiles*nr+winoSkew, m16*tiles+winoSkew
				ws.v = grow(ws.v, 16*xs)
				ws.p = grow(ws.p, 16*ys)
				pw := 2*tw + 2
				for n := b.n0; n < b.n1; n++ {
					t0 := (n - b.n0) * rows * tw
					ws.sv = x.channelsLast(ws.sv, &ws.m, n, grp, c.cin, cpad, 2*b.r0-c.pt, -c.pl, 2*rows+2, pw, c.h, c.w)
					for r := range rows {
						for i := 0; i < cpad; i += 16 {
							t := t0 + r*tw
							winoInputChannels(&ws.v[(i/nr*tiles+t)*nr+i%nr], xs, &ws.sv[2*r*pw*cpad+i], pw*cpad, cpad, tw, nr)
						}
					}
					ws.sp = dy.channelsLast(ws.sp, &ws.m, n, grp, c.cout, m16, 2*b.r0, 0, 2*rows, 2*tw, c.oh, c.ow)
					for r := range rows {
						for o := 0; o < m16; o += 16 {
							t := t0 + r*tw
							winoGradChannels(&ws.p[(o/mr*tiles+t)*mr], ys, &ws.sp[2*r*2*tw*m16+o], 2*tw*m16, m16, tw, mr, tiles*mr)
						}
					}
				}
				for xi := range 16 {
					for q := range cp {
						bp := &ws.v[xi*xs+q*tiles*nr]
						for r := range mp {
							g.tile(tiles, &ws.p[xi*ys+r*tiles*mr], bp, &acc[(xi*mpad+r*mr)*cpad+q*nr], cpad, bi > b0)
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
	return be.winoWeightSums(ctx, c, sums, parts, mpad, cpad, out)
}

// channelsLast returns dst, grown to rows×cols×cp, holding channels
// [0, ch) of group g of item n of x from row y0 and column x0 on, with
// channels innermost, cp apart: zero outside the image, which is h×w,
// and past the last channel. tmp holds the transposed rows of the image.
func (x winoImage) channelsLast(dst []float32, tmp *[]float32, n, g, ch, cp, y0, x0, rows, cols, h, w int) []float32 {
	dst = grow(dst, rows*cols*cp)
	clear(dst)
	iy0, iy1 := max(0, y0), min(h, y0+rows) // the image's rows read
	lo, hi := max(0, -x0), min(cols, w-x0)  // and their columns
	if iy0 >= iy1 || lo >= hi {
		return dst
	}
	base := x.off + n*x.sn + g*x.sg
	if x.sw != 1 || x.sh != w || transpose8x8 == nil {
		for c := range ch {
			for y := iy0; y < iy1; y++ {
				src := base + c*x.sc + y*x.sh + (x0+lo)*x.sw
				row := dst[((y-y0)*cols+lo)*cp+c:]
				for j := range hi - lo {
					row[j*cp] = x.data[src+j*x.sw]
				}
			}
		}
		return dst
	}
	// The rows read, whole, with channels innermost.
	span := (iy1 - iy0) * w
	*tmp = grow(*tmp, span*cp+packSlack)
	t := *tmp
	whole := ch &^ 7
	for c := 0; c < whole; c += 8 {
		transpose8x8(&t[c], cp, &x.data[base+c*x.sc+iy0*w], x.sc, span/8)
	}
	for c := range ch {
		from := span &^ 7
		if c >= whole {
			from = 0
		}
		src := x.data[base+c*x.sc+iy0*w:]
		for p := from; p < span; p++ {
			t[p*cp+c] = src[p]
		}
	}
	if ch < cp {
		for p := range span {
			clear(t[p*cp+ch : (p+1)*cp])
		}
	}
	for y := iy0; y < iy1; y++ {
		copy(dst[((y-y0)*cols+lo)*cp:][:(hi-lo)*cp], t[((y-iy0)*w+x0+lo)*cp:])
	}
	return dst
}

// zeroTail zeroes elements [used, tiles) of the 16 rows xs apart at row.
func zeroTail(row []float32, xs, used, tiles int) {
	for xi := range 16 {
		r := row[xi*xs:][:tiles]
		for t := used; t < tiles; t++ {
			r[t] = 0
		}
	}
}

// winoPanels transposes src, lines rows of tiles elements, tiles a
// multiple of 8, into panels of w rows each, w a multiple of 8: element
// (l, t) at (l/w·tiles+t)·w + l%w.
func winoPanels(dst, src []float32, lines, tiles, w int) {
	for l0 := 0; l0 < lines; l0 += w {
		panel := dst[l0*tiles:][:w*tiles]
		for l := l0; l < l0+w; l += 8 {
			if transpose8x8 != nil && w%8 == 0 {
				transpose8x8(&panel[l-l0], w, &src[l*tiles], tiles, tiles/8)
				continue
			}
			for i := l; i < min(l+8, l0+w); i++ {
				for t, v := range src[i*tiles:][:tiles] {
					panel[t*w+i-l0] = v
				}
			}
		}
	}
}
