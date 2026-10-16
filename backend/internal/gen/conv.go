package gen

import (
	"fmt"

	"github.com/gsoultan/mimir/backend"
)

// Convolutions run as tiled products (see tiled) whose factors read the
// windows of the image where they lie, so that the windows never take a
// buffer: implicit GEMM. A group's threads copy each element of the
// windows into shared memory once per tile, computing where it lies in
// the image, or that it is padding, with divisions by multiplication.
//
// When every item of the batch is multiplied by the same weights, as a
// layer's are, the items become columns of one product per group, so
// that the tiles span items and a small image still fills them.

// convPlan is what the convolution kernels share: the kernel's geometry,
// its window as parameters, and divisors of its shapes.
type convPlan struct {
	g         *gen
	c         backend.Conv
	kernel    []int    // the window's size along each dimension
	stride    []string // parameters
	dilation  []string
	pad       []string
	image     []string // the image's size along each windowed dimension
	positions []string
	cols      divisor // the number of window positions
	window    divisor // the elements of a window
}

func newConvPlan(g *gen) (*convPlan, error) {
	c, err := backend.ConvOf(g.k)
	if err != nil {
		return nil, err
	}
	a := g.k.Attrs
	p := &convPlan{g: g, c: c, kernel: a.Kernel, cols: g.divisor(max(c.Cols(), 1)), window: g.divisor(c.Window)}
	for d := range a.Kernel {
		p.stride = append(p.stride, g.param(a.Stride[d]))
		p.dilation = append(p.dilation, g.param(a.Dilation[d]))
		p.pad = append(p.pad, g.param(a.Pads[d][0]))
		p.image = append(p.image, g.param(c.Image[d]))
		p.positions = append(p.positions, g.param(c.Positions[d]))
	}
	return p, nil
}

// viewParams returns the offset and strides of v as parameters.
func (g *gen) viewParams(v backend.View) (string, []string) {
	strides := make([]string, len(v.Strides))
	for i, s := range v.Strides {
		strides[i] = g.param(s)
	}
	return g.param(v.Offset), strides
}

// shared reports whether every item of a batch reads the same matrix
// through v, whose leading dimension is the item.
func shared(v backend.View) bool { return v.Shape[0] == 1 || v.Strides[0] == 0 }

// windowAt emits code that reads the window element of the image, input
// 1, at position start[d] + offset[d] along each windowed dimension, at
// base plus those positions times strides, and returns its expression:
// 0 in the padding.
func (p *convPlan) windowAt(base string, start, offset, strides []string) string {
	g := p.g
	off, inside := g.v("off"), g.v("in")
	g.line("int %s = %s;", off, base)
	g.line("bool %s = true;", inside)
	for d := range start {
		x := g.v("x")
		g.line("int %s = %s + %s;", x, start[d], offset[d])
		g.line("%s = %s && %s >= 0 && %s < %s;", inside, inside, x, x, p.image[d])
		g.line("%s += %s * %s;", off, x, strides[d])
	}
	return fmt.Sprintf("(%s ? i1[%s] : 0.0f)", inside, off)
}

// planConv computes OpConv: per group, the weights, In[0], times the
// windows of the image, In[1]; the weights' rows are the output's rows
// and the windows' positions its columns.
func planConv(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	g := newGen(d, k, outs, ins)
	p, err := newConvPlan(g)
	if err != nil {
		return nil, err
	}
	c, va, vx := p.c, k.In[0], k.In[1]
	if c.N*c.G*c.M*c.Cols() == 0 {
		return nil, nil
	}
	fold := shared(va)
	items, batches := 1, c.N*c.G // items per product, and products
	if fold {
		items, batches = c.N, c.G
	}
	aOff, aS := g.viewParams(va)
	xOff, xS := g.viewParams(vx)
	cg := g.param(c.C)
	rows, cols := c.M, items*c.Cols()
	t := chooseTiling(rows, cols)
	return g.tiled(product{
		batches: batches,
		batch: func(b string) {
			if fold {
				g.line("int grp_ = %s, item_ = 0;", b)
			} else {
				g.line("int grp_ = %s %% %s, item_ = %s / %s;", b, g.param(c.G), b, g.param(c.G))
			}
		},
		m: g.param(rows), n: g.param(cols), k: g.param(c.Rows()),
		tilesM: ceilDiv(rows, t.height()), tilesN: ceilDiv(cols, t.width()), t: t,
		a: factor{
			kFast: true,
			fixed: func(m string) []string {
				f := g.v("fa")
				g.line("int %s = %s + item_ * %s + grp_ * %s + %s * %s;", f, aOff, aS[0], aS[1], m, aS[2])
				return []string{f}
			},
			at: func(f []string, j string) string { return fmt.Sprintf("i0[%s + %s * %s]", f[0], j, aS[3]) },
		},
		b: factor{
			fixed: func(q string) []string {
				n, o := "item_", q
				if fold {
					n, o = g.divmod(q, p.cols)
				}
				base := g.v("fb")
				g.line("int %s = %s + %s * %s + grp_ * %s * %s;", base, xOff, n, xS[0], cg, xS[1])
				out := []string{base}
				for dim, od := range g.coords(o, c.Positions) {
					h := g.v("h")
					g.line("int %s = %s * %s - %s;", h, od, p.stride[dim], p.pad[dim])
					out = append(out, h)
				}
				return out
			},
			at: func(f []string, j string) string {
				ch, kidx := g.divmod(j, p.window)
				base := g.v("off")
				g.line("int %s = %s + %s * %s;", base, f[0], ch, xS[1])
				ks := g.coords(kidx, p.kernel)
				for dim, kd := range ks {
					ks[dim] = g.v("h")
					g.line("int %s = %s * %s;", ks[dim], kd, p.dilation[dim])
				}
				return p.windowAt(base, f[1:], ks, xS[2:])
			},
		},
		out: func(row, col string) string {
			n, o := "item_", col
			if fold {
				n, o = g.divmod(col, p.cols)
			}
			i := g.v("i")
			g.line("int %s = ((%s * %s + grp_) * %s + %s) * %s + %s;", i, n, g.param(c.G), g.param(c.M), row, p.cols.d, o)
			return i
		},
	})
}

// planConvWeight computes OpConvWeight: per part and group, the
// gradient, In[0], times the windows of the image, In[1], transposed,
// over the part's pairs (n, o) of item and position.
func planConvWeight(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	g := newGen(d, k, outs, ins)
	p, err := newConvPlan(g)
	if err != nil {
		return nil, err
	}
	c, va, vx := p.c, k.In[0], k.In[1]
	rows, cols := c.M, c.Rows()
	if c.Parts*c.G*rows*cols == 0 {
		return nil, nil
	}
	pairs := c.N * c.Cols()
	size := ceilDiv(pairs, c.Parts)
	if size*c.Parts > MaxElems {
		return nil, fmt.Errorf("%d parts of %d pairs exceed the %d a GPU kernel indexes: %w", c.Parts, size, MaxElems, backend.ErrShape)
	}
	aOff, aS := g.viewParams(va)
	xOff, xS := g.viewParams(vx)
	cg := g.param(c.C)
	t := chooseTiling(rows, cols)
	// pair emits the item and position of pair t of the part.
	pair := func(t string) (string, string) {
		pt := g.v("pt")
		g.line("int %s = lo_ + %s;", pt, t)
		return g.divmod(pt, p.cols)
	}
	return g.tiled(product{
		batches: c.Parts * c.G,
		batch: func(b string) {
			g.line("int grp_ = %s %% %s, part_ = %s / %s;", b, g.param(c.G), b, g.param(c.G))
			size, pairs := g.param(size), g.param(pairs)
			g.line("int lo_ = part_ * %s, kc_ = %s - lo_;", size, pairs)
			g.line("if (kc_ > %s) kc_ = %s;", size, size)
		},
		m: g.param(rows), n: g.param(cols), k: "kc_",
		tilesM: ceilDiv(rows, t.height()), tilesN: ceilDiv(cols, t.width()), t: t,
		a: factor{
			kFast: true,
			fixed: func(m string) []string {
				f := g.v("fa")
				g.line("int %s = %s + grp_ * %s + %s * %s;", f, aOff, aS[1], m, aS[2])
				return []string{f}
			},
			at: func(f []string, t string) string {
				n, o := pair(t)
				return fmt.Sprintf("i0[%s + %s * %s + %s * %s]", f[0], n, aS[0], o, aS[3])
			},
		},
		b: factor{
			kFast: true,
			fixed: func(j string) []string {
				ch, kidx := g.divmod(j, p.window)
				base := g.v("fb")
				g.line("int %s = %s + (grp_ * %s + %s) * %s;", base, xOff, cg, ch, xS[1])
				out := []string{base}
				for dim, kd := range g.coords(kidx, p.kernel) {
					h := g.v("h")
					g.line("int %s = %s * %s - %s;", h, kd, p.dilation[dim], p.pad[dim])
					out = append(out, h)
				}
				return out
			},
			at: func(f []string, t string) string {
				n, o := pair(t)
				base := g.v("off")
				g.line("int %s = %s + %s * %s;", base, f[0], n, xS[0])
				// The position o along each dimension, times the stride,
				// plus the start fixed computed: windowAt's start times a
				// dilation of 1.
				os := g.coords(o, c.Positions)
				starts := make([]string, len(os))
				for dim, od := range os {
					starts[dim] = g.v("h")
					g.line("int %s = %s * %s;", starts[dim], od, p.stride[dim])
				}
				return p.windowAt(base, starts, f[1:], xS[2:])
			},
		},
		out: func(row, col string) string {
			i := g.v("i")
			g.line("int %s = ((part_ * %s + grp_) * %s + %s) * %s + %s;", i, g.param(c.G), g.param(rows), row, g.param(cols), col)
			return i
		},
	})
}

// planConvInput computes OpConvInput: per group, the transposed weights,
// In[0], times the gradient, In[1], read at the window positions whose
// windows cover each element of the image: element x along a windowed
// dimension sums window element k at position o where o*stride - pad +
// k*dilation is x. The output's rows are the group's channels, its
// columns the image's elements, and the inner index (m, k) runs over the
// weights' columns and the window's elements.
func planConvInput(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	g := newGen(d, k, outs, ins)
	p, err := newConvPlan(g)
	if err != nil {
		return nil, err
	}
	c, va, vy := p.c, k.In[0], k.In[1]
	pixels, _ := backend.NumElements(c.Image)
	if c.N*c.G*c.C*pixels == 0 {
		return nil, nil
	}
	fold := shared(va)
	items, batches := 1, c.N*c.G
	if fold {
		items, batches = c.N, c.G
	}
	aOff, aS := g.viewParams(va)
	yOff, yS := g.viewParams(vy)
	strided := false
	for _, s := range k.Attrs.Stride {
		strided = strided || s > 1
	}
	var strides []divisor
	if strided {
		for _, s := range k.Attrs.Stride {
			strides = append(strides, g.divisor(s))
		}
	}
	pix := g.divisor(pixels)
	rows, cols := c.C, items*pixels
	t := chooseTiling(rows, cols)
	return g.tiled(product{
		batches: batches,
		batch: func(b string) {
			if fold {
				g.line("int grp_ = %s, item_ = 0;", b)
			} else {
				g.line("int grp_ = %s %% %s, item_ = %s / %s;", b, g.param(c.G), b, g.param(c.G))
			}
		},
		m: g.param(rows), n: g.param(cols), k: g.param(c.M * c.Window),
		tilesM: ceilDiv(rows, t.height()), tilesN: ceilDiv(cols, t.width()), t: t,
		a: factor{
			kFast: true,
			fixed: func(ch string) []string {
				f := g.v("fa")
				g.line("int %s = %s + item_ * %s + grp_ * %s + %s * %s * %s;", f, aOff, aS[0], aS[1], ch, p.window.d, aS[2])
				return []string{f}
			},
			at: func(f []string, t string) string {
				m, kidx := g.divmod(t, p.window)
				return fmt.Sprintf("i0[%s + %s * %s + %s * %s]", f[0], kidx, aS[2], m, aS[3])
			},
		},
		b: factor{
			fixed: func(q string) []string {
				n, x := "item_", q
				if fold {
					n, x = g.divmod(q, pix)
				}
				base := g.v("fb")
				g.line("int %s = %s + %s * %s + grp_ * %s;", base, yOff, n, yS[0], yS[1])
				out := []string{base}
				for dim, xd := range g.coords(x, c.Image) {
					h := g.v("h")
					g.line("int %s = %s + %s;", h, xd, p.pad[dim])
					out = append(out, h)
				}
				return out
			},
			at: func(f []string, t string) string {
				m, kidx := g.divmod(t, p.window)
				ok, o := g.v("in"), g.v("o")
				g.line("bool %s = true;", ok)
				g.line("int %s = 0;", o)
				for dim, kd := range g.coords(kidx, p.kernel) {
					td := g.v("t")
					g.line("int %s = %s - %s * %s;", td, f[1+dim], kd, p.dilation[dim])
					g.line("%s = %s && %s >= 0;", ok, ok, td)
					g.line("if (%s < 0) %s = 0;", td, td)
					if strided {
						q, r := g.divmod(td, strides[dim])
						g.line("%s = %s && %s == 0;", ok, ok, r)
						td = q
					}
					g.line("%s = %s && %s < %s;", ok, ok, td, p.positions[dim])
					g.line("%s = %s * %s + %s;", o, o, p.positions[dim], td)
				}
				return fmt.Sprintf("(%s ? i1[%s + %s * %s + %s * %s] : 0.0f)", ok, f[0], m, yS[2], o, yS[3])
			},
		},
		out: func(row, col string) string {
			n, x := "item_", col
			if fold {
				n, x = g.divmod(col, pix)
			}
			i := g.v("i")
			g.line("int %s = ((%s * %s + grp_) * %s + %s) * %s + %s;", i, n, g.param(c.G), g.param(c.C), row, pix.d, x)
			return i
		},
	})
}

// convGroups is the number of groups a weight gradient's parts aim to
// give the device: a few per multiprocessor of a mid-sized GPU.
const convGroups = 160

// convPairs is the fewest pairs (n, o) a part of a weight gradient sums.
const convPairs = 256

// ConvParts returns the number of parts into which the weight gradient
// k, an OpConvWeight kernel of one part, splits so that its tiles and
// parts give the device about convGroups groups, each summing at least
// convPairs pairs, or 1 if its tiles alone are as many.
func ConvParts(k *backend.Kernel) int {
	c, err := backend.ConvOf(k)
	if err != nil || k.Op != backend.OpConvWeight || c.Parts != 1 {
		return 1
	}
	t := chooseTiling(c.M, c.Rows())
	tiles := c.G * ceilDiv(c.M, t.height()) * ceilDiv(c.Rows(), t.width())
	pairs := c.N * c.Cols()
	return max(1, min(ceilDiv(convGroups, tiles), pairs/convPairs))
}
