package gen

import (
	"fmt"
	"math/bits"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// Tiled products. A group computes one tile of a product, of tm × tn
// elements per thread, and steps along the inner dimension tileK
// elements at a time: its threads copy the tile's part of each factor
// into shared memory, each element once, wait for each other, and each
// multiplies the rows and columns of its own elements, keeping its sums
// in registers. A thread's rows lie rows threads apart and its columns
// cols threads apart, so that the threads of a SIMD group read shared
// memory without conflicts and write adjacent output elements.
//
// The factors need not be matrices in memory: each reads its elements
// with code of its own, so a convolution reads the windows of its image
// where they lie (see factor). The elements a thread copies sit at the
// same row or column of the tile at every step, so a factor computes once
// what depends only on that, and only the part that depends on the inner
// index at every step.

// tileK is the depth of the slices of the factors a group copies at
// once.
const tileK = 16

// tiling is how a group divides a tile among its threads: rows × cols
// threads, each computing tm × tn elements.
type tiling struct {
	rows, cols, tm, tn int
}

func (t tiling) height() int  { return t.rows * t.tm }
func (t tiling) width() int   { return t.cols * t.tn }
func (t tiling) threads() int { return t.rows * t.cols }

// tilings are the tilings products choose from, best first when they
// compute alike.
var tilings = []tiling{
	{16, 16, 4, 4}, // 64 × 64
	{8, 32, 4, 4},  // 32 × 128
	{32, 8, 4, 4},  // 128 × 32
	{16, 16, 2, 2}, // 32 × 32, for small products
}

// chooseTiling returns the tiling of an m × n product that computes the
// fewest elements beyond the product's own, in its last tiles, counting
// those of a tiling whose threads compute 2 × 2 elements twice: they
// read shared memory once per multiply-add rather than once per two.
// Ties go to the earlier tiling.
func chooseTiling(m, n int) tiling {
	best, work := tilings[0], -1
	for _, t := range tilings {
		w := ceilDiv(m, t.height()) * t.height() * ceilDiv(n, t.width()) * t.width()
		if t.tm*t.tn < 16 {
			w *= 2
		}
		if work < 0 || w < work {
			best, work = t, w
		}
	}
	return best
}

// factor is how a tiled product reads one of its factors. Element (q, k)
// is at row q of the left factor or column q of the right one, and at k
// along the inner dimension.
type factor struct {
	// kFast loads the elements of consecutive k on consecutive threads,
	// rather than of consecutive q, for the order in which they lie in
	// memory.
	kFast bool
	// fixed emits code that computes what reading the elements of q
	// needs of q alone, and returns its variables.
	fixed func(q string) []string
	// at emits code that reads element (q, k), given what fixed
	// computed for q, and returns an expression of its value, 0 outside
	// the factor. It is called only for q and k inside the product.
	at func(fixed []string, k string) string
}

// product is a batch of products that a tiled kernel computes.
type product struct {
	// batches is the number of products. batch emits code that computes
	// what the factors and out need of the product's number, b.
	batches int
	batch   func(b string)
	// m, n and k are expressions of the products' dimensions, which
	// may depend on what batch computes.
	m, n, k string
	// tilesM and tilesN are the tiles of the largest product along
	// its rows and columns.
	tilesM, tilesN int
	t              tiling
	a, b           factor
	// out emits code that computes the output element of the product's
	// element (row, col) and returns its index.
	out func(row, col string) string
}

// tiled emits the kernel that computes p, writing the result through
// the epilogue of g.k, if it has one, and returns its launch.
func (g *gen) tiled(p product) (*Launch, error) {
	t, typ := p.t, typeName(g.k.DType)
	h, w := t.height(), t.width()
	lda, ldb := h+1, w+1 // padded so that copying a slice along k is free of bank conflicts
	ep, err := g.epilogueViews()
	if err != nil {
		return nil, err
	}
	g.line("int tn_ = int(bid) %% %s, t_ = int(bid) / %s;", g.param(p.tilesN), g.param(p.tilesN))
	g.line("int tm_ = t_ %% %s, b_ = t_ / %s;", g.param(p.tilesM), g.param(p.tilesM))
	g.line("int r0_ = tm_ * %d, c0_ = tn_ * %d;", h, w)
	g.line("int tr_ = int(tid) / %d, tc_ = int(tid) %% %d;", t.cols, t.cols)
	p.batch("b_")
	g.line("int m_ = %s, n_ = %s, k_ = %s;", p.m, p.n, p.k)
	g.line("%s %s as_[%d];", g.d.shared, typ, tileK*lda)
	g.line("%s %s bs_[%d];", g.d.shared, typ, tileK*ldb)
	g.line("%s acc_[%d];", typ, t.tm*t.tn)
	g.line("for (int e_ = 0; e_ < %d; e_++) acc_[e_] = %s;", t.tm*t.tn, literal(0, g.k.DType))

	// The elements each thread copies: slot r of a factor is element
	// tid + threads*r of its slice, numbered with k or q fastest.
	type slot struct {
		q, k, at string // the variables of q, of k within the slice, and of the shared element
		fixed    []string
	}
	slots := func(f factor, name string, size, ld int, origin string) []slot {
		var out []slot
		for r := range size * tileK / t.threads() {
			e := fmt.Sprintf("(int(tid) + %d)", r*t.threads())
			var s slot
			s.q, s.k, s.at = g.v(name+"q_"), g.v(name+"k_"), g.v(name+"s_")
			if f.kFast {
				g.line("int %s = %s / %d, %s = %s %% %d;", s.q, e, tileK, s.k, e, tileK)
			} else {
				g.line("int %s = %s %% %d, %s = %s / %d;", s.q, e, size, s.k, e, size)
			}
			g.line("int %s = %s * %d + %s;", s.at, s.k, ld, s.q)
			g.line("%s += %s;", s.q, origin)
			s.fixed = f.fixed(s.q)
			out = append(out, s)
		}
		return out
	}
	sa := slots(p.a, "a", h, lda, "r0_")
	sb := slots(p.b, "b", w, ldb, "c0_")
	g.line("for (int k0_ = 0; k0_ < k_; k0_ += %d) {", tileK)
	copySlice := func(f factor, ss []slot, limit, shared string) {
		for _, s := range ss {
			k, v := g.v("k_"), g.v("v_")
			g.line("int %s = k0_ + %s;", k, s.k)
			g.line("%s %s = %s;", typ, v, literal(0, g.k.DType))
			g.line("if (%s < %s && %s < k_) {", s.q, limit, k)
			g.line("\t%s = %s;", v, f.at(s.fixed, k))
			g.line("}")
			g.line("%s[%s] = %s;", shared, s.at, v)
		}
	}
	copySlice(p.a, sa, "m_", "as_")
	copySlice(p.b, sb, "n_", "bs_")
	g.line("%s;", g.d.sync)
	g.line("#pragma unroll")
	g.line("for (int kk_ = 0; kk_ < %d; kk_++) {", tileK)
	for i := range t.tm {
		g.line("\t%s a%d_ = as_[kk_ * %d + tr_ + %d];", typ, i, lda, i*t.rows)
	}
	for j := range t.tn {
		g.line("\t%s b%d_ = bs_[kk_ * %d + tc_ + %d];", typ, j, ldb, j*t.cols)
	}
	for i := range t.tm {
		for j := range t.tn {
			e := i*t.tn + j
			if g.k.DType == backend.Float32 {
				g.line("\tacc_[%d] = fma(a%d_, b%d_, acc_[%d]);", e, i, j, e)
			} else {
				g.line("\tacc_[%d] = acc_[%d] + a%d_ * b%d_;", e, e, i, j)
			}
		}
	}
	g.line("}")
	g.line("%s;", g.d.sync)
	g.line("}")

	g.line("#pragma unroll")
	g.line("for (int e_ = 0; e_ < %d; e_++) {", t.tm*t.tn)
	g.line("int row_ = r0_ + tr_ + %d * (e_ / %d), col_ = c0_ + tc_ + %d * (e_ %% %d);", t.rows, t.tn, t.cols, t.tn)
	g.line("if (row_ < m_ && col_ < n_) {")
	idx := p.out("row_", "col_")
	g.store(ep, idx, "acc_[e_]")
	g.line("}")
	g.line("}")
	return g.finish(p.batches*p.tilesM*p.tilesN, true)
}

// divisor is a positive divisor d < 2³¹ as kernel parameters: d itself,
// and the multiplier and shift that divide by it without a division
// instruction.
type divisor struct{ d, mul, shift string }

// divisor returns d as parameters (see magic).
func (g *gen) divisor(d int) divisor {
	if d < 1 || d > MaxElems {
		g.err = fmt.Errorf("divisor %d: %w", d, backend.ErrInternal)
		d = 1
	}
	mul, shift := magic(d)
	return divisor{d: g.param(d), mul: g.param(int(int32(mul))), shift: g.param(shift)}
}

// magic returns the multiplier and shift that divide by d, in [1, 2³¹):
// for 0 <= x < 2³¹, x/d is mulhi(2x, mul) >> shift, the high 32 bits of
// the 64-bit product shifted, which is ⌊x·mul / 2^p⌋ with p = 31 + shift.
// With shift = ⌈log₂ d⌉ and mul = ⌈2^p / d⌉, below 2³², mul·d is 2^p + e
// with e < d ≤ 2^shift, so x·mul / 2^p exceeds x/d by x·e / (d·2^p) <
// 1/d, which never carries the quotient past ⌊x/d⌋.
func magic(d int) (mul uint32, shift int) {
	shift = bits.Len(uint(d - 1))
	return uint32((uint64(1)<<(31+shift) + uint64(d) - 1) / uint64(d)), shift
}

// divmod emits code that divides x, which is in [0, 2³¹), by dv and
// returns the variables of the quotient and the remainder.
func (g *gen) divmod(x string, dv divisor) (q, r string) {
	q, r = g.v("q"), g.v("m")
	g.line("int %s = int(mulhi(uint(%s) << 1, uint(%s)) >> uint(%s));", q, x, dv.mul, dv.shift)
	g.line("int %s = %s - %s * %s;", r, x, q, dv.d)
	return q, r
}

// coords emits code that splits x, an index in [0, Π shape) in row-major
// order, into its coordinates along shape, and returns their variables.
func (g *gen) coords(x string, shape []int) []string {
	out := make([]string, len(shape))
	for d := len(shape) - 1; d > 0; d-- {
		x, out[d] = g.divmod(x, g.divisor(shape[d]))
	}
	if len(shape) > 0 {
		out[0] = x
	}
	return out
}

// epilogueViews checks the epilogue of g.k, a product that writes an
// element through its epilogue program, which reads the product as
// input 0 and inputs 2 on as inputs 1 on, through views of a shape that
// lists the output's elements in row-major order. It returns their
// coalesced shape and views, or nil if there is no epilogue.
func (g *gen) epilogueViews() (*epilogueReads, error) {
	k := g.k
	if k.Attrs.Program == nil {
		return nil, nil
	}
	n, _ := backend.NumElements(k.Shape)
	shape := k.Shape
	if len(k.In) > 2 {
		shape = k.In[2].Shape
	}
	views := k.In[2:]
	for _, v := range views {
		if !slices.Equal(v.Shape, shape) {
			return nil, fmt.Errorf("epilogue inputs of shapes %v and %v: %w", shape, v.Shape, backend.ErrInternal)
		}
	}
	if e, _ := backend.NumElements(shape); e != n {
		return nil, fmt.Errorf("epilogue of shape %v for %d elements: %w", shape, n, backend.ErrInternal)
	}
	types := append([]backend.DType{k.DType}, g.ins[2:]...)
	if err := checkProgram(k.Attrs.Program, nil, k.DType, types); err != nil {
		return nil, err
	}
	cshape, cviews := backend.Coalesce(shape, views)
	return &epilogueReads{cshape, cviews}, nil
}

// epilogueReads are the coalesced views through which an epilogue reads
// its inputs.
type epilogueReads struct {
	shape []int
	views []backend.View
}

// store emits code that writes the product's value acc into output
// element idx, through the epilogue ep if it is not nil.
func (g *gen) store(ep *epilogueReads, idx, acc string) {
	if ep == nil {
		g.line("o0[%s] = %s;", idx, acc)
		return
	}
	offs := g.offsets(idx, ep.shape, ep.views)
	vals := g.program(g.k.Attrs.Program, func(a int) string {
		if a == 0 {
			return acc
		}
		return fmt.Sprintf("i%d[%s]", a+1, offs[a-1])
	})
	g.line("o0[%s] = %s;", idx, vals[len(vals)-1])
}
