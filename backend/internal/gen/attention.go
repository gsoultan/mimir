package gen

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// The attention kernels follow FlashAttention. A group of the forward
// kernel computes a tile of query rows of one head: it walks the keys in
// blocks, staging each block of keys and then of values in group memory,
// and keeps each row's running maximum, sum and result, rescaling them as
// the maximum grows, so no more than a tile of scores ever exists. The
// rows are spread over the group's threads, a few threads per row, each
// summing its share of the result's columns in registers.
//
// The backward kernel is one launch of two kinds of groups, so that no
// two groups write one gradient and none needs atomics. A key group holds
// a block of keys and values of a key head and walks the queries of every
// query head that reads it, accumulating the blocks' key and value
// gradients; a query group holds a tile of queries and walks the keys,
// accumulating their gradient. Both recompute the probabilities from the
// logsumexp the forward kernel saved.
//
// Scores are computed in base 2, the queries scaled by log2(e) along with
// the attention's scale, so the softmax is exp2 of their differences.
// Tile sizes depend on the head dimensions, which are constants of the
// source: kernels of one head size share their program.

// attnShared is the number of floats of group memory an attention kernel
// uses at most by default: 28 KiB, under Metal's 32, which leaves room
// for two groups on a CUDA multiprocessor of 64 KiB.
const attnShared = 7168

// attnBudget returns the floats of group memory the attention kernels of
// d may use: attnShared, or the larger number MIMIR_ATTN_SHARED sets,
// which benchmarks set to compare tile sizes, up to the dialect's limit:
// 32 KiB on Metal, the 48 KiB of a CUDA group's static shared memory and
// 64 KiB on HIP.
func attnBudget(d *Dialect) int {
	n := attnSharedEnv()
	if n <= attnShared {
		return attnShared
	}
	limit := 12288
	switch {
	case d == Metal:
		limit = 8192
	case strings.HasPrefix(d.Name, "hip"):
		limit = 16384
	}
	return min(n, limit)
}

var attnSharedEnv = sync.OnceValue(func() int {
	n, _ := strconv.Atoi(os.Getenv("MIMIR_ATTN_SHARED"))
	return n
})

// attnIn is how an attention kernel reads one of its inputs: element
// (x, y) of query head g of a problem, a batch item and key head, is at
// base + g·gs + x·xs + y·ys.
type attnIn struct {
	idx   int
	dt    backend.DType
	split backend.View // of [B..., HK, G, X, Y]

	// Set by attnBases.
	base       string
	gs, xs, ys string
}

// attnLayout is the geometry of an attention kernel and how it reads its
// inputs.
type attnLayout struct {
	s      backend.AttentionShape
	in     []*attnIn
	masked bool
	lead   []int // B..., HK
}

// newAttnLayout checks the inputs of the attention kernel k against its
// geometry; mask is the index of the optional mask input.
func newAttnLayout(k *backend.Kernel, ins []backend.Buffer, mask int) (*attnLayout, error) {
	if len(k.In) < 3 || len(k.In) != len(ins) {
		return nil, fmt.Errorf("%s of %d inputs: %w", k.Op, len(k.In), backend.ErrInternal)
	}
	var maskShape []int
	if len(k.In) > mask {
		maskShape = k.In[mask].Shape
	}
	s, err := backend.NewAttentionShape(k.In[0].Shape, k.In[1].Shape, k.In[2].Shape, maskShape)
	if err != nil {
		return nil, err
	}
	want := []int{s.Size()}
	if k.Op == backend.OpAttentionGrad {
		want = []int{s.GradSize()}
		if len(k.In) < 6 || !slices.Equal(k.In[3].Shape, s.OutShape()) || !slices.Equal(k.In[4].Shape, s.LSEShape()) ||
			!slices.Equal(k.In[5].Shape, s.OutShape()) {
			return nil, fmt.Errorf("%s of shape %v: %w", k.Op, s, backend.ErrInternal)
		}
	}
	if k.DType != backend.Float32 || !slices.Equal(k.Shape, want) {
		return nil, fmt.Errorf("%s into %s%v: %w", k.Op, k.DType, k.Shape, backend.ErrInternal)
	}
	if len(k.In) > mask+1 {
		return nil, fmt.Errorf("%s of %d inputs: %w", k.Op, len(k.In), backend.ErrInternal)
	}
	for i, b := range ins {
		if !b.DType().IsFloat() {
			return nil, fmt.Errorf("%s of %s input %d: %w", k.Op, b.DType(), i, backend.ErrDType)
		}
	}
	l := &attnLayout{s: s, masked: len(k.In) > mask, lead: slices.Concat(s.Batch, []int{s.KVHeads})}
	for i, v := range k.In {
		if k.Op == backend.OpAttentionGrad && i == 4 {
			// The logsumexp, [B..., H, Q], as [B..., H, Q, 1].
			v = backend.View{Shape: append(slices.Clone(v.Shape), 1), Strides: append(slices.Clone(v.Strides), 0), Offset: v.Offset}
		}
		l.in = append(l.in, &attnIn{idx: i, dt: ins[i].DType(), split: s.SplitHeads(v)})
	}
	return l, nil
}

// attnBases emits the offsets of problem p, a batch item and key head, in
// every input, and sets the inputs' strides.
func (g *gen) attnBases(l *attnLayout, p string) {
	views := make([]backend.View, len(l.in))
	for i, in := range l.in {
		views[i] = backend.View{Shape: l.lead, Strides: in.split.Strides[:len(l.lead)], Offset: in.split.Offset}
	}
	shape, cviews := backend.Coalesce(l.lead, views)
	offs := g.offsets(p, shape, cviews)
	for i, in := range l.in {
		r := len(in.split.Strides)
		in.base = offs[i]
		in.gs, in.xs, in.ys = g.param(in.split.Strides[r-3]), g.param(in.split.Strides[r-2]), g.param(in.split.Strides[r-1])
	}
}

// at returns the expression of element (x, y) of query head gq of the
// current problem of in, as a float.
func (in *attnIn) at(gq, x, y string) string {
	return load(in.dt, fmt.Sprintf("i%d[%s + %s * %s + %s * %s + %s * %s]", in.idx, in.base, gq, in.gs, x, in.xs, y, in.ys))
}

// attnTile is a tile of rows × cols.
type attnTile struct{ rows, cols int }

// pickTile returns the tile, of at most maxRows rows, that holds the most
// scores among those whose group memory, need floats, fits budget, and
// the squarest of those, whose loops are the least short.
func pickTile(maxRows, budget int, need func(rows, cols int) int) (attnTile, bool) {
	var best attnTile
	for _, rows := range []int{64, 32, 16, 8, 4} {
		if rows > maxRows {
			continue
		}
		for _, cols := range []int{64, 32, 16, 8} {
			more, square := rows*cols > best.rows*best.cols, rows*cols == best.rows*best.cols && min(rows, cols) > min(best.rows, best.cols)
			if need(rows, cols) <= budget && (more || square) {
				best = attnTile{rows, cols}
			}
		}
	}
	return best, best.rows > 0
}

// rowCap returns the rows a tile needs at most for n rows: the smallest
// power of two from 4 that holds them.
func rowCap(n int) int {
	c := 4
	for c < n && c < 64 {
		c *= 2
	}
	return c
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

// negInf is the expression of -Inf.
const negInf = "fbits(0xff800000u)"

// visible returns the condition that query i may see key j.
func (g *gen) attnVisible(i, j string) string {
	if !g.k.Attrs.Causal {
		return "true"
	}
	return fmt.Sprintf("%s <= %s + %s", j, i, g.param(g.k.Attrs.CausalOffset))
}

// attnHi emits hi, the end of the keys a tile of queries from q0 may see.
func (g *gen) attnHi(q0 string, rows int, l *attnLayout) {
	g.line("int hi = nk;")
	if g.k.Attrs.Causal {
		g.line("{")
		g.line("\tint end = %s + %d;", q0, rows)
		g.line("\tint last = (end < nq ? end : nq) - 1 + %s + 1;", g.param(g.k.Attrs.CausalOffset))
		g.line("\thi = last < 0 ? 0 : last < hi ? last : hi;")
		g.line("}")
	}
}

// planAttention returns the forward attention kernel.
//
// For each block of keys, the threads compute the tile of scores as the
// backward kernel's do, a few rows and columns each in registers, and
// write them to group memory; the threads of each row then take the
// row's new maximum, exponentiate the row's scores and sum them; and the
// threads multiply the probabilities by the block of values, each
// accumulating a few rows and columns of the result in registers,
// rescaled as the rows' maxima grow.
func planAttention(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	l, err := newAttnLayout(k, ins, 3)
	if err != nil {
		return nil, err
	}
	s := l.s
	dk, dv := s.D, s.DV
	// A tile stages the queries, a block of keys and then of values in
	// the same place, the scores, and each row's rescaling, maximum and
	// sum.
	need := func(r, c int) int { return r*(dk+1) + c*max(dk, dv) + c + r*(c+1) + 3*r }
	t, ok := pickTile(rowCap(s.Q), attnBudget(d), need)
	if !ok {
		return nil, fmt.Errorf("attention of head size %d and %d exceeds the group memory of a GPU kernel: %w", dk, dv, backend.ErrShape)
	}
	br, bc := t.rows, t.cols
	tpr := GroupSize / br
	st := newScoreTile(br, bc, GroupSize)
	at := newAccTile(br, dv, bc, GroupSize)
	nqb := ceilDiv(s.Q, br)
	groups := s.Batches() * s.Heads * nqb
	qs := 0
	kv := qs + br*(dk+1)
	ps := kv + bc*(max(dk, dv)+1)
	as := ps + br*(bc+1)
	ms := as + br
	lsum := ms + br
	size := lsum + br

	g := newGen(d, k, outs, ins)
	qIn, kIn, vIn := l.in[0], l.in[1], l.in[2]
	g.line("%s float sh[%d];", d.shared, size)
	g.line("int qb = int(bid) %% %s, pg = int(bid) / %s;", g.param(nqb), g.param(nqb))
	g.line("int gq = pg %% %s, pb = pg / %s;", g.param(s.Group()), g.param(s.Group()))
	g.attnBases(l, "pb")
	g.line("int nq = %s, nk = %s;", g.param(s.Q), g.param(s.K))
	g.line("float sc = %s;", g.constant(k.Attrs.Scale*math.Log2E, backend.Float32))
	// Each row's statistics are kept by its tpr threads alike.
	g.line("int r = int(tid) / %d, lane = int(tid) %% %d;", tpr, tpr)
	g.line("float m = %s, l = 0.0f;", negInf)
	// The thread's scores, rows a, a+gr, ... and cols b, b+gc, ...
	g.line("int a = int(tid) / %d, b = int(tid) %% %d;", st.gc(), st.gc())
	g.line("bool act = int(tid) < %d;", st.gr()*st.gc())
	// The thread's results, rows ar, ar+gr, ... and cols ad, ad+gd, ...
	g.line("float acc[%d];", at.tr*at.td)
	g.line("for (int t = 0; t < %d; t++) acc[t] = 0.0f;", at.tr*at.td)
	g.line("int ar = int(tid) / %d, ad = int(tid) %% %d;", at.gd, at.gd)
	g.attnHi(fmt.Sprintf("qb * %d", br), br, l)
	g.line("for (int e = int(tid); e < %d; e += %d) {", br*dk, GroupSize)
	g.line("\tint rr = e / %d, c = e %% %d, ii = qb * %d + rr;", max(dk, 1), max(dk, 1), br)
	g.line("\tsh[%d + rr * %d + c] = ii < nq ? %s * sc : 0.0f;", qs, dk+1, qIn.at("gq", "ii", "c"))
	g.line("}")
	g.line("for (int j0 = 0; j0 < hi; j0 += %d) {", bc)
	g.line("\t%s;", d.sync)
	g.line("\tfor (int e = int(tid); e < %d; e += %d) {", bc*dk, GroupSize)
	g.line("\t\tint c = e / %d, dd = e %% %d, j = j0 + c;", max(dk, 1), max(dk, 1))
	g.line("\t\tsh[%d + c * %d + dd] = j < nk ? %s : 0.0f;", kv, dk+1, kIn.at("gq", "j", "dd"))
	g.line("\t}")
	g.line("\t%s;", d.sync)
	g.line("\tif (act) {")
	n := st.tr * st.tc
	g.line("\t\tfloat sv[%d];", n)
	g.line("\t\tfor (int t = 0; t < %d; t++) sv[t] = 0.0f;", n)
	g.line("\t\tfor (int dd = 0; dd < %d; dd++) {", dk)
	for i := range st.tr {
		g.line("\t\t\tfloat x%d = sh[%d + (a + %d) * %d + dd];", i, qs, i*st.gr(), dk+1)
	}
	for j := range st.tc {
		g.line("\t\t\tfloat y%d = sh[%d + (b + %d) * %d + dd];", j, kv, j*st.gc(), dk+1)
	}
	for i := range st.tr {
		for j := range st.tc {
			g.line("\t\t\tsv[%d] = fma(x%d, y%d, sv[%d]);", i*st.tc+j, i, j, i*st.tc+j)
		}
	}
	g.line("\t\t}")
	for i := range st.tr {
		for j := range st.tc {
			g.line("\t\t{")
			g.line("\t\t\tint rr = a + %d, c = b + %d, i = qb * %d + rr, j = j0 + c;", i*st.gr(), j*st.gc(), br)
			g.line("\t\t\tbool vis = i < nq && j < nk && %s;", g.attnVisible("i", "j"))
			g.line("\t\t\tfloat s = sv[%d];", i*st.tc+j)
			if l.masked {
				g.line("\t\t\tif (vis) s += %s * %s;", l.in[3].at("gq", "i", "j"), literal(math.Log2E, backend.Float32))
			}
			g.line("\t\t\tsh[%d + rr * %d + c] = vis ? s : %s;", ps, bc+1, negInf)
			g.line("\t\t}")
		}
	}
	g.line("\t}")
	g.line("\t%s;", d.sync)
	g.line("\tfloat mb = %s;", negInf)
	g.line("\tfor (int c = 0; c < %d; c++) mb = mx(mb, sh[%d + r * %d + c]);", bc, ps, bc+1)
	g.line("\tfloat mn = mx(m, mb);")
	g.line("\tfloat alpha = mn == %s ? 1.0f : exp2(m - mn);", negInf)
	g.line("\t%s;", d.sync)
	g.line("\tfor (int c = lane; c < %d; c += %d) {", bc, tpr)
	g.line("\t\tfloat s = sh[%d + r * %d + c];", ps, bc+1)
	g.line("\t\tsh[%d + r * %d + c] = s == %s ? 0.0f : exp2(s - mn);", ps, bc+1, negInf)
	g.line("\t}")
	g.line("\tif (lane == 0) sh[%d + r] = alpha;", as)
	g.line("\tfor (int e = int(tid); e < %d; e += %d) {", bc*dv, GroupSize)
	g.line("\t\tint c = e / %d, dd = e %% %d, j = j0 + c;", max(dv, 1), max(dv, 1))
	g.line("\t\tsh[%d + c * %d + dd] = j < nk ? %s : 0.0f;", kv, dv+1, vIn.at("gq", "j", "dd"))
	g.line("\t}")
	g.line("\t%s;", d.sync)
	g.line("\tfloat ls = 0.0f;")
	g.line("\tfor (int c = 0; c < %d; c++) ls += sh[%d + r * %d + c];", bc, ps, bc+1)
	g.line("\tl = l * alpha + ls;")
	g.line("\tm = mn;")
	g.line("\tif (%s) {", at.active("a"))
	for i := range at.tr {
		g.line("\t\t{")
		g.line("\t\t\tfloat al = sh[%d + ar + %d];", as, i*at.gr())
		g.line("\t\t\tfor (int t = 0; t < %d; t++) acc[%d + t] *= al;", at.td, i*at.td)
		g.line("\t\t}")
	}
	g.line("\t\tfor (int c = 0; c < %d; c++) {", bc)
	for i := range at.tr {
		g.line("\t\t\tfloat x%d = sh[%d + (ar + %d) * %d + c];", i, ps, i*at.gr(), bc+1)
	}
	for u := range at.td {
		col := fmt.Sprintf("ad + %d", u*at.gd)
		if dv%at.gd != 0 {
			g.line("\t\t\tfloat y%d = %s < %d ? sh[%d + c * %d + %s] : 0.0f;", u, col, dv, kv, dv+1, col)
		} else {
			g.line("\t\t\tfloat y%d = sh[%d + c * %d + %s];", u, kv, dv+1, col)
		}
	}
	for i := range at.tr {
		for u := range at.td {
			g.line("\t\t\tacc[%d] = fma(x%d, y%d, acc[%d]);", i*at.td+u, i, u, i*at.td+u)
		}
	}
	g.line("\t\t}")
	g.line("\t}")
	g.line("}")
	g.line("if (lane == 0) {")
	g.line("\tsh[%d + r] = m;", ms)
	g.line("\tsh[%d + r] = l;", lsum)
	g.line("}")
	g.line("%s;", d.sync)
	g.line("if (%s) {", at.active("a"))
	for i := range at.tr {
		for u := range at.td {
			g.line("\t{")
			g.line("\t\tint rr = ar + %d, col = ad + %d, i = qb * %d + rr;", i*at.gr(), u*at.gd, br)
			g.line("\t\tfloat ll = sh[%d + rr], inv = ll > 0.0f ? 1.0f / ll : 0.0f;", lsum)
			g.line("\t\tif (i < nq && col < %d) o0[(pg * nq + i) * %d + col] = acc[%d] * inv;", dv, dv, i*at.td+u)
			g.line("\t}")
		}
	}
	g.line("}")
	g.line("if (lane == 0 && qb * %d + r < nq) {", br)
	g.line("\tint o = pg * nq + qb * %d + r;", br)
	g.line("\to0[%s + o] = l > 0.0f ? (m + log2(l)) * %s : %s;",
		g.param(s.Batches()*s.Heads*s.Q*dv), literal(math.Ln2, backend.Float32), negInf)
	g.line("}")
	return g.finish(groups, true)
}

// planAttentionGrad returns the backward attention kernel.
//
// Both kinds of group compute a tile of scores, keys by queries in a key
// group and queries by keys in a query group, and then multiply it by a
// block of rows: the probabilities and their gradients by the result
// gradients and the queries in a key group, the probabilities' gradients
// by the keys in a query group. Each thread computes a few scores, and a
// few rows and columns of the products, in registers, so that it reads
// group memory less than once per multiply-add. The sums of the result
// gradients times the results, which the probabilities' gradients
// subtract, are summed by all the group's threads, a few per query.
func planAttentionGrad(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	l, err := newAttnLayout(k, ins, 6)
	if err != nil {
		return nil, err
	}
	s := l.s
	dk, dv := s.D, s.DV
	const nt = GroupSize
	// A tile of rows × cols scores stages both blocks of rows of the two
	// products that make them, and the scores, which also hold the
	// partial sums of the queries' statistics, and the statistics of the
	// queries, the rows of a query group and the cols of a key group.
	need := func(rows, cols, queries int) int {
		return (rows+cols)*(dk+dv+2) + max(rows*cols, nt) + 2*queries
	}
	kt, ok := pickTile(rowCap(s.K), attnBudget(d), func(r, c int) int { return need(r, c, c) })
	qt, ok2 := pickTile(rowCap(s.Q), attnBudget(d), func(r, c int) int { return need(r, c, r) })
	if !ok || !ok2 {
		return nil, fmt.Errorf("attention gradient of head size %d and %d exceeds the group memory of a GPU kernel: %w", dk, dv, backend.ErrShape)
	}
	nkb, nqb := ceilDiv(s.K, kt.rows), ceilDiv(s.Q, qt.rows)
	keyGroups := s.Batches() * s.KVHeads * nkb
	groups := keyGroups + s.Batches()*s.Heads*nqb
	nq := s.Batches() * s.Heads * s.Q * dk
	nk := s.Batches() * s.KVHeads * s.K * dk
	log2e, ln2 := literal(math.Log2E, backend.Float32), literal(math.Ln2, backend.Float32)

	g := newGen(d, k, outs, ins)
	qIn, kIn, vIn, oIn, lseIn, goIn := l.in[0], l.in[1], l.in[2], l.in[3], l.in[4], l.in[5]
	var maskIn *attnIn
	if l.masked {
		maskIn = l.in[6]
	}
	// stage emits the loads of rows [x0, x0+n) of in, of cols columns, into
	// the tile at off, scaled by mul, with rows width+1 apart.
	stage := func(in *attnIn, x0 string, n, cols, off int, limit, mul string) {
		g.line("for (int e = int(tid); e < %d; e += %d) {", n*cols, nt)
		g.line("\tint rr = e / %d, c = e %% %d, x = %s + rr;", max(cols, 1), max(cols, 1), x0)
		g.line("\tsh[%d + rr * %d + c] = x < %s ? %s%s : 0.0f;", off, cols+1, limit, in.at("gq", "x", "c"), mul)
		g.line("}")
	}
	// stats emits the loads of the logsumexps, in base 2, of the n
	// queries from x0 into ls, and the sums of their result gradients
	// times their results into ds, which the group's threads sum in parts
	// in the scratch at part, followed by a barrier.
	stats := func(x0 string, n, part, ls, ds int) {
		tpd := nt / n
		g.line("{")
		g.line("\tint e = int(tid) / %d, w = int(tid) %% %d, x = %s + e;", tpd, tpd, x0)
		g.line("\tfloat a = 0.0f;")
		g.line("\tif (x < nq) for (int dd = w; dd < %d; dd += %d) a = fma(%s, %s, a);", dv, tpd, goIn.at("gq", "x", "dd"), oIn.at("gq", "x", "dd"))
		g.line("\tsh[%d + int(tid)] = a;", part)
		g.line("\tif (w == 0) sh[%d + e] = x < nq ? %s * %s : %s;", ls, lseIn.at("gq", "x", "0"), log2e, negInf)
		g.line("}")
		g.line("%s;", d.sync)
		g.line("for (int e = int(tid); e < %d; e += %d) {", n, nt)
		g.line("\tfloat a = 0.0f;")
		g.line("\tfor (int w = 0; w < %d; w++) a += sh[%d + e * %d + w];", tpd, part, tpd)
		g.line("\tsh[%d + e] = a;", ds)
		g.line("}")
		g.line("%s;", d.sync)
	}
	// scores emits the scores of the thread's tile of t, which holds its
	// rows at rowK and rowV and its cols at colK and colV; their
	// probabilities end in sv and their gradients in pv. The query and key
	// of row i and col j are those query(i, j) and key(i, j) return,
	// with qx the query's place in the statistics at ls and ds.
	scores := func(t scoreTile, rowK, rowV, colK, colV, ls, ds int, keysAreRows bool, q0, k0 string) {
		n := t.tr * t.tc
		g.line("float sv[%d], pv[%d];", n, n)
		g.line("for (int t = 0; t < %d; t++) { sv[t] = 0.0f; pv[t] = 0.0f; }", n)
		g.line("int a = int(tid) / %d, b = int(tid) %% %d;", t.gc(), t.gc())
		g.line("bool act = int(tid) < %d;", t.gr()*t.gc())
		g.line("if (act) {")
		dot := func(rows, cols, width int, acc string) {
			g.line("\tfor (int dd = 0; dd < %d; dd++) {", width)
			for i := range t.tr {
				g.line("\t\tfloat x%d = sh[%d + (a + %d) * %d + dd];", i, rows, i*t.gr(), width+1)
			}
			for j := range t.tc {
				g.line("\t\tfloat y%d = sh[%d + (b + %d) * %d + dd];", j, cols, j*t.gc(), width+1)
			}
			for i := range t.tr {
				for j := range t.tc {
					g.line("\t\t%s[%d] = fma(x%d, y%d, %s[%d]);", acc, i*t.tc+j, i, j, acc, i*t.tc+j)
				}
			}
			g.line("\t}")
		}
		dot(rowK, colK, dk, "sv")
		dot(rowV, colV, dv, "pv")
		g.line("}")
		for i := range t.tr {
			for j := range t.tc {
				row, col := fmt.Sprintf("(a + %d)", i*t.gr()), fmt.Sprintf("(b + %d)", j*t.gc())
				qx, kx := col, row
				if !keysAreRows {
					qx, kx = row, col
				}
				e := i*t.tc + j
				g.line("{")
				g.line("\tint ii = %s + %s, jj = %s + %s;", q0, qx, k0, kx)
				g.line("\tfloat lse = act ? sh[%d + %s] : %s;", ls, qx, negInf)
				g.line("\tbool vis = act && ii < nq && jj < nk && lse != %s && %s;", negInf, g.attnVisible("ii", "jj"))
				g.line("\tfloat s = sv[%d];", e)
				if maskIn != nil {
					g.line("\tif (vis) s += %s * %s;", maskIn.at("gq", "ii", "jj"), log2e)
				}
				g.line("\tfloat pr = vis ? exp2(s - lse) : 0.0f;")
				g.line("\tpv[%d] = vis ? pr * (pv[%d] - sh[%d + %s]) : 0.0f;", e, e, ds, qx)
				g.line("\tsv[%d] = pr;", e)
				g.line("}")
			}
		}
	}
	// store emits the stores of the values v of the thread's tile of t
	// into the scores at off, rows cols apart.
	store := func(t scoreTile, v string, off int) {
		g.line("if (act) {")
		for i := range t.tr {
			for j := range t.tc {
				g.line("\tsh[%d + (a + %d) * %d + b + %d] = %s[%d];", off, i*t.gr(), t.cols, j*t.gc(), v, i*t.tc+j)
			}
		}
		g.line("}")
	}
	// product emits acc += the scores at off, rows of a, times the block of
	// cols-rows of width+1 at y, for the thread's part of a, named name.
	product := func(a accTile, name string, off, y int) {
		g.line("if (%s) for (int c = 0; c < %d; c++) {", a.active(name), a.inner)
		for i := range a.tr {
			g.line("\tfloat x%d = sh[%d + (%sr + %d) * %d + c];", i, off, name, i*a.gr(), a.inner)
		}
		for t := range a.td {
			col := fmt.Sprintf("%sd + %d", name, t*a.gd)
			if a.width%a.gd != 0 {
				g.line("\tfloat y%d = %s < %d ? sh[%d + c * %d + %s] : 0.0f;", t, col, a.width, y, a.width+1, col)
			} else {
				g.line("\tfloat y%d = sh[%d + c * %d + %s];", t, y, a.width+1, col)
			}
		}
		for i := range a.tr {
			for t := range a.td {
				g.line("\t%s[%d] = fma(x%d, y%d, %s[%d]);", name, i*a.td+t, i, t, name, i*a.td+t)
			}
		}
		g.line("}")
	}
	// declare emits the accumulators of a, named name, and the thread's
	// row and column in it.
	declare := func(a accTile, name string) {
		g.line("float %s[%d];", name, a.tr*a.td)
		g.line("for (int t = 0; t < %d; t++) %s[t] = 0.0f;", a.tr*a.td, name)
		g.line("int %sr = int(tid) / %d, %sd = int(tid) %% %d;", name, a.gd, name, a.gd)
	}
	// write emits the stores of the accumulators of a, named name, of rows
	// from x0 of at most limit, into the output from base, times mul.
	write := func(a accTile, name, x0, limit, base, mul string) {
		g.line("if (%s) {", a.active(name))
		for i := range a.tr {
			for t := range a.td {
				g.line("\t{")
				g.line("\t\tint x = %s + %sr + %d, col = %sd + %d;", x0, name, i*a.gr(), name, t*a.gd)
				g.line("\t\tif (x < %s && col < %d) o0[%s + x * %d + col] = %s[%d]%s;", limit, a.width, base, a.width, name, i*a.td+t, mul)
				g.line("\t}")
			}
		}
		g.line("}")
	}

	g.line("%s float sh[%d];", d.shared, max(need(kt.rows, kt.cols, kt.cols), need(qt.rows, qt.cols, qt.rows)))
	g.line("float sc = %s;", g.constant(k.Attrs.Scale*math.Log2E, backend.Float32))
	g.line("int nq = %s, nk = %s;", g.param(s.Q), g.param(s.K))
	g.line("if (int(bid) < %s) {", g.param(keyGroups))
	{
		br, bc := kt.rows, kt.cols
		t := newScoreTile(br, bc, nt)
		av, ak := newAccTile(br, dv, bc, nt), newAccTile(br, dk, bc, nt)
		ks := 0
		vs := ks + br*(dk+1)
		qs := vs + br*(dv+1)
		os := qs + bc*(dk+1)
		ps := os + bc*(dv+1)
		ls := ps + max(br*bc, nt)
		ds := ls + bc
		g.line("int kb = int(bid) %% %s, pb = int(bid) / %s;", g.param(nkb), g.param(nkb))
		g.attnBases(l, "pb")
		g.line("int gq = 0;")
		declare(av, "gv")
		declare(ak, "gk")
		stage(kIn, fmt.Sprintf("kb * %d", br), br, dk, ks, "nk", "")
		stage(vIn, fmt.Sprintf("kb * %d", br), br, dv, vs, "nk", "")
		g.line("int qlo = 0;")
		if k.Attrs.Causal {
			g.line("qlo = kb * %d - %s;", br, g.param(k.Attrs.CausalOffset))
			g.line("qlo = (qlo > 0 ? qlo : 0) / %d;", bc)
		}
		g.line("for (gq = 0; gq < %s; gq++) {", g.param(s.Group()))
		g.line("for (int qb = qlo; qb * %d < nq; qb++) {", bc)
		g.line("%s;", d.sync)
		stage(qIn, fmt.Sprintf("qb * %d", bc), bc, dk, qs, "nq", " * sc")
		stage(goIn, fmt.Sprintf("qb * %d", bc), bc, dv, os, "nq", "")
		stats(fmt.Sprintf("qb * %d", bc), bc, ps, ls, ds)
		g.line("{")
		scores(t, ks, vs, qs, os, ls, ds, true, fmt.Sprintf("qb * %d", bc), fmt.Sprintf("kb * %d", br))
		store(t, "sv", ps)
		g.line("%s;", d.sync)
		product(av, "gv", ps, os)
		g.line("%s;", d.sync)
		store(t, "pv", ps)
		g.line("%s;", d.sync)
		product(ak, "gk", ps, qs)
		g.line("}")
		g.line("}")
		g.line("}")
		write(ak, "gk", fmt.Sprintf("kb * %d", br), "nk", fmt.Sprintf("%s + pb * nk * %d", g.param(nq), dk), " * "+ln2)
		write(av, "gv", fmt.Sprintf("kb * %d", br), "nk", fmt.Sprintf("%s + pb * nk * %d", g.param(nq+nk), dv), "")
	}
	g.line("} else {")
	{
		br, bc := qt.rows, qt.cols
		t := newScoreTile(br, bc, nt)
		aq := newAccTile(br, dk, bc, nt)
		qs := 0
		os := qs + br*(dk+1)
		ks := os + br*(dv+1)
		vs := ks + bc*(dk+1)
		dss := vs + bc*(dv+1)
		ls := dss + max(br*bc, nt)
		ds := ls + br
		g.line("int b0 = int(bid) - %s;", g.param(keyGroups))
		g.line("int qb = b0 %% %s, pg = b0 / %s;", g.param(nqb), g.param(nqb))
		g.line("int gq = pg %% %s, pb = pg / %s;", g.param(s.Group()), g.param(s.Group()))
		g.attnBases(l, "pb")
		declare(aq, "gd")
		stage(qIn, fmt.Sprintf("qb * %d", br), br, dk, qs, "nq", " * sc")
		stage(goIn, fmt.Sprintf("qb * %d", br), br, dv, os, "nq", "")
		stats(fmt.Sprintf("qb * %d", br), br, dss, ls, ds)
		g.attnHi(fmt.Sprintf("qb * %d", br), br, l)
		g.line("for (int j0 = 0; j0 < hi; j0 += %d) {", bc)
		g.line("%s;", d.sync)
		stage(kIn, "j0", bc, dk, ks, "nk", "")
		stage(vIn, "j0", bc, dv, vs, "nk", "")
		g.line("%s;", d.sync)
		g.line("{")
		scores(t, qs, os, ks, vs, ls, ds, false, fmt.Sprintf("qb * %d", br), "j0")
		store(t, "pv", dss)
		g.line("%s;", d.sync)
		product(aq, "gd", dss, ks)
		g.line("}")
		g.line("}")
		g.line("float scale = %s;", g.constant(k.Attrs.Scale, backend.Float32))
		write(aq, "gd", fmt.Sprintf("qb * %d", br), "nq", fmt.Sprintf("pg * nq * %d", dk), " * scale")
	}
	g.line("}")
	return g.finish(groups, true)
}

// scoreTile is how the threads of a group split a tile of rows × cols
// scores: thread (a, b) computes the tr × tc scores of rows a, a+gr, ...
// and cols b, b+gc, ..., so that it reads tr + tc elements of group
// memory for every tr·tc multiply-adds.
type scoreTile struct{ rows, cols, tr, tc int }

func newScoreTile(rows, cols, threads int) scoreTile {
	t := scoreTile{rows: rows, cols: cols, tr: 1, tc: 1}
	for t.tr*t.tc*threads < rows*cols {
		if t.tr <= t.tc && t.tr < rows || t.tc >= cols {
			t.tr *= 2
		} else {
			t.tc *= 2
		}
	}
	return t
}

func (t scoreTile) gr() int { return t.rows / t.tr }
func (t scoreTile) gc() int { return t.cols / t.tc }

// accTile is how the threads of a group split the product of rows ×
// inner scores and an inner × width block: thread (r, d) accumulates the
// tr × td elements of rows r, r+gr, ... and columns d, d+gd, ...
type accTile struct{ rows, width, inner, gd, tr, td int }

func newAccTile(rows, width, inner, threads int) accTile {
	gd := 1
	for gd < 16 && gd < width {
		gd *= 2
	}
	a := accTile{rows: rows, width: width, inner: inner, gd: gd}
	a.tr, a.td = max(ceilDiv(rows, a.gr()), 1), max(ceilDiv(width, gd), 1)
	return a
}

func (a accTile) gr() int { return GroupSize / a.gd }

// active returns the condition that the thread has rows of a, whose row
// in a is the variable name+"r".
func (a accTile) active(name string) string {
	if a.rows >= a.gr() {
		return "true"
	}
	return fmt.Sprintf("%sr < %d", name, a.rows)
}

// planSliceScatter copies input 0 with the block Attrs.Ranges replaced by
// input 1. Written over input 0, it writes the block alone.
func planSliceScatter(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer, n int) (*Launch, error) {
	if len(k.In) != 2 || len(k.Attrs.Ranges) != len(k.Shape) || len(k.In[1].Shape) != len(k.Shape) {
		return nil, fmt.Errorf("slice scatter of %d inputs: %w", len(k.In), backend.ErrInternal)
	}
	block := make([]int, len(k.Shape))
	lo := make([]int, len(k.Shape))
	for d, r := range k.Attrs.Ranges {
		block[d], lo[d] = r[1]-r[0], r[0]
	}
	if !slices.Equal(block, k.In[1].Shape) {
		return nil, fmt.Errorf("slice scatter of %v into %v: %w", k.In[1].Shape, k.Attrs.Ranges, backend.ErrInternal)
	}
	g := newGen(d, k, outs, ins)
	dst := backend.Contiguous(k.Shape).Shrink(k.Attrs.Ranges)
	if outs[0] == ins[0] {
		// In place: one thread per element of the block.
		m, _ := backend.NumElements(block)
		if m == 0 {
			return nil, nil
		}
		g.line("if (gid >= uint(%s)) return;", g.param(m))
		g.line("int i = int(gid);")
		shape, views := backend.Coalesce(block, []backend.View{dst, k.In[1]})
		offs := g.offsets("i", shape, views)
		g.line("o0[%s] = i1[%s];", offs[0], offs[1])
		return g.perElement(m)
	}
	g.line("if (gid >= uint(%s)) return;", g.param(n))
	g.line("int i = int(gid);")
	coords := g.index("i", k.Shape)
	g.line("bool inside = true;")
	g.line("int src = %s;", g.param(k.In[1].Offset))
	for d, c := range coords {
		g.line("inside = inside && %s >= %s && %s < %s;", c, g.param(lo[d]), c, g.param(lo[d]+block[d]))
		g.line("src += (%s - %s) * %s;", c, g.param(lo[d]), g.param(k.In[1].Strides[d]))
	}
	shape, views := backend.Coalesce(k.Shape, k.In[:1])
	off := g.offsets("i", shape, views)[0]
	g.line("o0[i] = inside ? i1[src] : i0[%s];", off)
	return g.perElement(n)
}
