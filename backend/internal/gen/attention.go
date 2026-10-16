package gen

import (
	"fmt"
	"math"
	"slices"

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
// uses at most: 28 KiB, under Metal's 32.
const attnShared = 7168

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
// scores among those whose group memory, need floats, fits attnShared.
func pickTile(maxRows int, need func(rows, cols int) int) (attnTile, bool) {
	var best attnTile
	for _, rows := range []int{64, 32, 16, 8, 4} {
		if rows > maxRows {
			continue
		}
		for _, cols := range []int{64, 32, 16, 8} {
			if need(rows, cols) <= attnShared && rows*cols > best.rows*best.cols {
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
func planAttention(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	l, err := newAttnLayout(k, ins, 3)
	if err != nil {
		return nil, err
	}
	s := l.s
	dk, dv := s.D, s.DV
	t, ok := pickTile(rowCap(s.Q), func(r, c int) int { return r*(dk+1) + c*max(dk+1, dv) + r*(c+1) })
	if !ok {
		return nil, fmt.Errorf("attention of head size %d and %d exceeds the group memory of a GPU kernel: %w", dk, dv, backend.ErrShape)
	}
	br, bc := t.rows, t.cols
	tpr := GroupSize / br
	cpt := max(ceilDiv(dv, tpr), 1)
	nqb := ceilDiv(s.Q, br)
	groups := s.Batches() * s.Heads * nqb
	qs, kv, ss := 0, br*(dk+1), br*(dk+1)+bc*max(dk+1, dv)
	size := ss + br*(bc+1)

	g := newGen(d, k, outs, ins)
	qIn, kIn, vIn := l.in[0], l.in[1], l.in[2]
	g.line("%s float sh[%d];", d.shared, size)
	g.line("int qb = int(bid) %% %s, pg = int(bid) / %s;", g.param(nqb), g.param(nqb))
	g.line("int gq = pg %% %s, pb = pg / %s;", g.param(s.Group()), g.param(s.Group()))
	g.attnBases(l, "pb")
	g.line("int r = int(tid) / %d, lane = int(tid) %% %d;", tpr, tpr)
	g.line("int nq = %s, nk = %s;", g.param(s.Q), g.param(s.K))
	g.line("int i = qb * %d + r;", br)
	g.line("bool row = i < nq;")
	g.line("float sc = %s;", g.constant(k.Attrs.Scale*math.Log2E, backend.Float32))
	g.line("float m = %s, l = 0.0f;", negInf)
	g.line("float acc[%d];", cpt)
	g.line("for (int t = 0; t < %d; t++) acc[t] = 0.0f;", cpt)
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
	g.line("\tfor (int c = lane; c < %d; c += %d) {", bc, tpr)
	g.line("\t\tint j = j0 + c;")
	g.line("\t\tfloat s = 0.0f;")
	g.line("\t\tfor (int dd = 0; dd < %d; dd++) s = fma(sh[%d + r * %d + dd], sh[%d + c * %d + dd], s);", dk, qs, dk+1, kv, dk+1)
	g.line("\t\tbool vis = row && j < nk && %s;", g.attnVisible("i", "j"))
	if l.masked {
		g.line("\t\tif (vis) s += %s * %s;", l.in[3].at("gq", "i", "j"), literal(math.Log2E, backend.Float32))
	}
	g.line("\t\tsh[%d + r * %d + c] = vis ? s : %s;", ss, bc+1, negInf)
	g.line("\t}")
	g.line("\t%s;", d.sync)
	g.line("\tfloat mb = %s;", negInf)
	g.line("\tfor (int c = 0; c < %d; c++) mb = mx(mb, sh[%d + r * %d + c]);", bc, ss, bc+1)
	g.line("\tfloat mn = mx(m, mb);")
	g.line("\tfloat alpha = mn == %s ? 1.0f : exp2(m - mn);", negInf)
	g.line("\t%s;", d.sync)
	g.line("\tfor (int c = lane; c < %d; c += %d) {", bc, tpr)
	g.line("\t\tfloat s = sh[%d + r * %d + c];", ss, bc+1)
	g.line("\t\tsh[%d + r * %d + c] = s == %s ? 0.0f : exp2(s - mn);", ss, bc+1, negInf)
	g.line("\t}")
	g.line("\tfor (int e = int(tid); e < %d; e += %d) {", bc*dv, GroupSize)
	g.line("\t\tint c = e / %d, dd = e %% %d, j = j0 + c;", max(dv, 1), max(dv, 1))
	g.line("\t\tsh[%d + c * %d + dd] = j < nk ? %s : 0.0f;", kv, dv, vIn.at("gq", "j", "dd"))
	g.line("\t}")
	g.line("\t%s;", d.sync)
	g.line("\tfloat ls = 0.0f;")
	g.line("\tfor (int c = 0; c < %d; c++) ls += sh[%d + r * %d + c];", bc, ss, bc+1)
	g.line("\tl = l * alpha + ls;")
	g.line("\tm = mn;")
	g.line("\tfor (int t = 0; t < %d; t++) {", cpt)
	g.line("\t\tint col = lane + t * %d;", tpr)
	g.line("\t\tif (col >= %d) break;", dv)
	g.line("\t\tfloat a = acc[t] * alpha;")
	g.line("\t\tfor (int c = 0; c < %d; c++) a = fma(sh[%d + r * %d + c], sh[%d + c * %d + col], a);", bc, ss, bc+1, kv, dv)
	g.line("\t\tacc[t] = a;")
	g.line("\t}")
	g.line("}")
	g.line("if (row) {")
	g.line("\tfloat inv = l > 0.0f ? 1.0f / l : 0.0f;")
	g.line("\tint o = pg * nq + i;")
	g.line("\tfor (int t = 0; t < %d; t++) {", cpt)
	g.line("\t\tint col = lane + t * %d;", tpr)
	g.line("\t\tif (col < %d) o0[o * %d + col] = acc[t] * inv;", dv, dv)
	g.line("\t}")
	g.line("\tif (lane == 0) o0[%s + o] = l > 0.0f ? (m + log2(l)) * %s : %s;",
		g.param(s.Batches()*s.Heads*s.Q*dv), literal(math.Ln2, backend.Float32), negInf)
	g.line("}")
	return g.finish(groups, true)
}

// planAttentionGrad returns the backward attention kernel.
func planAttentionGrad(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	l, err := newAttnLayout(k, ins, 6)
	if err != nil {
		return nil, err
	}
	s := l.s
	dk, dv := s.D, s.DV
	// A key group: a block of keys, the rows, and a tile of queries.
	kt, ok := pickTile(rowCap(s.K), func(r, c int) int {
		return (r+c)*(dk+1) + (r+c)*(dv+1) + 2*r*(c+1) + 2*c
	})
	// A query group: a tile of queries, the rows, and a block of keys.
	qt, ok2 := pickTile(rowCap(s.Q), func(r, c int) int {
		return (r+c)*(dk+1) + (r+c)*(dv+1) + r*(c+1) + 2*r
	})
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
	// score emits the probability pr and its gradient ds of query ii
	// (row qr of the query tile at qs, the result gradients' at os) and
	// key j (row kr of the key block at ks, the values' at vs).
	score := func(indent, qr, kr, ii, j, valid string, qs, os, ks, vs, ls, ds int) {
		line := func(format string, args ...any) { g.line(indent+format, args...) }
		line("float s = 0.0f, dp = 0.0f;")
		line("for (int dd = 0; dd < %d; dd++) s = fma(sh[%d + %s * %d + dd], sh[%d + %s * %d + dd], s);", dk, ks, kr, dk+1, qs, qr, dk+1)
		line("for (int dd = 0; dd < %d; dd++) dp = fma(sh[%d + %s * %d + dd], sh[%d + %s * %d + dd], dp);", dv, vs, kr, dv+1, os, qr, dv+1)
		line("float lse = sh[%d + %s];", ls, qr)
		line("bool vis = %s && lse != %s && %s;", valid, negInf, g.attnVisible(ii, j))
		if maskIn != nil {
			line("if (vis) s += %s * %s;", maskIn.at("gq", ii, j), log2e)
		}
		line("float pr = vis ? exp2(s - lse) : 0.0f;")
		line("float ds = pr * (dp - sh[%d + %s]);", ds, qr)
	}
	// stage emits the loads of rows [x0, x0+n) of in, of cols columns, into
	// the tile at off, scaled by mul, with rows width+1 apart.
	stage := func(in *attnIn, x0 string, n, cols, off int, limit, mul string) {
		g.line("for (int e = int(tid); e < %d; e += %d) {", n*cols, GroupSize)
		g.line("\tint rr = e / %d, c = e %% %d, x = %s + rr;", max(cols, 1), max(cols, 1), x0)
		g.line("\tsh[%d + rr * %d + c] = x < %s ? %s%s : 0.0f;", off, cols+1, limit, in.at("gq", "x", "c"), mul)
		g.line("}")
	}
	// rowStats emits the loads of the logsumexps, in base 2, of rows
	// [x0, x0+n) of the query tile, into ls, and, after a barrier, the
	// sums of their result gradients times results into ds.
	rowStats := func(x0 string, n, os, ls, ds int) {
		g.line("for (int e = int(tid); e < %d; e += %d) {", n, GroupSize)
		g.line("\tint x = %s + e;", x0)
		g.line("\tsh[%d + e] = x < nq ? %s * %s : %s;", ls, lseIn.at("gq", "x", "0"), log2e, negInf)
		g.line("}")
		g.line("%s;", d.sync)
		g.line("for (int e = int(tid); e < %d; e += %d) {", n, GroupSize)
		g.line("\tint x = %s + e;", x0)
		g.line("\tfloat a = 0.0f;")
		g.line("\tif (x < nq) for (int dd = 0; dd < %d; dd++) a = fma(sh[%d + e * %d + dd], %s, a);", dv, os, dv+1, oIn.at("gq", "x", "dd"))
		g.line("\tsh[%d + e] = a;", ds)
		g.line("}")
	}

	g.line("%s float sh[%d];", d.shared, max(
		(kt.rows+kt.cols)*(dk+dv+2)+2*kt.rows*(kt.cols+1)+2*kt.cols,
		(qt.rows+qt.cols)*(dk+dv+2)+qt.rows*(qt.cols+1)+2*qt.rows))
	g.line("float sc = %s;", g.constant(k.Attrs.Scale*math.Log2E, backend.Float32))
	g.line("int nq = %s, nk = %s;", g.param(s.Q), g.param(s.K))
	g.line("if (int(bid) < %s) {", g.param(keyGroups))
	{
		br, bc := kt.rows, kt.cols
		tpc := GroupSize / br
		ck, cv := max(ceilDiv(dk, tpc), 1), max(ceilDiv(dv, tpc), 1)
		ks := 0
		vs := ks + br*(dk+1)
		qs := vs + br*(dv+1)
		os := qs + bc*(dk+1)
		ps := os + bc*(dv+1)
		dss := ps + br*(bc+1)
		ls := dss + br*(bc+1)
		ds := ls + bc
		g.line("int kb = int(bid) %% %s, pb = int(bid) / %s;", g.param(nkb), g.param(nkb))
		g.attnBases(l, "pb")
		g.line("int c = int(tid) / %d, lane = int(tid) %% %d;", tpc, tpc)
		g.line("int j = kb * %d + c;", br)
		g.line("float dk[%d], dv[%d];", ck, cv)
		g.line("for (int t = 0; t < %d; t++) dk[t] = 0.0f;", ck)
		g.line("for (int t = 0; t < %d; t++) dv[t] = 0.0f;", cv)
		g.line("int gq = 0;")
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
		rowStats(fmt.Sprintf("qb * %d", bc), bc, os, ls, ds)
		g.line("%s;", d.sync)
		g.line("for (int rr = lane; rr < %d; rr += %d) {", bc, tpc)
		g.line("\tint ii = qb * %d + rr;", bc)
		score("\t", "rr", "c", "ii", "j", "j < nk && ii < nq", qs, os, ks, vs, ls, ds)
		g.line("\tsh[%d + c * %d + rr] = pr;", ps, bc+1)
		g.line("\tsh[%d + c * %d + rr] = ds;", dss, bc+1)
		g.line("}")
		g.line("%s;", d.sync)
		g.line("for (int t = 0; t < %d; t++) {", cv)
		g.line("\tint col = lane + t * %d;", tpc)
		g.line("\tif (col >= %d) break;", dv)
		g.line("\tfloat a = dv[t];")
		g.line("\tfor (int rr = 0; rr < %d; rr++) a = fma(sh[%d + c * %d + rr], sh[%d + rr * %d + col], a);", bc, ps, bc+1, os, dv+1)
		g.line("\tdv[t] = a;")
		g.line("}")
		g.line("for (int t = 0; t < %d; t++) {", ck)
		g.line("\tint col = lane + t * %d;", tpc)
		g.line("\tif (col >= %d) break;", dk)
		g.line("\tfloat a = dk[t];")
		g.line("\tfor (int rr = 0; rr < %d; rr++) a = fma(sh[%d + c * %d + rr], sh[%d + rr * %d + col], a);", bc, dss, bc+1, qs, dk+1)
		g.line("\tdk[t] = a;")
		g.line("}")
		g.line("}")
		g.line("}")
		g.line("if (j < nk) {")
		g.line("\tint o = pb * nk + j;")
		g.line("\tfor (int t = 0; t < %d; t++) {", ck)
		g.line("\t\tint col = lane + t * %d;", tpc)
		g.line("\t\tif (col < %d) o0[%s + o * %d + col] = dk[t] * %s;", dk, g.param(nq), dk, ln2)
		g.line("\t}")
		g.line("\tfor (int t = 0; t < %d; t++) {", cv)
		g.line("\t\tint col = lane + t * %d;", tpc)
		g.line("\t\tif (col < %d) o0[%s + o * %d + col] = dv[t];", dv, g.param(nq+nk), dv)
		g.line("\t}")
		g.line("}")
	}
	g.line("} else {")
	{
		br, bc := qt.rows, qt.cols
		tpr := GroupSize / br
		cq := max(ceilDiv(dk, tpr), 1)
		qs := 0
		os := qs + br*(dk+1)
		ks := os + br*(dv+1)
		vs := ks + bc*(dk+1)
		dss := vs + bc*(dv+1)
		ls := dss + br*(bc+1)
		ds := ls + br
		g.line("int b = int(bid) - %s;", g.param(keyGroups))
		g.line("int qb = b %% %s, pg = b / %s;", g.param(nqb), g.param(nqb))
		g.line("int gq = pg %% %s, pb = pg / %s;", g.param(s.Group()), g.param(s.Group()))
		g.attnBases(l, "pb")
		g.line("int r = int(tid) / %d, lane = int(tid) %% %d;", tpr, tpr)
		g.line("int i = qb * %d + r;", br)
		g.line("float dq[%d];", cq)
		g.line("for (int t = 0; t < %d; t++) dq[t] = 0.0f;", cq)
		stage(qIn, fmt.Sprintf("qb * %d", br), br, dk, qs, "nq", " * sc")
		stage(goIn, fmt.Sprintf("qb * %d", br), br, dv, os, "nq", "")
		rowStats(fmt.Sprintf("qb * %d", br), br, os, ls, ds)
		g.attnHi(fmt.Sprintf("qb * %d", br), br, l)
		g.line("for (int j0 = 0; j0 < hi; j0 += %d) {", bc)
		g.line("%s;", d.sync)
		stage(kIn, "j0", bc, dk, ks, "nk", "")
		stage(vIn, "j0", bc, dv, vs, "nk", "")
		g.line("%s;", d.sync)
		g.line("for (int c = lane; c < %d; c += %d) {", bc, tpr)
		g.line("\tint j = j0 + c;")
		score("\t", "r", "c", "i", "j", "i < nq && j < nk", qs, os, ks, vs, ls, ds)
		g.line("\tsh[%d + r * %d + c] = ds;", dss, bc+1)
		g.line("}")
		g.line("%s;", d.sync)
		g.line("for (int t = 0; t < %d; t++) {", cq)
		g.line("\tint col = lane + t * %d;", tpr)
		g.line("\tif (col >= %d) break;", dk)
		g.line("\tfloat a = dq[t];")
		g.line("\tfor (int c = 0; c < %d; c++) a = fma(sh[%d + r * %d + c], sh[%d + c * %d + col], a);", bc, dss, bc+1, ks, dk+1)
		g.line("\tdq[t] = a;")
		g.line("}")
		g.line("}")
		g.line("if (i < nq) {")
		g.line("\tfloat scale = %s;", g.constant(k.Attrs.Scale, backend.Float32))
		g.line("\tfor (int t = 0; t < %d; t++) {", cq)
		g.line("\t\tint col = lane + t * %d;", tpr)
		g.line("\t\tif (col < %d) o0[(pg * nq + i) * %d + col] = dq[t] * scale;", dk, dk)
		g.line("\t}")
		g.line("}")
	}
	g.line("}")
	return g.finish(groups, true)
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
