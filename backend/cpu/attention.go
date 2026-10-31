package cpu

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// Attention follows FlashAttention. The scores of a block of queries and a
// block of keys are computed into a tile with the GEMM micro-kernels, the
// softmax is applied to the tile online, rescaling what the query rows
// summed over earlier blocks whenever their maximum grows, and the tile's
// probabilities are multiplied into the result at once, so no more than a
// tile of scores ever exists. The forward kernel saves the logsumexp of
// each row; the backward kernel recomputes each tile's probabilities from
// it instead of storing them.
//
// The work is split by key head: the query heads that share a key head
// are rows of one problem, so their keys and values are packed once for
// all of them, and a decoding step of several query heads is one tile.
// The kernels walk the blocks of keys in the outer loop and the tiles of
// queries in the inner one, which packs every block once. A problem's
// rows, of the result or of the query gradient, are contiguous in the
// output, so the kernels accumulate into it directly.
//
// A problem of few query rows, such as a decoding step, has too few to
// fill a tile: packing the keys transposed for the micro-kernel would
// cost more than the scores. Its scores are dot products of each query
// with the keys, read in place, and its whole rows of probabilities are
// multiplied by the values at once; see forwardFew.
const (
	attnRows = 64  // query rows of a tile
	attnKeys = 128 // keys of a block
	attnFew  = 16  // query rows of a problem that forwardFew computes
	// attnAxpyRows is the most query rows of a problem whose products
	// with the values forwardFew sums row by row instead of as a tile.
	attnAxpyRows = 4
	// attnSerial is the most multiply-adds of a kernel of few rows that
	// one goroutine computes alone, in less time than the helpers take to
	// join it.
	attnSerial = 1 << 14
)

// attnGeom is the layout of an attention kernel's inputs: per problem,
// the offset of its first row in each input, and the strides that walk
// the problem's query heads (g), positions (x) and features (y).
type attnGeom struct {
	s        backend.AttentionShape
	problems int // batches × key heads
	rows     int // query rows of a problem: group × queries
	in       []attnView
}

type attnView struct {
	data    []float32
	half    []uint16      // the elements of a half input that forwardFew converts as it reads them, instead of data
	dt      backend.DType // the dtype of half
	offs    []int         // of each problem
	g, x, y int           // strides
}

// at returns the offset of element (x, y) of query head g of problem p.
func (v *attnView) at(p, g, x, y int) int { return v.offs[p] + g*v.g + x*v.x + y*v.y }

// newAttnGeom returns the layout of the inputs ins of the attention
// kernel k, read as float32: a half input is converted first.
func newAttnGeom(ctx context.Context, b *Backend, k *backend.Kernel, ins []backend.Buffer) (*attnGeom, error) {
	return newAttnGeomFew(ctx, b, k, ins, false)
}

// newAttnGeomFew is newAttnGeom that, with few set, leaves the half keys
// and values of a problem of few rows, whose features are contiguous, for
// forwardFew to convert: it converts the keys its rows see, which in a
// decoding step from a cache that is mostly empty are few.
func newAttnGeomFew(ctx context.Context, b *Backend, k *backend.Kernel, ins []backend.Buffer, few bool) (*attnGeom, error) {
	q, kk, v := k.In[0].Shape, k.In[1].Shape, k.In[2].Shape
	var mask []int
	maskAt := 3
	if k.Op == backend.OpAttentionGrad {
		maskAt = 6
	}
	if len(k.In) > maskAt {
		mask = k.In[maskAt].Shape
	}
	s, err := backend.NewAttentionShape(q, kk, v, mask)
	if err != nil {
		return nil, err
	}
	g := &attnGeom{s: s, problems: s.Batches() * s.KVHeads, rows: s.Group() * s.Q}
	lead := len(s.Batch) + 1 // the dimensions of the problems: B..., HK
	few = few && g.rows <= attnFew && k.Op == backend.OpAttention
	for i, view := range k.In {
		var av attnView
		if h, dt, ok := halfOf(ins[i]); ok && few && (i == 1 || i == 2) && view.Strides[len(view.Strides)-1] == 1 {
			av.half, av.dt = h, dt
		} else {
			data, err := floats(ctx, b, ins[i])
			if err != nil {
				return nil, err
			}
			av.data = data
		}
		if k.Op == backend.OpAttentionGrad && i == 4 {
			// The logsumexp, [B..., H, Q], as [B..., H, Q, 1].
			view = backend.View{Shape: append(slices.Clone(view.Shape), 1), Strides: append(slices.Clone(view.Strides), 0), Offset: view.Offset}
		}
		if len(view.Shape) != len(s.Batch)+3 {
			return nil, fmt.Errorf("attention input %d of shape %v: %w", i, view.Shape, backend.ErrInternal)
		}
		split := s.SplitHeads(view)
		r := len(split.Strides)
		av.offs = batchOffsets(split.Shape[:lead], split)
		av.g, av.x, av.y = split.Strides[r-3], split.Strides[r-2], split.Strides[r-1]
		g.in = append(g.in, av)
	}
	return g, nil
}

// floats returns the elements of a float buffer as float32s, converting a
// half buffer.
func floats(ctx context.Context, b *Backend, buf backend.Buffer) ([]float32, error) {
	switch buf := buf.(type) {
	case *buffer[float32]:
		return buf.data, nil
	case *buffer[bf16]:
		return halfFloats(ctx, b, backend.BFloat16, bitsOf(buf.data))
	case *buffer[f16]:
		return halfFloats(ctx, b, backend.Float16, bitsOf(buf.data))
	}
	return nil, fmt.Errorf("attention of %s: %w", buf.DType(), backend.ErrDType)
}

// halfOf returns the elements of a half buffer, and its dtype.
func halfOf(buf backend.Buffer) ([]uint16, backend.DType, bool) {
	switch buf := buf.(type) {
	case *buffer[bf16]:
		return bitsOf(buf.data), backend.BFloat16, true
	case *buffer[f16]:
		return bitsOf(buf.data), backend.Float16, true
	}
	return nil, 0, false
}

// rows returns the n rows of features of problem pr of v from position
// 0, each m features long: in place, at off, rows v.x and features v.y
// apart, or converted from half into buf, contiguous, as an operand.
func (v *attnView) rows(buf *[]float32, pr, n, m int) operand {
	off := v.at(pr, 0, 0, 0)
	if v.half == nil {
		return operand{data: v.data, off: off, rs: v.x, cs: v.y}
	}
	*buf = grow(*buf, n*m)
	for j := range n {
		loadHalf(v.dt, (*buf)[j*m:][:m], v.half, off+j*v.x, 1)
	}
	return operand{data: *buf, rs: m, cs: 1}
}

// extent returns how many of the first keys some row of problem pr sees:
// causality or a mask of -Inf hides every key after them from every row.
func (g *attnGeom) extent(k *backend.Kernel, pr int) int {
	n := 0
	for r := range g.rows {
		hd, i := r/g.s.Q, r%g.s.Q
		e := visible(k, i, 0, g.s.K)
		if len(g.in) > 3 {
			m := &g.in[3]
			off := m.at(pr, hd, i, 0)
			for e > n && math.IsInf(float64(m.data[off+(e-1)*m.y]), -1) {
				e--
			}
		}
		n = max(n, e)
	}
	return n
}

func halfFloats(ctx context.Context, b *Backend, dt backend.DType, h []uint16) ([]float32, error) {
	out := make([]float32, len(h))
	err := b.parallelFor(ctx, len(h), elementGrain, func(lo, hi int) error {
		loadHalf(dt, out[lo:hi], h, lo, 1)
		return nil
	})
	return out, err
}

// attnTile is a tile of query rows [r0, r1) of a problem.
type attnTile struct {
	r0, r1 int
	hi     int // the largest query position among them
}

// tileRows returns the query rows of a tile for the micro-kernel gk: a
// whole number of its panels.
func tileRows(gk *gemmKernel) int { return attnRows / gk.mr * gk.mr }

// tiles returns the tiles of n rows that cover the rows [r0, r1) of a
// problem.
func (g *attnGeom) tiles(r0, r1, n int) []attnTile {
	var ts []attnTile
	for t0 := r0; t0 < r1; t0 += n {
		t := attnTile{r0: t0, r1: min(t0+n, r1)}
		// The positions of rows t0, ... are t0%Q, ..., wrapping at Q.
		if t.r1-t.r0 >= g.s.Q || t0%g.s.Q+(t.r1-t.r0) > g.s.Q {
			t.hi = g.s.Q - 1
		} else {
			t.hi = (t.r1 - 1) % g.s.Q
		}
		ts = append(ts, t)
	}
	return ts
}

// visible returns the keys [j0, j1) that the query at position i may
// attend to among keys [j0, j1), as far as causality hides them: j1 is
// lowered to the first hidden key.
func visible(k *backend.Kernel, i, j0, j1 int) int {
	if !k.Attrs.Causal {
		return j1
	}
	return max(j0, min(j1, i+k.Attrs.CausalOffset+1))
}

// attnWorkspace holds a worker's packed panels and tiles.
type attnWorkspace struct {
	q, k, v, kr []float32 // packed panels
	qb, ob      []float32 // packed B panels of every tile's queries and result gradients
	oa          []float32 // packed A panels of the result gradients
	s, p, pp    []float32 // tiles of scores and probabilities, and packed probabilities
	m, l, d     []float32 // per row
	tile        [maxTile]float32
}

var attnWorkspaces = sync.Pool{New: func() any { return new(attnWorkspace) }}

// split returns how many parts the work of each of problems problems,
// spanning units units, splits into so that every worker has work.
func (b *Backend) split(problems, units int) int {
	if problems >= b.workers {
		return 1
	}
	return max(1, min(units, (b.workers+problems-1)/problems))
}

// prepare masks and scales a tile of scores s, with rows ld elements
// apart, of the rows [t.r0, t.r1) of problem pr and the keys [j0, j0+bc):
// it adds the mask, hides the keys causality hides, and leaves each row
// ready for the softmax.
func (g *attnGeom) prepare(k *backend.Kernel, pr int, t attnTile, j0, bc int, s []float32, ld int, maskAt int) {
	ninf := float32(math.Inf(-1))
	for r := t.r0; r < t.r1; r++ {
		row := s[(r-t.r0)*ld:][:bc]
		hd, i := r/g.s.Q, r%g.s.Q
		if maskAt < len(g.in) {
			m := &g.in[maskAt]
			off := m.at(pr, hd, i, j0)
			j := 0
			if rv := rowVector; rv != nil && m.y == 1 && len(row) >= rv.lanes {
				j = len(row) &^ (rv.lanes - 1)
				rv.addVV(&row[0], &row[0], &m.data[off], j)
			}
			for ; j < len(row); j++ {
				row[j] += m.data[off+j*m.y]
			}
		}
		for j := visible(k, i, j0, j0+bc) - j0; j < bc; j++ {
			row[j] = ninf
		}
	}
}

// attention computes the attention kernel k into out: the result, then
// the logsumexp of each query row.
func attention(ctx context.Context, b *Backend, k *backend.Kernel, out []float32, ins []backend.Buffer) error {
	g, err := newAttnGeomFew(ctx, b, k, ins, k.In[1].Strides[len(k.In[1].Strides)-1] == 1)
	if err != nil {
		return err
	}
	s := g.s
	if s.Size() != len(out) {
		return fmt.Errorf("attention output of %d elements, want %d: %w", len(out), s.Size(), backend.ErrInternal)
	}
	if g.rows == 0 || g.problems == 0 {
		return nil
	}
	if g.rows <= attnFew && g.in[1].y == 1 {
		if g.problems*g.rows*s.K*s.D < attnSerial {
			ws := attnWorkspaces.Get().(*attnWorkspace)
			defer attnWorkspaces.Put(ws)
			for pr := range g.problems {
				g.forwardFew(k, ws, pr, out)
			}
			return nil
		}
		return b.parallelFor(ctx, g.problems, 1, func(lo, hi int) error {
			ws := attnWorkspaces.Get().(*attnWorkspace)
			defer attnWorkspaces.Put(ws)
			for pr := lo; pr < hi; pr++ {
				g.forwardFew(k, ws, pr, out)
			}
			return nil
		})
	}
	tr := tileRows(gemmKernels[0])
	tiles := (g.rows + tr - 1) / tr
	parts := b.split(g.problems, tiles)
	per := (tiles + parts - 1) / parts * tr
	return b.parallelFor(ctx, g.problems*parts, 1, func(lo, hi int) error {
		ws := attnWorkspaces.Get().(*attnWorkspace)
		defer attnWorkspaces.Put(ws)
		for u := lo; u < hi; u++ {
			pr, part := u/parts, u%parts
			r0, r1 := part*per, min((part+1)*per, g.rows)
			if r0 < r1 {
				g.forward(k, ws, pr, r0, r1, out)
			}
		}
		return nil
	})
}

// forward computes the rows [r0, r1) of problem pr.
func (g *attnGeom) forward(k *backend.Kernel, ws *attnWorkspace, pr, r0, r1 int, out []float32) {
	gk := gemmKernels[0]
	s := g.s
	q, kk, v := &g.in[0], &g.in[1], &g.in[2]
	scale := float32(k.Attrs.Scale)
	rows := r1 - r0
	o := out[(pr*g.rows+r0)*s.DV:][:rows*s.DV]
	lse := out[s.Size()-s.Batches()*s.Heads*s.Q+pr*g.rows+r0:][:rows]
	clear(o)
	ws.m, ws.l = grow(ws.m, rows), grow(ws.l, rows)
	for i := range rows {
		ws.m[i], ws.l[i] = float32(math.Inf(-1)), 0
	}

	// The queries, scaled, as panels of mr rows.
	ws.q = grow(ws.q, roundUp(rows, gk.mr)*s.D+packSlack)
	for r := r0; r < r1; r += gk.mr {
		panel := ws.q[(r-r0)*s.D:][:gk.mr*s.D]
		n := min(gk.mr, r1-r)
		for i := range n {
			off := q.at(pr, (r+i)/s.Q, (r+i)%s.Q, 0)
			for p := range s.D {
				panel[p*gk.mr+i] = q.data[off+p*q.y] * scale
			}
		}
		for i := n; i < gk.mr; i++ {
			for p := range s.D {
				panel[p*gk.mr+i] = 0
			}
		}
	}

	tiles := g.tiles(r0, r1, tileRows(gk))
	last := s.K // the keys any row may attend to end before last
	if k.Attrs.Causal {
		hi := 0
		for _, t := range tiles {
			hi = max(hi, t.hi)
		}
		last = visible(k, hi, 0, s.K)
	}
	ws.s = grow(ws.s, attnRows*attnKeys)
	for j0 := 0; j0 < last; j0 += attnKeys {
		bc := min(attnKeys, s.K-j0)
		// Kᵀ, D × bc, and V, bc × DV, as panels of nr columns.
		ws.k = grow(ws.k, roundUp(bc, gk.nr)*s.D+packSlack)
		packB(ws.k, operand{data: kk.data, off: kk.at(pr, 0, j0, 0), rs: kk.y, cs: kk.x}, 0, s.D, 0, bc, gk.nr)
		ws.v = grow(ws.v, roundUp(s.DV, gk.nr)*bc+packSlack)
		packB(ws.v, operand{data: v.data, off: v.at(pr, 0, j0, 0), rs: v.x, cs: v.y}, 0, bc, 0, s.DV, gk.nr)
		for _, t := range tiles {
			if visible(k, t.hi, j0, j0+bc) == j0 {
				continue // every key of the block is hidden from every row
			}
			br := t.r1 - t.r0
			sc := ws.s[:br*attnKeys]
			macroKernel(gk, ws.q[(t.r0-r0)*s.D:], ws.k, s.D, br, bc, sc, attnKeys, false, &ws.tile)
			g.prepare(k, pr, t, j0, bc, sc, attnKeys, 3)
			for i := range br {
				row := sc[i*attnKeys:][:bc]
				r := t.r0 - r0 + i
				mOld := ws.m[r]
				mNew := max(mOld, rowMax(row))
				if math.IsInf(float64(mNew), -1) {
					clear(row) // no key visible so far
					continue
				}
				sum := expShift(row, mNew)
				if mOld != mNew {
					alpha := expf(mOld - mNew)
					ws.l[r] *= alpha
					scaleRow(o[r*s.DV:][:s.DV], alpha)
				}
				ws.l[r] += sum
				ws.m[r] = mNew
			}
			ws.p = grow(ws.p, roundUp(br, gk.mr)*bc+packSlack)
			// ws.s holds the rows past the tile that packing may read.
			packA(ws.p, operand{data: ws.s, rs: attnKeys, cs: 1}, 0, br, 0, bc, gk.mr)
			macroKernel(gk, ws.p, ws.v, bc, br, s.DV, o[(t.r0-r0)*s.DV:], s.DV, true, &ws.tile)
		}
	}
	for r := range rows {
		l := ws.l[r]
		if l == 0 {
			clear(o[r*s.DV:][:s.DV])
			lse[r] = float32(math.Inf(-1))
			continue
		}
		scaleRow(o[r*s.DV:][:s.DV], 1/l)
		lse[r] = ws.m[r] + float32(math.Log(float64(l)))
	}
}

// forwardFew computes problem pr, whose rows are few and whose keys have
// contiguous features, from whole rows of scores: each the dot products
// of a query, scaled, with the keys it may see.
func (g *attnGeom) forwardFew(k *backend.Kernel, ws *attnWorkspace, pr int, out []float32) {
	gk := gemmKernels[0]
	s := g.s
	q, kk, v := &g.in[0], &g.in[1], &g.in[2]
	scale := float32(k.Attrs.Scale)
	// The keys after the first n, which no row sees, add nothing.
	rows, n := g.rows, g.extent(k, pr)
	o := out[pr*rows*s.DV:][:rows*s.DV]
	lse := out[s.Size()-s.Batches()*s.Heads*s.Q+pr*rows:][:rows]
	// Packing reads whole panels of rows.
	ws.s = grow(ws.s, roundUp(rows, gk.mr)*n)
	ws.q = grow(ws.q, s.D)
	ws.l = grow(ws.l, rows)
	ko := kk.rows(&ws.k, pr, n, s.D)
	keys, kx := ko.data[ko.off:], ko.rs
	var d [4]float32
	for r := range rows {
		hd, i := r/s.Q, r%s.Q
		qr := ws.q[:s.D]
		off := q.at(pr, hd, i, 0)
		p := 0
		if rv := rowVector; rv != nil && q.y == 1 && s.D >= rv.lanes {
			p = s.D &^ (rv.lanes - 1)
			rv.mulVS(&qr[0], &q.data[off], scale, p)
		}
		for ; p < len(qr); p++ {
			qr[p] = q.data[off+p*q.y] * scale
		}
		row := ws.s[r*n:][:n]
		seen := visible(k, i, 0, n)
		j := 0
		if rv := rowVector; rv != nil && rv.dotRows != nil && seen >= 4 && s.D%rv.lanes == 0 && s.D > 0 {
			j = seen &^ 3
			_ = keys[(j-1)*kx+s.D-1]
			rv.dotRows(&qr[0], &keys[0], kx, s.D, j, &row[0])
		}
		for ; j+4 <= seen; j += 4 {
			dots4(&d, qr, keys[j*kx:], kx)
			copy(row[j:], d[:])
		}
		for ; j < seen; j++ {
			dots4(&d, qr, keys[j*kx:], 0)
			row[j] = d[0]
		}
		g.prepare(k, pr, attnTile{r0: r, r1: r + 1, hi: i}, 0, n, row, n, 3)
		m := rowMax(row)
		if math.IsInf(float64(m), -1) {
			clear(row) // no key is visible
			ws.l[r], lse[r] = 0, float32(math.Inf(-1))
			continue
		}
		ws.l[r] = expShift(row, m)
		lse[r] = m + float32(math.Log(float64(ws.l[r])))
	}
	if n == 0 {
		clear(o)
		return
	}
	vo := v.rows(&ws.kr, pr, n, s.DV)
	if rows <= attnAxpyRows && vo.cs == 1 {
		// A few rows, as one query of each head of a group decoding a
		// token: summing the values row by row, each read once for every
		// query, costs less than packing them for the tile kernel.
		clear(o)
		vals := vo.data[vo.off:]
		_ = vals[(n-1)*vo.rs+s.DV-1]
		c := 0
		if rv := rowVector; rv != nil && rv.axpyRows != nil {
			for ; c+4*rv.lanes <= s.DV; c += 4 * rv.lanes {
				for r := range rows {
					rv.axpyRows(&ws.s[r*n], &vals[c], vo.rs, n, &o[r*s.DV+c])
				}
			}
		}
		if c < s.DV {
			for j := range n {
				vrow := vals[j*vo.rs+c:][:s.DV-c]
				for r := range rows {
					if p := ws.s[r*n+j]; p != 0 {
						axpy(p, vrow, o[r*s.DV+c:][:s.DV-c])
					}
				}
			}
		}
	} else {
		ws.p = grow(ws.p, roundUp(rows, gk.mr)*n+packSlack)
		packA(ws.p, operand{data: ws.s, rs: n, cs: 1}, 0, rows, 0, n, gk.mr)
		ws.v = grow(ws.v, roundUp(s.DV, gk.nr)*n+packSlack)
		packB(ws.v, vo, 0, n, 0, s.DV, gk.nr)
		macroKernel(gk, ws.p, ws.v, n, rows, s.DV, o, s.DV, false, &ws.tile)
	}
	for r, l := range ws.l[:rows] {
		if l == 0 {
			clear(o[r*s.DV:][:s.DV])
			continue
		}
		scaleRow(o[r*s.DV:][:s.DV], 1/l)
	}
}

// attentionGrad computes the gradient kernel k into out: the gradients of
// the queries, the keys and the values.
func attentionGrad(ctx context.Context, b *Backend, k *backend.Kernel, out []float32, ins []backend.Buffer) error {
	g, err := newAttnGeom(ctx, b, k, ins)
	if err != nil {
		return err
	}
	s := g.s
	if s.GradSize() != len(out) {
		return fmt.Errorf("attention gradient of %d elements, want %d: %w", len(out), s.GradSize(), backend.ErrInternal)
	}
	if g.problems == 0 {
		return nil
	}
	blocks := (s.K + attnKeys - 1) / attnKeys
	parts := b.split(g.problems, max(blocks, 1))
	per := (blocks + parts - 1) / parts * attnKeys
	// Each part of a problem computes the key and value gradients of its
	// keys, and its share of the query gradient: the first part into
	// out, and the others into partial sums added to it afterwards.
	dq := out[:s.Batches()*s.Heads*s.Q*s.D]
	var partials []float32
	if parts > 1 {
		partials = make([]float32, (parts-1)*len(dq))
	}
	err = b.parallelFor(ctx, g.problems*parts, 1, func(lo, hi int) error {
		ws := attnWorkspaces.Get().(*attnWorkspace)
		defer attnWorkspaces.Put(ws)
		for u := lo; u < hi; u++ {
			pr, part := u/parts, u%parts
			j0, j1 := part*per, min((part+1)*per, s.K)
			rows := dq[pr*g.rows*s.D:][:g.rows*s.D]
			if part > 0 {
				rows = partials[(part-1)*len(dq)+pr*g.rows*s.D:][:g.rows*s.D]
			}
			g.backward(k, ws, pr, j0, max(j0, j1), rows, out)
		}
		return nil
	})
	if err != nil || parts == 1 {
		return err
	}
	return b.parallelFor(ctx, len(dq), elementGrain, func(lo, hi int) error {
		for p := range parts - 1 {
			for i, v := range partials[p*len(dq)+lo : p*len(dq)+hi] {
				dq[lo+i] += v
			}
		}
		return nil
	})
}

// backward computes the gradients of the keys [j0, j1) of problem pr,
// and adds their share of its query gradient to dq.
func (g *attnGeom) backward(k *backend.Kernel, ws *attnWorkspace, pr, j0, j1 int, dq, out []float32) {
	gk := gemmKernels[0]
	s := g.s
	q, kk, v, o, lse, do := &g.in[0], &g.in[1], &g.in[2], &g.in[3], &g.in[4], &g.in[5]
	scale := float32(k.Attrs.Scale)
	nq := s.Batches() * s.Heads * s.Q * s.D
	nk := s.Batches() * s.KVHeads * s.K * s.D
	dk := out[nq+pr*s.K*s.D:][:s.K*s.D][j0*s.D : j1*s.D]
	dv := out[nq+nk+pr*s.K*s.DV:][:s.K*s.DV][j0*s.DV : j1*s.DV]
	clear(dq)
	clear(dk)
	clear(dv)
	if j0 == j1 || g.rows == 0 {
		return
	}

	// Per row: the logsumexp, and D = Σ dO·O, the term of the softmax
	// gradient that the row's probabilities share.
	ws.l, ws.d = grow(ws.l, g.rows), grow(ws.d, g.rows)
	for r := range g.rows {
		hd, i := r/s.Q, r%s.Q
		ws.l[r] = lse.data[lse.at(pr, hd, i, 0)]
		oo, od := o.at(pr, hd, i, 0), do.at(pr, hd, i, 0)
		var d float32
		for c := range s.DV {
			d += o.data[oo+c*o.y] * do.data[od+c*do.y]
		}
		ws.d[r] = d
	}
	// The scaled queries as A panels and as B panels of each tile, and
	// the result gradients likewise.
	tiles := g.tiles(0, g.rows, tileRows(gk))
	ws.q = grow(ws.q, roundUp(g.rows, gk.mr)*s.D+packSlack)
	ws.oa = grow(ws.oa, roundUp(g.rows, gk.mr)*s.DV+packSlack)
	ws.qb = grow(ws.qb, len(tiles)*attnRows*roundUp(s.D, gk.nr)+packSlack)
	ws.ob = grow(ws.ob, len(tiles)*attnRows*roundUp(s.DV, gk.nr)+packSlack)
	packRows(ws.q, gk.mr, g.rows, s.D, func(r int) (int, []float32, int) { return q.at(pr, r/s.Q, r%s.Q, 0), q.data, q.y }, scale)
	packRows(ws.oa, gk.mr, g.rows, s.DV, func(r int) (int, []float32, int) { return do.at(pr, r/s.Q, r%s.Q, 0), do.data, do.y }, 1)
	for ti, t := range tiles {
		packCols(ws.qb[ti*attnRows*roundUp(s.D, gk.nr):], gk.nr, t.r0, t.r1, s.D, func(r int) (int, []float32, int) { return q.at(pr, r/s.Q, r%s.Q, 0), q.data, q.y }, scale)
		packCols(ws.ob[ti*attnRows*roundUp(s.DV, gk.nr):], gk.nr, t.r0, t.r1, s.DV, func(r int) (int, []float32, int) { return do.at(pr, r/s.Q, r%s.Q, 0), do.data, do.y }, 1)
	}

	ws.s = grow(ws.s, attnRows*attnKeys)
	ws.p = grow(ws.p, attnRows*attnKeys)
	ninf := float32(math.Inf(-1))
	for c0 := j0; c0 < j1; c0 += attnKeys {
		bc := min(attnKeys, j1-c0)
		// Kᵀ, D × bc, for the scores; Vᵀ, DV × bc, for the probability
		// gradients; and K, bc × D, for the query gradient.
		ws.k = grow(ws.k, roundUp(bc, gk.nr)*s.D+packSlack)
		packB(ws.k, operand{data: kk.data, off: kk.at(pr, 0, c0, 0), rs: kk.y, cs: kk.x}, 0, s.D, 0, bc, gk.nr)
		ws.v = grow(ws.v, roundUp(bc, gk.nr)*s.DV+packSlack)
		packB(ws.v, operand{data: v.data, off: v.at(pr, 0, c0, 0), rs: v.y, cs: v.x}, 0, s.DV, 0, bc, gk.nr)
		ws.kr = grow(ws.kr, roundUp(s.D, gk.nr)*bc+packSlack)
		packB(ws.kr, operand{data: kk.data, off: kk.at(pr, 0, c0, 0), rs: kk.x, cs: kk.y}, 0, bc, 0, s.D, gk.nr)
		dkc, dvc := dk[(c0-j0)*s.D:][:bc*s.D], dv[(c0-j0)*s.DV:][:bc*s.DV]
		for ti, t := range tiles {
			if visible(k, t.hi, c0, c0+bc) == c0 {
				continue
			}
			br := t.r1 - t.r0
			sc, dp := ws.s[:br*attnKeys], ws.p[:br*attnKeys]
			// P = e^(S - logsumexp).
			macroKernel(gk, ws.q[t.r0*s.D:], ws.k, s.D, br, bc, sc, attnKeys, false, &ws.tile)
			g.prepare(k, pr, t, c0, bc, sc, attnKeys, 6)
			for i := range br {
				row := sc[i*attnKeys:][:bc]
				if l := ws.l[t.r0+i]; l == ninf {
					clear(row)
				} else {
					expShift(row, l)
				}
			}
			// dV += Pᵀ dO.
			ws.pp = grow(ws.pp, roundUp(bc, gk.mr)*br+packSlack)
			packA(ws.pp, operand{data: sc, rs: 1, cs: attnKeys}, 0, bc, 0, br, gk.mr)
			macroKernel(gk, ws.pp, ws.ob[ti*attnRows*roundUp(s.DV, gk.nr):], br, bc, s.DV, dvc, s.DV, true, &ws.tile)
			// dP = dO Vᵀ, and dS = P·(dP - D) over P.
			macroKernel(gk, ws.oa[t.r0*s.DV:], ws.v, s.DV, br, bc, dp, attnKeys, false, &ws.tile)
			for i := range br {
				softmaxGrad(sc[i*attnKeys:][:bc], dp[i*attnKeys:], ws.d[t.r0+i])
			}
			// dK += dSᵀ (scale·Q).
			packA(ws.pp, operand{data: sc, rs: 1, cs: attnKeys}, 0, bc, 0, br, gk.mr)
			macroKernel(gk, ws.pp, ws.qb[ti*attnRows*roundUp(s.D, gk.nr):], br, bc, s.D, dkc, s.D, true, &ws.tile)
			// dQ += dS K, scaled below.
			ws.pp = grow(ws.pp, max(roundUp(bc, gk.mr)*br, roundUp(br, gk.mr)*bc)+packSlack)
			packA(ws.pp, operand{data: ws.s, rs: attnKeys, cs: 1}, 0, br, 0, bc, gk.mr)
			macroKernel(gk, ws.pp, ws.kr, bc, br, s.D, dq[t.r0*s.D:], s.D, true, &ws.tile)
		}
	}
	scaleRow(dq, scale)
}

// packRows packs rows [0, n) of a matrix of cols columns as A panels of mr
// rows (see packA), each element multiplied by scale. row returns where
// row r starts, in which data, and the stride of its columns.
func packRows(dst []float32, mr, n, cols int, row func(r int) (int, []float32, int), scale float32) {
	for r0 := 0; r0 < n; r0 += mr {
		panel := dst[r0*cols:][:mr*cols]
		m := min(mr, n-r0)
		for i := range m {
			off, data, stride := row(r0 + i)
			for p := range cols {
				panel[p*mr+i] = data[off+p*stride] * scale
			}
		}
		for i := m; i < mr; i++ {
			for p := range cols {
				panel[p*mr+i] = 0
			}
		}
	}
}

// packCols packs rows [r0, r1) of a matrix of cols columns as the B panels
// of a product that sums over the rows (see packB), each element
// multiplied by scale.
func packCols(dst []float32, nr, r0, r1, cols int, row func(r int) (int, []float32, int), scale float32) {
	kc := r1 - r0
	for j0 := 0; j0 < cols; j0 += nr {
		panel := dst[j0*kc:][:kc*nr]
		n := min(nr, cols-j0)
		for p := range kc {
			off, data, stride := row(r0 + p)
			dst := panel[p*nr:][:nr]
			for j := range n {
				dst[j] = data[off+(j0+j)*stride] * scale
			}
			clear(dst[n:])
		}
	}
}
