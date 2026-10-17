package cpu

import (
	"context"
	"math"
)

// QR factorizations reflect columns with Householder reflectors chosen
// as LAPACK's slarfg chooses them, so that R's diagonal has LAPACK's
// signs. They work on the transpose of the matrix, whose rows are its
// columns, so that a reflector and the columns it reflects are
// contiguous. A block of linalgBlock reflectors is applied to the rest of
// the matrix at once, as I - Y·T·Yᵀ with T from slarft, by three products.

// reflector sets the Householder reflector of x = [alpha; x[1:]]: on
// return x[0] is beta, the one element of H·x that is not zero, x[1:]
// holds v[1:], where v[0] is 1, and tau is returned, so that H = I -
// tau·v·vᵀ. A zero x[1:] gives tau 0, H = I.
func reflector(x []float32) float32 {
	if len(x) <= 1 {
		return 0
	}
	alpha := float64(x[0])
	xnorm := math.Sqrt(dot64(x[1:], x[1:]))
	if xnorm == 0 {
		return 0
	}
	beta := -math.Copysign(math.Hypot(alpha, xnorm), alpha)
	tau := (beta - alpha) / beta
	scaleRow(x[1:], float32(1/(alpha-beta)))
	x[0] = float32(beta)
	return float32(tau)
}

// reflectRows applies the reflector of row j of t, whose element j is
// taken as 1, to the parts from j on of rows [lo, hi) of t.
func reflectRows(t mat, j int, tau float32, lo, hi int) {
	v := t.row(j)[j:]
	beta := v[0]
	v[0] = 1
	defer func() { v[0] = beta }()
	if tau == 0 {
		return
	}
	for c := lo; c < hi; c++ {
		r := t.row(c)[j:]
		axpy(-tau*dot(v, r), v, r)
	}
}

// blockReflector returns Y, whose row r is reflector j0+r of the rows of
// t, from position j0 on, with its implied ones and zeros, and the upper
// triangular T of slarft, so that H(j0)·H(j0+1)···H(j1-1) = I - Yᵀ·T·Y.
func blockReflector(t mat, j0, j1 int, taus []float32) (y, tm mat) {
	jb, l := j1-j0, t.cols-j0
	y = newMat(jb, l)
	for r := range jb {
		dst := y.row(r)
		dst[r] = 1
		copy(dst[r+1:], t.row(j0 + r)[j0+r+1:])
	}
	tm = newMat(jb, jb)
	w := make([]float32, jb)
	for i := range jb {
		tau := taus[j0+i]
		if tau == 0 {
			continue
		}
		// T[0:i, i] = T[0:i, 0:i]·(-tau·Y[0:i]·y[i]).
		yi := y.row(i)[i:]
		for r := range i {
			w[r] = -tau * dot(y.row(r)[i:], yi)
		}
		for r := range i {
			tr := tm.row(r)
			var s float32
			for c := r; c < i; c++ {
				s += tr[c] * w[c]
			}
			tr[i] = s
		}
		tm.row(i)[i] = tau
	}
	return y, tm
}

// householderQR reduces the rows [0, k) of t, the transpose of an m×n
// matrix with m = t.cols, by reflectors: reflector j zeroes the elements
// after j of row j and is stored there, with beta at j, and taus[j] is
// its tau. It returns the block reflectors Y and T of each block of
// linalgBlock reflectors, which apply to the rows of t after the block.
func (b *Backend) householderQR(ctx context.Context, par bool, t mat, k int, taus []float32) (ys, ts []mat, err error) {
	n, m := t.rows, t.cols
	for j0 := 0; j0 < k; j0 += linalgBlock {
		j1 := min(j0+linalgBlock, k)
		for j := j0; j < j1; j++ {
			taus[j] = reflector(t.row(j)[j:])
			err := b.maybeParallel(ctx, par && (m-j)*(j1-j) > 1<<15, j1-j-1, 8, func(lo, hi int) error {
				reflectRows(t, j, taus[j], j+1+lo, j+1+hi)
				return nil
			})
			if err != nil {
				return nil, nil, err
			}
		}
		y, tm := blockReflector(t, j0, j1, taus)
		ys, ts = append(ys, y), append(ts, tm)
		if j1 == n {
			continue
		}
		// The rows after the block, C, become C·(I - Yᵀ·T·Y).
		jb, rows := j1-j0, n-j1
		c := t.sub(j1, j0, rows, m-j0)
		w := newMat(rows, jb)
		if err := b.mulAdd(ctx, par, w, false, false, c.op(false), y.op(true), m-j0); err != nil {
			return nil, nil, err
		}
		w2 := newMat(rows, jb)
		if err := b.mulAdd(ctx, par, w2, false, false, w.op(false), tm.op(false), jb); err != nil {
			return nil, nil, err
		}
		if err := b.mulAdd(ctx, par, c, true, true, w2.op(false), y.op(false), jb); err != nil {
			return nil, nil, err
		}
	}
	return ys, ts, nil
}

// applyReflectors multiplies the columns from j0 on of the rows from j0
// on of the k×m matrix qt by I - Yᵀ·Tᵀ·Y for the block reflectors of
// householderQR, each block's j0, from the last block to the first.
// Started from [I 0], qt becomes the transpose of the first k columns of
// Q = H(0)···H(k-1).
func (b *Backend) applyReflectors(ctx context.Context, par bool, qt mat, ys, ts []mat) error {
	for blk := len(ys) - 1; blk >= 0; blk-- {
		y, tm := ys[blk], ts[blk]
		j0 := blk * linalgBlock
		jb := y.rows
		rows := qt.rows - j0
		sub := qt.sub(j0, j0, rows, qt.cols-j0)
		w := newMat(rows, jb)
		if err := b.mulAdd(ctx, par, w, false, false, sub.op(false), y.op(true), y.cols); err != nil {
			return err
		}
		w2 := newMat(rows, jb)
		if err := b.mulAdd(ctx, par, w2, false, false, w.op(false), tm.op(true), jb); err != nil {
			return err
		}
		if err := b.mulAdd(ctx, par, sub, true, true, w2.op(false), y.op(false), jb); err != nil {
			return err
		}
	}
	return nil
}

// transposed returns the transpose of a in a matrix of its own.
func transposed(a mat) mat {
	t := newMat(a.cols, a.rows)
	for i := range a.rows {
		for j, v := range a.row(i) {
			t.data[j*t.ld+i] = v
		}
	}
	return t
}

// qr factorizes the m×n a into the m×k q, with orthonormal columns, and
// the upper triangular k×n r, k = min(m, n).
func (b *Backend) qr(ctx context.Context, par bool, a, q, r mat) error {
	m, n := a.rows, a.cols
	k := min(m, n)
	t := transposed(a)
	taus := make([]float32, k)
	ys, ts, err := b.householderQR(ctx, par, t, k, taus)
	if err != nil {
		return err
	}
	for i := range k {
		row := r.row(i)
		clear(row[:i])
		for c := i; c < n; c++ {
			row[c] = t.data[c*t.ld+i]
		}
	}
	qt := newMat(k, m)
	for i := range k {
		qt.data[i*qt.ld+i] = 1
	}
	if err := b.applyReflectors(ctx, par, qt, ys, ts); err != nil {
		return err
	}
	for i := range m {
		row := q.row(i)
		for c := range k {
			row[c] = qt.data[c*qt.ld+i]
		}
	}
	return nil
}
