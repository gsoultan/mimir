package cpu

import (
	"context"
	"math"
)

// The SVD reduces the matrix to an upper bidiagonal B = Qᵀ·A·P by
// Householder reflectors from the left and the right, as LAPACK's
// sgebrd does, after a QR factorization when it is much taller than
// wide. B's singular triplets are the eigenpairs of the tridiagonal
// Golub-Kahan matrix with zero diagonal and off-diagonal d0, e0, d1, e1,
// ..., whose eigenvector for σ interleaves B's right and left singular
// vectors, (v0, u0, v1, u1, ...)/√2; the symmetric eigensolver's QL and
// inverse iteration find them, as LAPACK's sbdsvdx does. Q and P then
// map them to A's, a block of reflectors at a time, by products. The
// singular values below 1e-10 of the largest, whose pairs of
// eigenvectors are too close to separate, get singular vectors that
// complete the others' to orthonormal bases instead: they contribute
// nothing that float32 can hold to A. A wide matrix is factorized
// through its transpose.

// svd sets u, s and vh to the reduced singular value decomposition of
// the m×n a: u is m×k, s holds k values in descending order and vh is
// k×n, for k = min(m, n).
func (b *Backend) svd(ctx context.Context, par bool, a, u mat, s []float32, vh mat) error {
	m, n := a.rows, a.cols
	if m == 0 || n == 0 {
		return nil
	}
	if m < n {
		// Aᵀ = U'·S·Vh' makes A = Vh'ᵀ·S·U'ᵀ.
		ut, vht := newMat(n, m), newMat(m, m)
		if err := b.svd(ctx, par, transposed(a), ut, s, vht); err != nil {
			return err
		}
		for i := range m {
			for c := range m {
				u.data[i*u.ld+c] = vht.data[c*vht.ld+i]
			}
		}
		for r := range m {
			row := vh.row(r)
			for c := range n {
				row[c] = ut.data[c*ut.ld+r]
			}
		}
		return nil
	}
	if 3*m >= 5*n {
		// A = Q·R and R = U'·S·Vh make U = Q·U'.
		q, r, ur := newMat(m, n), newMat(n, n), newMat(n, n)
		if err := b.qr(ctx, par, a, q, r); err != nil {
			return err
		}
		if err := b.svd(ctx, par, r, ur, s, vh); err != nil {
			return err
		}
		return b.mulAdd(ctx, par, u, false, false, q.op(false), ur.op(false), n)
	}
	g := newMat(m, n)
	for i := range m {
		copy(g.row(i), a.row(i))
	}
	left := newMat(n, m)
	tl, tr := make([]float32, n), make([]float32, n)
	d, e := make([]float64, n), make([]float64, n)
	if err := b.bidiagonalize(ctx, par, g, left, tl, tr, d, e); err != nil {
		return err
	}
	norm := 0.0
	off := make([]float64, 2*n)
	for j := range n {
		off[2*j], off[2*j+1] = d[j], e[j]
		norm = max(norm, math.Abs(d[j]), math.Abs(e[j]))
	}
	small := 1e-10 * norm
	vals, vecs, err := tridiagonalEigen(ctx, b, par, make([]float64, 2*n), off, func(l float64) bool { return l > small })
	if err != nil {
		return err
	}
	// The rows of ub and vb are B's left and right singular vectors.
	ub, vb := newMat(n, n), newMat(n, n)
	var rest []int
	r := 0
	for i := 2*n - 1; i >= 0 && r < n; i-- {
		s[r] = float32(math.Abs(vals[i]))
		v := vecs[i]
		if v.x == nil {
			rest = append(rest, r)
			r++
			continue
		}
		ur, vr := ub.row(r), vb.row(r)
		for k, x := range v.x {
			if at := v.lo + k; at%2 == 0 {
				vr[at/2] = float32(x)
			} else {
				ur[at/2] = float32(x)
			}
		}
		scaleRow(ur, float32(1/math.Sqrt(dot64(ur, ur))))
		scaleRow(vr, float32(1/math.Sqrt(dot64(vr, vr))))
		r++
	}
	completeRows(ub, rest)
	completeRows(vb, rest)
	// U = H(0)···H(n-1)·[Ub; 0] and V = G(0)···G(n-2)·Vb for the left
	// reflectors H in the rows of left and the right ones G in those of
	// g after the diagonal.
	zu := newMat(m, n)
	for i := range n {
		for c := range n {
			zu.data[i*zu.ld+c] = ub.data[c*ub.ld+i]
		}
	}
	if err := b.reflectBack(ctx, par, left, n, tl, zu); err != nil {
		return err
	}
	for i := range m {
		copy(u.row(i), zu.row(i))
	}
	zv := transposed(vb)
	if n > 1 {
		if err := b.reflectBack(ctx, par, g.sub(0, 1, n-1, n-1), n-1, tr, zv.sub(1, 0, n-1, n)); err != nil {
			return err
		}
	}
	for i := range n {
		row := vh.row(i)
		for c := range n {
			row[c] = zv.data[c*zv.ld+i]
		}
	}
	return nil
}

// bidiagonalize reduces the m×n a, m ≥ n, to the upper bidiagonal
// Hᵀ·a·G with diagonal d and superdiagonal e, e[n-1] = 0. Left reflector
// k, H(k) = I - tl[k]·v·vᵀ, zeroes column k below the diagonal; v[0] = 1
// and v[1:] is stored in row k of left from position k+1 on. Right
// reflector k, G(k) = I - tr[k]·v·vᵀ, zeroes row k after k+1; v[0] = 1
// and v[1:] is stored in row k of a from position k+2 on.
func (b *Backend) bidiagonalize(ctx context.Context, par bool, a, left mat, tl, tr []float32, d, e []float64) error {
	m, n := a.rows, a.cols
	col := make([]float32, m)
	for k := range n {
		v := col[:m-k]
		for i := range v {
			v[i] = a.data[(k+i)*a.ld+k]
		}
		tau := reflector(v)
		tl[k], d[k] = tau, float64(v[0])
		copy(left.row(k)[k+1:], v[1:])
		if cols := n - k - 1; tau != 0 && cols > 0 {
			// The columns after k, C, become C - tau·v·(vᵀ·C), by
			// columns split between the workers.
			v[0] = 1
			err := b.maybeParallel(ctx, par && len(v)*cols > 1<<17, cols, 64, func(lo, hi int) error {
				w := make([]float32, hi-lo)
				c0, c1 := k+1+lo, k+1+hi
				for i, vi := range v {
					axpy(vi, a.row(k + i)[c0:c1], w)
				}
				for i, vi := range v {
					axpy(-tau*vi, w, a.row(k + i)[c0:c1])
				}
				return nil
			})
			if err != nil {
				return err
			}
		}
		if k == n-1 {
			e[k] = 0
			continue
		}
		x := a.row(k)[k+1:]
		tau = reflector(x)
		tr[k], e[k] = tau, float64(x[0])
		if tau == 0 {
			continue
		}
		// The rows after k become R - tau·(R·v)·vᵀ.
		beta := x[0]
		x[0] = 1
		err := b.maybeParallel(ctx, par && (m-k-1)*len(x) > 1<<17, m-k-1, 32, func(lo, hi int) error {
			for i := k + 1 + lo; i < k+1+hi; i++ {
				row := a.row(i)[k+1:]
				axpy(-tau*dot(row, x), x, row)
			}
			return nil
		})
		if err != nil {
			return err
		}
		x[0] = beta
	}
	return nil
}

// completeRows sets the rows zero of a, whose other rows are orthonormal,
// to unit vectors orthogonal to every other row: standard basis vectors
// with the other rows' parts taken out by Gram-Schmidt, each time the
// one with the most left.
func completeRows(a mat, zero []int) {
	if len(zero) == 0 {
		return
	}
	valid := make([]bool, a.rows)
	for i := range valid {
		valid[i] = true
	}
	for _, r := range zero {
		valid[r] = false
	}
	// rest[t] is the squared length of basis vector t outside the rows.
	rest := make([]float64, a.cols)
	for t := range rest {
		rest[t] = 1
	}
	for q := range a.rows {
		if valid[q] {
			for t, v := range a.row(q) {
				rest[t] -= float64(v) * float64(v)
			}
		}
	}
	x := make([]float64, a.cols)
	for _, r := range zero {
		best := 0
		for t, l := range rest {
			if l > rest[best] {
				best = t
			}
		}
		clear(x)
		x[best] = 1
		for range 2 {
			for q := range a.rows {
				if !valid[q] {
					continue
				}
				row := a.row(q)
				p := 0.0
				for i, v := range row {
					p += float64(v) * x[i]
				}
				for i, v := range row {
					x[i] -= p * float64(v)
				}
			}
		}
		norm := math.Sqrt(dotF64(x, x))
		row := a.row(r)
		for i := range row {
			row[i] = float32(x[i] / norm)
		}
		valid[r] = true
		for t, v := range row {
			rest[t] -= float64(v) * float64(v)
		}
	}
}
