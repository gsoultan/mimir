package cpu

import (
	"context"
	"math"
	"slices"
	"sync/atomic"
)

// The SVD factorizes a tall matrix A = Q·R first, and then R by one-sided
// Jacobi (Hestenes): rotations of pairs of rows of G, which starts as R,
// make every pair orthogonal, and the same rotations of the rows of W,
// which starts as I, accumulate them, so that W·R = G at every step.
// Once the rows of G are orthogonal they are σ_i times the right singular
// vectors, R = Wᵀ·diag(σ)·Vh, and A's left singular vectors are Q·Wᵀ.
// The pairs follow a round-robin tournament, whose rounds pair every row
// with another disjoint from the round's other pairs, so the workers
// rotate a round's pairs in parallel. Factorizing by QR first makes the
// rows of R closer to orthogonal, which saves sweeps. A wide matrix is
// factorized through its transpose.

// jacobiSweeps bounds the sweeps over every pair of rows.
const jacobiSweeps = 40

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
	q, g := newMat(m, n), newMat(n, n)
	if err := b.qr(ctx, par, a, q, g); err != nil {
		return err
	}
	w := newMat(n, n)
	for i := range n {
		w.data[i*w.ld+i] = 1
	}
	if err := b.jacobi(ctx, par, g, w); err != nil {
		return err
	}
	sigma := make([]float64, n)
	for i := range n {
		sigma[i] = math.Sqrt(dot64(g.row(i), g.row(i)))
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(i, j int) int {
		switch {
		case sigma[i] > sigma[j]:
			return -1
		case sigma[i] < sigma[j]:
			return 1
		}
		return 0
	})
	wp := newMat(n, n)
	var zero []int
	for r, i := range order {
		s[r] = float32(sigma[i])
		copy(wp.row(r), w.row(i))
		if sigma[i] == 0 || sigma[i] < sigma[order[0]]*0x1p-60 {
			zero = append(zero, r)
			continue
		}
		row := vh.row(r)
		copy(row, g.row(i))
		scaleRow(row, float32(1/sigma[i]))
	}
	completeRows(vh, zero)
	return b.mulAdd(ctx, par, u, false, false, q.op(false), wp.op(true), n)
}

// completeRows sets the rows zero of a, whose other rows are orthonormal,
// to unit vectors orthogonal to every other row: the standard basis
// vectors with the other rows' parts taken out, by Gram-Schmidt.
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
	x := make([]float64, a.cols)
	t := 0
	for _, r := range zero {
		for ; t < a.cols; t++ {
			clear(x)
			x[t] = 1
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
			if norm := math.Sqrt(dotF64(x, x)); norm > 0.5 {
				row := a.row(r)
				for i := range row {
					row[i] = float32(x[i] / norm)
				}
				valid[r] = true
				t++
				break
			}
		}
	}
}

// jacobi rotates pairs of rows of g, and the same rows of w, until every
// pair of rows of g is orthogonal to working precision.
func (b *Backend) jacobi(ctx context.Context, par bool, g, w mat) error {
	n := g.rows
	tol := 0x1p-24 * max(8, math.Sqrt(float64(g.cols)))
	norms := make([]float64, n)
	players := n + n%2
	seat := make([]int, players)
	for i := range seat {
		seat[i] = i
	}
	for range jacobiSweeps {
		for i := range n {
			norms[i] = dot64(g.row(i), g.row(i))
		}
		var rotated atomic.Int64
		for range players - 1 {
			err := b.maybeParallel(ctx, par, players/2, max(1, 1<<12/max(g.cols, 1)), func(lo, hi int) error {
				count := 0
				for k := lo; k < hi; k++ {
					p, q := seat[k], seat[players-1-k]
					if p < n && q < n && rotate(g, w, norms, p, q, tol) {
						count++
					}
				}
				rotated.Add(int64(count))
				return nil
			})
			if err != nil {
				return err
			}
			// Every seat but the first moves one along.
			last := seat[players-1]
			copy(seat[2:], seat[1:players-1])
			seat[1] = last
		}
		if rotated.Load() == 0 {
			break
		}
	}
	return nil
}

// rotate makes rows p and q of g orthogonal by a rotation, which it also
// applies to rows p and q of w, unless they are orthogonal to within tol
// already, and updates their squared norms. It reports whether it
// rotated.
func rotate(g, w mat, norms []float64, p, q int, tol float64) bool {
	alpha, beta := norms[p], norms[q]
	if alpha == 0 || beta == 0 {
		return false
	}
	gp, gq := g.row(p), g.row(q)
	bound := tol * math.Sqrt(alpha*beta)
	gamma := float64(dot(gp, gq))
	if math.Abs(gamma) <= 16*bound {
		// Near the threshold the float32 sum's rounding matters.
		gamma = dot64(gp, gq)
	}
	if math.Abs(gamma) <= bound {
		return false
	}
	zeta := (beta - alpha) / (2 * gamma)
	t := math.Copysign(1/(math.Abs(zeta)+math.Hypot(1, zeta)), zeta)
	c := 1 / math.Sqrt(1+t*t)
	sn := c * t
	rotateRows(gp, gq, float32(c), float32(sn))
	rotateRows(w.row(p), w.row(q), float32(c), float32(sn))
	norms[p], norms[q] = alpha-t*gamma, beta+t*gamma
	return true
}

// rotateRows sets x and y to c·x - s·y and s·x + c·y.
func rotateRows(x, y []float32, c, s float32) {
	y = y[:len(x)]
	for i, xi := range x {
		yi := y[i]
		x[i] = c*xi - s*yi
		y[i] = s*xi + c*yi
	}
}
