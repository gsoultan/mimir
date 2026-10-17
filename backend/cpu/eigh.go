package cpu

import (
	"context"
	"math"
	"slices"
)

// The symmetric eigensolver reduces the matrix to a tridiagonal T = Qᵀ·A·Q
// by Householder reflectors, in float32 with the products of the
// reflections split between the workers. It finds T's eigenvalues by the
// implicit QL algorithm and its eigenvectors by inverse iteration, both
// in float64, the eigenvectors of eigenvalues closer than 1e-5 of T's
// norm orthogonalized against each other, as LAPACK's sstein does with a
// wider threshold: float64 iterations keep the eigenvectors of farther
// eigenvalues orthogonal to float32's precision.
// Q then maps them to A's eigenvectors, a block of reflectors at a time,
// by products. Inverse iteration costs O(n²) for every eigenvector
// together, so the reduction and the products, O(n³), take the time.

// eigh sets w to the eigenvalues of the symmetric a, read from its lower
// triangle, in ascending order, and a to the matrix of their
// eigenvectors, one per column.
func (b *Backend) eigh(ctx context.Context, par bool, a mat, w []float32) error {
	n := a.rows
	if n == 0 {
		return nil
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			a.data[i*a.ld+j] = a.data[j*a.ld+i]
		}
	}
	d, e := make([]float64, n), make([]float64, n)
	taus := make([]float32, max(n-1, 0))
	if err := b.tridiagonalize(ctx, par, a, d, e, taus); err != nil {
		return err
	}
	vals, vecs, err := tridiagonalEigen(ctx, b, par, d, e, nil)
	if err != nil {
		return err
	}
	z := newMat(n, n)
	for c, v := range vecs {
		w[c] = float32(vals[c])
		for i, x := range v.x {
			z.data[(v.lo+i)*z.ld+c] = float32(x)
		}
	}
	// V = Q·Z with Q = H(0)···H(n-2), H(j) acting on the elements after
	// j: reflector j's v sits in row j of a from position j+1 on, as the
	// QR's reflectors sit in their rows of the transpose.
	if n > 1 {
		if err := b.reflectBack(ctx, par, a.sub(0, 1, n-1, n-1), n-1, taus, z.sub(1, 0, n-1, n)); err != nil {
			return err
		}
	}
	for i := range n {
		copy(a.row(i), z.row(i))
	}
	return nil
}

// reflectBack multiplies z by H(0)···H(k-1), the reflectors of the rows
// [0, k) of refl as blockReflector reads them, with refl.cols = z.rows:
// H(j) acts on the rows of z from j on. It applies linalgBlock
// reflectors at a time, as I - Yᵀ·T·Y, from the last block to the first.
func (b *Backend) reflectBack(ctx context.Context, par bool, refl mat, k int, taus []float32, z mat) error {
	for j0 := (k - 1) / linalgBlock * linalgBlock; k > 0 && j0 >= 0; j0 -= linalgBlock {
		j1 := min(j0+linalgBlock, k)
		y, tm := blockReflector(refl, j0, j1, taus)
		sub := z.sub(j0, 0, z.rows-j0, z.cols)
		wm := newMat(j1-j0, z.cols)
		if err := b.mulAdd(ctx, par, wm, false, false, y.op(false), sub.op(false), y.cols); err != nil {
			return err
		}
		w2 := newMat(j1-j0, z.cols)
		if err := b.mulAdd(ctx, par, w2, false, false, tm.op(false), wm.op(false), j1-j0); err != nil {
			return err
		}
		if err := b.mulAdd(ctx, par, sub, true, true, y.op(true), w2.op(false), j1-j0); err != nil {
			return err
		}
	}
	return nil
}

// tridiagonalize reduces the symmetric a, both triangles of which it
// holds, to the tridiagonal Qᵀ·a·Q with diagonal d and off-diagonal e,
// e[n-1] = 0. Reflector j, H(j) = I - taus[j]·v·vᵀ, zeroes the elements of
// row and column j after j+1; v[0] = 1 and v[1:] is stored in row j of a
// from position j+2 on.
//
// Each step updates the rest of the matrix, A22, to H·A22·H and makes
// the next reflector from its first row, and the same pass over the
// other rows that updates them also multiplies them by the next
// reflector, so that every step reads the matrix once.
func (b *Backend) tridiagonalize(ctx context.Context, par bool, a mat, d, e []float64, taus []float32) error {
	n := a.rows
	if n == 1 {
		d[0], e[0] = float64(a.data[0]), 0
		return nil
	}
	p, q := make([]float32, n), make([]float32, n)
	// start makes reflector j from row j, leaving its v in row j from
	// position j+1 on, v[0] = 1, and returns beta, which belongs at j+1.
	start := func(j int) float32 {
		x := a.row(j)[j+1:]
		taus[j] = reflector(x)
		d[j] = float64(a.row(j)[j])
		beta := x[0]
		e[j] = float64(beta)
		x[0] = 1
		return beta
	}
	beta := start(0)
	// p = tau·A22·v for reflector 0.
	err := b.maybeParallel(ctx, par && n*n > 1<<18, n-1, 32, func(lo, hi int) error {
		v := a.row(0)[1:]
		for i := lo; i < hi; i++ {
			p[i] = taus[0] * dot(a.row(1 + i)[1:], v)
		}
		return nil
	})
	if err != nil {
		return err
	}
	for j := range n - 1 {
		v := a.row(j)[j+1:]
		l := len(v)
		tau := taus[j]
		// A22 becomes A22 - v·wᵀ - w·vᵀ with w = p - (tau/2)(pᵀv)·v.
		w := p[:l]
		if tau != 0 {
			axpy(-0.5*tau*dot(w, v), v, w)
			row := a.row(j + 1)[j+1:]
			axpy(-v[0], w, row)
			axpy(-w[0], v, row)
		}
		last := j+1 == n-1
		var next []float32
		nextBeta := float32(0)
		if !last {
			nextBeta = start(j + 1)
			next = a.row(j + 1)[j+2:]
		}
		err := b.maybeParallel(ctx, par && l*l > 1<<18, l-1, 32, func(lo, hi int) error {
			for i := lo + 1; i < hi+1; i++ {
				row := a.row(j + 1 + i)[j+1:]
				if tau != 0 {
					axpy(-v[i], w, row)
					axpy(-w[i], v, row)
				}
				if next != nil {
					q[i-1] = taus[j+1] * dot(row[1:], next)
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
		v[0] = beta
		beta = nextBeta
		p, q = q, p
	}
	d[n-1] = float64(a.row(n - 1)[n-1])
	e[n-1] = 0
	return nil
}

// eigvec is an eigenvector of a tridiagonal matrix, x, zero outside the
// elements [lo, lo+len(x)).
type eigvec struct {
	lo int
	x  []float64
}

// tridiagonalEigen returns the eigenvalues of the symmetric tridiagonal
// matrix with diagonal d and off-diagonal e, in ascending order, and
// their eigenvectors. It splits the matrix where an off-diagonal element
// is negligible, and finds each block's eigenvalues by QL and the
// eigenvectors of each cluster of close eigenvalues by inverse iteration,
// the clusters in parallel. If keep is not nil, it finds only the
// eigenvectors of the eigenvalues it keeps, and the others' x are nil.
func tridiagonalEigen(ctx context.Context, b *Backend, par bool, d, e []float64, keep func(float64) bool) ([]float64, []eigvec, error) {
	n := len(d)
	type cluster struct {
		lo, hi int       // the block
		vals   []float64 // its eigenvalues, ascending
		at     int       // where they go among every eigenvalue
	}
	var clusters []cluster
	vals := make([]float64, 0, n)
	for lo := 0; lo < n; {
		hi := lo + 1
		for hi < n && math.Abs(e[hi-1]) > 0x1p-53*(math.Abs(d[hi-1])+math.Abs(d[hi])) {
			hi++
		}
		bv := qlEigenvalues(slices.Clone(d[lo:hi]), slices.Clone(e[lo:hi]))
		slices.Sort(bv)
		norm := 0.0
		for i := lo; i < hi; i++ {
			s := math.Abs(d[i]) + math.Abs(e[i])
			if i > lo {
				s += math.Abs(e[i-1])
			}
			norm = max(norm, s)
		}
		start := -1
		for i := 0; i <= len(bv); i++ {
			kept := i < len(bv) && (keep == nil || keep(bv[i]))
			if start >= 0 && (!kept || bv[i]-bv[i-1] > 1e-5*norm) {
				clusters = append(clusters, cluster{lo: lo, hi: hi, vals: bv[start:i], at: len(vals) + start})
				start = -1
			}
			if kept && start < 0 {
				start = i
			}
		}
		vals = append(vals, bv...)
		lo = hi
	}
	vecs := make([]eigvec, n)
	err := b.maybeParallel(ctx, par, len(clusters), 1, func(lo, hi int) error {
		for _, c := range clusters[lo:hi] {
			xs := inverseIteration(d[c.lo:c.hi], e[c.lo:c.hi], c.vals)
			for i, x := range xs {
				vecs[c.at+i] = eigvec{lo: c.lo, x: x}
			}
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(i, j int) int {
		switch {
		case vals[i] < vals[j]:
			return -1
		case vals[i] > vals[j]:
			return 1
		}
		return 0
	})
	sortedVals, sortedVecs := make([]float64, n), make([]eigvec, n)
	for i, o := range order {
		sortedVals[i], sortedVecs[i] = vals[o], vecs[o]
	}
	return sortedVals, sortedVecs, nil
}

// qlEigenvalues returns the eigenvalues of the unreduced symmetric
// tridiagonal matrix with diagonal d and off-diagonal e, whose last
// element is unused, by the implicit QL algorithm with Wilkinson's shift
// (tqli), overwriting d and e.
func qlEigenvalues(d, e []float64) []float64 {
	n := len(d)
	if n > 0 {
		e[n-1] = 0
	}
	for l := 0; l < n; l++ {
		for iter := 0; iter < 64; iter++ {
			m := l
			for ; m < n-1; m++ {
				if math.Abs(e[m]) <= 0x1p-53*(math.Abs(d[m])+math.Abs(d[m+1])) {
					break
				}
			}
			if m == l {
				break
			}
			g := (d[l+1] - d[l]) / (2 * e[l])
			r := math.Sqrt(g*g + 1)
			g = d[m] - d[l] + e[l]/(g+math.Copysign(r, g))
			s, c, p := 1.0, 1.0, 0.0
			i := m - 1
			for ; i >= l; i-- {
				f, bb := s*e[i], c*e[i]
				r = math.Sqrt(f*f + g*g)
				e[i+1] = r
				if r == 0 {
					d[i+1] -= p
					e[m] = 0
					break
				}
				s, c = f/r, g/r
				g = d[i+1] - p
				r = (d[i]-g)*s + 2*c*bb
				p = s * r
				d[i+1] = g + p
				g = c*r - bb
			}
			if r == 0 && i >= l {
				continue
			}
			d[l] -= p
			e[l] = g
			e[m] = 0
		}
	}
	return d
}

// inverseIteration returns unit eigenvectors of the unreduced symmetric
// tridiagonal matrix with diagonal d and off-diagonal e, whose last
// element is unused, for the cluster of eigenvalues vals, ascending.
// Each is orthogonalized against those before it, and an eigenvalue too
// close to the one before is moved apart from it, as in LAPACK's sstein.
// Like sstein it stops one iteration after the residual of the vector is
// small, which usually takes two.
func inverseIteration(d, e, vals []float64) [][]float64 {
	n := len(d)
	out := make([][]float64, len(vals))
	if n == 1 {
		for i := range out {
			out[i] = []float64{1}
		}
		return out
	}
	norm := 0.0
	for i := range n {
		s := math.Abs(d[i])
		if i > 0 {
			s += math.Abs(e[i-1])
		}
		if i < n-1 {
			s += math.Abs(e[i])
		}
		norm = max(norm, s)
	}
	tiny := 0x1p-52 * max(norm, math.SmallestNonzeroFloat64)
	// A solve whose result is this large has a residual of 1e-10 of the
	// norm or less.
	converged := 1e10 / max(norm, math.SmallestNonzeroFloat64)
	// inv holds the reciprocals of U's diagonal.
	inv, u1, u2, l := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
	swap := make([]bool, n)
	prev := math.Inf(-1)
	seed := uint64(1)
	for v, lam := range vals {
		if pertol := 10 * 0x1p-52 * max(math.Abs(lam), norm); lam-prev < pertol {
			lam = prev + pertol
		}
		prev = lam
		// LU factorization of T - λI with partial pivoting: U has three
		// diagonals, and the row carried down has two elements.
		alpha, beta := d[0]-lam, e[0]
		for i := range n - 1 {
			c, a1, b1 := e[i], d[i+1]-lam, 0.0
			if i+1 < n-1 {
				b1 = e[i+1]
			}
			if math.Abs(c) <= math.Abs(alpha) {
				if alpha == 0 {
					alpha = tiny
				}
				l[i], swap[i] = c/alpha, false
				inv[i], u1[i], u2[i] = 1/alpha, beta, 0
				alpha, beta = a1-l[i]*beta, b1
			} else {
				l[i], swap[i] = alpha/c, true
				inv[i], u1[i], u2[i] = 1/c, a1, b1
				alpha, beta = beta-l[i]*a1, -l[i]*b1
			}
		}
		if alpha == 0 {
			alpha = tiny
		}
		inv[n-1] = 1 / alpha
		// A pseudo-random start, which is orthogonal to no eigenvector
		// but by accident, scaled to unit length.
		x := make([]float64, n)
		for i := range x {
			seed = seed*6364136223846793005 + 1442695040888963407
			x[i] = float64(seed>>11)*0x1p-53 - 0.5
		}
		scale(x, 1/math.Sqrt(dotF64(x, x)))
		extra := false
		for range 5 {
			for i := range n - 1 {
				if swap[i] {
					x[i], x[i+1] = x[i+1], x[i]-l[i]*x[i+1]
				} else {
					x[i+1] -= l[i] * x[i]
				}
			}
			x[n-1] *= inv[n-1]
			x[n-2] = (x[n-2] - u1[n-2]*x[n-1]) * inv[n-2]
			for i := n - 3; i >= 0; i-- {
				x[i] = (x[i] - u1[i]*x[i+1] - u2[i]*x[i+2]) * inv[i]
			}
			for range 2 {
				for _, q := range out[:v] {
					s := dotF64(q, x)
					for i, qi := range q {
						x[i] -= s * qi
					}
				}
			}
			big := 0.0
			for _, xi := range x {
				big = math.Max(big, math.Abs(xi))
			}
			scale(x, 1/big)
			scale(x, 1/math.Sqrt(dotF64(x, x)))
			if extra {
				break
			}
			extra = big >= converged
		}
		out[v] = x
	}
	return out
}

// scale multiplies every element of x by a.
func scale(x []float64, a float64) {
	for i := range x {
		x[i] *= a
	}
}

func dotF64(a, b []float64) float64 {
	b = b[:len(a)]
	var s float64
	for i, x := range a {
		s += x * b[i]
	}
	return s
}
