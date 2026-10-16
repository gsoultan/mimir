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
// in float64, the eigenvectors of eigenvalues closer than a thousandth of
// T's norm orthogonalized against each other, as LAPACK's sstein does.
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
	vals, vecs, err := tridiagonalEigen(ctx, b, par, d, e)
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
	refl := a.sub(0, 1, n-1, n-1)
	for j0 := (n - 2) / linalgBlock * linalgBlock; n > 1 && j0 >= 0; j0 -= linalgBlock {
		j1 := min(j0+linalgBlock, n-1)
		y, tm := blockReflector(refl, j0, j1, taus)
		sub := z.sub(j0+1, 0, n-1-j0, n)
		wm := newMat(j1-j0, n)
		if err := b.mulAdd(ctx, par, wm, false, false, y.op(false), sub.op(false), y.cols); err != nil {
			return err
		}
		w2 := newMat(j1-j0, n)
		if err := b.mulAdd(ctx, par, w2, false, false, tm.op(false), wm.op(false), j1-j0); err != nil {
			return err
		}
		if err := b.mulAdd(ctx, par, sub, true, true, y.op(true), w2.op(false), j1-j0); err != nil {
			return err
		}
	}
	for i := range n {
		copy(a.row(i), z.row(i))
	}
	return nil
}

// tridiagonalize reduces the symmetric a, both triangles of which it
// holds, to the tridiagonal Qᵀ·a·Q with diagonal d and off-diagonal e,
// e[n-1] = 0. Reflector j, H(j) = I - taus[j]·v·vᵀ, zeroes the elements of
// row and column j after j+1; v[0] = 1 and v[1:] is stored in row j of a
// from position j+2 on.
func (b *Backend) tridiagonalize(ctx context.Context, par bool, a mat, d, e []float64, taus []float32) error {
	n := a.rows
	p := make([]float32, n)
	for j := range n - 1 {
		x := a.row(j)[j+1:]
		tau := reflector(x)
		taus[j] = tau
		d[j], e[j] = float64(a.row(j)[j]), float64(x[0])
		if tau == 0 {
			continue
		}
		// The rest becomes H·A22·H = A22 - v·wᵀ - w·vᵀ with
		// w = p - (tau/2)(pᵀv)·v and p = tau·A22·v.
		l := n - j - 1
		beta := x[0]
		x[0] = 1
		v := x
		pp := p[:l]
		big := par && l*l > 1<<16
		err := b.maybeParallel(ctx, big, l, 32, func(lo, hi int) error {
			for i := lo; i < hi; i++ {
				pp[i] = tau * dot(a.row(j + 1 + i)[j+1:], v)
			}
			return nil
		})
		if err != nil {
			return err
		}
		k := -0.5 * tau * dot(pp, v)
		axpy(k, v, pp)
		err = b.maybeParallel(ctx, big, l, 32, func(lo, hi int) error {
			for i := lo; i < hi; i++ {
				row := a.row(j + 1 + i)[j+1:]
				axpy(-v[i], pp, row)
				axpy(-pp[i], v, row)
			}
			return nil
		})
		if err != nil {
			return err
		}
		x[0] = beta
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
// the clusters in parallel.
func tridiagonalEigen(ctx context.Context, b *Backend, par bool, d, e []float64) ([]float64, []eigvec, error) {
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
		start := 0
		for i := 1; i <= len(bv); i++ {
			if i == len(bv) || bv[i]-bv[i-1] > 1e-3*norm {
				clusters = append(clusters, cluster{lo: lo, hi: hi, vals: bv[start:i], at: len(vals) + start})
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
			r := math.Hypot(g, 1)
			g = d[m] - d[l] + e[l]/(g+math.Copysign(r, g))
			s, c, p := 1.0, 1.0, 0.0
			i := m - 1
			for ; i >= l; i-- {
				f, bb := s*e[i], c*e[i]
				r = math.Hypot(f, g)
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
	u0, u1, u2, l := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
	swap := make([]bool, n)
	prev := math.Inf(-1)
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
				u0[i], u1[i], u2[i] = alpha, beta, 0
				alpha, beta = a1-l[i]*beta, b1
			} else {
				l[i], swap[i] = alpha/c, true
				u0[i], u1[i], u2[i] = c, a1, b1
				alpha, beta = beta-l[i]*a1, -l[i]*b1
			}
		}
		if alpha == 0 {
			alpha = tiny
		}
		u0[n-1] = alpha
		x := make([]float64, n)
		// A fixed start that is not orthogonal to any eigenvector but by
		// accident.
		for i := range x {
			x[i] = 1 + 0.5*math.Sin(float64(3*i+v+1))
		}
		for range 4 {
			for i := range n - 1 {
				if swap[i] {
					x[i], x[i+1] = x[i+1], x[i]-l[i]*x[i+1]
				} else {
					x[i+1] -= l[i] * x[i]
				}
			}
			x[n-1] /= u0[n-1]
			x[n-2] = (x[n-2] - u1[n-2]*x[n-1]) / u0[n-2]
			for i := n - 3; i >= 0; i-- {
				x[i] = (x[i] - u1[i]*x[i+1] - u2[i]*x[i+2]) / u0[i]
			}
			for range 2 {
				for _, q := range out[:v] {
					s := 0.0
					for i, qi := range q {
						s += qi * x[i]
					}
					for i, qi := range q {
						x[i] -= s * qi
					}
				}
			}
			s := 0.0
			for _, xi := range x {
				s = math.Max(s, math.Abs(xi))
			}
			for i := range x {
				x[i] /= s
			}
			s = math.Sqrt(dotF64(x, x))
			for i := range x {
				x[i] /= s
			}
		}
		out[v] = x
	}
	return out
}

func dotF64(a, b []float64) float64 {
	b = b[:len(a)]
	var s float64
	for i, x := range a {
		s += x * b[i]
	}
	return s
}
