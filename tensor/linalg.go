package tensor

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// Linear algebra works on batches of matrices, the last two dimensions
// of a tensor, like torch.linalg, in Float32: half inputs are cast. Batch
// dimensions of the operands of Solve and TriangularSolve broadcast.
//
// The factorizations run on the CPU, blocked around its matrix products.
// Matrices on a GPU are copied to the CPU when the factorization is
// built, which computes them there and then, and its results copied
// back, as To does; the products and elementwise steps of their
// gradients stay on the GPU.
//
// The gradients are those of torch.linalg. Where a factorization is not
// unique they follow its conventions: eigenvectors and singular vectors
// differ between implementations by their signs, so a function whose
// gradient flows through them should not depend on the signs. The
// gradients of repeated eigenvalues or singular values are not defined:
// those that flow through the vectors of a matrix with repeated values,
// or through U and V of an SVD with a zero singular value and more rows
// or columns than values, are infinite or NaN, as in PyTorch.

// ErrSingular reports a singular matrix that a solve or an inverse needs
// invertible, or one that Cholesky needs positive definite.
var ErrSingular = backend.ErrSingular

// Cholesky returns the lower triangular L with a positive diagonal and
// L·Lᵀ = t, for symmetric positive definite matrices t read from their
// lower triangle, like torch.linalg.cholesky. A matrix that is not
// positive definite fails with ErrSingular when computed. The gradient
// is symmetric.
func (t *Tensor) Cholesky() *Tensor {
	a, err := t.matrices("cholesky", true)
	if err != nil {
		return fail(err)
	}
	l := a.factor(backend.OpCholesky)
	return track(l, func(g *Tensor) []*Tensor { return grads(choleskyBackward(l, g)) }, a)
}

// TriangularSolve returns X of a·X = b for triangular matrices a, read
// from their lower triangle or, if upper, their upper one, and matrices
// b, like torch.linalg.solve_triangular(a, b, upper=upper). A zero on
// the diagonal fails with ErrSingular when computed. The gradient of a
// is triangular too.
func TriangularSolve(a, b *Tensor, upper bool) *Tensor {
	am, err := a.matrices("triangular solve", true)
	if err != nil {
		return fail(err)
	}
	bm, err := b.matrices("triangular solve", false)
	if err != nil {
		return fail(err)
	}
	if am, bm, err = solveOperands(am, bm); err != nil {
		return fail(err)
	}
	ad, bd := am.detached(), bm.detached()
	x := triSolve(ad, bd, upper, false)
	return track(x, func(g *Tensor) []*Tensor { return triangularSolveBackward(ad, x, g, upper) }, am, bm)
}

// Solve returns X of a·X = b for square matrices a, like
// torch.linalg.solve(a, b): b holds matrices, or vectors if it has one
// dimension or a's shape without the last. It factorizes a by LU with
// partial pivoting. A singular a fails with ErrSingular when computed.
func Solve(a, b *Tensor) *Tensor {
	am, err := a.matrices("solve", true)
	if err != nil {
		return fail(err)
	}
	if err := firstErr(b); err != nil {
		return fail(err)
	}
	r, vector := len(am.node.Shape), false
	if len(b.node.Shape) == 1 || len(b.node.Shape) == r-1 && slices.Equal(b.node.Shape, am.node.Shape[:r-1]) {
		b, vector = b.Unsqueeze(-1), true
	}
	bm, err := b.matrices("solve", false)
	if err != nil {
		return fail(err)
	}
	if am, bm, err = solveOperands(am, bm); err != nil {
		return fail(err)
	}
	x := solve(am, bm)
	if vector {
		return x.Squeeze(-1)
	}
	return x
}

// Inv returns the inverses of the square matrices t, like
// torch.linalg.inv. A singular matrix fails with ErrSingular when
// computed.
func (t *Tensor) Inv() *Tensor {
	a, err := t.matrices("inverse", true)
	if err != nil {
		return fail(err)
	}
	return solve(a, eye(a.node.Shape[len(a.node.Shape)-1]).broadcastTo(a.node.Shape))
}

// Det returns the determinants of the square matrices t, like
// torch.linalg.det, from an LU factorization. The gradient of a singular
// matrix's determinant fails with ErrSingular when computed.
func (t *Tensor) Det() *Tensor {
	a, err := t.matrices("det", true)
	if err != nil {
		return fail(err)
	}
	ad := a.detached()
	lu, perm, sign := ad.lu()
	sign, logAbs := slogDet(lu, sign)
	det := sign.Mul(logAbs.Exp())
	return track(det, func(g *Tensor) []*Tensor { return grads(logDetBackward(lu, perm, g.Mul(det))) }, a)
}

// SlogDet returns the signs of the determinants of the square matrices t
// and the logarithms of their absolute values, like torch.linalg.slogdet:
// a singular matrix has sign 0 and logAbsDet -Inf. The gradient flows
// through logAbsDet; that of a singular matrix fails with ErrSingular
// when computed.
func (t *Tensor) SlogDet() (sign, logAbsDet *Tensor) {
	a, err := t.matrices("slogdet", true)
	if err != nil {
		return fail(err), fail(err)
	}
	ad := a.detached()
	lu, perm, s := ad.lu()
	sign, logAbs := slogDet(lu, s)
	logAbs = track(logAbs, func(g *Tensor) []*Tensor { return grads(logDetBackward(lu, perm, g)) }, a)
	return sign, logAbs
}

// QR returns the reduced QR factorization of the matrices t, [..., m, n],
// like torch.linalg.qr: Q [..., m, k] with orthonormal columns and the
// upper triangular R [..., k, n], k = min(m, n), by Householder
// reflections, R's diagonal with LAPACK's signs. The gradient needs the
// first k columns of t independent, as PyTorch's does.
func (t *Tensor) QR() (q, r *Tensor) {
	a, err := t.matrices("qr", false)
	if err != nil {
		return fail(err), fail(err)
	}
	flat := a.factor(backend.OpQR)
	p := factorParts(flat, backend.OpQR, a.node.Shape)
	qd, rd := p[0], p[1]
	flat = track(flat, func(g *Tensor) []*Tensor {
		gp := factorParts(g, backend.OpQR, a.node.Shape)
		return grads(qrBackward(qd, rd, gp[0], gp[1]))
	}, a)
	p = factorParts(flat, backend.OpQR, a.node.Shape)
	return p[0], p[1]
}

// SVD returns the reduced singular value decomposition of the matrices
// t, [..., m, n], like torch.linalg.svd(t, full_matrices=False): U
// [..., m, k] and Vh [..., k, n] with orthonormal columns and rows and
// the k = min(m, n) singular values S in descending order, t =
// U·diag(S)·Vh. Each pair of singular vectors is unique up to a common
// sign.
func (t *Tensor) SVD() (u, s, vh *Tensor) {
	a, err := t.matrices("svd", false)
	if err != nil {
		return fail(err), fail(err), fail(err)
	}
	flat := a.factor(backend.OpSVD)
	p := factorParts(flat, backend.OpSVD, a.node.Shape)
	ud, sd, vhd := p[0], p[1], p[2]
	flat = track(flat, func(g *Tensor) []*Tensor {
		gp := factorParts(g, backend.OpSVD, a.node.Shape)
		return grads(svdBackward(ud, sd, vhd, gp[0], gp[1], gp[2]))
	}, a)
	p = factorParts(flat, backend.OpSVD, a.node.Shape)
	return p[0], p[1], p[2]
}

// Eigh returns the eigenvalues W, in ascending order, and eigenvectors V,
// one per column, of the symmetric matrices t read from their lower
// triangle, like torch.linalg.eigh: t = V·diag(W)·Vᵀ. Each eigenvector is
// unique up to its sign when its eigenvalue is not repeated. The
// gradient is symmetric.
func (t *Tensor) Eigh() (w, v *Tensor) {
	a, err := t.matrices("eigh", true)
	if err != nil {
		return fail(err), fail(err)
	}
	flat := a.factor(backend.OpEigh)
	p := factorParts(flat, backend.OpEigh, a.node.Shape)
	wd, vd := p[0], p[1]
	flat = track(flat, func(g *Tensor) []*Tensor {
		gp := factorParts(g, backend.OpEigh, a.node.Shape)
		return grads(eighBackward(wd, vd, gp[0], gp[1]))
	}, a)
	p = factorParts(flat, backend.OpEigh, a.node.Shape)
	return p[0], p[1]
}

// matrices returns t, a batch of matrices of a float dtype, square if
// square is set, as Float32.
func (t *Tensor) matrices(what string, square bool) (*Tensor, error) {
	if err := firstErr(t); err != nil {
		return nil, err
	}
	if batchLevel(t) != 0 {
		return nil, fmt.Errorf("tensor: %s: %w", what, ErrNotBatchable)
	}
	s := t.node.Shape
	r := len(s)
	if r < 2 || square && s[r-2] != s[r-1] {
		return nil, fmt.Errorf("tensor: %s of %v: %w", what, s, ErrShape)
	}
	if !t.node.DType.IsFloat() {
		return nil, fmt.Errorf("tensor: %s of %s: %w", what, t.node.DType, ErrDType)
	}
	return t.Cast(Float32), nil
}

// solveOperands checks that the matrices b have as many rows as the
// square a and broadcasts their batch dimensions.
func solveOperands(a, b *Tensor) (*Tensor, *Tensor, error) {
	ra, rb := len(a.node.Shape), len(b.node.Shape)
	if a.node.Shape[ra-1] != b.node.Shape[rb-2] {
		return nil, nil, fmt.Errorf("tensor: solve of %v by %v: %w", a.node.Shape, b.node.Shape, ErrShape)
	}
	batch, err := broadcastShape(a.node.Shape[:ra-2], b.node.Shape[:rb-2])
	if err != nil {
		return nil, nil, err
	}
	a = a.broadcastTo(slices.Concat(batch, a.node.Shape[ra-2:]))
	b = b.broadcastTo(slices.Concat(batch, b.node.Shape[rb-2:]))
	return a, b, firstErr(a, b)
}

// onCPU returns t on the CPU, untracked, and the device it was on.
func (t *Tensor) onCPU() (*Tensor, Device) {
	x, d := t.detached(), t.Device()
	if d.backend() == cpuBackend {
		return x, d
	}
	return x.To(CPU()), d
}

// factor returns the untracked result of the factorization op of the
// Float32 matrices t, on t's device: the flat parts of
// backend.FactorParts, or Cholesky's L.
func (t *Tensor) factor(op backend.Op) *Tensor {
	x, d := t.onCPU()
	if err := firstErr(x); err != nil {
		return fail(err)
	}
	return wrap(graph.Factor(op, x.node)).To(d)
}

// factorParts returns the parts of the flat result of a factorization
// op of matrices of shape.
func factorParts(flat *Tensor, op backend.Op, shape []int) []*Tensor {
	ps, _, err := backend.FactorParts(op, shape)
	if err != nil {
		return slices.Repeat([]*Tensor{fail(err)}, 3)
	}
	out := make([]*Tensor, len(ps))
	at := 0
	for i, p := range ps {
		n, _ := backend.NumElements(p)
		out[i] = flat.shrink([][2]int{{at, at + n}}).reshape(p)
		at += n
	}
	return out
}

// triSolve is X of a·X = b for the triangular Float32 matrices a, with
// a unit diagonal if unit, and matrices b of the same batch shape,
// untracked.
func triSolve(a, b *Tensor, upper, unit bool) *Tensor {
	x, d := a.onCPU()
	y, _ := b.onCPU()
	if err := firstErr(x, y); err != nil {
		return fail(err)
	}
	return wrap(graph.TriangularSolve(x.node, y.node, upper, unit)).To(d)
}

// lu returns the LU factorization of the square Float32 matrices t,
// untracked: L and U packed, the Int32 permutation whose element i is
// the row of t that row i of P·t is, and the sign of P.
func (t *Tensor) lu() (lu, perm, sign *Tensor) {
	p := factorParts(t.factor(backend.OpLU), backend.OpLU, t.node.Shape)
	return p[0], p[1].Cast(Int32), p[2]
}

// solve is X of a·X = b for square Float32 matrices a and matrices b of
// the same batch shape, by LU.
func solve(a, b *Tensor) *Tensor {
	ad, bd := a.detached(), b.detached()
	lu, perm, _ := ad.lu()
	x := luSolve(lu, perm, bd)
	return track(x, func(g *Tensor) []*Tensor { return solveBackward(lu, perm, x, g) }, a, b)
}

// luSolve is X of A·X = b for P·A = L·U.
func luSolve(lu, perm, b *Tensor) *Tensor {
	pb := b.Gather(-2, perm.Unsqueeze(-1).broadcastTo(b.node.Shape))
	return triSolve(lu, triSolve(lu, pb, false, true), true, false)
}

// luSolveT is X of Aᵀ·X = b for P·A = L·U: Aᵀ = Uᵀ·Lᵀ·P.
func luSolveT(lu, perm, b *Tensor) *Tensor {
	v := triSolve(mT(lu), triSolve(mT(lu), b, false, false), true, true)
	return Zeros(Float32, v.node.Shape...).ScatterAdd(-2, perm.Unsqueeze(-1).broadcastTo(v.node.Shape), v)
}

// slogDet returns the sign and the logarithm of the absolute value of
// the determinants of the matrices whose LU factorization is lu, with P's
// sign: those of the product of U's diagonal.
func slogDet(lu, sign *Tensor) (*Tensor, *Tensor) {
	d := diagonal(lu)
	logAbs := d.Abs().Log().Sum(-1)
	negative := d.Less(Scalar[float32](0)).Sum(-1).Fmod(Scalar[float32](2))
	sign = sign.Mul(Scalar[float32](1).Sub(negative.Mul(Scalar[float32](2))))
	singular := d.Equal(Scalar[float32](0)).Sum(-1)
	return Where(singular, Scalar[float32](0), sign), logAbs
}

// mT returns the transposes of the matrices t.
func mT(t *Tensor) *Tensor { return t.Transpose(-2, -1) }

// positions returns the Int32 values 0 to n-1, which, unlike a tensor
// uploaded from data, belong to no device.
func positions(n int) *Tensor { return Ones(Int32, n).cumsum(0).Sub(Scalar[int32](1)) }

// eye returns the n×n Float32 identity matrix.
func eye(n int) *Tensor {
	p := positions(n)
	return p.Unsqueeze(1).Equal(p.Unsqueeze(0)).Cast(Float32)
}

// triangle returns the lower triangle of the matrices t, or the upper,
// with the diagonal unless strict, and zeros elsewhere.
func triangle(t *Tensor, upper, strict bool) *Tensor {
	s := t.node.Shape
	i, j := positions(s[len(s)-2]).Unsqueeze(1), positions(s[len(s)-1]).Unsqueeze(0)
	if upper {
		i, j = j, i
	}
	keep := j.Less(i)
	if !strict {
		keep = j.Less(i.Add(Scalar[int32](1)))
	}
	return Where(keep, t, Scalar[float32](0))
}

// diagonal returns the diagonals of the square matrices t.
func diagonal(t *Tensor) *Tensor { return t.Mul(eye(t.node.Shape[len(t.node.Shape)-1])).Sum(-1) }

// diagEmbed returns the diagonal matrices whose diagonals are t.
func diagEmbed(t *Tensor) *Tensor { return t.Unsqueeze(-1).Mul(eye(t.node.Shape[len(t.node.Shape)-1])) }
