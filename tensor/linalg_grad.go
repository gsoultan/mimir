package tensor

// The gradient rules of the linear algebra, those of torch.linalg's
// derivatives. Each takes the untracked factors of the forward pass and
// the gradient of its result.

// choleskyBackward is the gradient of A = L·Lᵀ given that of L:
// L⁻ᵀ·Φ·L⁻¹, for Φ the symmetric matrix whose lower triangle is that of
// Lᵀ·gL, its diagonal halved.
func choleskyBackward(l, g *Tensor) *Tensor {
	p := triangle(mT(l).MatMul(g), false, false)
	p = p.Add(mT(triangle(p, false, true))).Mul(Scalar[float32](0.5))
	ga := triSolve(mT(l), p, true, false)
	return mT(triSolve(mT(l), mT(ga), true, false))
}

// triangularSolveBackward is the gradient of X = A⁻¹·B for the
// triangular A given that of X: gB = A⁻ᵀ·gX and gA = -gB·Xᵀ, on A's
// triangle.
func triangularSolveBackward(a, x, g *Tensor, upper bool) []*Tensor {
	gb := triSolve(mT(a), g, !upper, false)
	return grads(triangle(gb.MatMul(mT(x)).Neg(), upper, false), gb)
}

// solveBackward is the gradient of X = A⁻¹·B, P·A = L·U, given that of
// X: gB = A⁻ᵀ·gX and gA = -gB·Xᵀ.
func solveBackward(lu, perm, x, g *Tensor) []*Tensor {
	gb := luSolveT(lu, perm, g)
	return grads(gb.MatMul(mT(x)).Neg(), gb)
}

// logDetBackward is the gradient of log|det A|, P·A = L·U, given g: g·A⁻ᵀ.
// That of det A is the same for g·det A.
func logDetBackward(lu, perm, g *Tensor) *Tensor {
	inv := luSolve(lu, perm, eye(lu.node.Shape[len(lu.node.Shape)-1]).broadcastTo(lu.node.Shape))
	return mT(inv).Mul(g.Unsqueeze(-1).Unsqueeze(-1))
}

// qrBackward is the gradient of A = Q·R, given those of Q and R, as
// PyTorch's linalg_qr_backward computes it.
func qrBackward(q, r, gq, gr *Tensor) *Tensor {
	s := q.node.Shape
	m, n := s[len(s)-2], r.node.Shape[len(s)-1]
	if m < n {
		// A = [X Y] for a square X = Q·R1 and Y = Q·R2: the gradient of
		// X is that of the square QR with gQ + Y·gR2ᵀ, and Y's is Q·gR2.
		r1, r2 := r.shrink(axisRange(r.node.Shape, len(s)-1, 0, m)), r.shrink(axisRange(r.node.Shape, len(s)-1, m, n))
		g1, g2 := gr.shrink(axisRange(r.node.Shape, len(s)-1, 0, m)), gr.shrink(axisRange(r.node.Shape, len(s)-1, m, n))
		y := q.MatMul(r2)
		return Cat(-1, qrBackward(q, r1, gq.Add(y.MatMul(mT(g2))), g1), q.MatMul(g2))
	}
	// gA = (gQ + Q·copyltu(M))·R⁻ᵀ for M = triu(gR·Rᵀ - Qᵀ·gQ), its
	// upper triangle mirrored below.
	mm := triangle(gr.MatMul(mT(r)).Sub(mT(q).MatMul(gq)), true, false)
	mm = mm.Add(mT(triangle(mm, true, true)))
	c := gq.Add(q.MatMul(mm))
	return mT(triSolve(r, mT(c), true, false))
}

// svdBackward is the gradient of A = U·diag(S)·Vh, given those of U, S
// and Vh, as PyTorch's svd_backward computes it.
func svdBackward(u, s, vh, gu, gs, gvh *Tensor) *Tensor {
	m, n, k := u.node.Shape[len(u.node.Shape)-2], vh.node.Shape[len(vh.node.Shape)-1], s.node.Shape[len(s.node.Shape)-1]
	v, gv := mT(vh), mT(gvh)
	s2 := s.Mul(s)
	e := s2.Unsqueeze(-2).Sub(s2.Unsqueeze(-1)).Add(eye(k)) // e[i][j] = s[j]² - s[i]²
	utgu, vtgv := mT(u).MatMul(gu), vh.MatMul(gv)
	inner := divOrZero(utgu.Sub(mT(utgu)), e).Mul(s.Unsqueeze(-2)).
		Add(s.Unsqueeze(-1).Mul(divOrZero(vtgv.Sub(mT(vtgv)), e))).
		Add(diagEmbed(gs))
	ga := u.MatMul(inner).MatMul(vh)
	if m > k {
		ga = ga.Add(divOrZero(gu.Sub(u.MatMul(utgu)), s.Unsqueeze(-2)).MatMul(vh))
	}
	if n > k {
		ga = ga.Add(u.MatMul(mT(divOrZero(gv.Sub(v.MatMul(vtgv)), s.Unsqueeze(-2)))))
	}
	return ga
}

// eighBackward is the gradient of A = V·diag(W)·Vᵀ, given those of W and
// V, as PyTorch's linalg_eig_backward computes it for a symmetric A.
func eighBackward(w, v, gw, gv *Tensor) *Tensor {
	n := w.node.Shape[len(w.node.Shape)-1]
	vtgv := mT(v).MatMul(gv)
	skew := vtgv.Sub(mT(vtgv)).Mul(Scalar[float32](0.5))
	e := w.Unsqueeze(-2).Sub(w.Unsqueeze(-1)).Add(eye(n)) // e[i][j] = w[j] - w[i]
	inner := divOrZero(skew, e).Add(diagEmbed(gw))
	return v.MatMul(inner).MatMul(mT(v))
}

// divOrZero returns x / y, with 0 where x is 0 whatever y is, so that a
// part of a gradient that receives nothing contributes nothing even
// where y is 0.
func divOrZero(x, y *Tensor) *Tensor {
	return Where(x.Equal(Scalar[float32](0)), Scalar[float32](0), x.Div(y))
}
