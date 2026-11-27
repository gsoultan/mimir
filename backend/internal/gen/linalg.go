// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mimir, a pure-Go machine learning framework,
// created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

package gen

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// Linear algebra kernels factorize, or solve, one matrix of a batch per
// group, in the output: each group copies its matrix there and works on
// it in place, its threads sharing the work of each step and waiting for
// each other between steps. The output has room for every algorithm's
// working matrices, so the kernels need no memory of their own beyond a
// group's shared reductions.
//
// Cholesky, LU with partial pivoting and triangular solves take the
// unblocked right-looking loops of LAPACK's spotf2, sgetf2 and strsm: a
// step per column, whose update of the rest of the matrix the threads
// split element by element. QR reflects columns by Householder
// reflectors chosen as LAPACK's slarfg chooses them, so that R's
// diagonal has LAPACK's signs, and then forms Q in place from them as
// sorg2r does. The SVD and the symmetric eigensolver are one-sided
// Jacobi (Hestenes): sweeps of rotations, each pair of columns in a
// round-robin order whose rounds rotate disjoint pairs at once, until
// every pair is orthogonal. Eigh rotates A + σI for a σ that makes it
// positive definite, whose singular vectors are A's eigenvectors and
// whose singular values are A's eigenvalues plus σ. Signs of the vectors
// follow the convention of the CPU backend (see negativeLead there).
//
// Everything computes in float, where the CPU backend sums some dot
// products in float64, so results differ from the CPU's in their last
// bits, and those of the SVD and eigh, by other algorithms, by more.

// jacobiSweeps bounds the sweeps of the Jacobi kernels, which converge in
// fewer than a dozen for matrices whose singular values are not
// clustered.
const jacobiSweeps = 30

// planLinalg plans a factorization or triangular solve of each matrix of
// the batch of input 0, a group per matrix.
func planLinalg(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	v := k.In[0]
	r := len(v.Shape)
	if r < 2 || k.DType != backend.Float32 || ins[0].DType() != backend.Float32 {
		return nil, fmt.Errorf("%s of %s%v: %w", k.Op, ins[0].DType(), v.Shape, backend.ErrInternal)
	}
	batch, m, n := v.Shape[:r-2], v.Shape[r-2], v.Shape[r-1]
	items, _ := backend.NumElements(batch)
	views := []backend.View{{Strides: v.Strides[:r-2], Offset: v.Offset}}
	if k.Op == backend.OpTriangularSolve {
		if len(k.In) < 2 || len(ins) < 2 || ins[1].DType() != backend.Float32 || len(k.In[1].Shape) != r || k.In[1].Shape[r-2] != n || m != n {
			return nil, fmt.Errorf("triangular solve of %v: %w", v.Shape, backend.ErrInternal)
		}
		views = append(views, backend.View{Strides: k.In[1].Strides[:r-2], Offset: k.In[1].Offset})
	}
	if (k.Op == backend.OpCholesky || k.Op == backend.OpLU || k.Op == backend.OpEigh) && m != n {
		return nil, fmt.Errorf("%s of %v: %w", k.Op, v.Shape, backend.ErrInternal)
	}
	bshape, bviews := backend.Coalesce(slices.Clone(batch), views)

	g := newGen(d, k, outs, ins)
	g.helpers = fmt.Sprintf("%s float fab(float x) { return x < 0.0f ? -x : x; }\n\n", d.decl)
	g.line("int item = int(bid), t = int(tid);")
	offs := g.offsets("item", bshape, bviews)
	g.line("int src = %s, rs = %s, cs = %s;", offs[0], g.param(v.Strides[r-2]), g.param(v.Strides[r-1]))
	g.line("%s float red[%d];", d.shared, GroupSize)
	g.line("%s int redi[%d];", d.shared, GroupSize)
	switch k.Op {
	case backend.OpCholesky:
		g.cholesky(n)
	case backend.OpTriangularSolve:
		g.triangularSolve(n, k.In[1], offs[1])
	case backend.OpLU:
		g.lu(items, n)
	case backend.OpQR:
		g.qr(items, m, n)
	case backend.OpSVD:
		g.svd(items, m, n)
	case backend.OpEigh:
		g.eigh(items, n)
	default:
		return nil, fmt.Errorf("%s is not linear algebra: %w", k.Op, backend.ErrInternal)
	}
	return g.finish(items, true)
}

// barrier makes the group wait for each other and see each other's
// writes to device and shared memory.
func (g *gen) barrier() {
	g.line("%s;", g.d.syncDevice)
	if g.d.sync != g.d.syncDevice {
		g.line("%s;", g.d.sync)
	}
}

// groupReduce emits the reduction of the float x of every thread of the
// group into the float dst, which every thread then holds: their sum, or
// their maximum if max is set.
func (g *gen) groupReduce(dst, x string, max bool) {
	step := "red[t] += red[t + hs];"
	if max {
		step = "if (red[t + hs] > red[t]) red[t] = red[t + hs];"
	}
	g.line("red[t] = %s;", x)
	g.line("%s;", g.d.sync)
	g.line("for (int hs = %d; hs > 0; hs >>= 1) {", GroupSize/2)
	g.line("\tif (t < hs) { %s }", step)
	g.line("\t%s;", g.d.sync)
	g.line("}")
	g.line("float %s = red[0];", dst)
	g.line("%s;", g.d.sync)
}

// in returns the element (i, j) of the group's input matrix.
func in(i, j string) string { return fmt.Sprintf("i0[src + (%s) * rs + (%s) * cs]", i, j) }

// cholesky factorizes the lower triangle of the n×n matrix into L, a
// column at a time, reporting ErrSingular at the first diagonal element
// that is not positive and finite.
func (g *gen) cholesky(n int) {
	g.line("int nn = %s, base = item * nn * nn;", g.param(n))
	at := func(i, j string) string { return fmt.Sprintf("o0[base + (%s) * nn + (%s)]", i, j) }
	g.line("for (int e = t; e < nn * nn; e += %d) {", GroupSize)
	g.line("\tint i = e / nn, j = e %% nn;")
	g.line("\to0[base + e] = j <= i ? %s : 0.0f;", in("i", "j"))
	g.line("}")
	g.barrier()
	g.line("for (int j = 0; j < nn; j++) {")
	g.line("\tfloat dj = %s;", at("j", "j"))
	g.line("\tif (!(dj > 0.0f && dj <= 3.40282347e38f)) {")
	g.line("\t\tif (t == 0) report(err, %du);", ErrSingular)
	g.line("\t\treturn;")
	g.line("\t}")
	g.line("\tfloat lj = sqrt(dj);")
	g.line("\tfor (int i = j + 1 + t; i < nn; i += %d) %s /= lj;", GroupSize, at("i", "j"))
	g.barrier()
	g.line("\tif (t == 0) %s = lj;", at("j", "j"))
	g.line("\tint rest = nn - j - 1;")
	g.line("\tfor (int e = t; e < rest * rest; e += %d) {", GroupSize)
	g.line("\t\tint i = j + 1 + e / rest, c = j + 1 + e %% rest;")
	g.line("\t\tif (c <= i) %s -= %s * %s;", at("i", "c"), at("i", "j"), at("c", "j"))
	g.line("\t}")
	g.barrier()
	g.line("}")
}

// triangularSolve replaces the n×k right-hand sides of input 1, copied
// to the output, by X of A·X = B, a row of X at a time, reporting
// ErrSingular for a zero on A's diagonal before it starts.
func (g *gen) triangularSolve(n int, bv backend.View, off1 string) {
	r := len(bv.Shape)
	upper, unit := g.k.Attrs.Upper, g.k.Attrs.Unit
	g.line("int nn = %s, kc = %s, base = item * nn * kc, src1 = %s, brs = %s, bcs = %s;",
		g.param(n), g.param(bv.Shape[r-1]), off1, g.param(bv.Strides[r-2]), g.param(bv.Strides[r-1]))
	x := func(i, c string) string { return fmt.Sprintf("o0[base + (%s) * kc + (%s)]", i, c) }
	if !unit {
		g.line("if (t == 0) redi[0] = 0;")
		g.line("%s;", g.d.sync)
		g.line("for (int i = t; i < nn; i += %d) if (%s == 0.0f) redi[0] = 1;", GroupSize, in("i", "i"))
		g.line("%s;", g.d.sync)
		g.line("if (redi[0] != 0) {")
		g.line("\tif (t == 0) report(err, %du);", ErrSingular)
		g.line("\treturn;")
		g.line("}")
	}
	g.line("for (int e = t; e < nn * kc; e += %d) {", GroupSize)
	g.line("\tint i = e / kc, c = e %% kc;")
	g.line("\to0[base + e] = i1[src1 + i * brs + c * bcs];")
	g.line("}")
	g.barrier()
	g.line("for (int s = 0; s < nn; s++) {")
	if upper {
		g.line("\tint i = nn - 1 - s, r0 = 0, cnt = i;")
	} else {
		g.line("\tint i = s, r0 = i + 1, cnt = nn - 1 - i;")
	}
	if !unit {
		g.line("\tfloat di = %s;", in("i", "i"))
		g.line("\tfor (int c = t; c < kc; c += %d) %s /= di;", GroupSize, x("i", "c"))
		g.barrier()
	}
	g.line("\tfor (int e = t; e < cnt * kc; e += %d) {", GroupSize)
	g.line("\t\tint q = r0 + e / kc, c = e %% kc;")
	g.line("\t\t%s -= %s * %s;", x("q", "c"), in("q", "i"), x("i", "c"))
	g.line("\t}")
	g.barrier()
	g.line("}")
}

// lu factorizes the n×n matrix in place into P·A = L·U with partial
// pivoting, choosing the first row of the largest magnitude as the CPU
// backend does, and writes the permutation and its sign after the
// batch's factors.
func (g *gen) lu(items, n int) {
	g.line("int nn = %s, lu = item * nn * nn, pm = %s + item * nn, sg = %s + item;",
		g.param(n), g.param(items*n*n), g.param(items*n*n+items*n))
	at := func(i, j string) string { return fmt.Sprintf("o0[lu + (%s) * nn + (%s)]", i, j) }
	g.line("for (int e = t; e < nn * nn; e += %d) o0[lu + e] = %s;", GroupSize, in("e / nn", "e % nn"))
	g.line("for (int e = t; e < nn; e += %d) o0[pm + e] = float(e);", GroupSize)
	g.line("if (t == 0) o0[sg] = 1.0f;")
	g.barrier()
	g.line("for (int j = 0; j < nn; j++) {")
	g.line("\tfloat bv = -1.0f;")
	g.line("\tint bi = j;")
	g.line("\tfor (int i = j + t; i < nn; i += %d) {", GroupSize)
	g.line("\t\tfloat a = fab(%s);", at("i", "j"))
	g.line("\t\tif (a > bv) { bv = a; bi = i; }")
	g.line("\t}")
	g.line("\tred[t] = bv;")
	g.line("\tredi[t] = bi;")
	g.line("\t%s;", g.d.sync)
	g.line("\tfor (int hs = %d; hs > 0; hs >>= 1) {", GroupSize/2)
	g.line("\t\tif (t < hs) {")
	g.line("\t\t\tfloat ov = red[t + hs];")
	g.line("\t\t\tint oi = redi[t + hs];")
	g.line("\t\t\tif (ov > red[t] || (ov == red[t] && oi < redi[t])) { red[t] = ov; redi[t] = oi; }")
	g.line("\t\t}")
	g.line("\t\t%s;", g.d.sync)
	g.line("\t}")
	g.line("\tint piv = redi[0];")
	// A NaN on the diagonal stays the pivot, as the CPU's scan keeps it.
	g.line("\tfloat dj = fab(%s);", at("j", "j"))
	g.line("\tif (!(dj == dj)) piv = j;")
	g.barrier()
	g.line("\tif (piv != j) {")
	g.line("\t\tfor (int c = t; c < nn; c += %d) {", GroupSize)
	g.line("\t\t\tfloat y = %s;", at("j", "c"))
	g.line("\t\t\t%s = %s;", at("j", "c"), at("piv", "c"))
	g.line("\t\t\t%s = y;", at("piv", "c"))
	g.line("\t\t}")
	g.line("\t\tif (t == 0) {")
	g.line("\t\t\tfloat y = o0[pm + j];")
	g.line("\t\t\to0[pm + j] = o0[pm + piv];")
	g.line("\t\t\to0[pm + piv] = y;")
	g.line("\t\t\to0[sg] = -o0[sg];")
	g.line("\t\t}")
	g.barrier()
	g.line("\t}")
	g.line("\tfloat dv = %s;", at("j", "j"))
	g.line("\tif (dv != 0.0f) {")
	g.line("\t\tfor (int i = j + 1 + t; i < nn; i += %d) %s /= dv;", GroupSize, at("i", "j"))
	g.barrier()
	g.line("\t\tint rest = nn - j - 1;")
	g.line("\t\tfor (int e = t; e < rest * rest; e += %d) {", GroupSize)
	g.line("\t\t\tint i = j + 1 + e / rest, c = j + 1 + e %% rest;")
	g.line("\t\t\t%s -= %s * %s;", at("i", "c"), at("i", "j"), at("j", "c"))
	g.line("\t\t}")
	g.barrier()
	g.line("\t}")
	g.line("}")
}

// qr factorizes the m×n matrix into Q, m×k, and R, k×n, k = min(m, n).
// The matrix is reduced in whichever of Q's and R's places holds m×n
// elements: Q's when m ≥ n, R's otherwise. Reflector j's v, whose v[0]
// is 1, stays below the diagonal of column j; its tau, 2/(vᵀv), or 0
// for a reflector that changes nothing, is computed again from v where
// sorg2r reads it.
func (g *gen) qr(items, m, n int) {
	k := min(m, n)
	g.line("int mm = %s, nc = %s, kk = %s;", g.param(m), g.param(n), g.param(k))
	g.line("int qb = item * mm * kk, rb = %s + item * kk * nc;", g.param(items*m*k))
	if m >= n {
		g.line("int wb = qb;")
	} else {
		g.line("int wb = rb;")
	}
	w := func(i, j string) string { return fmt.Sprintf("o0[wb + (%s) * nc + (%s)]", i, j) }
	g.line("for (int e = t; e < mm * nc; e += %d) o0[wb + e] = %s;", GroupSize, in("e / nc", "e % nc"))
	g.barrier()
	g.line("for (int j = 0; j < kk; j++) {")
	g.line("\tfloat part = 0.0f;")
	g.line("\tfor (int i = j + 1 + t; i < mm; i += %d) { float y = fab(%s); if (y > part) part = y; }", GroupSize, w("i", "j"))
	g.groupReduce("amax", "part", true)
	g.line("\tpart = 0.0f;")
	g.line("\tif (amax > 0.0f) for (int i = j + 1 + t; i < mm; i += %d) { float y = %s / amax; part += y * y; }", GroupSize, w("i", "j"))
	g.groupReduce("ss", "part", false)
	g.line("\tfloat alpha = %s;", w("j", "j"))
	g.barrier()
	g.line("\tif (amax > 0.0f) {")
	g.line("\t\tfloat xnorm = amax * sqrt(ss), big = fab(alpha) > xnorm ? fab(alpha) : xnorm;")
	g.line("\t\tfloat ha = alpha / big, hx = xnorm / big;")
	g.line("\t\tfloat hyp = big * sqrt(ha * ha + hx * hx);")
	g.line("\t\tfloat beta = (ubits(alpha) >> 31) != 0u ? hyp : -hyp;")
	g.line("\t\tfloat tau = (beta - alpha) / beta, sc = 1.0f / (alpha - beta);")
	g.line("\t\tfor (int i = j + 1 + t; i < mm; i += %d) %s *= sc;", GroupSize, w("i", "j"))
	g.line("\t\tif (t == 0) %s = beta;", w("j", "j"))
	g.barrier()
	g.line("\t\tfor (int c = j + 1 + t; c < nc; c += %d) {", GroupSize)
	g.line("\t\t\tfloat y = %s;", w("j", "c"))
	g.line("\t\t\tfor (int i = j + 1; i < mm; i++) y += %s * %s;", w("i", "j"), w("i", "c"))
	g.line("\t\t\ty *= tau;")
	g.line("\t\t\t%s -= y;", w("j", "c"))
	g.line("\t\t\tfor (int i = j + 1; i < mm; i++) %s -= y * %s;", w("i", "c"), w("i", "j"))
	g.line("\t\t}")
	g.barrier()
	g.line("\t}")
	g.line("}")
	if m >= n {
		g.line("for (int e = t; e < kk * nc; e += %d) {", GroupSize)
		g.line("\tint i = e / nc, c = e %% nc;")
		g.line("\to0[rb + e] = c >= i ? %s : 0.0f;", w("i", "c"))
		g.line("}")
	} else {
		g.line("for (int e = t; e < mm * mm; e += %d) o0[qb + e] = %s;", GroupSize, w("e / mm", "e % mm"))
		g.barrier()
		g.line("for (int e = t; e < kk * nc; e += %d) if (e %% nc < e / nc) o0[rb + e] = 0.0f;", GroupSize)
	}
	g.barrier()
	q := func(i, j string) string { return fmt.Sprintf("o0[qb + (%s) * kk + (%s)]", i, j) }
	g.line("for (int i = kk - 1; i >= 0; i--) {")
	g.line("\tfloat part = 0.0f;")
	g.line("\tfor (int r = i + 1 + t; r < mm; r += %d) { float y = fab(%s); if (y > part) part = y; }", GroupSize, q("r", "i"))
	g.groupReduce("vmax", "part", true)
	g.line("\tpart = 0.0f;")
	g.line("\tif (vmax > 0.0f) for (int r = i + 1 + t; r < mm; r += %d) { float y = %s / vmax; part += y * y; }", GroupSize, q("r", "i"))
	g.groupReduce("vs", "part", false)
	g.line("\tfloat tau = vmax > 0.0f ? 2.0f / (1.0f + vmax * vmax * vs) : 0.0f;")
	g.line("\tfor (int c = i + 1 + t; c < kk; c += %d) {", GroupSize)
	g.line("\t\tfloat y = %s;", q("i", "c"))
	g.line("\t\tfor (int r = i + 1; r < mm; r++) y += %s * %s;", q("r", "i"), q("r", "c"))
	g.line("\t\ty *= tau;")
	g.line("\t\t%s -= y;", q("i", "c"))
	g.line("\t\tfor (int r = i + 1; r < mm; r++) %s -= y * %s;", q("r", "c"), q("r", "i"))
	g.line("\t}")
	g.barrier()
	g.line("\tfor (int r = t; r < mm; r += %d) %s = r < i || tau == 0.0f ? (r == i ? 1.0f : 0.0f) : r == i ? 1.0f - tau : -tau * %s;", GroupSize, q("r", "i"), q("r", "i"))
	g.barrier()
	g.line("}")
}

// vectors addresses p vectors of l elements in the output: element e of
// vector j is at base + j*js + e*es, each an expression.
type vectors struct {
	base, js, es string
	p, l         string
}

func (v vectors) at(j, e string) string {
	return fmt.Sprintf("o0[%s + (%s) * %s + (%s) * %s]", v.base, j, v.js, e, v.es)
}

// jacobi orthogonalizes the p vectors w, each pair in turn, rotating the
// vectors acc, if any, alike, and then sets the p values at vals to the
// norms of w and normalizes w: a vector of norm below 1e-35 is set to
// zero, or, if complete is set, to a unit vector orthogonal to the
// others. It sorts the vectors by their values, in descending order if
// desc is set, and flips the signs of each vector of signs, w or acc,
// whose leading element is negative, and of the other vector with it.
// Rounds of the round-robin order pair the p vectors, and a dummy if p
// is odd, as players of a tournament whose last player stays put.
func (g *gen) jacobi(w vectors, acc *vectors, vals string, players, length int, complete, desc, signsOfAcc bool) {
	tol := max(1e-6, math.Sqrt(float64(length))*2.4e-7)
	pp := players + players%2
	g.line("int pp = %s;", g.param(pp))
	rotate := func(v vectors) {
		g.line("\t\t\t\tfor (int e = 0; e < %s; e++) {", v.l)
		g.line("\t\t\t\t\tfloat x = %s, y = %s;", v.at("a", "e"), v.at("b", "e"))
		g.line("\t\t\t\t\t%s = cc * x - sn * y;", v.at("a", "e"))
		g.line("\t\t\t\t\t%s = sn * x + cc * y;", v.at("b", "e"))
		g.line("\t\t\t\t}")
	}
	g.line("for (int sweep = 0; sweep < %d; sweep++) {", jacobiSweeps)
	g.line("\tif (t == 0) redi[0] = 0;")
	g.barrier()
	g.line("\tfor (int rnd = 0; rnd < pp - 1; rnd++) {")
	g.line("\t\tfor (int q = t; q < pp / 2; q += %d) {", GroupSize)
	g.line("\t\t\tint a = q == 0 ? rnd : (rnd + q) %% (pp - 1);")
	g.line("\t\t\tint b = q == 0 ? pp - 1 : (rnd - q + pp - 1) %% (pp - 1);")
	g.line("\t\t\tif (a > b) { int y = a; a = b; b = y; }")
	g.line("\t\t\tif (b >= %s) continue;", w.p)
	g.line("\t\t\tfloat al = 0.0f, be = 0.0f, ga = 0.0f;")
	g.line("\t\t\tfor (int e = 0; e < %s; e++) {", w.l)
	g.line("\t\t\t\tfloat x = %s, y = %s;", w.at("a", "e"), w.at("b", "e"))
	g.line("\t\t\t\tal += x * x;")
	g.line("\t\t\t\tbe += y * y;")
	g.line("\t\t\t\tga += x * y;")
	g.line("\t\t\t}")
	g.line("\t\t\tif (fab(ga) > %s * sqrt(al) * sqrt(be)) {", literal(tol, backend.Float32))
	g.line("\t\t\t\tredi[0] = 1;")
	g.line("\t\t\t\tfloat z = (be - al) / (2.0f * ga), az = fab(z);")
	g.line("\t\t\t\tfloat tt = az > 1e18f ? 0.5f / az : 1.0f / (az + sqrt(1.0f + az * az));")
	g.line("\t\t\t\tif (z < 0.0f) tt = -tt;")
	g.line("\t\t\t\tfloat cc = 1.0f / sqrt(1.0f + tt * tt), sn = cc * tt;")
	rotate(w)
	if acc != nil {
		rotate(*acc)
	}
	g.line("\t\t\t}")
	g.line("\t\t}")
	g.barrier()
	g.line("\t}")
	g.line("\tint rotated = redi[0];")
	g.barrier()
	g.line("\tif (rotated == 0) break;")
	g.line("}")

	val := func(j string) string { return fmt.Sprintf("o0[%s + (%s)]", vals, j) }
	g.line("for (int j = t; j < %s; j += %d) {", w.p, GroupSize)
	g.line("\tfloat s = 0.0f;")
	g.line("\tfor (int e = 0; e < %s; e++) { float x = %s; s += x * x; }", w.l, w.at("j", "e"))
	g.line("\t%s = sqrt(s);", val("j"))
	g.line("}")
	g.barrier()
	g.line("for (int e = t; e < %s * %s; e += %d) {", w.p, w.l, GroupSize)
	g.line("\tint j = e / %s, i = e %% %s;", w.l, w.l)
	g.line("\tfloat s = %s;", val("j"))
	g.line("\t%s = s > 1e-35f ? %s / s : 0.0f;", w.at("j", "i"), w.at("j", "i"))
	g.line("}")
	g.barrier()

	if complete {
		// The standard basis vector with the most left outside the
		// others, after two passes of Gram-Schmidt: one of them has more
		// than half the share each would have if they had the same.
		g.line("for (int r = 0; r < %s; r++) {", w.p)
		g.line("\tif (%s > 1e-35f) continue;", val("r"))
		g.line("\tfor (int bs = 0; bs < %s; bs++) {", w.l)
		g.line("\t\tfor (int e = t; e < %s; e += %d) %s = e == bs ? 1.0f : 0.0f;", w.l, GroupSize, w.at("r", "e"))
		g.barrier()
		g.line("\t\tfor (int pass = 0; pass < 2; pass++) {")
		g.line("\t\t\tfor (int q = 0; q < %s; q++) {", w.p)
		g.line("\t\t\t\tif (q == r) continue;")
		g.line("\t\t\t\tfloat part = 0.0f;")
		g.line("\t\t\t\tfor (int e = t; e < %s; e += %d) part += %s * %s;", w.l, GroupSize, w.at("r", "e"), w.at("q", "e"))
		g.groupReduce("pr", "part", false)
		g.line("\t\t\t\tfor (int e = t; e < %s; e += %d) %s -= pr * %s;", w.l, GroupSize, w.at("r", "e"), w.at("q", "e"))
		g.barrier()
		g.line("\t\t\t}")
		g.line("\t\t}")
		g.line("\t\tfloat part = 0.0f;")
		g.line("\t\tfor (int e = t; e < %s; e += %d) { float x = %s; part += x * x; }", w.l, GroupSize, w.at("r", "e"))
		g.groupReduce("nr", "part", false)
		g.line("\t\tif (nr > 0.5f / float(%s)) {", w.l)
		g.line("\t\t\tfloat inv = 1.0f / sqrt(nr);")
		g.line("\t\t\tfor (int e = t; e < %s; e += %d) %s *= inv;", w.l, GroupSize, w.at("r", "e"))
		g.barrier()
		g.line("\t\t\tbreak;")
		g.line("\t\t}")
		g.line("\t}")
		g.line("}")
	}

	// Selection sort, the swaps split between the threads.
	cmp := "<"
	if desc {
		cmp = ">"
	}
	swap := func(v vectors) {
		g.line("\t\tfor (int e = t; e < %s; e += %d) {", v.l, GroupSize)
		g.line("\t\t\tfloat y = %s;", v.at("pos", "e"))
		g.line("\t\t\t%s = %s;", v.at("pos", "e"), v.at("bi", "e"))
		g.line("\t\t\t%s = y;", v.at("bi", "e"))
		g.line("\t\t}")
	}
	g.line("for (int pos = 0; pos + 1 < %s; pos++) {", w.p)
	g.line("\tif (t == 0) {")
	g.line("\t\tint bi = pos;")
	g.line("\t\tfloat bv = %s;", val("pos"))
	g.line("\t\tfor (int j = pos + 1; j < %s; j++) { float y = %s; if (y %s bv) { bv = y; bi = j; } }", w.p, val("j"), cmp)
	g.line("\t\tredi[0] = bi;")
	g.line("\t}")
	g.barrier()
	g.line("\tint bi = redi[0];")
	g.line("\tif (bi != pos) {")
	swap(w)
	if acc != nil {
		swap(*acc)
	}
	g.line("\t\tif (t == 0) { float y = %s; %s = %s; %s = y; }", val("pos"), val("pos"), val("bi"), val("bi"))
	g.line("\t}")
	g.barrier()
	g.line("}")

	lead, other := w, acc
	if signsOfAcc {
		lead, other = *acc, &w
	}
	g.line("for (int j = t; j < %s; j += %d) {", w.p, GroupSize)
	g.line("\tfloat big = 0.0f, pick = 0.0f;")
	g.line("\tfor (int e = 0; e < %s; e++) { float y = fab(%s); if (y > big) big = y; }", lead.l, lead.at("j", "e"))
	g.line("\tfor (int e = 0; e < %s; e++) { float y = %s; if (fab(y) >= 0.5f * big) { pick = y; break; } }", lead.l, lead.at("j", "e"))
	g.line("\tif (pick < 0.0f) {")
	g.line("\t\tfor (int e = 0; e < %s; e++) %s = -%s;", lead.l, lead.at("j", "e"), lead.at("j", "e"))
	if other != nil {
		g.line("\t\tfor (int e = 0; e < %s; e++) %s = -%s;", other.l, other.at("j", "e"), other.at("j", "e"))
	}
	g.line("\t}")
	g.line("}")
}

// svd computes the reduced SVD of the m×n matrix by one-sided Jacobi:
// of its columns, in U's place, when m ≥ n, with Vh's rows rotated
// alike from the identity, or else of its rows, in Vh's place, with U's
// columns rotated alike.
func (g *gen) svd(items, m, n int) {
	k := min(m, n)
	g.line("int mm = %s, nc = %s, kk = %s;", g.param(m), g.param(n), g.param(k))
	g.line("int ub = item * mm * kk, sb = %s + item * kk, vb = %s + item * kk * nc;", g.param(items*m*k), g.param(items*m*k+items*k))
	ucols := vectors{base: "ub", js: "1", es: "kk", p: "kk", l: "mm"}
	vrows := vectors{base: "vb", js: "nc", es: "1", p: "kk", l: "nc"}
	if m >= n {
		g.line("for (int e = t; e < mm * nc; e += %d) o0[ub + e] = %s;", GroupSize, in("e / nc", "e % nc"))
		g.line("for (int e = t; e < kk * nc; e += %d) o0[vb + e] = e / nc == e %% nc ? 1.0f : 0.0f;", GroupSize)
		g.jacobi(ucols, &vrows, "sb", k, m, true, true, false)
		return
	}
	g.line("for (int e = t; e < mm * nc; e += %d) o0[vb + e] = %s;", GroupSize, in("e / nc", "e % nc"))
	g.line("for (int e = t; e < mm * kk; e += %d) o0[ub + e] = e / kk == e %% kk ? 1.0f : 0.0f;", GroupSize)
	g.jacobi(vrows, &ucols, "sb", k, n, true, true, true)
}

// eigh computes the eigenvalues and eigenvectors of the symmetric n×n
// matrix, read from its lower triangle, by one-sided Jacobi of A + σI,
// in V's place, with σ twice the largest absolute row sum, which bounds
// A's eigenvalues, or 1 for a zero A: the norms of the columns are then
// the eigenvalues plus σ, all positive.
func (g *gen) eigh(items, n int) {
	g.line("int nn = %s, wb = item * nn, vb = %s + item * nn * nn;", g.param(n), g.param(items*n))
	sym := func(i, j string) string {
		return fmt.Sprintf("(%[2]s <= %[1]s ? %[3]s : %[4]s)", i, j, in(i, j), in(j, i))
	}
	g.line("float part = 0.0f;")
	g.line("for (int i = t; i < nn; i += %d) {", GroupSize)
	g.line("\tfloat s = 0.0f;")
	g.line("\tfor (int c = 0; c < nn; c++) s += fab(%s);", sym("i", "c"))
	g.line("\tif (s > part) part = s;")
	g.line("}")
	g.groupReduce("rowmax", "part", true)
	g.line("float sigma = rowmax > 0.0f ? 2.0f * rowmax : 1.0f;")
	g.line("for (int e = t; e < nn * nn; e += %d) {", GroupSize)
	g.line("\tint i = e / nn, c = e %% nn;")
	g.line("\to0[vb + e] = %s + (i == c ? sigma : 0.0f);", sym("i", "c"))
	g.line("}")
	cols := vectors{base: "vb", js: "1", es: "nn", p: "nn", l: "nn"}
	g.jacobi(cols, nil, "wb", n, n, false, false, false)
	g.line("for (int j = t; j < nn; j += %d) o0[wb + j] -= sigma;", GroupSize)
}
