package cpu

import (
	"context"
	"fmt"
	"math"

	"github.com/gsoultan/mimir/backend"
)

// Linear algebra runs on Float32 matrices batched over leading
// dimensions. Each matrix is copied into a row-major matrix of its own,
// in the output where the output has room for it, and factorized there
// by blocked algorithms that leave most of their work to the GEMM
// micro-kernels: a block of linalgBlock columns is factorized by plain
// loops, and the rest of the matrix updated by one product. A batch of
// at least as many matrices as workers, or of small ones, gives each
// worker whole matrices; otherwise the products of each matrix are split
// between the workers.
//
// Cholesky, LU and triangular solves compute in float32 as LAPACK's
// single-precision routines do. QR, the symmetric eigensolver and the SVD
// are in qr.go, eigh.go and svd.go.

// linalgBlock is the number of columns that blocked factorizations
// factorize by plain loops before they update the rest of the matrix with
// a product.
const linalgBlock = 64

// linalgShared is the least work, in multiply-adds, of one matrix that is
// split between the workers rather than given to one.
const linalgShared = 1 << 21

// runLinalg runs a factorization or triangular solve.
func runLinalg(ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, ins []backend.Buffer) error {
	out, err := data[float32](outBuf)
	if err != nil {
		return err
	}
	x, err := data[float32](ins[0])
	if err != nil {
		return err
	}
	v := k.In[0]
	r := len(v.Shape)
	if r < 2 || k.DType != backend.Float32 {
		return fmt.Errorf("%s of %s%v: %w", k.Op, k.DType, v.Shape, backend.ErrInternal)
	}
	batch, m, n := v.Shape[:r-2], v.Shape[r-2], v.Shape[r-1]
	items, _ := backend.NumElements(batch)
	offs := batchOffsets(batch, v)
	rs, cs := v.Strides[r-2], v.Strides[r-1]
	// load copies item i of the input into a.
	load := func(a mat, i int) {
		for row := range a.rows {
			dst := a.row(row)
			src := offs[i] + row*rs
			for j := range dst {
				dst[j] = x[src+j*cs]
			}
		}
	}
	var each func(ctx context.Context, par bool, i int) error
	work := m * n * min(m, n)
	switch k.Op {
	case backend.OpCholesky:
		if m != n || len(out) != items*n*n {
			return fmt.Errorf("cholesky of %v: %w", v.Shape, backend.ErrInternal)
		}
		each = func(ctx context.Context, par bool, i int) error {
			a := mat{data: out[i*n*n:][:n*n], rows: n, cols: n, ld: n}
			load(a, i)
			return b.cholesky(ctx, par, a)
		}
	case backend.OpTriangularSolve:
		bv := k.In[1]
		y, err := data[float32](ins[1])
		if err != nil {
			return err
		}
		if m != n || len(bv.Shape) != r || bv.Shape[r-2] != n || len(out) != items*n*bv.Shape[r-1] {
			return fmt.Errorf("triangular solve of %v by %v: %w", v.Shape, bv.Shape, backend.ErrInternal)
		}
		cols := bv.Shape[r-1]
		bOffs := batchOffsets(batch, bv)
		work = n * n * cols
		each = func(ctx context.Context, par bool, i int) error {
			xm := mat{data: out[i*n*cols:][:n*cols], rows: n, cols: cols, ld: cols}
			for row := range n {
				dst, src := xm.row(row), bOffs[i]+row*bv.Strides[r-2]
				for j := range dst {
					dst[j] = y[src+j*bv.Strides[r-1]]
				}
			}
			a := operand{data: x, off: offs[i], rs: rs, cs: cs}
			return b.triangularSolve(ctx, par, a, k.Attrs.Upper, k.Attrs.Unit, xm)
		}
	default:
		parts, size, err := backend.FactorParts(k.Op, v.Shape)
		if err != nil || len(out) != size {
			return fmt.Errorf("%s of %v into %d elements: %w", k.Op, v.Shape, len(out), backend.ErrInternal)
		}
		// part returns item i of output part p.
		part := func(p, i int) []float32 {
			at := 0
			for _, s := range parts[:p] {
				n, _ := backend.NumElements(s)
				at += n
			}
			per, _ := backend.NumElements(parts[p][len(batch):])
			return out[at+i*per:][:per]
		}
		switch k.Op {
		case backend.OpLU:
			each = func(ctx context.Context, par bool, i int) error {
				a := mat{data: part(0, i), rows: n, cols: n, ld: n}
				load(a, i)
				return b.lu(ctx, par, a, part(1, i), &part(2, i)[0])
			}
		case backend.OpQR:
			kk := min(m, n)
			each = func(ctx context.Context, par bool, i int) error {
				a := newMat(m, n)
				load(a, i)
				q := mat{data: part(0, i), rows: m, cols: kk, ld: kk}
				rm := mat{data: part(1, i), rows: kk, cols: n, ld: n}
				return b.qr(ctx, par, a, q, rm)
			}
		case backend.OpSVD:
			kk := min(m, n)
			each = func(ctx context.Context, par bool, i int) error {
				a := newMat(m, n)
				load(a, i)
				u := mat{data: part(0, i), rows: m, cols: kk, ld: kk}
				vh := mat{data: part(2, i), rows: kk, cols: n, ld: n}
				return b.svd(ctx, par, a, u, part(1, i), vh)
			}
		case backend.OpEigh:
			each = func(ctx context.Context, par bool, i int) error {
				a := mat{data: part(1, i), rows: n, cols: n, ld: n}
				load(a, i)
				return b.eigh(ctx, par, a, part(0, i))
			}
		}
	}
	if items >= b.workers || work < linalgShared {
		return b.parallelFor(ctx, items, max(1, linalgShared/max(work, 1)), func(lo, hi int) error {
			for i := lo; i < hi; i++ {
				if err := each(ctx, false, i); err != nil {
					return err
				}
			}
			return nil
		})
	}
	for i := range items {
		if err := each(ctx, true, i); err != nil {
			return err
		}
	}
	return nil
}

func singular(op backend.Op, at int) error {
	if op == backend.OpCholesky {
		return fmt.Errorf("leading minor %d is not positive definite: %w", at+1, backend.ErrSingular)
	}
	return fmt.Errorf("zero diagonal element %d: %w", at, backend.ErrSingular)
}

// dot64 returns Σ a[i]·b[i] summed in float64.
func dot64(a, b []float32) float64 {
	b = b[:len(a)]
	var s float64
	for i, x := range a {
		s += float64(x) * float64(b[i])
	}
	return s
}

// axpyVector adds a times each of the n elements at x, a multiple of
// axpyLanes, to those at y, or is nil where there is no vector routine.
// The architecture-specific files set it at init.
var (
	axpyVector func(a float32, x, y *float32, n int)
	axpyLanes  int
)

// axpy adds a·x to y.
func axpy(a float32, x, y []float32) {
	y = y[:len(x)]
	n := 0
	if axpyVector != nil {
		if n = len(x) &^ (axpyLanes - 1); n > 0 {
			axpyVector(a, &x[0], &y[0], n)
		}
	}
	for i := n; i < len(x); i++ {
		y[i] += a * x[i]
	}
}

// cholesky replaces the lower triangle of the symmetric a with its
// Cholesky factor L, a = L·Lᵀ, and zeroes the upper triangle. It reads
// only the lower triangle.
func (b *Backend) cholesky(ctx context.Context, par bool, a mat) error {
	n := a.rows
	for j0 := 0; j0 < n; j0 += linalgBlock {
		j1 := min(j0+linalgBlock, n)
		// The diagonal block, by plain loops, summing in float64.
		for j := j0; j < j1; j++ {
			rj := a.row(j)
			d := float64(rj[j]) - dot64(rj[j0:j], rj[j0:j])
			if !(d > 0) || math.IsInf(d, 0) {
				return singular(backend.OpCholesky, j)
			}
			ljj := math.Sqrt(d)
			rj[j] = float32(ljj)
			for i := j + 1; i < j1; i++ {
				ri := a.row(i)
				ri[j] = float32((float64(ri[j]) - dot64(ri[j0:j], rj[j0:j])) / ljj)
			}
		}
		if j1 == n {
			break
		}
		// The block below it: L21·L11ᵀ = A21, row by row.
		err := b.maybeParallel(ctx, par, n-j1, 64, func(lo, hi int) error {
			for i := j1 + lo; i < j1+hi; i++ {
				ri := a.row(i)
				for j := j0; j < j1; j++ {
					rj := a.row(j)
					ri[j] = (ri[j] - dot(ri[j0:j], rj[j0:j])) / rj[j]
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
		// The lower triangle of the rest less L21·L21ᵀ, a block column
		// and up to 256 rows a task.
		type task struct{ c0, cb, r0, rb int }
		var tasks []task
		for c0 := j1; c0 < n; c0 += linalgBlock {
			cb := min(linalgBlock, n-c0)
			for r0 := c0; r0 < n; r0 += 256 {
				tasks = append(tasks, task{c0, cb, r0, min(256, n-r0)})
			}
		}
		jb := j1 - j0
		err = b.maybeParallel(ctx, par, len(tasks), 1, func(lo, hi int) error {
			for _, t := range tasks[lo:hi] {
				c := a.sub(t.r0, t.c0, t.rb, t.cb)
				x := a.sub(t.r0, j0, t.rb, jb)
				y := a.sub(t.c0, j0, t.cb, jb)
				if err := b.mulAdd(ctx, false, c, true, true, x.op(false), y.op(true), jb); err != nil {
					return err
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	for i := range n {
		clear(a.row(i)[i+1:])
	}
	return nil
}

// lu factorizes a in place into P·a = L·U with partial pivoting, L's unit
// diagonal implied, and sets perm[i] to the row of a that row i of P·a
// is, and sign to P's determinant. A zero pivot leaves its column of L
// zero, as LAPACK's sgetrf does.
func (b *Backend) lu(ctx context.Context, par bool, a mat, perm []float32, sign *float32) error {
	n := a.rows
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	s := float32(1)
	for j0 := 0; j0 < n; j0 += linalgBlock {
		j1 := min(j0+linalgBlock, n)
		// The panel of columns [j0, j1), with whole rows swapped.
		for j := j0; j < j1; j++ {
			piv, best := j, math.Abs(float64(a.data[j*a.ld+j]))
			for i := j + 1; i < n; i++ {
				if v := math.Abs(float64(a.data[i*a.ld+j])); v > best {
					piv, best = i, v
				}
			}
			if piv != j {
				ri, rp := a.row(j), a.row(piv)
				for c := range ri {
					ri[c], rp[c] = rp[c], ri[c]
				}
				p[j], p[piv] = p[piv], p[j]
				s = -s
			}
			rj := a.row(j)
			d := rj[j]
			if d == 0 {
				continue
			}
			err := b.maybeParallel(ctx, par && (n-j)*(j1-j) > 1<<16, n-j-1, 256, func(lo, hi int) error {
				for i := j + 1 + lo; i < j+1+hi; i++ {
					ri := a.row(i)
					l := ri[j] / d
					ri[j] = l
					axpy(-l, rj[j+1:j1], ri[j+1:j1])
				}
				return nil
			})
			if err != nil {
				return err
			}
		}
		if j1 == n {
			break
		}
		// U12 = L11⁻¹·A12, a chunk of columns each.
		err := b.maybeParallel(ctx, par, n-j1, 256, func(lo, hi int) error {
			for i := j0 + 1; i < j1; i++ {
				ri := a.row(i)
				for q := j0; q < i; q++ {
					axpy(-ri[q], a.row(q)[j1+lo:j1+hi], ri[j1+lo:j1+hi])
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
		jb := j1 - j0
		rest := n - j1
		if err := b.mulAdd(ctx, par, a.sub(j1, j1, rest, rest), true, true, a.sub(j1, j0, rest, jb).op(false), a.sub(j0, j1, jb, rest).op(false), jb); err != nil {
			return err
		}
	}
	for i, v := range p {
		perm[i] = float32(v)
	}
	*sign = s
	return nil
}

// triangularSolve replaces x, n×k, with A⁻¹·x for the triangular n×n
// matrix A at a, read from its lower or upper triangle, with ones on its
// diagonal if unit is set.
func (b *Backend) triangularSolve(ctx context.Context, par bool, a operand, upper, unit bool, x mat) error {
	n, k := x.rows, x.cols
	at := func(i, j int) float32 { return a.data[a.off+i*a.rs+j*a.cs] }
	if !unit {
		for i := range n {
			if at(i, i) == 0 {
				return singular(backend.OpTriangularSolve, i)
			}
		}
	}
	if k == 1 && a.cs == 1 {
		// One column, rows of A contiguous: dot products.
		v := x.data[:n]
		row := func(i int) []float32 { return a.data[a.off+i*a.rs:][:n] }
		solve := func(i int, s float32) {
			if !unit {
				s /= at(i, i)
			}
			v[i] = s
		}
		if upper {
			for i := n - 1; i >= 0; i-- {
				solve(i, v[i]-dot(row(i)[i+1:], v[i+1:]))
			}
		} else {
			for i := range n {
				solve(i, v[i]-dot(row(i)[:i], v[:i]))
			}
		}
		return nil
	}
	for blk := 0; blk < n; blk += linalgBlock {
		// The blocks go down for a lower A and up for an upper one.
		i0, i1 := blk, min(blk+linalgBlock, n)
		if upper {
			i0, i1 = max(n-blk-linalgBlock, 0), n-blk
		}
		err := b.maybeParallel(ctx, par && k >= 64, k, 64, func(lo, hi int) error {
			for t := range i1 - i0 {
				i := i0 + t
				if upper {
					i = i1 - 1 - t
				}
				ri := x.row(i)[lo:hi]
				if upper {
					for q := i + 1; q < i1; q++ {
						axpy(-at(i, q), x.row(q)[lo:hi], ri)
					}
				} else {
					for q := i0; q < i; q++ {
						axpy(-at(i, q), x.row(q)[lo:hi], ri)
					}
				}
				if !unit {
					scaleRow(ri, 1/at(i, i))
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
		// The rows still to solve less the block's share.
		r0, rows := i1, n-i1
		if upper {
			r0, rows = 0, i0
		}
		if rows == 0 {
			continue
		}
		sa := a
		sa.off += r0*a.rs + i0*a.cs
		if err := b.mulAdd(ctx, par, x.sub(r0, 0, rows, k), true, true, sa, x.sub(i0, 0, i1-i0, k).op(false), i1-i0); err != nil {
			return err
		}
	}
	return nil
}
