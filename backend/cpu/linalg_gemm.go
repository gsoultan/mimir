package cpu

import "context"

// mat is a row-major float32 matrix: element (i, j) is data[i*ld+j]. The
// factorizations work on matrices of their own in this form and keep
// their blocks as views of it.
type mat struct {
	data       []float32
	rows, cols int
	ld         int
}

// newMat returns a zeroed rows×cols matrix.
func newMat(rows, cols int) mat {
	return mat{data: make([]float32, rows*cols), rows: rows, cols: cols, ld: cols}
}

// sub returns the rows×cols block of a at (i, j), which shares its
// memory.
func (a mat) sub(i, j, rows, cols int) mat {
	if rows == 0 || cols == 0 {
		return mat{rows: rows, cols: cols, ld: a.ld}
	}
	return mat{data: a.data[i*a.ld+j : (i+rows-1)*a.ld+j+cols], rows: rows, cols: cols, ld: a.ld}
}

// row returns row i.
func (a mat) row(i int) []float32 { return a.data[i*a.ld:][:a.cols] }

// op returns a as a GEMM operand, or its transpose if t is set.
func (a mat) op(t bool) operand {
	if t {
		return operand{data: a.data, rs: 1, cs: a.ld}
	}
	return operand{data: a.data, rs: a.ld, cs: 1}
}

// linalgSmall is the most multiply-adds a product computes by plain
// loops: below it, packing for the GEMM costs more than it saves.
const linalgSmall = 1 << 13

// mulAdd sets c to c + s·x·y, or to s·x·y if acc is unset, where s is -1
// if neg is set and 1 otherwise, x is c.rows×k and y is k×c.cols. It
// multiplies with the GEMM micro-kernels, splitting the work between the
// workers if par is set; otherwise it runs on the calling goroutine.
func (b *Backend) mulAdd(ctx context.Context, par bool, c mat, neg, acc bool, x, y operand, k int) error {
	m, n := c.rows, c.cols
	if m == 0 || n == 0 {
		return nil
	}
	sign := float32(1)
	if neg {
		sign = -1
	}
	if k == 0 || m*n*k <= linalgSmall {
		for i := range m {
			row := c.row(i)
			for j := range row {
				var s float32
				for p := range k {
					s += x.data[x.off+i*x.rs+p*x.cs] * y.data[y.off+p*y.rs+j*y.cs]
				}
				if acc {
					row[j] += sign * s
				} else {
					row[j] = sign * s
				}
			}
		}
		return nil
	}
	g := gemmKernels[0]
	workers := 1
	if par {
		workers = b.workers
	}
	shared := gemmWorkspaces.Get().(*gemmWorkspace)
	defer gemmWorkspaces.Put(shared)
	for jc := 0; jc < n; jc += g.nc {
		nc := min(g.nc, n-jc)
		p := planGEMM(g, m, nc, k, workers)
		for pc := 0; pc < k; pc += g.kc {
			kc := min(g.kc, k-pc)
			shared.b = grow(shared.b, roundUp(nc, g.nr)*kc+packSlack)
			bp := shared.b
			panels := (nc + g.nr - 1) / g.nr
			err := b.maybeParallel(ctx, par, panels, packGrain/(kc*g.nr), func(lo, hi int) error {
				j0 := lo * g.nr
				packB(bp[j0*kc:hi*g.nr*kc], y, pc, kc, jc+j0, min(hi*g.nr, nc)-j0, g.nr)
				return nil
			})
			if err != nil {
				return err
			}
			accumulate := acc || pc > 0
			err = b.maybeParallel(ctx, par, p.mParts*p.nParts, 1, func(lo, hi int) error {
				ws := gemmWorkspaces.Get().(*gemmWorkspace)
				defer gemmWorkspaces.Put(ws)
				for u := lo; u < hi; u++ {
					i0, j0 := u/p.nParts*p.mStep, u%p.nParts*p.nStep
					i1, j1 := min(i0+p.mStep, m), min(j0+p.nStep, nc)
					for ic := i0; ic < i1; ic += g.mc {
						mc := min(g.mc, i1-ic)
						ws.a = grow(ws.a, roundUp(mc, g.mr)*kc+packSlack)
						packA(ws.a, x, ic, mc, pc, kc, g.mr)
						if neg {
							for i := range ws.a[:roundUp(mc, g.mr)*kc] {
								ws.a[i] = -ws.a[i]
							}
						}
						macroKernel(g, ws.a, bp[j0*kc:], kc, mc, j1-j0, c.data[ic*c.ld+jc+j0:], c.ld, accumulate, &ws.tile)
					}
				}
				return nil
			})
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// maybeParallel runs fn over [0, n) with every worker if par is set, and
// on the calling goroutine otherwise.
func (b *Backend) maybeParallel(ctx context.Context, par bool, n, grain int, fn func(lo, hi int) error) error {
	if !par {
		if n <= 0 {
			return nil
		}
		return fn(0, n)
	}
	return b.parallelFor(ctx, n, grain, fn)
}
