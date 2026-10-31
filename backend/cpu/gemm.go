package cpu

import (
	"context"
	"slices"
	"sync"
	"unsafe"
)

// Float32 matrix multiplication follows the BLIS design. Slices of A and B
// kc deep are packed into contiguous panels, mr rows of A and nr columns of
// B each, and a micro-kernel computes one mr×nr tile of the output per pair
// of panels while its accumulators stay in registers. Packing reads any
// strides, so transposed, broadcast and windowed inputs cost little more
// than contiguous ones.

// A gemmKernel computes one tile of C from packed panels.
type gemmKernel struct {
	name   string
	mr, nr int // tile shape
	// Cache blocking: a packed B panel of kc×nr stays in L1, a packed A
	// block of mc×kc in L2, and a packed B block of kc×nc in L3.
	kc, mc, nc int
	// tile sets the mr×nr tile at c, with rows ldc elements apart, to the
	// product of the kc-deep panels a (kc×mr) and b (kc×nr), adding it to
	// the tile's previous contents if accumulate is set. It must not
	// retain its pointers.
	tile func(kc int, a, b, c *float32, ldc int, accumulate bool)
}

// maxTile bounds mr*nr for every kernel, sizing the scratch tile used at
// the ragged edges of the output.
const maxTile = 16 * 32

// gemmKernels lists the kernels this CPU supports, fastest first. The
// architecture-specific files prepend to it at init.
var gemmKernels = []*gemmKernel{&genericKernel}

// genericKernel is the portable reference.
var genericKernel = gemmKernel{name: "generic", mr: 4, nr: 4, kc: 256, mc: 128, nc: 1024, tile: tileGeneric}

// operand is a strided matrix: element (i, j) is data[off+i*rs+j*cs].
type operand struct {
	data   []float32
	off    int
	rs, cs int
}

// gemmWorkspace holds a worker's packed panels and the tile it computes
// at the ragged edges of the output. The tile lives here because a local
// array passed to a kernel through a function value escapes to the heap.
type gemmWorkspace struct {
	a, b     []float32
	partials []float32 // sums of groups of products; see sgemmGrouped
	tile     [maxTile]float32
}

var gemmWorkspaces = sync.Pool{New: func() any { return new(gemmWorkspace) }}

// grow returns s with length n, reusing its storage when large enough.
func grow[E any](s []E, n int) []E {
	if cap(s) < n {
		return make([]E, n)
	}
	return s[:n]
}

// packSlack is the room past the last panel that lets packing use
// transposes throughout; see packTransposed.
const packSlack = 8

// transpose8x8, when the CPU supports it, sets dst[(8q+c)*ldd+r] to
// src[r*lds+8q+c] for r, c < 8 and q < blocks. It must not retain its
// pointers.
var transpose8x8 func(dst *float32, ldd int, src *float32, lds int, blocks int)

// minUnitFlops is the least work worth giving a worker of its own.
const minUnitFlops = 1 << 20

// packCost is the cost of packing an element of A, in multiply-adds.
const packCost = 8

// packGrain is the least number of elements worth packing in parallel.
const packGrain = 1 << 15

// maxGroups is the most groups that sgemm splits the products of an
// output into, and maxPartials the most elements of their sums that it
// keeps besides the output.
const (
	maxGroups   = 16
	maxPartials = 1 << 20
)

// sgemm computes the batched product out[o] = Σ a[i]·b[i] over the sums
// matrices i = o·sums, ..., o·sums+sums-1 of each output o, where matrix i
// of a starts at offset offA[i] and of b at offB[i], and out holds the m×n
// results contiguously. sums is 1 for a plain batched product. A non-nil
// epilogue rewrites each finished part of out.
//
// A batch whose items share a, and whose a is the larger operand, is one
// wide product when a block of its columns has work for every worker; see
// sgemmWide. Outputs that sum several products, and are small enough to
// keep partial sums of, sum groups of them in parallel; see sgemmGrouped.
// Otherwise a batch with an output for every worker computes its outputs
// in parallel, and smaller batches compute one output at a time with every
// worker; see sgemmShared. The order in which each output adds its
// products depends only on the shapes, and splitting an output's rows or
// columns between workers changes no element's sum, so the result does
// not depend on the number of workers.
func (be *Backend) sgemm(ctx context.Context, g *gemmKernel, out []float32, a operand, offA []int, b operand, offB []int, m, n, k, sums int, epi *evaluator) error {
	outputs := len(out) / max(m*n, 1)
	if sums == 0 || m*n == 0 {
		clear(out)
		return epi.apply(0, len(out))
	}
	terms := func(o int) ([]int, []int) { return offA[o*sums:][:sums], offB[o*sums:][:sums] }
	flops := max(2*m*n*k*sums, 1)
	groups := min(sums, maxGroups)
	if (groups-1)*len(out) > maxPartials {
		groups = 1
	}
	shared := !slices.ContainsFunc(offA, func(off int) bool { return off != offA[0] })
	switch {
	case sums == 1 && outputs > 1 && m >= n && shared && wideEnough(g, m, n, k, outputs, be.workers):
		a.off = offA[0]
		return be.sgemmWide(ctx, g, out, a, b, offB, m, n, k, epi)
	case groups > 1 || sums > 1 && outputs < be.workers:
		// Summing in groups, or, with too little memory for partial sums,
		// in one group whose parts compute every product independently:
		// the alternative, sgemmShared, waits for every worker after each
		// product, and so for the slowest one, many times.
		return be.sgemmGrouped(ctx, g, out, a, offA, b, offB, m, n, k, sums, groups, epi)
	case outputs >= be.workers || flops < 4*minUnitFlops:
		return be.parallelFor(ctx, outputs, max(1, minUnitFlops/flops), func(lo, hi int) error {
			ws := gemmWorkspaces.Get().(*gemmWorkspace)
			defer gemmWorkspaces.Put(ws)
			for o := lo; o < hi; o++ {
				ta, tb := terms(o)
				gemmBlock(g, ws, a, ta, b, tb, k, out[o*m*n:][:m*n], n, 0, m, 0, n)
				if err := epi.apply(o*m*n, (o+1)*m*n); err != nil {
					return err
				}
			}
			return nil
		})
	}
	if groups := deepGroups(g, m, n, k, outputs, be.workers); groups > 1 {
		return be.sgemmDeep(ctx, g, out, a, offA, b, offB, m, n, k, groups, epi)
	}
	for o := range outputs {
		ta, tb := terms(o)
		if err := be.sgemmShared(ctx, g, a, ta, b, tb, out[o*m*n:][:m*n], m, n, k, epi, o*m*n); err != nil {
			return err
		}
	}
	return nil
}

// sgemmGrouped computes out like sgemm, splitting the sums products of
// each output into groups of consecutive ones, or keeping them in one. It
// sums the groups of every output in parallel, the first into out and the
// others into partial sums, and then adds the partial sums to out in
// order. When there are fewer groups than workers it splits each group's
// output too; see groupWorkers. Each part packs the operands it needs.
func (be *Backend) sgemmGrouped(ctx context.Context, g *gemmKernel, out []float32, a operand, offA []int, b operand, offB []int, m, n, k, sums, groups int, epi *evaluator) error {
	size := m * n
	outputs := len(out) / size
	shared := gemmWorkspaces.Get().(*gemmWorkspace)
	defer gemmWorkspaces.Put(shared)
	shared.partials = grow(shared.partials, (groups-1)*len(out))
	partials := shared.partials
	depth := k * (sums / groups) // of a group's products, at least
	p := planParts(g, m, n, depth, groupWorkers(outputs*groups, be.workers))
	parts := p.mParts * p.nParts
	flops := max(2*size*depth/parts, 1)
	err := be.parallelFor(ctx, outputs*groups*parts, max(1, minUnitFlops/flops), func(lo, hi int) error {
		ws := gemmWorkspaces.Get().(*gemmWorkspace)
		defer gemmWorkspaces.Put(ws)
		for u := lo; u < hi; u++ {
			o, j, part := u/parts/groups, u/parts%groups, u%parts
			t0, t1 := o*sums+j*sums/groups, o*sums+(j+1)*sums/groups
			i0, j0 := part/p.nParts*p.mStep, part%p.nParts*p.nStep
			c := out[o*size:][:size]
			if j > 0 {
				c = partials[(j-1)*len(out)+o*size:][:size]
			}
			gemmBlock(g, ws, a, offA[t0:t1], b, offB[t0:t1], k, c, n, i0, min(i0+p.mStep, m), j0, min(j0+p.nStep, n))
		}
		return nil
	})
	if err != nil {
		return err
	}
	return be.parallelFor(ctx, len(out), elementGrain, func(lo, hi int) error {
		c := out[lo:hi]
		for j := range groups - 1 {
			for i, v := range partials[j*len(out)+lo:][:len(c)] {
				c[i] += v
			}
		}
		return epi.apply(lo, hi)
	})
}

// wideEnough reports whether a block of B of the wide product of a batch
// of outputs items, m×n and k deep, has work for every worker; see
// sgemmWide. The kernels compute the padding of each item's last panel
// too, so it counts as work.
func wideEnough(g *gemmKernel, m, n, k, outputs, workers int) bool {
	cols := min(g.nc/g.nr, outputs*((n+g.nr-1)/g.nr)) * g.nr
	return 2*m*cols*k >= workers*minUnitFlops
}

// groupWorkers returns the number of workers to split the output of each
// of groups groups of products between, so that the parts of all of them
// can be shared evenly by workers workers.
func groupWorkers(groups, workers int) int {
	if groups >= workers {
		return 1
	}
	a, b := groups, workers
	for b != 0 {
		a, b = b, a%b
	}
	return workers / a
}

// sgemmShared computes the m×n product c = Σ a[i]·b[i], where matrix i of
// a starts at offA[i] and of b at offB[i], with every worker: each
// computes parts of the output, packing the rows of A and the columns of
// B that its part needs. Packing B once for every part and reading the
// panels from the cores that packed them was measured slower on a VM
// where cores share cache lines slowly, even where it packed less; see
// planParts for how the parts limit the repeated packing. The epilogue
// rewrites each part once it is done; c starts at element base of the
// epilogue's output.
func (be *Backend) sgemmShared(ctx context.Context, g *gemmKernel, a operand, offA []int, b operand, offB []int, c []float32, m, n, k int, epi *evaluator, base int) error {
	if k == 0 {
		clear(c)
		return epi.apply(base, base+m*n)
	}
	p := planParts(g, m, n, k, be.workers)
	return be.parallelFor(ctx, p.mParts*p.nParts, 1, func(lo, hi int) error {
		ws := gemmWorkspaces.Get().(*gemmWorkspace)
		defer gemmWorkspaces.Put(ws)
		for u := lo; u < hi; u++ {
			i0, j0 := u/p.nParts*p.mStep, u%p.nParts*p.nStep
			i1, j1 := min(i0+p.mStep, m), min(j0+p.nStep, n)
			gemmBlock(g, ws, a, offA, b, offB, k, c, n, i0, i1, j0, j1)
			if j0 == 0 && j1 == n {
				// Whole rows: one contiguous range, which the epilogue
				// runs in wide tiles.
				if err := epi.apply(base+i0*n, base+i1*n); err != nil {
					return err
				}
				continue
			}
			for i := i0; i < i1; i++ {
				if err := epi.apply(base+i*n+j0, base+i*n+j1); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

// sgemmDeep computes each output's product like sgemmShared, with the
// sum over k split into groups slices computed in parallel, the first
// into out and the others into partial sums that are then added to out
// in order: the way to share a product of few rows and columns summed
// over many, such as a dense layer's over a batch, which parts of the
// output would share only by packing the same operands again and again.
// groups depends on the shapes alone (see deepGroups), so the result does
// not depend on the number of workers.
func (be *Backend) sgemmDeep(ctx context.Context, g *gemmKernel, out []float32, a operand, offA []int, b operand, offB []int, m, n, k, groups int, epi *evaluator) error {
	size := m * n
	outputs := len(out) / size
	shared := gemmWorkspaces.Get().(*gemmWorkspace)
	defer gemmWorkspaces.Put(shared)
	shared.partials = grow(shared.partials, (groups-1)*len(out))
	partials := shared.partials
	slice := (k + groups - 1) / groups
	err := be.parallelFor(ctx, outputs*groups, 1, func(lo, hi int) error {
		ws := gemmWorkspaces.Get().(*gemmWorkspace)
		defer gemmWorkspaces.Put(ws)
		for u := lo; u < hi; u++ {
			o, j := u/groups, u%groups
			k0 := j * slice
			c := out[o*size:][:size]
			if j > 0 {
				c = partials[(j-1)*len(out)+o*size:][:size]
			}
			gemmBlock(g, ws, a, []int{offA[o] + k0*a.cs}, b, []int{offB[o] + k0*b.rs}, min(slice, k-k0), c, n, 0, m, 0, n)
		}
		return nil
	})
	if err != nil {
		return err
	}
	return be.parallelFor(ctx, len(out), elementGrain, func(lo, hi int) error {
		c := out[lo:hi]
		for j := range groups - 1 {
			addVec(c, partials[j*len(out)+lo:][:len(c)])
		}
		return epi.apply(lo, hi)
	})
}

// addVec adds the elements of src to those of dst.
func addVec(dst, src []float32) {
	src = src[:len(dst)]
	if rowVector != nil {
		if n := vectorPart(len(dst)); n > 0 {
			rowVector.addVV(&dst[0], &dst[0], &src[0], n)
			dst, src = dst[n:], src[n:]
		}
	}
	for i, v := range src {
		dst[i] += v
	}
}

// reduceCost is the cost of summing one element of a partial sum into
// the output, in multiply-adds: the sums stream through memory.
const reduceCost = 16

// deepGroups returns the number of slices that sgemmDeep splits the sum
// over k of an m×n product into, or 1 where the product is better shared
// as parts of its output: the largest power of two of slices g.kc deep at
// least, up to maxGroups, whose partial sums fit maxPartials, when that
// costs the busiest of workers workers less than parts of the output
// would (see planCost). It depends on the shapes alone, besides workers,
// so that the result of a product does not depend on workers.
func deepGroups(g *gemmKernel, m, n, k, outputs, workers int) int {
	groups := 1
	for 2*groups <= min(maxGroups, k/g.kc) && (2*groups-1)*outputs*m*n <= maxPartials {
		groups *= 2
	}
	if groups == 1 {
		return 1
	}
	_, parts := planCost(g, m, n, k, workers, true)
	slice := (k + groups - 1) / groups
	rounds := (outputs*groups + workers - 1) / workers
	deep := rounds*(m*n*slice+(m+n)*slice*packCost) + (groups-1)*m*n*reduceCost
	if outputs*deep < parts {
		return groups
	}
	return 1
}

// sgemmWide computes out[i] = a·b[i] for every matrix i of b, which starts
// at offB[i], with every worker. Side by side, the products are one wide
// product a·[b[0] b[1] ...], whose columns it computes like sgemmShared:
// a block of A, packed once, then serves every item instead of being
// packed again for each. Each item's columns start a panel of their own.
// k must be positive.
func (be *Backend) sgemmWide(ctx context.Context, g *gemmKernel, out []float32, a, b operand, offB []int, m, n, k int, epi *evaluator) error {
	per := (n + g.nr - 1) / g.nr // panels of each item
	total := len(offB) * per
	block := max(1, g.nc/g.nr) // panels of a block of B
	// spans calls fn for each item's run of the panels [q0, q1): item i's
	// columns [j0, j0+cols), packed from panel q on.
	spans := func(q0, q1 int, fn func(q, i, j0, cols int)) {
		for q := q0; q < q1; {
			i, j0 := q/per, q%per*g.nr
			end := min(q1, (i+1)*per)
			fn(q, i, j0, min(n, (end-i*per)*g.nr)-j0)
			q = end
		}
	}
	shared := gemmWorkspaces.Get().(*gemmWorkspace)
	defer gemmWorkspaces.Put(shared)
	for q0 := 0; q0 < total; q0 += block {
		q1 := min(q0+block, total)
		p := spreadRows(g, planGEMM(g, m, (q1-q0)*g.nr, k, be.workers), m, min(k, g.kc), be.workers)
		for pc := 0; pc < k; pc += g.kc {
			kc := min(g.kc, k-pc)
			shared.b = grow(shared.b, (q1-q0)*g.nr*kc+packSlack)
			bp := shared.b
			err := be.parallelFor(ctx, q1-q0, packGrain/(kc*g.nr), func(lo, hi int) error {
				// Bound the panels so that packing stores nothing into those
				// of other chunks.
				end := hi * g.nr * kc
				spans(q0+lo, q0+hi, func(q, i, j0, cols int) {
					bi := b
					bi.off = offB[i]
					packB(bp[(q-q0)*g.nr*kc:end], bi, pc, kc, j0, cols, g.nr)
				})
				return nil
			})
			if err != nil {
				return err
			}
			err = be.parallelFor(ctx, p.mParts*p.nParts, 1, func(lo, hi int) error {
				ws := gemmWorkspaces.Get().(*gemmWorkspace)
				defer gemmWorkspaces.Put(ws)
				for u := lo; u < hi; u++ {
					i0, s0 := u/p.nParts*p.mStep, u%p.nParts*p.nStep/g.nr
					i1, s1 := min(i0+p.mStep, m), min(s0+p.nStep/g.nr, q1-q0)
					for ic := i0; ic < i1; ic += g.mc {
						mc := min(g.mc, i1-ic)
						ws.a = grow(ws.a, roundUp(mc, g.mr)*kc+packSlack)
						packA(ws.a, a, ic, mc, pc, kc, g.mr)
						spans(q0+s0, q0+s1, func(q, i, j0, cols int) {
							c := out[i*m*n:][:m*n]
							macroKernel(g, ws.a, bp[(q-q0)*g.nr*kc:], kc, mc, cols, c[ic*n+j0:], n, pc > 0, &ws.tile)
						})
					}
					if pc+kc < k {
						continue
					}
					var err error
					spans(q0+s0, q0+s1, func(q, i, j0, cols int) {
						if j0 == 0 && cols == n {
							// Whole rows of an item: one contiguous range.
							if err == nil {
								err = epi.apply(i*m*n+i0*n, i*m*n+i1*n)
							}
							return
						}
						for r := i0; r < i1 && err == nil; r++ {
							row := i*m*n + r*n + j0
							err = epi.apply(row, row+cols)
						}
					})
					if err != nil {
						return err
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

type gemmPlan struct {
	mStep, nStep   int
	mParts, nParts int
}

// planGEMM splits an m×n output between workers, only as far as needed to
// give each a share. Every part packs the rows of A it needs, so splitting
// the columns repeats that packing. The plan minimizes the work of the
// busiest worker, counting a packed element of A as packCost multiply-adds.
func planGEMM(g *gemmKernel, m, n, k, workers int) gemmPlan {
	return planSplit(g, m, n, k, workers, false)
}

// spreadParts is how many parts per worker spreadRows aims for.
const spreadParts = 4

// spreadFlops is the fewest multiply-adds, times two, that spreadRows
// leaves in a part: enough that claiming it costs little.
const spreadFlops = 32 * minUnitFlops

// spreadRows splits the rows of the parts of plan p, of an m-row output
// whose B panels are packed once for all parts, kc deep, into about
// spreadParts parts per worker. Rows cost no extra packing to split, as
// each part packs the rows of A it needs anyway, and the finer parts
// balance the work: a worker that finishes early, or that other
// processes on the machine leave more time, takes more of them instead of
// waiting for the slowest.
func spreadRows(g *gemmKernel, p gemmPlan, m, kc, workers int) gemmPlan {
	parts := p.mParts * p.nParts
	if parts < 2 {
		return p
	}
	want := (spreadParts*workers + p.nParts - 1) / p.nParts
	least := roundUp(max(1, spreadFlops/max(2*p.nStep*kc, 1)), g.mr)
	step := max(roundUp((m+want-1)/want, g.mr), least)
	if step >= p.mStep {
		return p
	}
	p.mStep, p.mParts = step, (m+step-1)/step
	return p
}

// planParts is planGEMM for parts that each pack the columns of B they
// need too, so that splitting the rows repeats that packing as well.
func planParts(g *gemmKernel, m, n, k, workers int) gemmPlan {
	return planSplit(g, m, n, k, workers, true)
}

func planSplit(g *gemmKernel, m, n, k, workers int, packsB bool) gemmPlan {
	p, _ := planCost(g, m, n, k, workers, packsB)
	return p
}

// planCost is planSplit with the cost of the plan for the busiest worker,
// in multiply-adds, counting a packed element as packCost of them.
func planCost(g *gemmKernel, m, n, k, workers int, packsB bool) (gemmPlan, int) {
	p := gemmPlan{mStep: max(m, 1), nStep: max(n, 1), mParts: 1, nParts: 1}
	mt, nt := (m+g.mr-1)/g.mr, (n+g.nr-1)/g.nr
	whole := mt * g.mr * (nt*g.nr + packCost)
	if packsB {
		whole += nt * g.nr * packCost
	}
	want := min(workers, 2*m*n*max(k, 1)/minUnitFlops)
	if want <= 1 {
		return p, whole * max(k, 1)
	}
	best := -1
	for pn := 1; pn <= min(want, nt); pn++ {
		pm := min((want+pn-1)/pn, mt)
		mTiles, nTiles := (mt+pm-1)/pm, (nt+pn-1)/pn
		rounds := (pm*pn + workers - 1) / workers
		// In multiply-adds per element of k, for rows of A packed and
		// columns of B packed by each part.
		cost := rounds * mTiles * g.mr * (nTiles*g.nr + packCost)
		if packsB {
			cost += rounds * nTiles * g.nr * packCost
		}
		if best < 0 || cost < best {
			best = cost
			p.mStep, p.nStep = mTiles*g.mr, nTiles*g.nr
		}
	}
	p.mParts, p.nParts = (m+p.mStep-1)/p.mStep, (n+p.nStep-1)/p.nStep
	return p, best * max(k, 1)
}

// gemmBlock computes rows [i0, i1) and columns [j0, j1) of c = Σ a[i]·b[i],
// where matrix i of a starts at offA[i] and of b at offB[i], a's have k
// columns and c is row-major with rows ldc elements apart.
func gemmBlock(g *gemmKernel, ws *gemmWorkspace, a operand, offA []int, b operand, offB []int, k int, c []float32, ldc, i0, i1, j0, j1 int) {
	if k == 0 || len(offA) == 0 {
		for i := i0; i < i1; i++ {
			clear(c[i*ldc+j0 : i*ldc+j1])
		}
		return
	}
	for jc := j0; jc < j1; jc += g.nc {
		nc := min(g.nc, j1-jc)
		for t := range offA {
			at, bt := a, b
			at.off, bt.off = offA[t], offB[t]
			for pc := 0; pc < k; pc += g.kc {
				kc := min(g.kc, k-pc)
				ws.b = grow(ws.b, roundUp(nc, g.nr)*kc+packSlack)
				packB(ws.b, bt, pc, kc, jc, nc, g.nr)
				for ic := i0; ic < i1; ic += g.mc {
					mc := min(g.mc, i1-ic)
					ws.a = grow(ws.a, roundUp(mc, g.mr)*kc+packSlack)
					packA(ws.a, at, ic, mc, pc, kc, g.mr)
					macroKernel(g, ws.a, ws.b, kc, mc, nc, c[ic*ldc+jc:], ldc, t > 0 || pc > 0, &ws.tile)
				}
			}
		}
	}
}

// macroKernel runs the micro-kernel over every tile of an mc×nc block of
// c, kc deep, from the packed panels of a, mc rows, and b, nc columns. c
// starts at the block's first element. Edge tiles go through scratch.
func macroKernel(g *gemmKernel, a, b []float32, kc, mc, nc int, c []float32, ldc int, accumulate bool, scratch *[maxTile]float32) {
	mr, nr := g.mr, g.nr
	for jr := 0; jr < nc; jr += nr {
		cols := min(nr, nc-jr)
		bp := &b[jr*kc]
		for ir := 0; ir < mc; ir += mr {
			rows := min(mr, mc-ir)
			ap := &a[ir*kc]
			if rows == mr && cols == nr {
				// The full tile lies inside c: its last element is
				// (ir+mr-1)*ldc + jr+nr-1 < len(c).
				_ = c[(ir+mr-1)*ldc+jr+nr-1]
				g.tile(kc, ap, bp, &c[ir*ldc+jr], ldc, accumulate)
				continue
			}
			g.tile(kc, ap, bp, &scratch[0], nr, false)
			for i := range rows {
				dst := c[(ir+i)*ldc+jr:][:cols]
				src := scratch[i*nr:][:cols]
				if accumulate {
					for j, v := range src {
						dst[j] += v
					}
				} else {
					copy(dst, src)
				}
			}
		}
	}
}

// packA copies rows [i0, i0+m) and columns [p0, p0+kc) of a into dst as
// panels of mr rows, each stored column by column (kc×mr), padding the
// last panel with zeros. It may store garbage into up to packSlack
// elements of dst past the panels, and runs faster when they exist.
func packA(dst []float32, a operand, i0, m, p0, kc, mr int) {
	for ir := 0; ir < m; ir += mr {
		rows := min(mr, m-ir)
		panel := dst[ir*kc:][:kc*mr]
		base := a.off + (i0+ir)*a.rs + p0*a.cs
		switch {
		case a.cs == 1: // rows of a are contiguous
			done := packTransposed(dst[ir*kc:], mr, a.data, base, a.rs, rows, kc)
			for i := range rows {
				src := a.data[base+i*a.rs:][done:kc]
				for p, v := range src {
					panel[(done+p)*mr+i] = v
				}
			}
		case a.rs == 1: // columns of a are contiguous
			for p := range kc {
				copy(panel[p*mr:][:rows], a.data[base+p*a.cs:][:rows])
			}
		default:
			for p := range kc {
				off := base + p*a.cs
				for i := range rows {
					panel[p*mr+i] = a.data[off+i*a.rs]
				}
			}
		}
		if rows < mr {
			for p := range kc {
				clear(panel[p*mr+rows : p*mr+mr])
			}
		}
	}
}

// packB copies rows [p0, p0+kc) and columns [j0, j0+n) of b into dst as
// panels of nr columns, each stored row by row (kc×nr), padding the last
// panel with zeros. It may store garbage into up to packSlack elements of
// dst past the panels, and runs faster when they exist.
func packB(dst []float32, b operand, p0, kc, j0, n, nr int) {
	for jr := 0; jr < n; jr += nr {
		cols := min(nr, n-jr)
		panel := dst[jr*kc:][:kc*nr]
		base := b.off + p0*b.rs + (j0+jr)*b.cs
		switch {
		case b.cs == 1: // rows of b are contiguous
			for p := range kc {
				copy(panel[p*nr:][:cols], b.data[base+p*b.rs:][:cols])
			}
		case b.rs == 1: // columns of b are contiguous
			done := packTransposed(dst[jr*kc:], nr, b.data, base, b.cs, cols, kc)
			for j := range cols {
				src := b.data[base+j*b.cs:][done:kc]
				for p, v := range src {
					panel[(done+p)*nr+j] = v
				}
			}
		default:
			for p := range kc {
				off := base + p*b.rs
				for j := range cols {
					panel[p*nr+j] = b.data[off+j*b.cs]
				}
			}
		}
		if cols < nr {
			for p := range kc {
				clear(panel[p*nr+cols : p*nr+nr])
			}
		}
	}
}

// packTransposed sets panel[p*ld+l] = data[base+l*stride+p] for lines
// l < lines ≤ ld and p < done with 8×8 transposes, and returns done, the
// largest multiple of 8 no greater than kc, or 0 if it cannot transpose.
//
// The transposes move 8 lines at a time. In a final group of fewer than 8
// lines they read the lines that follow, which must exist, and store
// garbage past the last line of each row: into the row's padding, which
// the caller clears, and, in rows narrower than the group, into the first
// elements of the next row. Packing that group first lets the stores of
// the next row overwrite the garbage. The last row of the panel can spill
// past it, into the next panel, which is packed later, or into packSlack;
// packTransposed stores nothing past the end of panel.
func packTransposed(panel []float32, ld int, data []float32, base, stride, lines, kc int) int {
	blocks := kc / 8
	if transpose8x8 == nil || blocks == 0 || stride < 0 {
		return 0
	}
	groups := roundUp(lines, 8)
	// Bounds of the lines read and the elements stored.
	if base+(groups-1)*stride+8*blocks > len(data) || (8*blocks-1)*ld+groups > len(panel) {
		return 0
	}
	if g := lines &^ 7; g < lines {
		transpose8x8(&panel[g], ld, &data[base+g*stride], stride, blocks)
	}
	for g := 0; g+8 <= lines; g += 8 {
		transpose8x8(&panel[g], ld, &data[base+g*stride], stride, blocks)
	}
	return 8 * blocks
}

// tileGeneric is the portable 4×4 micro-kernel.
func tileGeneric(kc int, a, b, c *float32, ldc int, accumulate bool) {
	const mr, nr = 4, 4
	ap := unsafe.Slice(a, kc*mr)
	bp := unsafe.Slice(b, kc*nr)
	var c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33 float32
	for p := range kc {
		av := ap[p*mr : p*mr+mr : p*mr+mr]
		bv := bp[p*nr : p*nr+nr : p*nr+nr]
		a0, a1, a2, a3 := av[0], av[1], av[2], av[3]
		b0, b1, b2, b3 := bv[0], bv[1], bv[2], bv[3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}
	cs := unsafe.Slice(c, 3*ldc+nr)
	rows := [mr][nr]float32{
		{c00, c01, c02, c03},
		{c10, c11, c12, c13},
		{c20, c21, c22, c23},
		{c30, c31, c32, c33},
	}
	for i, row := range rows {
		dst := cs[i*ldc : i*ldc+nr : i*ldc+nr]
		if accumulate {
			for j, v := range row {
				dst[j] += v
			}
		} else {
			copy(dst, row[:])
		}
	}
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }
