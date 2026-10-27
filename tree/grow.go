package tree

import (
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
)

// A grower grows one tree from gradients, the way XGBoost does: each row i
// has a gradient g and a hessian h for each of k outputs, a leaf answers
// -G/(H+lambda) per output, from the sums G and H of its rows, and a split
// gains G²/(H+lambda) summed over its two sides less the same for the
// node it splits. With g = -y and h = 1 a leaf is the mean answer and the
// gain is the fall in squared error, as in a regression tree; with y a
// row's class one-hot, the fall in Gini impurity, as in a classification
// tree.
type grower struct {
	b       *binned
	k       int
	g, h    []float32 // n×k, row by row
	feats   []int     // the features the tree may split on
	perNode int       // features to draw at each node, or 0 for all of feats
	rng     *rand.Rand
	workers int
	cfg     *config
	gain    []float64 // the gain of the splits on each feature
	t       Tree
	free    [][]float64
	// leafOf, when not nil, is set to the leaf each grown row ends in.
	leafOf []int32
	og, oh []float32 // the gradients and hessians of a node's rows, in order
	tmp    []int32
}

// stride is the length of one bin's sums: G and H for each output, then
// the row count.
func (gr *grower) stride() int { return 2*gr.k + 1 }

type split struct {
	gain     float64
	feature  int
	bin      int // rows in bins 1..bin go left
	missLeft bool
	left     []float64 // the left side's sums of G and H
}

// grow grows a tree from rows, which it reorders.
func (gr *grower) grow(rows []int32) Tree {
	gr.t = Tree{}
	k := gr.k
	sums := make([]float64, 2*k)
	for _, r := range rows {
		for j := range k {
			sums[j] += float64(gr.g[int(r)*k+j])
			sums[k+j] += float64(gr.h[int(r)*k+j])
		}
	}
	gr.node(rows, nil, sums, 0)
	return gr.t
}

// node grows the subtree of rows at depth, whose sums of G and H are
// sums, and returns its root. hist, when not nil, is rows' histogram over
// gr.feats, which node takes over.
func (gr *grower) node(rows []int32, hist, sums []float64, depth int) int32 {
	k := gr.k
	id := int32(len(gr.t.Feature))
	gr.t.Feature = append(gr.t.Feature, -1)
	gr.t.Threshold = append(gr.t.Threshold, 0)
	gr.t.MissingLeft = append(gr.t.MissingLeft, false)
	gr.t.Left = append(gr.t.Left, -1)
	gr.t.Right = append(gr.t.Right, -1)
	for j := range k {
		gr.t.Value = append(gr.t.Value, float32(-sums[j]/(sums[k+j]+gr.cfg.l2)))
	}
	cfg := gr.cfg
	if depth >= cfg.maxDepth || len(rows) < 2*cfg.minLeaf {
		gr.release(hist)
		gr.leaf(rows, id)
		return id
	}
	feats := gr.feats
	if gr.perNode > 0 && gr.perNode < len(feats) {
		feats = slices.Clone(feats)
		gr.rng.Shuffle(len(feats), func(a, b int) { feats[a], feats[b] = feats[b], feats[a] })
		feats = feats[:gr.perNode]
		gr.release(hist)
		hist = nil
	}
	if hist == nil {
		hist = gr.histogram(rows, feats)
	}
	best := gr.best(hist, feats, sums, len(rows))
	if best.gain <= cfg.minGain {
		gr.release(hist)
		gr.leaf(rows, id)
		return id
	}
	lo := gr.partition(rows, best)
	left, right := rows[:lo], rows[lo:]
	gr.gain[best.feature] += best.gain
	gr.t.Feature[id] = int32(best.feature)
	gr.t.Threshold[id] = gr.b.edges[best.feature][best.bin-1]
	gr.t.MissingLeft[id] = best.missLeft
	var hl, hr []float64
	if gr.perNode == 0 {
		// The larger side's histogram is the parent's less the smaller's.
		small := left
		if len(right) < len(left) {
			small = right
		}
		hs := gr.histogram(small, feats)
		for i, v := range hs {
			hist[i] -= v
		}
		hl, hr = hs, hist
		if len(right) < len(left) {
			hl, hr = hist, hs
		}
	} else {
		gr.release(hist)
	}
	rs := make([]float64, 2*k)
	for i := range rs {
		rs[i] = sums[i] - best.left[i]
	}
	l := gr.node(left, hl, best.left, depth+1)
	r := gr.node(right, hr, rs, depth+1)
	gr.t.Left[id], gr.t.Right[id] = l, r
	return id
}

// histogram sums each row's gradients, hessians and count into its bin of
// every feature in feats.
func (gr *grower) histogram(rows []int32, feats []int) []float64 {
	k := gr.k
	// Gather the rows' gradients once, so they are read in order.
	gr.og, gr.oh = slices.Grow(gr.og[:0], len(rows)*k)[:len(rows)*k], slices.Grow(gr.oh[:0], len(rows)*k)[:len(rows)*k]
	parallel(len(rows), gr.workers, 16384, func(lo, hi int) {
		for i, r := range rows[lo:hi] {
			copy(gr.og[(lo+i)*k:], gr.g[int(r)*k:int(r)*k+k])
			copy(gr.oh[(lo+i)*k:], gr.h[int(r)*k:int(r)*k+k])
		}
	})
	const grain = 8192
	work := min(max(gr.workers, 1), max(len(rows)/grain, 1))
	if gr.workers <= 0 {
		work = min(runtime.GOMAXPROCS(0), max(len(rows)/grain, 1))
	}
	hist := gr.alloc()
	if work == 1 {
		// Too few rows to split among goroutines: split the features
		// among them instead, once there are enough rows to pay for it.
		w := gr.workers
		if len(rows) < 1024 {
			w = 1
		}
		parallel(len(feats), w, 4, func(lo, hi int) {
			gr.accumulate(hist, rows, gr.og, gr.oh, feats[lo:hi])
		})
		return hist
	}
	// Each goroutine sums a run of rows into a histogram of its own.
	parts := make([][]float64, work)
	parts[0] = hist
	for w := 1; w < work; w++ {
		parts[w] = gr.alloc()
	}
	step := (len(rows) + work - 1) / work
	var wg sync.WaitGroup
	for w := range work {
		lo, hi := min(w*step, len(rows)), min((w+1)*step, len(rows))
		wg.Go(func() { gr.accumulate(parts[w], rows[lo:hi], gr.og[lo*k:hi*k], gr.oh[lo*k:hi*k], feats) })
	}
	wg.Wait()
	for _, p := range parts[1:] {
		for i, v := range p {
			hist[i] += v
		}
		gr.release(p)
	}
	return hist
}

// accumulate adds rows, whose gradients and hessians are og and oh, into
// hist for the features in feats.
func (gr *grower) accumulate(hist []float64, rows []int32, og, oh []float32, feats []int) {
	b, k, s := gr.b, gr.k, gr.stride()
	f, off := b.f, b.off
	all := len(feats) == f // then feats[j] is j
	if k == 1 {
		oh = oh[:len(og)]
		for i, r := range rows {
			row := b.bins[int(r)*f : int(r)*f+f]
			g, h := float64(og[i]), float64(oh[i])
			if all {
				off := off[:len(row)]
				for j, bn := range row {
					e := (off[j] + int(bn)) * 3
					c := hist[e : e+3 : e+3]
					c[0] += g
					c[1] += h
					c[2]++
				}
				continue
			}
			for _, j := range feats {
				e := (off[j] + int(row[j])) * 3
				hist[e] += g
				hist[e+1] += h
				hist[e+2]++
			}
		}
		return
	}
	for i, r := range rows {
		row := b.bins[int(r)*f : int(r)*f+f]
		g, h := og[i*k:i*k+k], oh[i*k:i*k+k]
		for _, j := range feats {
			e := (off[j] + int(row[j])) * s
			for o := range k {
				hist[e+o] += float64(g[o])
				hist[e+k+o] += float64(h[o])
			}
			hist[e+2*k]++
		}
	}
}

// best finds the split of most gain on any feature of feats, given the
// node's histogram, its sums of G and H and its row count.
func (gr *grower) best(hist []float64, feats []int, sums []float64, count int) split {
	k, s, cfg := gr.k, gr.stride(), gr.cfg
	score := func(v []float64) float64 {
		var sc float64
		for j := range k {
			sc += v[j] * v[j] / (v[k+j] + cfg.l2)
		}
		return sc
	}
	weight := func(v []float64) float64 {
		var w float64
		for j := range k {
			w += v[k+j]
		}
		return w
	}
	total := make([]float64, s)
	copy(total, sums)
	total[2*k] = float64(count)
	parent := score(total)
	results := make([]split, len(feats))
	work := gr.workers
	if count < 1024 {
		work = 1
	}
	parallel(len(feats), work, 4, func(lo, hi int) {
		cum, left, right := make([]float64, s), make([]float64, s), make([]float64, s)
		for fi := lo; fi < hi; fi++ {
			j := feats[fi]
			base := gr.b.off[j] * s
			if k == 1 {
				results[fi] = gr.best1(hist[base:base+gr.b.nb[j]*3], j, total, parent)
				continue
			}
			miss := hist[base : base+s]
			clear(cum)
			best := split{feature: j}
			try := func(bin int, missLeft bool) {
				for i := range s {
					left[i] = cum[i]
					if missLeft {
						left[i] += miss[i]
					}
					right[i] = total[i] - left[i]
				}
				if left[2*k] < float64(cfg.minLeaf) || right[2*k] < float64(cfg.minLeaf) ||
					weight(left) < cfg.minWeight || weight(right) < cfg.minWeight {
					return
				}
				if g := score(left) + score(right) - parent; g > best.gain {
					best = split{gain: g, feature: j, bin: bin, missLeft: missLeft, left: slices.Clone(left[:2*k])}
				}
			}
			for bin := 1; bin < gr.b.nb[j]-1; bin++ {
				e := hist[base+bin*s : base+bin*s+s]
				if e[2*k] == 0 {
					continue
				}
				for i := range s {
					cum[i] += e[i]
				}
				if miss[2*k] > 0 {
					try(bin, true)
					try(bin, false)
				} else {
					// No row of this node is missing the feature: send
					// missing values with the larger side.
					try(bin, cum[2*k] >= total[2*k]-cum[2*k])
				}
			}
			results[fi] = best
		}
	})
	var best split
	for _, r := range results {
		if r.gain > best.gain {
			best = r
		}
	}
	return best
}

// partition reorders rows so those that sp sends left come first, and
// returns how many do.
func (gr *grower) partition(rows []int32, sp split) int {
	bins, f := gr.b.bins[sp.feature:], gr.b.f
	goesLeft := func(r int32) bool {
		bn := bins[int(r)*f]
		return (bn == 0 && sp.missLeft) || (bn != 0 && int(bn) <= sp.bin)
	}
	const grain = 32768
	if len(rows) < 2*grain || gr.workers == 1 {
		lo, hi := 0, len(rows)
		for lo < hi {
			if goesLeft(rows[lo]) {
				lo++
			} else {
				hi--
				rows[lo], rows[hi] = rows[hi], rows[lo]
			}
		}
		return lo
	}
	// Each goroutine counts, then copies, its run of rows to the left
	// and right of where the runs before it end.
	work := gr.workers
	if work <= 0 {
		work = runtime.GOMAXPROCS(0)
	}
	work = min(work, len(rows)/grain)
	step := (len(rows) + work - 1) / work
	lefts := make([]int, work)
	var wg sync.WaitGroup
	for w := range work {
		wg.Go(func() {
			for _, r := range rows[min(w*step, len(rows)):min((w+1)*step, len(rows))] {
				if goesLeft(r) {
					lefts[w]++
				}
			}
		})
	}
	wg.Wait()
	total := 0
	for _, l := range lefts {
		total += l
	}
	gr.tmp = slices.Grow(gr.tmp[:0], len(rows))[:len(rows)]
	l, r := 0, total
	for w := range work {
		lw, rw := l, r
		l += lefts[w]
		r += min((w+1)*step, len(rows)) - min(w*step, len(rows)) - lefts[w]
		wg.Go(func() {
			for _, row := range rows[min(w*step, len(rows)):min((w+1)*step, len(rows))] {
				if goesLeft(row) {
					gr.tmp[lw] = row
					lw++
				} else {
					gr.tmp[rw] = row
					rw++
				}
			}
		})
	}
	wg.Wait()
	copy(rows, gr.tmp)
	return total
}

// leaf records that rows end in leaf id.
func (gr *grower) leaf(rows []int32, id int32) {
	if gr.leafOf != nil {
		for _, r := range rows {
			gr.leafOf[r] = id
		}
	}
}

func (gr *grower) alloc() []float64 {
	n := gr.b.total * gr.stride()
	if l := len(gr.free); l > 0 {
		h := gr.free[l-1]
		gr.free = gr.free[:l-1]
		clear(h)
		return h
	}
	return make([]float64, n)
}

func (gr *grower) release(h []float64) {
	if h != nil {
		gr.free = append(gr.free, h)
	}
}

// best1 is best's search of one feature for one output, given the
// feature's bins h, three sums each.
func (gr *grower) best1(h []float64, j int, total []float64, parent float64) split {
	cfg := gr.cfg
	l2, minLeaf, minWeight := cfg.l2, float64(cfg.minLeaf), cfg.minWeight
	tg, th, tc := total[0], total[1], total[2]
	mg, mh, mc := h[0], h[1], h[2]
	var cg, ch, cc float64
	best := split{feature: j}
	try := func(bin int, lg, lh, lc float64, missLeft bool) {
		rg, rh, rc := tg-lg, th-lh, tc-lc
		if lc < minLeaf || rc < minLeaf || lh < minWeight || rh < minWeight {
			return
		}
		if g := lg*lg/(lh+l2) + rg*rg/(rh+l2) - parent; g > best.gain {
			best = split{gain: g, feature: j, bin: bin, missLeft: missLeft, left: []float64{lg, lh}}
		}
	}
	for bin := 1; bin < len(h)/3-1; bin++ {
		if h[bin*3+2] == 0 {
			continue
		}
		cg += h[bin*3]
		ch += h[bin*3+1]
		cc += h[bin*3+2]
		switch {
		case mc > 0:
			try(bin, cg+mg, ch+mh, cc+mc, true)
			try(bin, cg, ch, cc, false)
		case cc >= tc-cc:
			try(bin, cg, ch, cc, true)
		default:
			try(bin, cg, ch, cc, false)
		}
	}
	return best
}
