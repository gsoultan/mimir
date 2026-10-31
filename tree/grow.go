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
	b *binned
	k int
	// gh holds each row's gradients, hessians and a count of one, a
	// histogram bin's stride apart, as interleave lays them out.
	gh      []float32
	feats   []int // the features the tree may split on
	perNode int   // features to draw at each node, or 0 for all of feats
	rng     *rand.Rand
	workers int
	cfg     *config
	gain    []float64 // the gain of the splits on each feature
	t       Tree
	free    [][]float64
	// leafOf, when not nil, is set to the leaf each grown row ends in.
	leafOf []int32
	tmp    []int32
}

func (gr *grower) workerCount() int {
	if gr.workers <= 0 {
		return runtime.GOMAXPROCS(0)
	}
	return gr.workers
}

// stride is the length of one bin's sums: G and H for each output, then
// the row count, padded to a multiple of four so a bin is whole vector
// registers and, for up to three outputs, one 32- or 64-byte line.
func (gr *grower) stride() int { return (2*gr.k + 1 + 3) &^ 3 }

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
	k, sz := gr.k, gr.stride()
	sums := make([]float64, 2*k)
	for _, r := range rows {
		for j := range k {
			sums[j] += float64(gr.gh[int(r)*sz+j])
			sums[k+j] += float64(gr.gh[int(r)*sz+k+j])
		}
	}
	if gr.perNode == 0 {
		gr.levels(rows, sums)
	} else {
		gr.node(rows, nil, sums, 0)
	}
	return gr.t
}

// newNode adds a leaf whose rows' sums of G and H are sums, and returns
// its id.
func (gr *grower) newNode(sums []float64) int32 {
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
	return id
}

// A level node is a node of the level levels is growing: its rows are
// rows[lo:hi] of the tree's rows.
type levelNode struct {
	lo, hi int
	id     int32
	sums   []float64
	hist   []float64
	// parent, when not nil, is the parent's histogram, from which this
	// node's is found by taking away its sibling's.
	parent  []float64
	sibling int // the index of the sibling in the level
	sp      split
}

// levels grows the tree from rows a level at a time, the way XGBoost's
// depthwise growth does, and gives the same tree node grows: every node
// of a level that may split has its histogram summed in one pass over
// the level's rows, shared evenly among the workers whatever the size of
// each node, and the nodes find and make their splits side by side. As
// in node, the larger child's histogram is its parent's less the smaller
// child's.
func (gr *grower) levels(rows []int32, sums []float64) {
	cfg, workers := gr.cfg, gr.workerCount()
	level := []levelNode{{lo: 0, hi: len(rows), id: gr.newNode(sums), sums: sums}}
	for depth := 0; len(level) > 0; depth++ {
		// The nodes that may split.
		var open []int
		isOpen := make([]bool, len(level))
		for i := range level {
			n := &level[i]
			if depth >= cfg.maxDepth || n.hi-n.lo < 2*cfg.minLeaf {
				gr.leaf(rows[n.lo:n.hi], n.id)
				continue
			}
			open = append(open, i)
			isOpen[i] = true
		}
		// Sum the histograms of the open nodes that have no parent's to
		// take theirs from, or are the smaller of two open siblings; the
		// others' are their parent's less their sibling's.
		var need []int
		for _, i := range open {
			n := &level[i]
			if n.parent == nil || !isOpen[n.sibling] {
				need = append(need, i)
				continue
			}
			mine, theirs := n.hi-n.lo, level[n.sibling].hi-level[n.sibling].lo
			if mine < theirs || (mine == theirs && i < n.sibling) {
				need = append(need, i)
			}
		}
		gr.levelHistograms(rows, level, need)
		taken := make([]bool, len(level))
		var derive []int
		for _, i := range open {
			if level[i].hist == nil {
				derive = append(derive, i)
				taken[i] = true
			}
		}
		parallel(len(derive), workers, 1, func(lo, hi int) {
			for _, i := range derive[lo:hi] {
				n := &level[i]
				for j, v := range level[n.sibling].hist {
					n.parent[j] -= v
				}
				n.hist = n.parent
			}
		})
		for i := range level {
			// Release each parent's histogram that no child took over,
			// once for the pair.
			if n := &level[i]; n.parent != nil && i < n.sibling && !taken[i] && !taken[n.sibling] {
				gr.release(n.parent)
			}
		}
		for i := range level {
			level[i].parent = nil
		}
		// Find each open node's best split, and send its rows left or right.
		inner := 1
		if len(open) < workers {
			inner = workers
		}
		parallel(len(open), workers/inner, 1, func(lo, hi int) {
			for _, i := range open[lo:hi] {
				n := &level[i]
				n.sp = gr.best(n.hist, gr.feats, n.sums, n.hi-n.lo, inner)
			}
		})
		var split []int
		for _, i := range open {
			n := &level[i]
			if n.sp.gain <= cfg.minGain {
				gr.leaf(rows[n.lo:n.hi], n.id)
				gr.release(n.hist)
				n.hist = nil
				continue
			}
			split = append(split, i)
		}
		mid := make([]int, len(level))
		if len(split) < workers {
			for _, i := range split {
				n := &level[i]
				mid[i] = n.lo + gr.partition(rows[n.lo:n.hi], n.sp)
			}
		} else {
			parallel(len(split), workers, 1, func(lo, hi int) {
				for _, i := range split[lo:hi] {
					n := &level[i]
					mid[i] = n.lo + partitionSerial(gr.b, rows[n.lo:n.hi], n.sp)
				}
			})
		}
		// The next level: each split node's two children, in order.
		next := make([]levelNode, 0, 2*len(split))
		k := gr.k
		for _, i := range split {
			n := &level[i]
			sp := n.sp
			gr.gain[sp.feature] += sp.gain
			gr.t.Feature[n.id] = int32(sp.feature)
			gr.t.Threshold[n.id] = gr.b.edges[sp.feature][sp.bin-1]
			gr.t.MissingLeft[n.id] = sp.missLeft
			rs := make([]float64, 2*k)
			for j := range rs {
				rs[j] = n.sums[j] - sp.left[j]
			}
			l, r := gr.newNode(sp.left), gr.newNode(rs)
			gr.t.Left[n.id], gr.t.Right[n.id] = l, r
			li := len(next)
			next = append(next,
				levelNode{lo: n.lo, hi: mid[i], id: l, sums: sp.left, parent: n.hist, sibling: li + 1},
				levelNode{lo: mid[i], hi: n.hi, id: r, sums: rs, parent: n.hist, sibling: li})
		}
		for _, i := range open {
			level[i].hist = nil
		}
		level = next
	}
}

// levelHistograms sums the histograms of the nodes need of level, over
// their rows, in one pass that the workers share evenly.
func (gr *grower) levelHistograms(rows []int32, level []levelNode, need []int) {
	if len(need) == 0 {
		return
	}
	sz, workers := gr.stride(), gr.workerCount()
	total := 0
	for _, i := range need {
		total += level[i].hi - level[i].lo
	}
	offs := gr.offsets(gr.feats)
	const grain = 8192
	work := min(workers, max(total/grain, 1))
	// Each worker takes an equal run of the nodes' rows, laid end to end.
	// A piece is the part of one node's rows that one worker sums: the
	// first piece of a node sums into the node's histogram, the others
	// into histograms of their own, added in afterwards.
	type piece struct {
		node   int // index into need
		lo, hi int // rows of the node, from rows' start
		at     int // where the piece starts end to end
		hist   []float64
	}
	pieces := make([][]piece, work)
	step := (total + work - 1) / work
	var extra []piece
	at := 0
	for ni, i := range need {
		n := &level[i]
		n.hist = gr.allocDirty()
		for lo := n.lo; lo < n.hi; {
			w := at / step
			hi := min(n.hi, lo+(w+1)*step-at)
			p := piece{node: ni, lo: lo, hi: hi, at: at, hist: n.hist}
			if lo != n.lo {
				p.hist = gr.allocDirty()
				extra = append(extra, p)
			}
			pieces[w] = append(pieces[w], p)
			at += hi - lo
			lo = hi
		}
	}
	parallel(work, work, 1, func(wlo, whi int) {
		for w := wlo; w < whi; w++ {
			for _, p := range pieces[w] {
				clear(p.hist)
				addRows(p.hist, gr.b, gr.feats, offs, rows[p.lo:p.hi], gr.gh, sz)
			}
		}
	})
	// Add the extra pieces in, each node's in order, the nodes side by side.
	var groups [][]piece
	for i, p := range extra {
		if i == 0 || p.node != extra[i-1].node {
			groups = append(groups, nil)
		}
		groups[len(groups)-1] = append(groups[len(groups)-1], p)
	}
	parallel(len(groups), workers, 1, func(lo, hi int) {
		for _, g := range groups[lo:hi] {
			dst := level[need[g[0].node]].hist
			for _, p := range g {
				for j, v := range p.hist {
					dst[j] += v
				}
			}
		}
	})
	for _, p := range extra {
		gr.release(p.hist)
	}
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
	best := gr.best(hist, feats, sums, len(rows), gr.workers)
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
	sz := gr.stride()
	offs := gr.offsets(feats)
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
			addRows(hist, gr.b, feats[lo:hi], offs[lo:hi], rows, gr.gh, sz)
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
		wg.Go(func() { addRows(parts[w], gr.b, feats, offs, rows[lo:hi], gr.gh, sz) })
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

// interleave lays out the gradients g and hessians h of k outputs a row
// at a time, a stride of sz apart: the row's gradients, its hessians, a
// count of one, and zeros, as a histogram bin holds them.
func interleave(g, h []float32, k, sz int) []float32 {
	n := len(g) / k
	gh := make([]float32, n*sz)
	for i := range n {
		d := gh[i*sz : i*sz+sz]
		copy(d, g[i*k:i*k+k])
		copy(d[k:], h[i*k:i*k+k])
		d[2*k] = 1
	}
	return gh
}

// offsets returns where in a histogram the bins of each of feats start.
func (gr *grower) offsets(feats []int) []int {
	offs := make([]int, len(feats))
	for i, j := range feats {
		offs[i] = gr.b.off[j]
	}
	return offs
}

// best finds the split of most gain on any feature of feats, given the
// node's histogram, its sums of G and H and its row count.
func (gr *grower) best(hist []float64, feats []int, sums []float64, count, workers int) split {
	k, s, cfg := gr.k, gr.stride(), gr.cfg
	score := func(v []float64) float64 {
		var sc float64
		for j := range k {
			sc += v[j] * v[j] / (v[k+j] + cfg.l2)
		}
		return sc
	}
	total := make([]float64, s)
	copy(total, sums)
	total[2*k] = float64(count)
	parent := score(total)
	results := make([]split, len(feats))
	work := workers
	if count < 1024 {
		work = 1
	}
	parallel(len(feats), work, 4, func(lo, hi int) {
		left, right := make([]float64, s), make([]float64, s)
		for fi := lo; fi < hi; fi++ {
			j := feats[fi]
			base := gr.b.off[j] * s
			if k == 1 {
				results[fi] = gr.best1(hist[base:base+gr.b.nb[j]*s], j, total, parent)
				continue
			}
			if k == 3 {
				results[fi] = gr.best3(hist[base:base+gr.b.nb[j]*s], j, total, parent)
				continue
			}
			results[fi] = gr.bestK(hist[base:base+gr.b.nb[j]*s], j, total, parent, left, right)
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
	const grain = 32768
	if len(rows) < 2*grain || gr.workers == 1 {
		return partitionSerial(gr.b, rows, sp)
	}
	// Each goroutine partitions its run of rows in place, then the runs'
	// left and right parts are copied together, lefts first.
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
			lefts[w] = partitionSerial(gr.b, rows[min(w*step, len(rows)):min((w+1)*step, len(rows))], sp)
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
		lo, hi := min(w*step, len(rows)), min((w+1)*step, len(rows))
		lw, rw := l, r
		l += lefts[w]
		r += hi - lo - lefts[w]
		wg.Go(func() {
			copy(gr.tmp[lw:], rows[lo:lo+lefts[w]])
			copy(gr.tmp[rw:], rows[lo+lefts[w]:hi])
		})
	}
	wg.Wait()
	parallel(len(rows), work, grain, func(lo, hi int) { copy(rows[lo:hi], gr.tmp[lo:hi]) })
	return total
}

// partitionSerial is partition on one goroutine.
func partitionSerial(b *binned, rows []int32, sp split) int {
	bins, f := b.bins[sp.feature:], b.f
	lo, hi := 0, len(rows)
	for lo < hi {
		bn := bins[int(rows[lo])*f]
		if (bn == 0 && sp.missLeft) || (bn != 0 && int(bn) <= sp.bin) {
			lo++
		} else {
			hi--
			rows[lo], rows[hi] = rows[hi], rows[lo]
		}
	}
	return lo
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

// allocDirty is alloc without the clearing, for a histogram its user
// clears.
func (gr *grower) allocDirty() []float64 {
	if l := len(gr.free); l > 0 {
		h := gr.free[l-1]
		gr.free = gr.free[:l-1]
		return h
	}
	return make([]float64, gr.b.total*gr.stride())
}

func (gr *grower) release(h []float64) {
	if h != nil {
		gr.free = append(gr.free, h)
	}
}

// bestK is best's search of one feature for more than one output, given
// the feature's bins h, a stride of sums each, and two stride-long
// scratch slices.
func (gr *grower) bestK(h []float64, j int, total []float64, parent float64, left, right []float64) split {
	k, s, cfg := gr.k, gr.stride(), gr.cfg
	l2, minLeaf, minWeight := cfg.l2, float64(cfg.minLeaf), cfg.minWeight
	miss := h[:s:s]
	cum := make([]float64, s)
	best := split{feature: j}
	var bestLeft []float64
	try := func(bin int, missLeft bool) {
		for i := range s {
			left[i] = cum[i]
			if missLeft {
				left[i] += miss[i]
			}
			right[i] = total[i] - left[i]
		}
		if left[2*k] < minLeaf || right[2*k] < minLeaf {
			return
		}
		var wl, wr, sl, sr float64
		for o := range k {
			wl += left[k+o]
			wr += right[k+o]
		}
		if wl < minWeight || wr < minWeight {
			return
		}
		for o := range k {
			sl += left[o] * left[o] / (left[k+o] + l2)
		}
		for o := range k {
			sr += right[o] * right[o] / (right[k+o] + l2)
		}
		if g := sl + sr - parent; g > best.gain {
			best = split{gain: g, feature: j, bin: bin, missLeft: missLeft}
			bestLeft = append(bestLeft[:0], left[:2*k]...)
		}
	}
	for bin := 1; bin < len(h)/s-1; bin++ {
		e := h[bin*s : bin*s+s : bin*s+s]
		if e[2*k] == 0 {
			continue
		}
		for i, v := range e {
			cum[i] += v
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
	best.left = bestLeft
	return best
}

// best3 is bestK for three outputs, written out so its sums stay in
// registers; it adds in the same order, so it finds the same split.
func (gr *grower) best3(h []float64, j int, total []float64, parent float64) split {
	cfg := gr.cfg
	l2, minLeaf, minWeight := cfg.l2, float64(cfg.minLeaf), cfg.minWeight
	t := total[:7:7]
	m := h[:7:7]
	var c0, c1, c2, c3, c4, c5, c6 float64
	best := split{feature: j}
	var b0, b1, b2, b3, b4, b5 float64
	try := func(bin int, missLeft bool) {
		g0, g1, g2, w0, w1, w2, n := c0, c1, c2, c3, c4, c5, c6
		if missLeft {
			g0, g1, g2, w0, w1, w2, n = g0+m[0], g1+m[1], g2+m[2], w0+m[3], w1+m[4], w2+m[5], n+m[6]
		}
		rg0, rg1, rg2, rw0, rw1, rw2, rn := t[0]-g0, t[1]-g1, t[2]-g2, t[3]-w0, t[4]-w1, t[5]-w2, t[6]-n
		if n < minLeaf || rn < minLeaf || (w0+w1)+w2 < minWeight || (rw0+rw1)+rw2 < minWeight {
			return
		}
		sl := (g0*g0/(w0+l2) + g1*g1/(w1+l2)) + g2*g2/(w2+l2)
		sr := (rg0*rg0/(rw0+l2) + rg1*rg1/(rw1+l2)) + rg2*rg2/(rw2+l2)
		if g := sl + sr - parent; g > best.gain {
			best = split{gain: g, feature: j, bin: bin, missLeft: missLeft}
			b0, b1, b2, b3, b4, b5 = g0, g1, g2, w0, w1, w2
		}
	}
	for bin := 1; bin < len(h)/8-1; bin++ {
		e := h[bin*8 : bin*8+7 : bin*8+7]
		if e[6] == 0 {
			continue
		}
		c0, c1, c2, c3, c4, c5, c6 = c0+e[0], c1+e[1], c2+e[2], c3+e[3], c4+e[4], c5+e[5], c6+e[6]
		if m[6] > 0 {
			try(bin, true)
			try(bin, false)
		} else {
			try(bin, c6 >= t[6]-c6)
		}
	}
	if best.gain > 0 {
		best.left = []float64{b0, b1, b2, b3, b4, b5}
	}
	return best
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
	for bin := 1; bin < len(h)/4-1; bin++ {
		if h[bin*4+2] == 0 {
			continue
		}
		cg += h[bin*4]
		ch += h[bin*4+1]
		cc += h[bin*4+2]
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
