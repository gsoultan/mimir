package tree

import (
	"math"
	"math/rand/v2"
	"reflect"
	"testing"
)

// TestAddRowsMatchesGo checks the histogram kernels against the Go loop
// bit for bit, over one, two and three outputs, a subset of features in
// an odd order, and rows repeated and out of order.
func TestAddRowsMatchesGo(t *testing.T) {
	for _, k := range []int{1, 2, 3} {
		d := table(1001, 7, uint64(k), quadrant)
		for i := 0; i < len(d.X); i += 13 {
			d.X[i] = float32(math.NaN())
		}
		r := rand.New(rand.NewPCG(uint64(k), 9))
		b := bin(d, 255, r, 1)
		g, h := make([]float32, b.n*k), make([]float32, b.n*k)
		for i := range g {
			g[i], h[i] = r.Float32()-0.5, r.Float32()
		}
		sz := (&grower{k: k}).stride()
		gh := interleave(g, h, k, sz)
		rows := make([]int32, 777)
		for i := range rows {
			rows[i] = int32(r.IntN(b.n))
		}
		feats := []int{5, 0, 3, 6}
		offs := make([]int, len(feats))
		for i, j := range feats {
			offs[i] = b.off[j]
		}
		for _, n := range []int{0, 1, 2, 3, len(rows)} {
			want, got := make([]float64, b.total*sz), make([]float64, b.total*sz)
			addRowsGo(want, b, feats, offs, rows[:n], gh, sz)
			addRows(got, b, feats, offs, rows[:n], gh, sz)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("k=%d, %d rows: the kernel's histogram differs from the Go loop's", k, n)
			}
		}
	}
}

// TestLevelsGrowWhatNodeGrows checks that growing a tree level by level
// on several workers makes the same splits and leaves as growing it node
// by node on one: the same answer for every row.
func TestLevelsGrowWhatNodeGrows(t *testing.T) {
	for _, k := range []int{1, 3} {
		d := table(40000, 6, uint64(10+k), quadrant)
		r := rand.New(rand.NewPCG(3, uint64(k)))
		b := bin(d, 255, r, 1)
		g, h := make([]float32, b.n*k), make([]float32, b.n*k)
		for i := range g {
			g[i], h[i] = r.Float32()-0.5, 0.1+r.Float32()
		}
		sz := (&grower{k: k}).stride()
		gh := interleave(g, h, k, sz)
		cfg := &config{maxDepth: 7, minLeaf: 20, minWeight: 1, l2: 1}
		feats := []int{0, 1, 2, 3, 4, 5}
		grow := func(workers int, levels bool) (Tree, []int32) {
			gr := &grower{b: b, k: k, gh: gh, feats: feats, workers: workers, cfg: cfg,
				gain: make([]float64, 6), leafOf: make([]int32, b.n)}
			rows := make([]int32, b.n)
			for i := range rows {
				rows[i] = int32(i)
			}
			if levels {
				return gr.grow(rows), gr.leafOf
			}
			gr.t = Tree{}
			sums := make([]float64, 2*k)
			for _, rw := range rows {
				for j := range k {
					sums[j] += float64(gh[int(rw)*sz+j])
					sums[k+j] += float64(gh[int(rw)*sz+k+j])
				}
			}
			gr.node(rows, nil, sums, 0)
			return gr.t, gr.leafOf
		}
		nt, nl := grow(1, false)
		lt, ll := grow(4, true)
		if nt.Nodes() != lt.Nodes() {
			t.Fatalf("k=%d: %d nodes level by level, %d node by node", k, lt.Nodes(), nt.Nodes())
		}
		for i := range b.n {
			a, c := nt.Value[int(nl[i])*k:int(nl[i])*k+k], lt.Value[int(ll[i])*k:int(ll[i])*k+k]
			if !reflect.DeepEqual(a, c) {
				t.Fatalf("k=%d: row %d ends in a leaf of %v level by level, %v node by node", k, i, c, a)
			}
		}
	}
}
