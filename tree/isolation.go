package tree

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Isolation is the link of an isolation forest: the answer is
// 2^-(mean path length / c(samples)), the anomaly score.
const Isolation Link = "isolation"

// IsolationForest grows trees that each cut a random sample of the rows
// of x, features numbers each, at random until every row stands alone,
// as Liu, Ting and Zhou's isolation forest and scikit-learn's
// IsolationForest do. Unusual rows are cut off in few steps, so the
// model's Predict answers each row with an anomaly score from 0 to 1:
// about 0.5 or less for ordinary rows, and near 1 for rows unlike the
// rest. Answers in d.Y are not read. Trees, Samples, Seed and Workers
// apply; a NaN goes right at every cut.
func IsolationForest(ctx context.Context, x []float32, features int, opts ...Option) (*Model, error) {
	c := &config{trees: 100, rate: 1, maxDepth: 1, minLeaf: 1, subsample: 1, colsample: 1, bins: 255, seed: 1, samples: 256}
	for _, o := range opts {
		o(c)
	}
	if features <= 0 || len(x) == 0 || len(x)%features != 0 {
		return nil, fmt.Errorf("tree: %d numbers for rows of %d features: %w", len(x), features, tensor.ErrShape)
	}
	if c.trees < 1 || c.samples < 2 {
		return nil, fmt.Errorf("tree: %d trees of %d samples: %w", c.trees, c.samples, tensor.ErrShape)
	}
	n, f := len(x)/features, features
	psi := min(c.samples, n)
	limit := int(math.Ceil(math.Log2(float64(max(psi, 2)))))
	m := &Model{Task: tabular.Regression, Features: f, Outputs: 1, Link: Isolation, Base: []float32{0},
		Scale: float32(1 / (float64(c.trees) * avgPath(psi))), Trees: make([]Tree, c.trees), Importance: make([]float64, f)}
	rng := rand.New(rand.NewPCG(c.seed, 0x69736f))
	seeds := make([]uint64, c.trees)
	for i := range seeds {
		seeds[i] = rng.Uint64()
	}
	workers := c.workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	next := make(chan int)
	done := 0
	for range min(workers, c.trees) {
		wg.Go(func() {
			for t := range next {
				r := rand.New(rand.NewPCG(seeds[t], uint64(t)))
				rows := r.Perm(n)[:psi]
				var tr Tree
				isolate(&tr, x, f, rows, 0, limit, r)
				m.Trees[t] = tr
				if c.progress != nil {
					mu.Lock()
					done++
					c.progress(done, c.trees)
					mu.Unlock()
				}
			}
		})
	}
	var err error
	for t := range c.trees {
		if err = ctx.Err(); err != nil {
			break
		}
		next <- t
	}
	close(next)
	wg.Wait()
	if err != nil {
		return nil, err
	}
	return m, nil
}

// isolate grows the subtree of rows at depth into t and returns its root.
func isolate(t *Tree, x []float32, f int, rows []int, depth, limit int, r *rand.Rand) int32 {
	id := int32(len(t.Feature))
	t.Feature = append(t.Feature, -1)
	t.Threshold = append(t.Threshold, 0)
	t.MissingLeft = append(t.MissingLeft, false)
	t.Left = append(t.Left, -1)
	t.Right = append(t.Right, -1)
	t.Value = append(t.Value, float32(float64(depth)+avgPath(len(rows))))
	if depth >= limit || len(rows) <= 1 {
		return id
	}
	// Cut a feature that still varies here, picked at random.
	for _, j := range r.Perm(f) {
		lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
		for _, i := range rows {
			if v := x[i*f+j]; v == v {
				lo, hi = min(lo, v), max(hi, v)
			}
		}
		if !(lo < hi) {
			continue
		}
		cut := lo + float32(r.Float64())*(hi-lo)
		if cut >= hi {
			cut = lo
		}
		var left, right []int
		for _, i := range rows {
			if v := x[i*f+j]; v <= cut {
				left = append(left, i)
			} else {
				right = append(right, i)
			}
		}
		t.Feature[id], t.Threshold[id] = int32(j), cut
		l := isolate(t, x, f, left, depth+1, limit, r)
		rt := isolate(t, x, f, right, depth+1, limit, r)
		t.Left[id], t.Right[id] = l, rt
		return id
	}
	return id
}

// avgPath is c(n), the mean path length of an unsuccessful search in a
// binary search tree of n rows, which a leaf of n rows adds to its depth.
func avgPath(n int) float64 {
	switch {
	case n <= 1:
		return 0
	case n == 2:
		return 1
	}
	return 2*(math.Log(float64(n-1))+0.5772156649015329) - 2*float64(n-1)/float64(n)
}
