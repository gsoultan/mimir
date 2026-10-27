package tree

import (
	"context"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"

	"github.com/gsoultan/mimir/tabular"
)

// RandomForest grows trees side by side, each from a bootstrap sample of
// the rows and choosing each split from a random few features, and
// averages their answers, as scikit-learn's random forests do. A
// Classification forest's leaves hold class probabilities and its splits
// lower Gini impurity; a Regression forest's leaves hold mean answers and
// its splits lower squared error.
func RandomForest(ctx context.Context, d tabular.Data, task tabular.Task, opts ...Option) (*Model, error) {
	perNode := d.Features
	if task == tabular.Classification {
		perNode = max(1, int(math.Round(math.Sqrt(float64(d.Features)))))
	}
	c := &config{trees: 100, rate: 1, maxDepth: math.MaxInt32, minLeaf: 1, subsample: 1, colsample: 1,
		bins: 255, seed: 1, bootstrap: true, perNode: min(perNode, max(d.Features, 0))}
	return forest(ctx, d, task, c, opts)
}

// DecisionTree grows one tree that may split on every feature at every
// node, from every row, as scikit-learn's DecisionTreeClassifier and
// DecisionTreeRegressor do.
func DecisionTree(ctx context.Context, d tabular.Data, task tabular.Task, opts ...Option) (*Model, error) {
	c := &config{trees: 1, rate: 1, maxDepth: math.MaxInt32, minLeaf: 1, subsample: 1, colsample: 1,
		bins: 255, seed: 1}
	return forest(ctx, d, task, c, opts)
}

func forest(ctx context.Context, d tabular.Data, task tabular.Task, c *config, opts []Option) (*Model, error) {
	for _, o := range opts {
		o(c)
	}
	if err := d.Check(task); err != nil {
		return nil, err
	}
	if err := c.check(d); err != nil {
		return nil, err
	}
	n, f := d.Rows(), d.Features
	m := &Model{Task: task, Features: f, Outputs: 1, Link: Identity, Scale: 1 / float32(c.trees)}
	// A leaf answers -G/H, so g = -y and h = 1 make it the mean answer.
	var g, h []float32
	if task == tabular.Classification {
		m.Classes = c.numClasses(d)
		m.Outputs = m.Classes
		g = make([]float32, n*m.Classes)
		for i, y := range d.Y {
			g[i*m.Classes+int(y)] = -1
		}
	} else {
		g = make([]float32, n)
		for i, y := range d.Y {
			g[i] = -y
		}
	}
	h = make([]float32, len(g))
	for i := range h {
		h[i] = 1
	}
	m.Base = make([]float32, m.Outputs)
	c.l2, c.minWeight = 0, 0
	perNode := c.perNode
	if perNode >= f {
		perNode = 0
	}

	rng := rand.New(rand.NewPCG(c.seed, 0x666f72657374))
	b := bin(d, c.bins, rng, c.workers)
	feats := make([]int, f)
	for i := range feats {
		feats[i] = i
	}
	seeds := make([]uint64, c.trees)
	for i := range seeds {
		seeds[i] = rng.Uint64()
	}
	m.Trees = make([]Tree, c.trees)
	gains := make([][]float64, c.trees)

	workers := c.workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	// Grow trees in parallel when there are several, else parallelize
	// within the one tree.
	inner := 1
	if c.trees == 1 {
		inner, workers = workers, 1
	}
	next := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done := 0
	for range min(workers, c.trees) {
		wg.Go(func() {
			for t := range next {
				r := rand.New(rand.NewPCG(seeds[t], uint64(t)))
				rows := make([]int32, n)
				for i := range rows {
					if c.bootstrap {
						rows[i] = int32(r.IntN(n))
					} else {
						rows[i] = int32(i)
					}
				}
				gr := &grower{b: b, k: m.Outputs, g: g, h: h, feats: feats, perNode: perNode, rng: r,
					workers: inner, cfg: c, gain: make([]float64, f)}
				m.Trees[t] = gr.grow(rows)
				gains[t] = gr.gain
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
	m.Importance = importance(gains, f)
	return m, nil
}
