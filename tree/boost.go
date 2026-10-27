package tree

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// GradientBoosting grows trees one after another, each fit to the
// gradients of the loss of the trees before it, as XGBoost and LightGBM
// do: squared error for Regression, log loss for Classification, with a
// sigmoid for two classes and a softmax for more.
func GradientBoosting(ctx context.Context, d tabular.Data, task tabular.Task, opts ...Option) (*Model, error) {
	c := &config{trees: 100, rate: 0.1, maxDepth: 6, minLeaf: 1, minWeight: 1, l2: 1,
		subsample: 1, colsample: 1, bins: 255, seed: 1}
	for _, o := range opts {
		o(c)
	}
	if err := d.Check(task); err != nil {
		return nil, err
	}
	if err := c.check(d); err != nil {
		return nil, err
	}
	if c.val != nil {
		if err := c.val.Check(task); err != nil {
			return nil, fmt.Errorf("tree: validation: %w", err)
		}
	}
	rng := rand.New(rand.NewPCG(c.seed, 0x6d696d6972))
	n, f := d.Rows(), d.Features
	m := &Model{Task: task, Features: f, Outputs: 1, Link: Identity, Scale: float32(c.rate)}
	if task == tabular.Classification {
		m.Classes = c.numClasses(d)
		if c.val != nil {
			for _, y := range c.val.Y {
				if int(y) >= m.Classes {
					return nil, fmt.Errorf("tree: validation class %v of %d: %w", y, m.Classes, tensor.ErrShape)
				}
			}
		}
		m.Link = Sigmoid
		if m.Classes > 2 {
			m.Link, m.Outputs = Softmax, m.Classes
		}
	}
	k := m.Outputs
	m.Base = base(m, d.Y)

	b := bin(d, c.bins, rng, c.workers)
	gr := &grower{b: b, k: k, g: make([]float32, n*k), h: make([]float32, n*k), rng: rng,
		workers: c.workers, cfg: c, gain: make([]float64, f)}
	if c.subsample == 1 {
		gr.leafOf = make([]int32, n)
	}
	raw := start(m, n)
	var vraw []float32
	if c.val != nil {
		vraw = start(m, c.val.Rows())
	}
	all := make([]int32, n)
	for i := range all {
		all[i] = int32(i)
	}
	rows := make([]int32, n)
	feats := make([]int, f)
	for i := range feats {
		feats[i] = i
	}
	bestLoss, bestRound := math.Inf(1), 0
	gains := [][]float64{}
	for round := range c.trees {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		gradients(m, raw, d.Y, gr.g, gr.h)
		rows = rows[:0]
		if c.subsample < 1 {
			for _, r := range all {
				if rng.Float64() < c.subsample {
					rows = append(rows, r)
				}
			}
			if len(rows) == 0 {
				rows = append(rows, int32(rng.IntN(n)))
			}
		} else {
			rows = append(rows, all...)
		}
		gr.feats = feats
		if c.colsample < 1 {
			rng.Shuffle(f, func(a, b int) { feats[a], feats[b] = feats[b], feats[a] })
			gr.feats = feats[:max(1, int(math.Round(c.colsample*float64(f))))]
		}
		before := append([]float64(nil), gr.gain...)
		t := gr.grow(rows)
		m.Trees = append(m.Trees, t)
		if gr.leafOf != nil {
			// Every row grew the tree, so each knows its leaf.
			for i, l := range gr.leafOf {
				v := t.Value[int(l)*k : int(l)*k+k]
				for j, vj := range v {
					raw[i*k+j] += m.Scale * vj
				}
			}
		} else {
			add(m, &t, d.X, raw, c.workers)
		}
		delta := make([]float64, f)
		for j := range delta {
			delta[j] = gr.gain[j] - before[j]
		}
		gains = append(gains, delta)
		if c.progress != nil {
			c.progress(round+1, c.trees)
		}
		if c.val == nil {
			continue
		}
		add(m, &t, c.val.X, vraw, c.workers)
		if l := logLoss(m, vraw, c.val.Y); l < bestLoss-1e-12 {
			bestLoss, bestRound = l, round
		} else if round-bestRound >= c.patience {
			break
		}
	}
	if c.val != nil {
		m.Trees = m.Trees[:bestRound+1]
		gains = gains[:bestRound+1]
	}
	m.Importance = importance(gains, f)
	return m, nil
}

// base returns the constant answer that best fits y: the mean for
// Regression, the log-odds or log prior of each class otherwise.
func base(m *Model, y []float32) []float32 {
	switch m.Link {
	case Identity:
		var s float64
		for _, v := range y {
			s += float64(v)
		}
		return []float32{float32(s / float64(len(y)))}
	case Sigmoid:
		var p float64
		for _, v := range y {
			p += float64(v)
		}
		p = math.Min(math.Max(p/float64(len(y)), 1e-6), 1-1e-6)
		return []float32{float32(math.Log(p / (1 - p)))}
	}
	count := make([]float64, m.Outputs)
	for _, v := range y {
		count[int(v)]++
	}
	b := make([]float32, m.Outputs)
	for j, c := range count {
		b[j] = float32(math.Log(math.Max(c/float64(len(y)), 1e-6)))
	}
	return b
}

// start returns n rows of m's base values.
func start(m *Model, n int) []float32 {
	raw := make([]float32, n*m.Outputs)
	for i := range n {
		copy(raw[i*m.Outputs:], m.Base)
	}
	return raw
}

// add adds t's scaled answer for each row of x to raw.
func add(m *Model, t *Tree, x []float32, raw []float32, workers int) {
	k, f := m.Outputs, m.Features
	parallel(len(x)/f, workers, 256, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			v := t.Value[int(t.leaf(x[i*f:(i+1)*f]))*k:]
			for j := range k {
				raw[i*k+j] += m.Scale * v[j]
			}
		}
	})
}

// gradients sets g and h to the gradient and hessian of m's loss at raw.
func gradients(m *Model, raw, y, g, h []float32) {
	k := m.Outputs
	switch m.Link {
	case Identity:
		for i, v := range raw {
			g[i], h[i] = v-y[i], 1
		}
	case Sigmoid:
		for i, v := range raw {
			p := sigmoid(v)
			g[i], h[i] = p-y[i], max(p*(1-p), 1e-6)
		}
	case Softmax:
		p := make([]float32, k)
		for i := range len(y) {
			copy(p, raw[i*k:(i+1)*k])
			softmax(p)
			for j, pj := range p {
				t := float32(0)
				if int(y[i]) == j {
					t = 1
				}
				g[i*k+j], h[i*k+j] = pj-t, max(pj*(1-pj), 1e-6)
			}
		}
	}
}

// logLoss returns m's mean loss at raw for answers y: squared error for
// Regression, log loss for Classification.
func logLoss(m *Model, raw, y []float32) float64 {
	var s float64
	switch m.Link {
	case Identity:
		for i, v := range raw {
			e := float64(v - y[i])
			s += e * e
		}
	default:
		p := m.answer(append([]float32(nil), raw...), len(y))
		for i, c := range y {
			s -= math.Log(math.Max(float64(p[i*m.Classes+int(c)]), 1e-15))
		}
	}
	return s / float64(len(y))
}

// importance sums the gain of every kept tree on each feature, as shares
// of the whole.
func importance(gains [][]float64, f int) []float64 {
	imp := make([]float64, f)
	var total float64
	for _, g := range gains {
		for j, v := range g {
			imp[j] += v
			total += v
		}
	}
	if total > 0 {
		for j := range imp {
			imp[j] /= total
		}
	}
	return imp
}
