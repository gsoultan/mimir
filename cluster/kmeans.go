// Package cluster finds groups in tables of numbers without answers to
// learn from: k-means, Gaussian mixtures and DBSCAN, as scikit-learn's
// cluster and mixture packages do, and the silhouette that scores how
// well rows sit in their groups.
//
//	m, err := cluster.KMeans(ctx, x, features, 5)
//	if err != nil { ... }
//	groups := m.Assign(rows) // the group of each row, 0 to 4
//
// Each standardizes the features first, so a column in grams and one in
// kilometres weigh alike, and counts a missing value, NaN, as the
// feature's mean. Models hold only exported fields, so encoding/json
// saves and loads them; KMeans and Mixture also convert to an nn.Module.
package cluster

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

// Option configures KMeans, Mixture and DBSCAN.
type Option func(*config)

type config struct {
	inits   int
	iters   int
	tol     float64
	seed    uint64
	workers int
	minRows int
}

// Inits sets how many times k-means starts afresh, keeping the best
// grouping. The default is 4.
func Inits(n int) Option { return func(c *config) { c.inits = n } }

// MaxIter caps the passes each start makes. The default is 300.
func MaxIter(n int) Option { return func(c *config) { c.iters = n } }

// Tolerance stops a start once no center moves by more than tol, in
// standard deviations, or the mixture's mean log-likelihood rises by
// less. The default is 1e-4.
func Tolerance(tol float64) Option { return func(c *config) { c.tol = tol } }

// Seed seeds the random starts. The default is 1.
func Seed(s uint64) Option { return func(c *config) { c.seed = s } }

// Workers sets how many goroutines run. The default is GOMAXPROCS.
func Workers(n int) Option { return func(c *config) { c.workers = n } }

// MinRows sets the rows within reach a DBSCAN row needs, itself
// included, to be a core of a group. The default is 5.
func MinRows(n int) Option { return func(c *config) { c.minRows = n } }

func options(x []float32, features, k int, opts []Option) (*config, int, error) {
	c := &config{inits: 4, iters: 300, tol: 1e-4, seed: 1, workers: runtime.GOMAXPROCS(0), minRows: 5}
	for _, o := range opts {
		o(c)
	}
	bad := func(name string, v any) error { return fmt.Errorf("cluster: %s %v: %w", name, v, tensor.ErrShape) }
	if features <= 0 || len(x) == 0 || len(x)%features != 0 {
		return nil, 0, fmt.Errorf("cluster: %d numbers for rows of %d features: %w", len(x), features, tensor.ErrShape)
	}
	n := len(x) / features
	switch {
	case k < 1 || k > n:
		return nil, 0, bad("groups", k)
	case c.inits < 1:
		return nil, 0, bad("starts", c.inits)
	case c.iters < 1:
		return nil, 0, bad("iterations", c.iters)
	case !(c.tol >= 0):
		return nil, 0, bad("tolerance", c.tol)
	case c.minRows < 1:
		return nil, 0, bad("min rows", c.minRows)
	}
	c.workers = max(c.workers, 1)
	return c, n, nil
}

// Means is a k-means grouping: each row belongs to the nearest of K
// centers.
type Means struct {
	Features int
	Scaler   tabular.Scaler
	// Centers holds K centers of Features standardized numbers each.
	Centers []float32
	// Inertia is the sum of squared standardized distances from the
	// training rows to their centers.
	Inertia    float64
	Iterations int
}

// K returns the number of groups.
func (m *Means) K() int { return len(m.Centers) / m.Features }

// KMeans groups the rows of x, features numbers each, around k centers by
// Lloyd's algorithm from k-means++ starts.
func KMeans(ctx context.Context, x []float32, features, k int, opts ...Option) (*Means, error) {
	c, n, err := options(x, features, k, opts)
	if err != nil {
		return nil, err
	}
	sc := tabular.FitScaler(x, features)
	z := sc.Transform(x)
	rng := rand.New(rand.NewPCG(c.seed, 0x6b6d65616e73))
	var best *Means
	for range c.inits {
		m, err := lloyd(ctx, z, n, features, k, c, rng)
		if err != nil {
			return nil, err
		}
		if best == nil || m.Inertia < best.Inertia {
			best = m
		}
	}
	best.Scaler = sc
	return best, nil
}

// lloyd runs one start of k-means over the standardized rows z.
func lloyd(ctx context.Context, z []float32, n, f, k int, c *config, rng *rand.Rand) (*Means, error) {
	centers := plusPlus(z, n, f, k, rng)
	assign := make([]int, n)
	m := &Means{Features: f, Centers: centers}
	for it := range c.iters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		m.Iterations = it + 1
		m.Inertia = nearestAll(z, centers, f, assign, c.workers)
		sums := make([]float64, k*f)
		counts := make([]int, k)
		for i, a := range assign {
			counts[a]++
			for j, v := range z[i*f : (i+1)*f] {
				sums[a*f+j] += float64(v)
			}
		}
		var moved float64
		for a := range k {
			if counts[a] == 0 {
				// An empty group takes the row farthest from its center.
				far, fd := 0, -1.0
				for i := range n {
					if d := dist(z[i*f:(i+1)*f], centers[assign[i]*f:(assign[i]+1)*f]); d > fd {
						far, fd = i, d
					}
				}
				copy(centers[a*f:(a+1)*f], z[far*f:(far+1)*f])
				assign[far] = a
				moved = math.Inf(1)
				continue
			}
			for j := range f {
				v := float32(sums[a*f+j] / float64(counts[a]))
				moved = math.Max(moved, math.Abs(float64(v-centers[a*f+j])))
				centers[a*f+j] = v
			}
		}
		if moved <= c.tol {
			break
		}
	}
	m.Inertia = nearestAll(z, centers, f, assign, c.workers)
	return m, nil
}

// plusPlus picks k starting centers from the rows of z, each new one with
// a chance in proportion to its squared distance from the nearest picked,
// as k-means++ does, over a sample of at most 100,000 rows.
func plusPlus(z []float32, n, f, k int, rng *rand.Rand) []float32 {
	rows := make([]int, 0, min(n, 100_000))
	if n <= 100_000 {
		for i := range n {
			rows = append(rows, i)
		}
	} else {
		for range 100_000 {
			rows = append(rows, rng.IntN(n))
		}
	}
	centers := make([]float32, 0, k*f)
	first := rows[rng.IntN(len(rows))]
	centers = append(centers, z[first*f:(first+1)*f]...)
	d := make([]float64, len(rows))
	for i, r := range rows {
		d[i] = dist(z[r*f:(r+1)*f], centers)
	}
	for len(centers) < k*f {
		var total float64
		for _, v := range d {
			total += v
		}
		pick := rows[rng.IntN(len(rows))]
		if total > 0 {
			t := rng.Float64() * total
			for i, v := range d {
				if t -= v; t <= 0 {
					pick = rows[i]
					break
				}
			}
		}
		c := z[pick*f : (pick+1)*f]
		centers = append(centers, c...)
		for i, r := range rows {
			d[i] = math.Min(d[i], dist(z[r*f:(r+1)*f], c))
		}
	}
	return centers
}

// nearestAll sets assign to the nearest center of each row of z and
// returns the sum of the squared distances.
func nearestAll(z, centers []float32, f int, assign []int, workers int) float64 {
	n := len(z) / f
	parts := make([]float64, workers)
	rows(n, workers, func(w, lo, hi int) {
		for i := lo; i < hi; i++ {
			a, d := nearest(z[i*f:(i+1)*f], centers, f)
			assign[i] = a
			parts[w] += d
		}
	})
	var s float64
	for _, v := range parts {
		s += v
	}
	return s
}

func nearest(row, centers []float32, f int) (int, float64) {
	best, bd := 0, math.Inf(1)
	for a := range len(centers) / f {
		if d := dist(row, centers[a*f:(a+1)*f]); d < bd {
			best, bd = a, d
		}
	}
	return best, bd
}

// dist returns the squared distance between a and b[:len(a)].
func dist(a, b []float32) float64 {
	var s float64
	for j, v := range a {
		d := float64(v - b[j])
		s += d * d
	}
	return s
}

// Assign returns the group of each row of x, rows of m.Features numbers.
func (m *Means) Assign(x []float32) []int {
	z := m.Scaler.Transform(x)
	out := make([]int, len(z)/m.Features)
	nearestAll(z, m.Centers, m.Features, out, runtime.GOMAXPROCS(0))
	return out
}

// Predict returns K numbers per row of x that are 1 for its group and 0
// for the others, so a Means is a tabular.Model whose classes are groups.
func (m *Means) Predict(x []float32) []float32 {
	k := m.K()
	out := make([]float32, len(x)/m.Features*k)
	for i, a := range m.Assign(x) {
		out[i*k+a] = 1
	}
	return out
}

// rows calls f over [0, n) in chunks on up to workers goroutines, each
// with its own index below workers.
func rows(n, workers int, f func(w, lo, hi int)) {
	workers = min(workers, max(n/2048, 1))
	if workers <= 1 {
		f(0, 0, n)
		return
	}
	var wg sync.WaitGroup
	step := (n + workers - 1) / workers
	for w := range workers {
		lo, hi := min(w*step, n), min((w+1)*step, n)
		wg.Go(func() { f(w, lo, hi) })
	}
	wg.Wait()
}
