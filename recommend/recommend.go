// Package recommend learns which items each user will like from what
// users did with items before, by matrix factorization with alternating
// least squares: for clicks, views and purchases as Hu, Koren and Volinsky
// do, and for ratings as Netflix-prize models do.
//
//	m, err := recommend.Fit(ctx, users, items, events)
//	if err != nil { ... }
//	top := m.Recommend(user, 10, seen) // the 10 items user is likeliest to want
//
// Users and items are indices from 0. Each gets Factors numbers, so the
// score of an item for a user is the dot product of theirs.
package recommend

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/tensor"
)

// Event is one thing a user did with an item: Value is how much, such as
// a count of purchases, or a rating with Ratings.
type Event struct {
	User, Item int
	Value      float32
}

// Option configures Fit.
type Option func(*config)

type config struct {
	factors, iters int
	l2, alpha      float64
	ratings        bool
	seed           uint64
	workers        int
}

// Factors sets the numbers each user and item gets. The default is 32.
func Factors(n int) Option { return func(c *config) { c.factors = n } }

// Iterations sets the passes over users and items. The default is 15.
func Iterations(n int) Option { return func(c *config) { c.iters = n } }

// L2 sets the penalty on the factors. The default is 0.1.
func L2(l float64) Option { return func(c *config) { c.l2 = l } }

// Alpha sets how much more an event counts than its absence, per unit of
// Value, for implicit events. The default is 10.
func Alpha(a float64) Option { return func(c *config) { c.alpha = a } }

// Ratings reads each Value as a rating to predict, not an amount of
// interest, and learns from the rated pairs only.
func Ratings() Option { return func(c *config) { c.ratings = true } }

// Seed seeds the starting factors. The default is 1.
func Seed(s uint64) Option { return func(c *config) { c.seed = s } }

// Workers sets how many goroutines run. The default is GOMAXPROCS.
func Workers(n int) Option { return func(c *config) { c.workers = n } }

// Model is a learned factorization.
type Model struct {
	Users, Items, Factors int
	UserFactors           []float32 // Users×Factors
	ItemFactors           []float32 // Items×Factors
	// Mean is added to every score of a Ratings model, the mean rating,
	// and is 0 otherwise.
	Mean    float32
	Ratings bool
}

// Fit learns users×items factors from events. Several events of one
// pair add up.
func Fit(ctx context.Context, users, items int, events []Event, opts ...Option) (*Model, error) {
	c := &config{factors: 32, iters: 15, l2: 0.1, alpha: 10, seed: 1, workers: runtime.GOMAXPROCS(0)}
	for _, o := range opts {
		o(c)
	}
	bad := func(name string, v any) error { return fmt.Errorf("recommend: %s %v: %w", name, v, tensor.ErrShape) }
	switch {
	case users < 1 || items < 1:
		return nil, fmt.Errorf("recommend: %d users and %d items: %w", users, items, tensor.ErrShape)
	case len(events) == 0:
		return nil, bad("events", 0)
	case c.factors < 1 || c.factors > 1024:
		return nil, bad("factors", c.factors)
	case c.iters < 1:
		return nil, bad("iterations", c.iters)
	case !(c.l2 > 0) || math.IsInf(c.l2, 0):
		return nil, bad("L2", c.l2)
	case !(c.alpha > 0) || math.IsInf(c.alpha, 0):
		return nil, bad("alpha", c.alpha)
	}
	for _, e := range events {
		if e.User < 0 || e.User >= users || e.Item < 0 || e.Item >= items || e.Value != e.Value || math.IsInf(float64(e.Value), 0) || (!c.ratings && e.Value < 0) {
			return nil, fmt.Errorf("recommend: event %+v of %d users and %d items: %w", e, users, items, tensor.ErrShape)
		}
	}
	c.workers = max(c.workers, 1)
	byUser, byItem := lists(users, items, events)
	k := c.factors
	m := &Model{Users: users, Items: items, Factors: k, Ratings: c.ratings,
		UserFactors: make([]float32, users*k), ItemFactors: make([]float32, items*k)}
	if c.ratings {
		var s float64
		for _, e := range events {
			s += float64(e.Value)
		}
		m.Mean = float32(s / float64(len(events)))
	}
	rng := rand.New(rand.NewPCG(c.seed, 0x616c73))
	for i := range m.ItemFactors {
		m.ItemFactors[i] = float32(rng.NormFloat64() * 0.01)
	}
	for i := range m.UserFactors {
		m.UserFactors[i] = float32(rng.NormFloat64() * 0.01)
	}
	for range c.iters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		solve(m.UserFactors, m.ItemFactors, byUser, k, c, m.Mean)
		solve(m.ItemFactors, m.UserFactors, byItem, k, c, m.Mean)
	}
	return m, nil
}

type entry struct {
	other int
	value float32
}

// lists returns each user's and each item's events, a pair's values
// summed.
func lists(users, items int, events []Event) (byUser, byItem [][]entry) {
	sum := map[[2]int]float32{}
	for _, e := range events {
		sum[[2]int{e.User, e.Item}] += e.Value
	}
	byUser, byItem = make([][]entry, users), make([][]entry, items)
	keys := make([][2]int, 0, len(sum))
	for p := range sum {
		keys = append(keys, p)
	}
	slices.SortFunc(keys, func(a, b [2]int) int {
		if a[0] != b[0] {
			return a[0] - b[0]
		}
		return a[1] - b[1]
	})
	for _, p := range keys {
		byUser[p[0]] = append(byUser[p[0]], entry{p[1], sum[p]})
		byItem[p[1]] = append(byItem[p[1]], entry{p[0], sum[p]})
	}
	return byUser, byItem
}

// solve sets each row of x, k factors, to the least-squares fit against
// the fixed factors y of what it has events with.
func solve(x, y []float32, lists [][]entry, k int, c *config, mean float32) {
	// For implicit events every pair counts with confidence 1, so YᵀY is
	// shared and each row adds only its events' extra confidence.
	var yty []float64
	if !c.ratings {
		yty = make([]float64, k*k)
		for o := range len(y) / k {
			v := y[o*k : (o+1)*k]
			for a := range k {
				for b := range a + 1 {
					yty[a*k+b] += float64(v[a]) * float64(v[b])
				}
			}
		}
	}
	n := len(lists)
	workers := min(c.workers, max(n/64, 1))
	var wg sync.WaitGroup
	step := (n + workers - 1) / workers
	for w := range workers {
		wg.Go(func() {
			a, rhs := make([]float64, k*k), make([]float64, k)
			for r := w * step; r < min((w+1)*step, n); r++ {
				es := lists[r]
				if len(es) == 0 && c.ratings {
					clear(x[r*k : (r+1)*k])
					continue
				}
				if c.ratings {
					clear(a)
				} else {
					copy(a, yty)
				}
				clear(rhs)
				for _, e := range es {
					v := y[e.other*k : (e.other+1)*k]
					conf, target := 1.0, float64(e.value-mean)
					if !c.ratings {
						conf, target = 1+c.alpha*float64(e.value), 1
						if e.value <= 0 {
							target = 0
						}
					}
					extra := conf
					if !c.ratings {
						extra = conf - 1 // YᵀY already counts it once
					}
					for i := range k {
						rhs[i] += conf * target * float64(v[i])
						for j := range i + 1 {
							a[i*k+j] += extra * float64(v[i]) * float64(v[j])
						}
					}
				}
				reg := c.l2
				if c.ratings {
					reg *= float64(max(len(es), 1)) // weighted-λ, as Zhou et al.
				}
				for i := range k {
					a[i*k+i] += reg
				}
				cholSolve(a, rhs, k)
				for i := range k {
					x[r*k+i] = float32(rhs[i])
				}
			}
		})
	}
	wg.Wait()
}

// cholSolve solves a·x = b for the symmetric positive definite a, given
// by its lower triangle, in place: b becomes x and a its Cholesky factor.
func cholSolve(a, b []float64, k int) {
	for j := range k {
		s := a[j*k+j]
		for p := range j {
			s -= a[j*k+p] * a[j*k+p]
		}
		d := math.Sqrt(math.Max(s, 1e-12))
		a[j*k+j] = d
		for i := j + 1; i < k; i++ {
			s := a[i*k+j]
			for p := range j {
				s -= a[i*k+p] * a[j*k+p]
			}
			a[i*k+j] = s / d
		}
	}
	for i := range k {
		s := b[i]
		for p := range i {
			s -= a[i*k+p] * b[p]
		}
		b[i] = s / a[i*k+i]
	}
	for i := k - 1; i >= 0; i-- {
		s := b[i]
		for p := i + 1; p < k; p++ {
			s -= a[p*k+i] * b[p]
		}
		b[i] = s / a[i*k+i]
	}
}

// Score returns the score of item for user: the predicted rating for a
// Ratings model, a preference near 0 to 1 otherwise.
func (m *Model) Score(user, item int) float32 {
	var s float32
	u, v := m.UserFactors[user*m.Factors:(user+1)*m.Factors], m.ItemFactors[item*m.Factors:(item+1)*m.Factors]
	for i := range u {
		s += u[i] * v[i]
	}
	return s + m.Mean
}

// Recommend returns the n items of highest score for user, the highest
// first, leaving out the items in skip, such as what the user already has.
func (m *Model) Recommend(user, n int, skip []int) []int {
	scores := make([]float32, m.Items)
	for it := range scores {
		scores[it] = m.Score(user, it)
	}
	for _, it := range skip {
		if it >= 0 && it < m.Items {
			scores[it] = float32(math.Inf(-1))
		}
	}
	idx := make([]int, m.Items)
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(a, b int) int {
		switch {
		case scores[a] > scores[b]:
			return -1
		case scores[a] < scores[b]:
			return 1
		}
		return 0
	})
	out := idx[:min(n, len(idx))]
	for len(out) > 0 && math.IsInf(float64(scores[out[len(out)-1]]), -1) {
		out = out[:len(out)-1]
	}
	return out
}

// Similar returns the n items whose factors point most nearly the way
// item's do, by cosine, leaving item out.
func (m *Model) Similar(item, n int) []int {
	k := m.Factors
	v := m.ItemFactors[item*k : (item+1)*k]
	norm := func(w []float32) float64 {
		var s float64
		for _, x := range w {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s) + 1e-12
	}
	nv := norm(v)
	type scored struct {
		item int
		cos  float64
	}
	var all []scored
	for it := range m.Items {
		if it == item {
			continue
		}
		w := m.ItemFactors[it*k : (it+1)*k]
		var d float64
		for i := range w {
			d += float64(v[i]) * float64(w[i])
		}
		all = append(all, scored{it, d / (nv * norm(w))})
	}
	slices.SortStableFunc(all, func(a, b scored) int {
		switch {
		case a.cos > b.cos:
			return -1
		case a.cos < b.cos:
			return 1
		}
		return 0
	})
	out := make([]int, 0, n)
	for _, s := range all[:min(n, len(all))] {
		out = append(out, s.item)
	}
	return out
}

// HitRate returns the share of held-out events whose item is among the n
// that m recommends for the event's user, skipping the items each user's
// training events name: recall at n, for leave-one-out evaluation.
func (m *Model) HitRate(train, held []Event, n int) float64 {
	seen := map[int][]int{}
	for _, e := range train {
		seen[e.User] = append(seen[e.User], e.Item)
	}
	hits := 0
	for _, e := range held {
		if slices.Contains(m.Recommend(e.User, n, seen[e.User]), e.Item) {
			hits++
		}
	}
	return float64(hits) / float64(max(len(held), 1))
}
