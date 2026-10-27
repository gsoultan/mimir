package svm

import (
	"container/list"
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"

	"github.com/gsoultan/mimir/tensor"
)

// kernel computes and caches rows of the kernel matrix of the
// standardized rows, shared by the machines fit at once.
type kernel struct {
	m    *Model
	x    []float32
	n    int
	diag []float64

	mu    sync.Mutex
	rows  map[int]*list.Element
	order *list.List // most recent first
	limit int
}

type cached struct {
	i   int
	row []float32
}

func newKernel(m *Model, x []float32, n int, c config) *kernel {
	k := &kernel{m: m, x: x, n: n, rows: map[int]*list.Element{}, order: list.New()}
	k.limit = max(2, c.cacheMB<<20/(4*max(n, 1)))
	k.diag = make([]float64, n)
	f := m.Features
	for i := range n {
		r := x[i*f : (i+1)*f]
		k.diag[i] = m.k(r, r)
	}
	return k
}

// row returns row i of the kernel matrix. It is read only.
func (k *kernel) row(i int) []float32 {
	k.mu.Lock()
	if e, ok := k.rows[i]; ok {
		k.order.MoveToFront(e)
		k.mu.Unlock()
		return e.Value.(*cached).row
	}
	k.mu.Unlock()
	f := k.m.Features
	xi := k.x[i*f : (i+1)*f]
	row := make([]float32, k.n)
	for j := range k.n {
		row[j] = float32(k.m.k(xi, k.x[j*f:(j+1)*f]))
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if e, ok := k.rows[i]; ok {
		return e.Value.(*cached).row
	}
	k.rows[i] = k.order.PushFront(&cached{i, row})
	for k.order.Len() > k.limit {
		last := k.order.Back()
		k.order.Remove(last)
		delete(k.rows, last.Value.(*cached).i)
	}
	return row
}

// problem is libsvm's dual: minimize ½αᵀQα + pᵀα subject to yᵀα = 0 and
// 0 ≤ α ≤ c, where Q[i][j] = y[i]·y[j]·K(i, j).
type problem struct {
	n    int
	y    []int8
	p    []float64
	c    float64
	diag []float64 // K(i, i)
	// row returns kernel row i over the problem's first wrap variables,
	// or over all n when wrap is 0; variable j reads column j % wrap.
	row  func(i int) []float32
	wrap int
}

// solve solves pr by SMO with libsvm's second-order working-set
// selection. It returns α, the offset ρ of the decision Σ y α K - ρ, and
// the steps taken.
func solve(ctx context.Context, pr problem, c config) (alpha []float64, rho float64, iters int, err error) {
	n := pr.n
	alpha = make([]float64, n)
	g := make([]float64, n)
	copy(g, pr.p)
	y := pr.y
	col := func(j int) int {
		if pr.wrap > 0 {
			return j % pr.wrap
		}
		return j
	}
	upper := func(i int) bool { return alpha[i] >= pr.c }
	lower := func(i int) bool { return alpha[i] <= 0 }
	const tau = 1e-12
	for iters = 0; iters < c.iters; iters++ {
		if iters%1000 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, iters, err
			}
		}
		// i maximizes -y·∇ among those that can move up.
		i, gmax := -1, math.Inf(-1)
		for t := range n {
			if y[t] == 1 {
				if !upper(t) && -g[t] >= gmax {
					i, gmax = t, -g[t]
				}
			} else if !lower(t) && g[t] >= gmax {
				i, gmax = t, g[t]
			}
		}
		if i < 0 {
			break
		}
		ki := pr.row(i)
		qii := pr.diag[i]
		// j most decreases the objective of the pair, by the second-order
		// estimate.
		j, gmax2, best := -1, math.Inf(-1), math.Inf(1)
		for t := range n {
			kit := float64(ki[col(t)])
			if y[t] == 1 {
				if lower(t) {
					continue
				}
				diff := gmax + g[t]
				gmax2 = max(gmax2, g[t])
				if diff > 0 {
					quad := qii + pr.diag[t] - 2*kit
					if o := -diff * diff / max(quad, tau); o <= best {
						j, best = t, o
					}
				}
			} else {
				if upper(t) {
					continue
				}
				diff := gmax - g[t]
				gmax2 = max(gmax2, -g[t])
				if diff > 0 {
					quad := qii + pr.diag[t] - 2*kit
					if o := -diff * diff / max(quad, tau); o <= best {
						j, best = t, o
					}
				}
			}
		}
		if gmax+gmax2 < c.tol || j < 0 {
			break
		}
		kj := pr.row(j)
		qij := float64(y[i]) * float64(y[j]) * float64(ki[col(j)])
		ai, aj := alpha[i], alpha[j]
		ci, cj := pr.c, pr.c
		if y[i] != y[j] {
			quad := max(pr.diag[i]+pr.diag[j]+2*qij, tau)
			delta := (-g[i] - g[j]) / quad
			diff := ai - aj
			ai += delta
			aj += delta
			if diff > 0 {
				if aj < 0 {
					aj, ai = 0, diff
				}
			} else if ai < 0 {
				ai, aj = 0, -diff
			}
			if diff > ci-cj {
				if ai > ci {
					ai, aj = ci, ci-diff
				}
			} else if aj > cj {
				aj, ai = cj, cj+diff
			}
		} else {
			quad := max(pr.diag[i]+pr.diag[j]-2*qij, tau)
			delta := (g[i] - g[j]) / quad
			sum := ai + aj
			ai -= delta
			aj += delta
			if sum > ci {
				if ai > ci {
					ai, aj = ci, sum-ci
				}
			} else if aj < 0 {
				aj, ai = 0, sum
			}
			if sum > cj {
				if aj > cj {
					aj, ai = cj, sum-cj
				}
			} else if ai < 0 {
				ai, aj = 0, sum
			}
		}
		dai, daj := ai-alpha[i], aj-alpha[j]
		alpha[i], alpha[j] = ai, aj
		yi, yj := float64(y[i]), float64(y[j])
		for t := range n {
			ct := col(t)
			g[t] += float64(y[t]) * (yi*float64(ki[ct])*dai + yj*float64(kj[ct])*daj)
		}
	}
	// ρ is the mean of y·∇ over the free variables, or the middle of its
	// bounds when none is free.
	ub, lb := math.Inf(1), math.Inf(-1)
	var sum float64
	free := 0
	for t := range n {
		yg := float64(y[t]) * g[t]
		switch {
		case upper(t):
			if y[t] == -1 {
				ub = min(ub, yg)
			} else {
				lb = max(lb, yg)
			}
		case lower(t):
			if y[t] == 1 {
				ub = min(ub, yg)
			} else {
				lb = max(lb, yg)
			}
		default:
			free++
			sum += yg
		}
	}
	if free > 0 {
		rho = sum / float64(free)
	} else {
		rho = (ub + lb) / 2
	}
	if math.IsNaN(rho) || math.IsInf(rho, 0) {
		return nil, 0, iters, fmt.Errorf("svm: the solver found no offset: %w", tensor.ErrShape)
	}
	return alpha, rho, iters, nil
}

// linearDual fits a linear machine with hinge loss and an unpenalized-ish
// bias, a constant feature of 1, by LIBLINEAR's dual coordinate descent.
// It returns the weights, the bias and the passes taken.
func linearDual(ctx context.Context, x []float32, f int, y []int8, c config) ([]float64, float64, int, error) {
	n := len(y)
	w := make([]float64, f+1) // the last is the bias
	alpha := make([]float64, n)
	qd := make([]float64, n)
	for i := range n {
		q := 1.0
		for _, v := range x[i*f : (i+1)*f] {
			q += float64(v) * float64(v)
		}
		qd[i] = q
	}
	r := rand.New(rand.NewPCG(uint64(n), 0x737663))
	order := r.Perm(n)
	passes := min(1000, max(c.iters/max(n, 1), 1))
	it := 0
	for ; it < passes; it++ {
		if err := ctx.Err(); err != nil {
			return nil, 0, it, err
		}
		r.Shuffle(n, func(a, b int) { order[a], order[b] = order[b], order[a] })
		pgMax, pgMin := math.Inf(-1), math.Inf(1)
		for _, i := range order {
			row := x[i*f : (i+1)*f]
			yi := float64(y[i])
			dot := w[f]
			for j, v := range row {
				dot += w[j] * float64(v)
			}
			grad := yi*dot - 1
			pg := grad
			switch {
			case alpha[i] == 0:
				pg = min(grad, 0)
			case alpha[i] == c.c:
				pg = max(grad, 0)
			}
			pgMax, pgMin = max(pgMax, pg), min(pgMin, pg)
			if math.Abs(pg) > 1e-12 {
				old := alpha[i]
				alpha[i] = min(max(old-grad/qd[i], 0), c.c)
				if d := (alpha[i] - old) * yi; d != 0 {
					for j, v := range row {
						w[j] += d * float64(v)
					}
					w[f] += d
				}
			}
		}
		if pgMax-pgMin < c.tol*10 {
			it++
			break
		}
	}
	return w[:f], w[f], it, nil
}
