package linear

import (
	"context"
	"math"

	"github.com/gsoultan/mimir/tabular"
)

// Regression fits y ≈ w·x + b by least squares, minimizing
//
//	1/(2n) Σ (y - w·x - b)² + l1 Σ|w| + l2/2 Σ w²
//
// over the standardized features by cyclic coordinate descent, as
// scikit-learn's ElasticNet does. With no penalty it is ordinary least
// squares; with L2 alone, ridge regression; with L1 alone, the lasso.
func Regression(ctx context.Context, d tabular.Data, opts ...Option) (*Model, error) {
	c, err := options(tabular.Regression, d, 0, opts)
	if err != nil {
		return nil, err
	}
	s := standardize(d)
	n, f := s.n, s.f
	var ymean float64
	for _, y := range d.Y {
		ymean += float64(y)
	}
	ymean /= float64(n)
	r := make([]float64, n) // the residual y - ymean - Σ w x
	for i, y := range d.Y {
		r[i] = float64(y) - ymean
	}
	w := make([]float64, f)
	it := 0
	for ; it < c.iters; it++ {
		if err := canceled(ctx, it); err != nil {
			return nil, err
		}
		var change float64
		for j := range f {
			if s.std[j] == 0 {
				continue
			}
			col := s.cols[j*n : (j+1)*n]
			// Each column has mean 0 and mean square 1, so the best w_j
			// alone is the mean of x_j times the residual without it.
			var rho float64
			for i, x := range col {
				rho += float64(x) * r[i]
			}
			rho = rho/float64(n) + w[j]
			wj := soft(rho, c.l1) / (1 + c.l2)
			if dw := wj - w[j]; dw != 0 {
				for i, x := range col {
					r[i] -= dw * float64(x)
				}
				change = math.Max(change, math.Abs(dw))
				w[j] = wj
			}
		}
		if change < c.tol {
			it++
			break
		}
	}
	m := s.model(tabular.Regression, 1, w, []float64{ymean})
	m.Link, m.Iterations = "identity", it
	return m, nil
}

// Logistic fits logistic regression, minimizing the mean log loss plus
//
//	l1 Σ|w| + l2/2 Σ w²
//
// over the standardized features: by L-BFGS without an L1 penalty, and by
// accelerated proximal gradient descent (FISTA) with a backtracking line
// search with one. Two classes share one set of
// weights, the log-odds of class 1; more are fit jointly by softmax, as
// scikit-learn's multinomial LogisticRegression.
func Logistic(ctx context.Context, d tabular.Data, opts ...Option) (*Model, error) {
	c, err := options(tabular.Classification, d, 1/float64(max(d.Rows(), 1)), opts)
	if err != nil {
		return nil, err
	}
	classes := max(c.classes, d.Classes())
	k := 1
	if classes > 2 {
		k = classes
	}
	s := standardize(d)
	n, f := s.n, s.f
	y := make([]int, n)
	for i, v := range d.Y {
		y[i] = int(v)
	}
	// The loss reads row by row.
	xs := make([]float32, n*f)
	for j := range f {
		for i, v := range s.cols[j*n : (j+1)*n] {
			xs[i*f+j] = v
		}
	}
	// The parameters are k rows of f weights, then k biases.
	size := k*f + k
	loss := func(p []float64, grad []float64) float64 {
		parts := make([]float64, c.workers)
		var grads [][]float64
		if grad != nil {
			grads = make([][]float64, c.workers)
		}
		rows(n, c.workers, func(wk, lo, hi int) {
			var g []float64
			if grad != nil {
				g = make([]float64, size)
				grads[wk] = g
			}
			z := make([]float64, k)
			var l float64
			for i := lo; i < hi; i++ {
				for o := range k {
					z[o] = p[k*f+o]
				}
				for j := range f {
					x := float64(xs[i*f+j])
					if x == 0 {
						continue
					}
					for o := range k {
						z[o] += p[o*f+j] * x
					}
				}
				if k == 1 {
					// log(1 + e^z) - y z, computed stably.
					t := float64(y[i])
					l += math.Max(z[0], 0) + math.Log1p(math.Exp(-math.Abs(z[0]))) - t*z[0]
					z[0] = 1/(1+math.Exp(-z[0])) - t
				} else {
					hi := z[0]
					for _, v := range z {
						hi = math.Max(hi, v)
					}
					var sum float64
					for _, v := range z {
						sum += math.Exp(v - hi)
					}
					l += hi + math.Log(sum) - z[y[i]]
					for o := range z {
						z[o] = math.Exp(z[o]-hi) / sum
					}
					z[y[i]]--
				}
				if g == nil {
					continue
				}
				for o, e := range z {
					g[k*f+o] += e
				}
				for j := range f {
					x := float64(xs[i*f+j])
					if x == 0 {
						continue
					}
					for o, e := range z {
						g[o*f+j] += e * x
					}
				}
			}
			parts[wk] = l
		})
		var l float64
		for _, v := range parts {
			l += v
		}
		l /= float64(n)
		if grad != nil {
			clear(grad)
			for _, g := range grads {
				for i, v := range g {
					grad[i] += v
				}
			}
			for i := range grad {
				grad[i] /= float64(n)
			}
		}
		// The L2 penalty is smooth, so it joins the loss; L1 is the
		// proximal step's.
		for i := range k * f {
			l += c.l2 / 2 * p[i] * p[i]
			if grad != nil {
				grad[i] += c.l2 * p[i]
			}
		}
		return l
	}

	w, prev, v := make([]float64, size), make([]float64, size), make([]float64, size)
	grad, next := make([]float64, size), make([]float64, size)
	if k == 1 {
		// Start from the bias that fits the classes' share.
		var p float64
		for _, c := range y {
			p += float64(c)
		}
		p = math.Min(math.Max(p/float64(n), 1e-6), 1-1e-6)
		w[f] = math.Log(p / (1 - p))
		copy(v, w)
	}
	it := 0
	if c.l1 == 0 {
		// Without an L1 penalty the loss is smooth: L-BFGS gets there in
		// far fewer passes.
		if it, err = lbfgs(ctx, loss, w, c.iters, c.tol); err != nil {
			return nil, err
		}
	}
	lip, t := 1.0, 1.0
	for ; c.l1 > 0 && it < c.iters; it++ {
		if err := canceled(ctx, it); err != nil {
			return nil, err
		}
		lv := loss(v, grad)
		// Backtrack until the quadratic model at v bounds the loss.
		for {
			for i := range next {
				next[i] = v[i] - grad[i]/lip
				if i < k*f {
					next[i] = soft(next[i], c.l1/lip)
				}
			}
			q := lv
			for i := range next {
				dlt := next[i] - v[i]
				q += grad[i]*dlt + lip/2*dlt*dlt
			}
			if loss(next, nil) <= q+1e-12 || lip > 1e12 {
				break
			}
			lip *= 2
		}
		var change float64
		for i := range w {
			change = math.Max(change, math.Abs(next[i]-w[i]))
		}
		copy(prev, w)
		copy(w, next)
		tn := (1 + math.Sqrt(1+4*t*t)) / 2
		for i := range v {
			v[i] = w[i] + (t-1)/tn*(w[i]-prev[i])
		}
		t = tn
		if change < c.tol {
			it++
			break
		}
	}
	m := s.model(tabular.Classification, k, w[:k*f], w[k*f:])
	m.Classes, m.Iterations, m.Link = classes, it, "sigmoid"
	if k > 1 {
		m.Link = "softmax"
	}
	return m, nil
}
