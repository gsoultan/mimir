package linear

import (
	"context"
	"math"
)

// lbfgs minimizes loss from w, which it overwrites with the minimum, by
// limited-memory BFGS with a backtracking line search. loss returns the
// loss at its first argument and, when its second is not nil, writes the
// gradient there. It stops after iters passes, or once no gradient
// component exceeds tol. It returns the passes it took.
func lbfgs(ctx context.Context, loss func(w, grad []float64) float64, w []float64, iters int, tol float64) (int, error) {
	const history = 10
	n := len(w)
	g, gNew, wNew, dir := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
	var ss, ys [][]float64
	var rhos []float64
	alpha := make([]float64, history)
	f := loss(w, g)
	dot := func(a, b []float64) float64 {
		var s float64
		for i, v := range a {
			s += v * b[i]
		}
		return s
	}
	for it := range iters {
		if err := canceled(ctx, it); err != nil {
			return it, err
		}
		var gmax float64
		for _, v := range g {
			gmax = math.Max(gmax, math.Abs(v))
		}
		if gmax <= tol {
			return it, nil
		}
		// The two-loop recursion turns the gradient into a step along
		// the inverse Hessian the history estimates.
		for i, v := range g {
			dir[i] = -v
		}
		for j := len(ss) - 1; j >= 0; j-- {
			alpha[j] = rhos[j] * dot(ss[j], dir)
			for i := range dir {
				dir[i] -= alpha[j] * ys[j][i]
			}
		}
		if l := len(ss); l > 0 {
			scale := dot(ss[l-1], ys[l-1]) / dot(ys[l-1], ys[l-1])
			for i := range dir {
				dir[i] *= scale
			}
		} else {
			for i := range dir {
				dir[i] /= math.Max(gmax, 1)
			}
		}
		for j := range ss {
			b := rhos[j] * dot(ys[j], dir)
			for i := range dir {
				dir[i] += (alpha[j] - b) * ss[j][i]
			}
		}
		slope := dot(g, dir)
		if slope >= 0 {
			// Not a descent direction: start the history over.
			for i, v := range g {
				dir[i] = -v / math.Max(gmax, 1)
			}
			slope = dot(g, dir)
			ss, ys, rhos = ss[:0], ys[:0], rhos[:0]
		}
		step, fNew := 1.0, 0.0
		for {
			for i := range w {
				wNew[i] = w[i] + step*dir[i]
			}
			fNew = loss(wNew, gNew)
			if fNew <= f+1e-4*step*slope || step < 1e-12 {
				break
			}
			step /= 2
		}
		s, y := make([]float64, n), make([]float64, n)
		for i := range w {
			s[i], y[i] = wNew[i]-w[i], gNew[i]-g[i]
		}
		if sy := dot(s, y); sy > 1e-12 {
			if len(ss) == history {
				ss, ys, rhos = ss[1:], ys[1:], rhos[1:]
			}
			ss, ys, rhos = append(ss, s), append(ys, y), append(rhos, 1/sy)
		}
		done := math.Abs(f-fNew) <= 1e-12*math.Max(1, math.Abs(f))
		copy(w, wNew)
		copy(g, gNew)
		f = fNew
		if done {
			return it + 1, nil
		}
	}
	return iters, nil
}
