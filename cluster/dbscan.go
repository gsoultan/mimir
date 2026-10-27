package cluster

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Noise is the group DBSCAN gives a row that belongs to none.
const Noise = -1

// Density is a DBSCAN grouping: rows with at least MinRows rows within
// Eps, themselves included, are cores, and a group is the cores that
// reach one another and the rows they reach. It finds groups of any
// shape and their number by itself, and leaves rows in sparse places
// out as Noise.
type Density struct {
	Features int
	Scaler   tabular.Scaler
	Eps      float64
	MinRows  int
	// Labels is the group of each training row, or Noise.
	Labels []int
	// Cores holds the core rows, standardized, and CoreLabels their
	// groups, for Assign.
	Cores      []float32
	CoreLabels []int
	Groups     int
}

// DBSCAN groups the rows of x, features numbers each, by density: eps is
// the reach, in standardized units, and MinRows sets how many rows make
// a core. It finds neighbors by sweeping the rows sorted along the first
// standardized feature, so it suits tables up to some hundred thousand
// rows.
func DBSCAN(ctx context.Context, x []float32, features int, eps float64, opts ...Option) (*Density, error) {
	c, n, err := options(x, features, 1, opts)
	if err != nil {
		return nil, err
	}
	if !(eps > 0) || math.IsInf(eps, 0) {
		return nil, fmt.Errorf("cluster: reach %v: %w", eps, tensor.ErrShape)
	}
	f := features
	sc := tabular.FitScaler(x, f)
	z := sc.Transform(x)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		switch va, vb := z[a*f], z[b*f]; {
		case va < vb:
			return -1
		case va > vb:
			return 1
		}
		return 0
	})
	pos := make([]int, n) // each row's place in order
	for p, i := range order {
		pos[i] = p
	}
	eps2 := eps * eps
	neighbors := func(i int, out []int) []int {
		out = out[:0]
		row := z[i*f : (i+1)*f]
		for _, dir := range []int{-1, 1} {
			for p := pos[i] + dir; p >= 0 && p < n; p += dir {
				j := order[p]
				if math.Abs(float64(z[j*f]-row[0])) > eps {
					break
				}
				if dist(row, z[j*f:(j+1)*f]) <= eps2 {
					out = append(out, j)
				}
			}
		}
		return out
	}
	core := make([]bool, n)
	var nb []int
	for i := range n {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		nb = neighbors(i, nb)
		core[i] = len(nb)+1 >= c.minRows
	}
	d := &Density{Features: f, Scaler: sc, Eps: eps, MinRows: c.minRows, Labels: make([]int, n)}
	for i := range d.Labels {
		d.Labels[i] = Noise
	}
	var stack []int
	for i := range n {
		if !core[i] || d.Labels[i] != Noise {
			continue
		}
		g := d.Groups
		d.Groups++
		d.Labels[i] = g
		stack = append(stack[:0], i)
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			nb = neighbors(p, nb)
			for _, q := range nb {
				if d.Labels[q] != Noise {
					continue
				}
				d.Labels[q] = g
				if core[q] {
					stack = append(stack, q)
				}
			}
		}
	}
	for i, isCore := range core {
		if isCore {
			d.Cores = append(d.Cores, z[i*f:(i+1)*f]...)
			d.CoreLabels = append(d.CoreLabels, d.Labels[i])
		}
	}
	return d, nil
}

// Assign returns the group of the nearest core within Eps of each row of
// x, or Noise.
func (d *Density) Assign(x []float32) []int {
	z := d.Scaler.Transform(x)
	f := d.Features
	out := make([]int, len(z)/f)
	for i := range out {
		out[i] = Noise
		a, dd := nearest(z[i*f:(i+1)*f], d.Cores, f)
		if len(d.CoreLabels) > 0 && dd <= d.Eps*d.Eps {
			out[i] = d.CoreLabels[a]
		}
	}
	return out
}
