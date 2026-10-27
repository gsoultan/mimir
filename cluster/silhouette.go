package cluster

import (
	"math"
	"math/rand/v2"
)

// silhouetteRows is the most rows Silhouette scores; it samples more.
const silhouetteRows = 5000

// Silhouette returns the mean silhouette of the rows of z, standardized
// rows of f numbers, in the groups labels gives them: for each row, how
// much nearer it is to its own group than to the next nearest, from -1
// to 1. Noise rows are left out, as are rows alone in their group, which
// score 0. Over more than 5,000 rows it scores a sample, drawn with seed.
func Silhouette(z []float32, f int, labels []int, seed uint64) float64 {
	n := len(labels)
	idx := make([]int, 0, min(n, silhouetteRows))
	for i := range n {
		if labels[i] != Noise {
			idx = append(idx, i)
		}
	}
	if len(idx) > silhouetteRows {
		r := rand.New(rand.NewPCG(seed, 0x73696c))
		r.Shuffle(len(idx), func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
		idx = idx[:silhouetteRows]
	}
	groups := 0
	for _, i := range idx {
		groups = max(groups, labels[i]+1)
	}
	if groups < 2 {
		return 0
	}
	var total float64
	sum := make([]float64, groups)
	count := make([]int, groups)
	for _, i := range idx {
		clear(sum)
		clear(count)
		for _, j := range idx {
			if j == i {
				continue
			}
			sum[labels[j]] += math.Sqrt(dist(z[i*f:(i+1)*f], z[j*f:(j+1)*f]))
			count[labels[j]]++
		}
		own := labels[i]
		if count[own] == 0 {
			continue
		}
		a := sum[own] / float64(count[own])
		b := math.Inf(1)
		for g := range groups {
			if g != own && count[g] > 0 {
				b = math.Min(b, sum[g]/float64(count[g]))
			}
		}
		if math.IsInf(b, 1) {
			continue
		}
		total += (b - a) / math.Max(a, b)
	}
	return total / float64(len(idx))
}

// Silhouette returns the silhouette of the rows of x in m's groups.
func (m *Means) Silhouette(x []float32, seed uint64) float64 {
	return Silhouette(m.Scaler.Transform(x), m.Features, m.Assign(x), seed)
}
