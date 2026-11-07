package cluster

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// MaxLinkageRows is the most rows Agglomerative groups: it keeps the
// distance between every pair of rows, 4 bytes a pair, so 10,000 rows
// take 200 MB. Sample a larger table first.
const MaxLinkageRows = 10000

// Merge is one step of a hierarchy: the groups A and B, numbered by
// their first row, joined at Height, Ward's increase in the sum of
// squared distances to the groups' centers, into a group of Size rows.
type Merge struct {
	A, B   int
	Height float64
	Size   int
}

// Linkage is a hierarchical grouping by Ward's method, as scikit-learn's
// AgglomerativeClustering with ward linkage: every row starts as its own
// group and the two groups whose joining costs least are joined, over
// and over, until one is left. Cut gives the groups at any number, so a
// table is grouped once and the number of groups picked afterwards, and
// Merges draw the dendrogram.
type Linkage struct {
	Features int
	Scaler   tabular.Scaler
	// Merges holds the n-1 joins in order, from the closest pair.
	Merges []Merge
	// Labels is the group of each training row at K groups, and Means
	// the groups' centers, which assign new rows to the nearest.
	Labels []int
	Means  *Means
}

// K returns the number of groups Labels and Means hold.
func (l *Linkage) K() int { return l.Means.K() }

// Agglomerative groups the rows of x, features numbers each, into k
// groups by Ward's method, with the nearest-neighbor chain algorithm,
// in time and memory proportional to the square of the rows.
func Agglomerative(ctx context.Context, x []float32, features, k int, opts ...Option) (*Linkage, error) {
	_, n, err := options(x, features, k, opts)
	if err != nil {
		return nil, err
	}
	if n > MaxLinkageRows {
		return nil, fmt.Errorf("cluster: %d rows, at most %d for a hierarchy: %w", n, MaxLinkageRows, tensor.ErrShape)
	}
	sc := tabular.FitScaler(x, features)
	z := sc.Transform(x)
	f := features
	// d holds the Ward distance between groups i < j, the pairs in row
	// order, starting as squared distances between rows.
	d := make([]float32, n*(n-1)/2)
	index := func(i, j int) int {
		if i > j {
			i, j = j, i
		}
		return i*n - i*(i+1)/2 + j - i - 1
	}
	for i := range n {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		ri := z[i*f : (i+1)*f]
		row := d[index(i, i+1):]
		for j := i + 1; j < n; j++ {
			row[j-i-1] = float32(dist(ri, z[j*f:(j+1)*f]))
		}
	}
	at := func(i, j int) float32 { return d[index(i, j)] }
	set := func(i, j int, v float32) { d[index(i, j)] = v }
	size := make([]int, n)
	active := make([]bool, n)
	for i := range n {
		size[i], active[i] = 1, true
	}
	l := &Linkage{Features: f, Scaler: sc, Merges: make([]Merge, 0, n-1)}
	// The chain: each group's nearest is the next, so the last two are
	// each other's nearest and join.
	var chain []int
	remaining := n
	for remaining > 1 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(chain) == 0 {
			for i := range n {
				if active[i] {
					chain = append(chain, i)
					break
				}
			}
		}
		for {
			a := chain[len(chain)-1]
			var prev = -1
			if len(chain) > 1 {
				prev = chain[len(chain)-2]
			}
			// The nearest active group to a, prev winning ties so that
			// the chain closes.
			b, best := prev, float32(0)
			if prev >= 0 {
				best = at(a, prev)
			}
			for j := range n {
				if j == a || !active[j] || j == prev {
					continue
				}
				if v := at(a, j); b < 0 || v < best {
					b, best = j, v
				}
			}
			if b == prev {
				break
			}
			chain = append(chain, b)
		}
		a, b := chain[len(chain)-1], chain[len(chain)-2]
		chain = chain[:len(chain)-2]
		if a > b {
			a, b = b, a
		}
		// Join b into a, the Lance-Williams update of Ward's distances.
		h := at(a, b)
		na, nb := float32(size[a]), float32(size[b])
		for j := range n {
			if !active[j] || j == a || j == b {
				continue
			}
			nj := float32(size[j])
			v := ((na+nj)*at(a, j) + (nb+nj)*at(b, j) - nj*h) / (na + nb + nj)
			set(a, j, v)
		}
		active[b] = false
		size[a] += size[b]
		remaining--
		l.Merges = append(l.Merges, Merge{A: a, B: b, Height: float64(h), Size: size[a]})
	}
	l.Labels = l.Cut(k)
	l.Means = l.means(z, l.Labels, k)
	return l, nil
}

// Cut returns the group of each training row, 0 to k-1 numbered by first
// row, with the hierarchy cut into k groups, or into one when k is more
// than the rows or less than 1.
func (l *Linkage) Cut(k int) []int {
	n := len(l.Merges) + 1
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	find := func(i int) int {
		for parent[i] != i {
			parent[i] = parent[parent[i]]
			i = parent[i]
		}
		return i
	}
	k = min(max(k, 1), n)
	for _, m := range l.Merges[:n-k] {
		parent[find(m.B)] = find(m.A)
	}
	labels := make([]int, n)
	group := map[int]int{}
	for i := range n {
		r := find(i)
		g, ok := group[r]
		if !ok {
			g = len(group)
			group[r] = g
		}
		labels[i] = g
	}
	return labels
}

// means returns the centers of the k groups of the standardized rows z.
func (l *Linkage) means(z []float32, labels []int, k int) *Means {
	f := l.Features
	sums := make([]float64, k*f)
	counts := make([]int, k)
	for i, g := range labels {
		counts[g]++
		for j, v := range z[i*f : (i+1)*f] {
			sums[g*f+j] += float64(v)
		}
	}
	m := &Means{Features: f, Scaler: l.Scaler, Centers: make([]float32, k*f)}
	for g := range k {
		for j := range f {
			m.Centers[g*f+j] = float32(sums[g*f+j] / float64(max(counts[g], 1)))
		}
	}
	assign := make([]int, len(labels))
	m.Inertia = nearestAll(z, m.Centers, f, assign, 1)
	return m
}

// Assign returns the group of each row of x, the nearest center at K
// groups.
func (l *Linkage) Assign(x []float32) []int { return l.Means.Assign(x) }

// Heights returns the heights of the merges, rising, whose largest jumps
// suggest where to cut: a jump after the merge i from the end means i+1
// groups stand well apart.
func (l *Linkage) Heights() []float64 {
	h := make([]float64, len(l.Merges))
	for i, m := range l.Merges {
		h[i] = m.Height
	}
	slices.Sort(h)
	return h
}
