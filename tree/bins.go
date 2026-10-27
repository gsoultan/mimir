package tree

import (
	"math/rand/v2"
	"slices"
	"sort"

	"github.com/gsoultan/mimir/tabular"
)

// binned is a table with each feature sorted into bins: bin 0 holds the
// missing values, and bin b > 0 the values in (edges[b-2], edges[b-1]].
type binned struct {
	n, f  int
	bins  []uint8 // row by row: bins[i*f+j] is row i's bin of feature j
	edges [][]float32
	nb    []int // bins of each feature, the missing bin included
	off   []int // where each feature's bins start in a histogram
	total int   // bins of every feature
}

// sampleRows is how many rows quantiles are taken from.
const sampleRows = 200_000

// bin sorts d's features into at most maxBins bins of values each.
func bin(d tabular.Data, maxBins int, rng *rand.Rand, workers int) *binned {
	n, f := d.Rows(), d.Features
	b := &binned{n: n, f: f, bins: make([]uint8, n*f), edges: make([][]float32, f), nb: make([]int, f), off: make([]int, f)}
	sample := make([]int, 0, min(n, sampleRows))
	if n <= sampleRows {
		for i := range n {
			sample = append(sample, i)
		}
	} else {
		for range sampleRows {
			sample = append(sample, rng.IntN(n))
		}
	}
	parallel(f, workers, 1, func(lo, hi int) {
		vals := make([]float32, 0, len(sample))
		for j := lo; j < hi; j++ {
			vals = vals[:0]
			for _, i := range sample {
				if v := d.X[i*f+j]; v == v {
					vals = append(vals, v)
				}
			}
			b.edges[j] = edges(vals, maxBins)
		}
	})
	parallel(n, workers, 1024, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			for j, v := range d.X[i*f : (i+1)*f] {
				if v == v {
					e := b.edges[j]
					b.bins[i*f+j] = uint8(1 + sort.Search(len(e), func(k int) bool { return v <= e[k] }))
				}
			}
		}
	})
	for j := range f {
		b.nb[j] = len(b.edges[j]) + 2
		b.off[j] = b.total
		b.total += b.nb[j]
	}
	return b
}

// edges returns at most maxBins-1 increasing bin edges for vals, which it
// sorts: one between each two distinct values when there are few enough,
// and the quantiles otherwise.
func edges(vals []float32, maxBins int) []float32 {
	slices.Sort(vals)
	distinct := slices.Compact(slices.Clone(vals))
	if len(distinct) <= maxBins {
		e := make([]float32, 0, max(len(distinct)-1, 0))
		for i := 1; i < len(distinct); i++ {
			e = append(e, distinct[i-1]+(distinct[i]-distinct[i-1])/2)
		}
		return slices.Compact(e)
	}
	e := make([]float32, 0, maxBins-1)
	for q := 1; q < maxBins; q++ {
		e = append(e, vals[q*len(vals)/maxBins])
	}
	e = slices.Compact(e)
	if e[len(e)-1] >= distinct[len(distinct)-1] {
		e = e[:len(e)-1] // the last bin above the top edge would be empty
	}
	return e
}
