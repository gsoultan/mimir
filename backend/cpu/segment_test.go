package cpu

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

func TestSegmentSumsInListOrderOnAnyWorkers(t *testing.T) {
	// Skewed segments, one holding most rows, and rows of random floats,
	// so that splitting the work by segments or summing out of order
	// changes the sums.
	rng := rand.New(rand.NewPCG(9, 10))
	const rows, width, segs = 500, 40, 3000
	offsets := []int32{0}
	var list []int32
	for s := range segs {
		n := rng.IntN(4)
		if s == 7 {
			n = 20000
		}
		for range n {
			list = append(list, int32(rng.IntN(rows)))
		}
		offsets = append(offsets, int32(len(list)))
	}
	src := make([]float32, rows*width)
	for i := range src {
		src[i] = rng.Float32()*2 - 1
	}
	for _, c := range []backend.Combine{backend.CombineSum, backend.CombineMax} {
		want := make([]float32, segs*width)
		for s := range segs {
			for j := offsets[s]; j < offsets[s+1]; j++ {
				for d := range width {
					v := src[int(list[j])*width+d]
					switch {
					case j == offsets[s]:
						want[s*width+d] = v
					case c == backend.CombineSum:
						want[s*width+d] += v
					default:
						want[s*width+d] = max(want[s*width+d], v)
					}
				}
			}
		}
		k := &backend.Kernel{
			Op: backend.OpSegment, DType: backend.Float32, Shape: []int{segs, width},
			In:    []backend.View{backend.Contiguous([]int{rows, width}), backend.Contiguous([]int{segs + 1}), backend.Contiguous([]int{len(list)})},
			Attrs: backend.Attrs{Combine: c},
		}
		for _, workers := range []int{1, 3, 8} {
			got := runOn[float32](t, workers, k, f32buf(src...), i32buf(offsets...), i32buf(list...))
			if !slices.Equal(got, want) {
				t.Fatalf("%s on %d workers differs from combining in list order", c, workers)
			}
		}
	}
}
