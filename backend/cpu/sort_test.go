package cpu

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// refArgsort returns the positions of the first k elements of the stable
// sort of each line of x, read through v, along axis: NaN last ascending
// and first descending.
func refArgsort(x []float64, v backend.View, axis, k int, descending bool) []int32 {
	shape := slices.Clone(v.Shape)
	n := shape[axis]
	shape[axis] = k
	size, _ := backend.NumElements(shape)
	out := make([]int32, size)
	strides := backend.ContiguousStrides(shape)
	lines := slices.Clone(v.Shape)
	lines[axis] = 1
	eachIndex(lines, func(idx []int) {
		pos := make([]int, n)
		vals := make([]float64, n)
		at := slices.Clone(idx)
		for j := range n {
			pos[j] = j
			at[axis] = j
			vals[j] = x[offsetOf(v, at)]
		}
		slices.SortStableFunc(pos, func(a, b int) int {
			x, y := vals[a], vals[b]
			c := cmp.Compare(x, y) // NaN is less than everything
			if math.IsNaN(x) != math.IsNaN(y) {
				c = -c
			}
			if descending {
				c = -c
			}
			return c
		})
		o := 0
		for d, i := range idx {
			o += i * strides[d]
		}
		for j := range k {
			out[o+j*strides[axis]] = int32(pos[j])
		}
	})
	return out
}

// sortInput returns x as a buffer of dt and the values the buffer holds.
func sortInput(x []float64, dt backend.DType) (backend.Buffer, []float64) {
	vals := slices.Clone(x)
	switch dt {
	case backend.Float32:
		v := make([]float32, len(x))
		for i, e := range x {
			v[i] = float32(e)
		}
		return f32buf(v...), vals
	case backend.Int32:
		v := make([]int32, len(x))
		for i, e := range x {
			if math.IsNaN(e) {
				e = math.MaxInt32
				vals[i] = e
			}
			v[i] = int32(e)
		}
		return i32buf(v...), vals
	default:
		v := make([]bf16, len(x))
		for i, e := range x {
			v[i] = bf16(f32ToBF16(float32(e)))
		}
		return &buffer[bf16]{data: v}, vals
	}
}

func TestArgsortMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, s := range []struct {
		shape []int
		axis  int
		ks    []int // the axis' length if nil
	}{
		{[]int{7}, 0, []int{0, 1, 3, 7}},
		{[]int{3, 5, 4}, 1, nil},
		{[]int{3, 5, 4}, 2, []int{1, 4}},
		{[]int{4, 300}, 1, []int{5, 100, 300}},
		{[]int{300, 3}, 0, []int{7, 300}},
		{[]int{2, 70000}, 1, []int{10, 70000}}, // long lines that the workers share
	} {
		ks := s.ks
		if ks == nil {
			ks = []int{s.shape[s.axis]}
		}
		views := stridedViews(s.shape)
		if s.shape[len(s.shape)-1] > 1000 {
			views = views[:1]
		}
		for _, sv := range views {
			x := make([]float64, sv.size)
			for i := range x {
				x[i] = float64(rng.IntN(20) - 10) // ties
				switch rng.IntN(50) {
				case 0:
					x[i] = math.NaN()
				case 1:
					x[i] = math.Copysign(0, -1)
				}
			}
			for _, dt := range []backend.DType{backend.Float32, backend.Int32, backend.BFloat16} {
				buf, vals := sortInput(x, dt)
				for _, k := range ks {
					for _, desc := range []bool{false, true} {
						want := refArgsort(vals, sv.view, s.axis, k, desc)
						shape := slices.Clone(s.shape)
						shape[s.axis] = k
						kern := &backend.Kernel{Op: backend.OpArgsort, DType: backend.Int32, Shape: shape, In: []backend.View{sv.view},
							Attrs: backend.Attrs{Axis: s.axis, Descending: desc}}
						for _, w := range []int{1, 4} {
							if got := runOn[int32](t, w, kern, buf); !slices.Equal(got, want) {
								t.Fatalf("%v %s %s axis %d k %d descending %v workers %d: got %v, want %v",
									s.shape, sv.name, dt, s.axis, k, desc, w, head(got), head(want))
							}
						}
					}
				}
			}
		}
	}
}

func head[T any](s []T) string {
	if len(s) > 20 {
		return fmt.Sprint(s[:20], "...")
	}
	return fmt.Sprint(s)
}
