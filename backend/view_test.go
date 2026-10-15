package backend

import (
	"slices"
	"testing"
)

// offsets lists the buffer offset of each element of v in row-major order.
func offsets(v View) []int {
	n := v.Size()
	out := make([]int, 0, n)
	idx := make([]int, len(v.Shape))
	for range n {
		off := v.Offset
		for d, i := range idx {
			off += i * v.Strides[d]
		}
		out = append(out, off)
		for d := len(idx) - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < v.Shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

func TestReshapeKeepsElementOrder(t *testing.T) {
	base := Contiguous([]int{4, 6, 5})
	views := map[string]View{
		"contiguous":       base,
		"shrunk rows":      base.Shrink([][2]int{{1, 3}, {0, 6}, {0, 5}}),
		"shrunk inner":     base.Shrink([][2]int{{0, 4}, {0, 6}, {1, 4}}),
		"permuted":         base.Permute([]int{1, 0, 2}),
		"expanded":         Contiguous([]int{4, 1, 5}).Expand([]int{4, 3, 5}),
		"flipped":          base.Flip([]int{0}),
		"shrunk and split": Contiguous([]int{3, 20}).Shrink([][2]int{{0, 3}, {0, 12}}),
	}
	shapes := [][]int{{120}, {24, 5}, {4, 30}, {2, 2, 6, 5}, {4, 6, 5, 1}, {1, 4, 6, 5}, {8, 3, 5}, {2, 6, 5}, {12, 5}, {2, 30}, {10}, {6, 5}, {36}, {3, 2, 6}, {3, 3, 4}, {3, 12}}
	var ok int
	for name, v := range views {
		want := offsets(v)
		for _, s := range shapes {
			if n, _ := NumElements(s); n != v.Size() {
				continue
			}
			r, done := v.Reshape(s)
			if !done {
				continue
			}
			ok++
			if !slices.Equal(r.Shape, s) || !slices.Equal(offsets(r), want) {
				t.Errorf("%s %v -> %v reads %v, want %v", name, v.Shape, s, offsets(r), want)
			}
		}
	}
	if ok < 15 {
		t.Errorf("only %d reshapes avoided a copy", ok)
	}
	// Splitting a shrunk dimension needs no copy; merging across a gap does.
	sh := Contiguous([]int{3, 20}).Shrink([][2]int{{0, 3}, {0, 12}})
	if _, done := sh.Reshape([]int{3, 3, 4}); !done {
		t.Error("splitting a shrunk dimension copied")
	}
	if _, done := sh.Reshape([]int{36}); done {
		t.Error("merging across a gap did not copy")
	}
}
