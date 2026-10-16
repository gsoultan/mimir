package metal

import (
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// TestGEMMPacked checks which batched products MPS may run: those whose
// matrices lie rows × ld apart, and no others.
func TestGEMMPacked(t *testing.T) {
	f := backend.Float32
	for name, c := range map[string]struct {
		x    backend.View
		n    int
		want bool
	}{
		"contiguous": {backend.Contiguous([]int{2, 4, 6}), 48, true},
		"padded":     {backend.View{Shape: []int{2, 4, 6}, Strides: []int{48, 8, 1}, Offset: 9}, 4 * 48, false},
	} {
		k := &backend.Kernel{Op: backend.OpMatMul, DType: f, Shape: []int{2, 4, 5},
			In: []backend.View{c.x, backend.Contiguous([]int{2, 6, 5})}}
		g, ok := gen.PlanGEMM(k, []int{c.n, 60})
		if !ok {
			t.Fatalf("%s: not planned", name)
		}
		if packed(g) != c.want {
			t.Errorf("%s: packed %t, want %t", name, packed(g), c.want)
		}
	}
}
