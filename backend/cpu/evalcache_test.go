package cpu

import (
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// TestKeptEvaluatorRebinds runs one fused kernel again with new buffers,
// a new constant and an input that is its output too, as a replayed
// program does, and checks that each run computes with what it was given.
func TestKeptEvaluatorRebinds(t *testing.T) {
	be := New()
	k := &backend.Kernel{Op: backend.OpFused, DType: backend.Float32, Shape: []int{4},
		In: []backend.View{backend.Contiguous([]int{4})},
		Attrs: backend.Attrs{Program: []backend.Instr{
			{Op: backend.OpLoad, DType: backend.Float32, Args: []int{0}},
			{Op: backend.OpConst, DType: backend.Float32, Value: 2},
			{Op: backend.OpMul, DType: backend.Float32, Args: []int{0, 1}},
		}}}
	check := func(in backend.Buffer, out backend.Buffer, want ...float32) {
		t.Helper()
		if err := be.Run(t.Context(), k, []backend.Buffer{out}, []backend.Buffer{in}); err != nil {
			t.Fatal(err)
		}
		if got := out.(*buffer[float32]).data; !slices.Equal(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	}
	out := f32buf(0, 0, 0, 0)
	check(f32buf(1, 2, 3, 4), out, 2, 4, 6, 8)
	check(f32buf(5, 6, 7, 8), f32buf(0, 0, 0, 0), 10, 12, 14, 16)
	k.Attrs.Program[1].Value = 3
	check(f32buf(1, 1, 1, 1), f32buf(0, 0, 0, 0), 3, 3, 3, 3)
	check(out, out, 6, 12, 18, 24) // in place
	k.Attrs.Program[1].Value = 2
	check(f32buf(1, 2, 3, 4), f32buf(0, 0, 0, 0), 2, 4, 6, 8)
}
