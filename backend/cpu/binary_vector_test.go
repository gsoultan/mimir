package cpu

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// TestBinaryVectorRoutines checks every set of vector routines of the
// binary float32 ops against the Go loops, on data with NaN, infinities,
// signed zeros and ties, in lengths that leave a tail for the loops.
func TestBinaryVectorRoutines(t *testing.T) {
	sets := rowRoutineSets()
	if len(sets) == 0 {
		t.Skip("no vector routines on this CPU")
	}
	saved := rowVector
	defer func() { rowVector = saved }()
	rng := rand.New(rand.NewPCG(5, 6))
	special := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 0, float32(math.Copysign(0, -1)), 1, -1}
	fill := func(x []float32) {
		for i := range x {
			if rng.IntN(4) == 0 {
				x[i] = special[rng.IntN(len(special))]
			} else {
				x[i] = rng.Float32()*4 - 2
			}
		}
	}
	// The maximum of -0 and +0 takes either sign.
	same := func(a, b float32) bool { return a == b || a != a && b != b }
	ops := []backend.Op{backend.OpAdd, backend.OpMul, backend.OpMax, backend.OpCmpLT, backend.OpCmpEQ}
	for name, set := range sets {
		for _, n := range []int{1, 15, 16, 17, 64, 100, 333} {
			a, b, want, got := make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n)
			fill(a)
			fill(b)
			for _, s := range []float32{0, 0.5, float32(math.NaN())} {
				for _, op := range ops {
					for kind := range 3 {
						run := func(dst []float32) error {
							switch kind {
							case 0:
								return binaryVV(op, dst, a, b)
							case 1:
								return binaryVS(op, dst, a, s)
							}
							return binarySV(op, dst, s, b)
						}
						rowVector = nil
						if err := run(want); err != nil {
							t.Fatal(err)
						}
						rowVector = set
						if err := run(got); err != nil {
							t.Fatal(err)
						}
						for i := range want {
							if !same(want[i], got[i]) {
								t.Fatalf("%s: %s kind %d n=%d s=%v: element %d = %v, want %v (a=%v, b=%v)", name, op, kind, n, s, i, got[i], want[i], a[i], b[i])
							}
						}
					}
				}
			}
		}
	}
}
