package cpu

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

func TestExp2fMatchesMathExp2(t *testing.T) {
	check := func(x float32) bool {
		want := float32(math.Exp2(float64(x)))
		got := exp2f(x)
		switch {
		case want != want:
			return got != got
		case got == want:
			return true
		}
		// One ulp at most, the rounding of a value within 1e-11 of a
		// midpoint.
		return math.Abs(float64(got)-float64(want)) <= float64(math.Nextafter32(want, float32(math.Inf(1)))-want)
	}
	special := []float32{0, float32(math.Copysign(0, -1)), 1, -1, 0.5, -0.5, 127, 127.99999, 128, 129, -126, -149, -149.5, -150, -150.5, -151, 1e-8, -1e-8,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), math.MaxFloat32, -math.MaxFloat32}
	for _, x := range special {
		if !check(x) {
			t.Errorf("exp2f(%v) = %v, want %v", x, exp2f(x), math.Exp2(float64(x)))
		}
	}
	r := rand.New(rand.NewPCG(1, 2))
	differ := 0
	const n = 1 << 20
	for range n {
		x := float32(r.Float64()*300 - 160)
		if !check(x) {
			t.Fatalf("exp2f(%v) = %v, want %v", x, exp2f(x), math.Exp2(float64(x)))
		}
		if exp2f(x) != float32(math.Exp2(float64(x))) {
			differ++
		}
	}
	if differ > n/1000 {
		t.Errorf("%d of %d results differ from math.Exp2 by an ulp", differ, n)
	}
}

func BenchmarkExp2(b *testing.B) {
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i)/256 - 8
	}
	dst := make([]float32, len(x))
	b.SetBytes(int64(4 * len(x)))
	b.Run("exp2f", func(b *testing.B) {
		for b.Loop() {
			unaryLoop(backend.OpExp2, dst, x)
		}
	})
	b.Run("math", func(b *testing.B) {
		for b.Loop() {
			for i, v := range x {
				dst[i] = float32(math.Exp2(float64(v)))
			}
		}
	})
}
