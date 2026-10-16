package gen

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestMagicDivides checks the multipliers and shifts that kernels divide
// with against division, at the edges of the dividends and divisors and
// at random.
func TestMagicDivides(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	divisors := []int{1, 2, 3, 5, 7, 9, 196, 784, 1 << 20, 1<<20 + 1, math.MaxInt32 - 1, math.MaxInt32}
	for range 1000 {
		divisors = append(divisors, 1+rng.IntN(5000), 1+rng.IntN(math.MaxInt32))
	}
	for _, d := range divisors {
		mul, shift := magic(d)
		xs := []int{0, 1, d - 1, d, d + 1, 2*d - 1, math.MaxInt32, math.MaxInt32 - 1, math.MaxInt32 / d * d, math.MaxInt32/d*d - 1}
		for range 200 {
			xs = append(xs, rng.IntN(math.MaxInt32))
		}
		for _, x := range xs {
			if x < 0 || x > math.MaxInt32 {
				continue
			}
			hi := uint64(uint32(x)<<1) * uint64(mul) >> 32
			if q := int(uint32(hi) >> shift); q != x/d {
				t.Fatalf("%d / %d: got %d, want %d (mul %d, shift %d)", x, d, q, x/d, mul, shift)
			}
		}
	}
}
