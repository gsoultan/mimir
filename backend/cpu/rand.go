package cpu

import (
	"context"
	"math"
	"math/bits"
)

// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2,
// 3", SC 2011) is a counter-based generator: block c under key k is a pure
// function, so any element of a random tensor can be computed independently
// and every backend produces the same numbers for the same seed.
const (
	philoxM0 = 0xD2511F53
	philoxM1 = 0xCD9E8D57
	philoxW0 = 0x9E3779B9
	philoxW1 = 0xBB67AE85
)

func philox(ctr [4]uint32, key [2]uint32) [4]uint32 {
	for round := range 10 {
		if round > 0 {
			key[0] += philoxW0
			key[1] += philoxW1
		}
		hi0, lo0 := bits.Mul32(philoxM0, ctr[0])
		hi1, lo1 := bits.Mul32(philoxM1, ctr[2])
		ctr = [4]uint32{hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0}
	}
	return ctr
}

// philoxBlock returns the four random words of block i under seed.
func philoxBlock(seed uint64, i int) [4]uint32 {
	key := [2]uint32{uint32(seed), uint32(seed >> 32)}
	return philox([4]uint32{uint32(uint64(i)), uint32(uint64(i) >> 32)}, key)
}

// Uniform floats use the top 24 bits of a word: [0, 1) with spacing 2^-24,
// exactly representable in float32.
const twoNeg24 = 1.0 / (1 << 24)

// fillRand writes random float32 values to out. Element i comes from word
// i%4 of block i/4. Uniform values lie in [0, 1). Normal values pair words
// (0, 1) and (2, 3) of a block through Box-Muller; the even element of a pair
// takes the cosine and the odd one the sine.
func (b *Backend) fillRand(ctx context.Context, out []float32, seed uint64, normal bool) error {
	blocks := (len(out) + 3) / 4
	return b.parallelFor(ctx, blocks, elementGrain/4, func(lo, hi int) error {
		for blk := lo; blk < hi; blk++ {
			w := philoxBlock(seed, blk)
			dst := out[blk*4 : min(len(out), blk*4+4)]
			if !normal {
				for l := range dst {
					dst[l] = float32(float64(w[l]>>8) * twoNeg24)
				}
				continue
			}
			for l := range dst {
				p := l &^ 1
				u1 := float64(w[p]>>8+1) * twoNeg24 // (0, 1], so the log is finite
				u2 := float64(w[p+1]>>8) * twoNeg24
				r := math.Sqrt(-2 * math.Log(u1))
				theta := 2 * math.Pi * u2
				if l&1 == 0 {
					dst[l] = float32(r * math.Cos(theta))
				} else {
					dst[l] = float32(r * math.Sin(theta))
				}
			}
		}
		return nil
	})
}
