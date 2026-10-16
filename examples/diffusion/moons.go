package main

import (
	"math"
	"math/rand/v2"

	"github.com/gsoultan/mimir/tensor"
)

// moons returns n points, [n, 2], of scikit-learn's two interleaving half
// circles with Gaussian noise of standard deviation noise, scaled into
// [-1, 1] about the origin: a toy data set a DDPM learns in seconds.
func moons(n int, noise float64, seed uint64) *tensor.Tensor {
	r := rand.New(rand.NewPCG(seed, 0))
	xy := make([]float32, 2*n)
	for i := range n {
		a := math.Pi * r.Float64()
		x, y := math.Cos(a), math.Sin(a)
		if i%2 == 1 {
			x, y = 1-x, 0.5-y
		}
		x += noise * r.NormFloat64()
		y += noise * r.NormFloat64()
		// The moons span [-1, 2]×[-0.5, 1]; centre and shrink them.
		xy[2*i], xy[2*i+1] = float32((x-0.5)/1.8), float32((y-0.25)/1.8)
	}
	return tensor.FromSlice(xy, n, 2)
}
