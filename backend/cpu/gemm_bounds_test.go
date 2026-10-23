package cpu

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestGEMMKernelsWriteOnlyTheirOutput runs every kernel over many random
// shapes, batches and sums with the output placed between two zones of a
// marker value, and checks that no kernel wrote past either end and that
// every element is the reference product. The assembly tiles handle the
// ragged edges of the output themselves, where Go's bounds checks do not
// reach, so an off-by-one there is a silent memory corruption without this.
func TestGEMMKernelsWriteOnlyTheirOutput(t *testing.T) {
	be := New()
	const guard = 64
	marker := float32(math.Float32frombits(0x7fc0dead)) // a NaN with a payload
	isMarker := func(v float32) bool { return math.Float32bits(v) == 0x7fc0dead }
	layouts := []string{"row-major", "column-major", "padded", "strided", "broadcast rows"}
	for _, g := range gemmKernels {
		t.Run(g.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(11, 13))
			for trial := 0; trial < 120; trial++ {
				// Sizes cluster around the tile and block edges, where the
				// tail handling lives, and range wider now and then.
				edge := func(unit int) int {
					switch rng.IntN(4) {
					case 0:
						return max(1, unit*rng.IntN(3)+rng.IntN(3)-1)
					case 1:
						return 1 + rng.IntN(unit)
					default:
						return 1 + rng.IntN(3*unit+7)
					}
				}
				m, n, k := edge(g.mr), edge(g.nr), edge(max(g.kc/8, 1))
				if trial%17 == 0 {
					k = g.kc + rng.IntN(g.kc) // more than one k block
				}
				if trial%23 == 0 {
					m = g.mc + rng.IntN(g.mr*2) // more than one m block
				}
				sums, outputs := 1+rng.IntN(3), 1+rng.IntN(4)
				if trial%5 == 0 {
					sums, outputs = 1, 1+rng.IntN(2*be.workers+1)
				}
				la, lb := layouts[rng.IntN(len(layouts))], layouts[rng.IntN(len(layouts))]
				if la == "broadcast rows" {
					la = "row-major"
				}
				var as, bs []operand
				var offA, offB []int
				for range outputs * sums {
					a, b := layout(rng, la, m, k), layout(rng, lb, k, n)
					as, bs = append(as, a), append(bs, b)
				}
				// Every term reads from one buffer: the terms' data are
				// laid end to end, with offsets into the whole.
				a, b := as[0], bs[0]
				a.data, b.data = nil, nil
				for i := range as {
					offA, offB = append(offA, len(a.data)+as[i].off), append(offB, len(b.data)+bs[i].off)
					a.data, b.data = append(a.data, as[i].data...), append(b.data, bs[i].data...)
				}
				whole := make([]float32, guard+outputs*m*n+guard)
				for i := range whole {
					whole[i] = marker
				}
				out := whole[guard : guard+outputs*m*n]
				if err := be.sgemm(t.Context(), g, out, a, offA, b, offB, m, n, k, sums, nil); err != nil {
					t.Fatal(err)
				}
				for i, v := range whole[:guard] {
					if !isMarker(v) {
						t.Fatalf("%dx%dx%d, %d sums, %d outputs, a %s, b %s: wrote %d elements before the output", m, n, k, sums, outputs, la, lb, guard-i)
					}
				}
				for i, v := range whole[guard+outputs*m*n:] {
					if !isMarker(v) {
						t.Fatalf("%dx%dx%d, %d sums, %d outputs, a %s, b %s: wrote %d elements past the output", m, n, k, sums, outputs, la, lb, i+1)
					}
				}
				for o := range outputs {
					want := make([]float64, m*n)
					scale := make([]float64, m*n)
					for s := range sums {
						ta, tb := as[o*sums+s], bs[o*sums+s]
						w, sc := refGEMM(ta, tb, m, n, k)
						for e := range want {
							want[e] += w[e]
							scale[e] += sc[e]
						}
					}
					for e, got := range out[o*m*n:][:m*n] {
						if diff := math.Abs(float64(got) - want[e]); !(diff <= 1e-5*scale[e]+1e-7) {
							t.Fatalf("%dx%dx%d, %d sums, %d outputs, a %s, b %s: output %d element %d = %v, want %v", m, n, k, sums, outputs, la, lb, o, e, got, want[e])
						}
					}
				}
			}
		})
	}
}
