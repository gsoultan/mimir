package cpu

import (
	"math"
	"testing"
)

// TestRowRoutinesWriteOnlyTheirRows calls every vector row routine of
// every instruction set this machine has, for one to four whole vectors
// (the routines take whole vectors only; vectorPart and the Go loops
// around them handle the tails), with the rows placed between zones of a
// marker value, and checks that nothing outside the rows changed. The
// routines take raw pointers and lengths, so Go's bounds checks do not
// reach them.
func TestRowRoutinesWriteOnlyTheirRows(t *testing.T) {
	const guard = 32
	const marker = 0x7fc0dead // a NaN with a payload
	sets := rowRoutineSets()
	if len(sets) == 0 {
		t.Skip("no vector row routines on this machine")
	}
	for name, r := range sets {
		t.Run(name, func(t *testing.T) {
			for n := r.lanes; n <= 4*r.lanes; n += r.lanes {
				// Each buffer is n elements between two guards; the
				// routines are given the elements and told n.
				fresh := func(fill func(i int) float32) []float32 {
					b := make([]float32, guard+n+guard)
					for i := range b {
						b[i] = math.Float32frombits(marker)
					}
					for i := range n {
						b[guard+i] = fill(i)
					}
					return b
				}
				row := func(b []float32) *float32 { return &b[guard] }
				check := func(what string, bufs ...[]float32) {
					t.Helper()
					for _, b := range bufs {
						for i := range guard {
							if math.Float32bits(b[i]) != marker {
								t.Fatalf("%s, n=%d: wrote %d elements before the row", what, n, guard-i)
							}
							if math.Float32bits(b[guard+n+i]) != marker {
								t.Fatalf("%s, n=%d: wrote %d elements past the row", what, n, i+1)
							}
						}
					}
				}
				val := func(i int) float32 { return float32(i%7) - 3 }
				pos := func(i int) float32 { return float32(i%5) + 0.5 }

				x := fresh(val)
				r.max(row(x), n)
				check("max", x)
				x = fresh(val)
				r.expShift(row(x), n, 1)
				check("expShift", x)
				x = fresh(val)
				r.scale(row(x), n, 2)
				check("scale", x)
				p, dp := fresh(val), fresh(pos)
				r.softmaxGrad(row(p), row(dp), n, 0.5)
				check("softmaxGrad", p, dp)
				dst, src := fresh(val), fresh(val)
				r.exp2(row(dst), row(src), n)
				check("exp2", dst, src)
				dst, a, b := fresh(val), fresh(val), fresh(pos)
				r.addVV(row(dst), row(a), row(b), n)
				check("addVV", dst, a, b)
				dst, a, b = fresh(val), fresh(val), fresh(pos)
				r.mulVV(row(dst), row(a), row(b), n)
				check("mulVV", dst, a, b)
				dst, a = fresh(val), fresh(val)
				r.addVS(row(dst), row(a), 3, n)
				check("addVS", dst, a)
				dst, a = fresh(val), fresh(val)
				r.mulVS(row(dst), row(a), 3, n)
				check("mulVS", dst, a)
				dst, a = fresh(val), fresh(pos)
				r.recip(row(dst), row(a), n)
				check("recip", dst, a)
				dst, a = fresh(val), fresh(pos)
				r.sqrt(row(dst), row(a), n)
				check("sqrt", dst, a)

				// dot4 reads four rows of y, stride apart, and writes
				// four sums.
				{
					stride := n + 3
					y := make([]float32, guard+3*stride+n+guard)
					for i := range y {
						y[i] = math.Float32frombits(marker)
					}
					for row := range 4 {
						for i := range n {
							y[guard+row*stride+i] = val(i + row)
						}
					}
					out := make([]float32, guard+4+guard)
					for i := range out {
						out[i] = math.Float32frombits(marker)
					}
					x = fresh(val)
					r.dot4(row(x), &y[guard], stride, n, &out[guard])
					check("dot4", x)
					for i := range guard {
						if math.Float32bits(out[i]) != marker || math.Float32bits(out[guard+4+i]) != marker {
							t.Fatalf("dot4, n=%d: wrote outside its four sums", n)
						}
					}
					for i := range guard {
						if math.Float32bits(y[i]) != marker || math.Float32bits(y[len(y)-1-i]) != marker {
							t.Fatalf("dot4, n=%d: wrote outside y", n)
						}
					}
					for row := range 4 {
						var want float32
						for i := range n {
							want += val(i) * val(i+row)
						}
						if got := out[guard+row]; math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-4 {
							t.Fatalf("dot4, n=%d: row %d = %v, want %v", n, row, got, want)
						}
					}
				}
			}
		})
	}
}
