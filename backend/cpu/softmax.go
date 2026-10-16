package cpu

import "math"

// The attention kernels apply softmax to rows of scores a block at a time.
// These row operations run on vector routines where the CPU has them; the
// routines take a whole number of vectors, and the Go code here handles
// the rest of each row, computing what the routines compute.

// rowVector holds the vector routines this CPU supports, or nil for none.
// The architecture-specific files set it at init.
var rowVector *rowRoutines

type rowRoutines struct {
	lanes       int
	max         func(x *float32, n int) float32
	expShift    func(x *float32, n int, shift float32) float32
	scale       func(x *float32, n int, a float32)
	softmaxGrad func(p, dp *float32, n int, d float32)
	dot         func(x, y *float32, n int) float32
	exp2        func(dst, src *float32, n int)
}

// vectorPart returns how many leading elements of a row of n the vector
// routines handle.
func vectorPart(n int) int {
	if rowVector == nil {
		return 0
	}
	return n &^ (rowVector.lanes - 1)
}

// rowMax returns the largest element of x, which is not empty.
func rowMax(x []float32) float32 {
	m := float32(math.Inf(-1))
	n := vectorPart(len(x))
	if n > 0 {
		m = rowVector.max(&x[0], n)
	}
	for _, v := range x[n:] {
		m = max(m, v)
	}
	return m
}

// expShift sets each element v of x to e^(v-shift) and returns their sum.
func expShift(x []float32, shift float32) float32 {
	var sum float32
	n := vectorPart(len(x))
	if n > 0 {
		sum = rowVector.expShift(&x[0], n, shift)
	}
	for i := n; i < len(x); i++ {
		e := expf(x[i] - shift)
		x[i] = e
		sum += e
	}
	return sum
}

// exp2Floats sets each element of dst to 2 to the power of the element of
// a, which may be dst, within 2 ulp of math.Exp2, with the vector routines,
// which must exist. They compute arguments in [-126, 127], whose powers
// are normal floats, and the rest, and NaNs, take math.Exp2. Every
// element takes the same path wherever it is in a, so the result does
// not depend on how a kernel splits its elements.
func exp2Floats(dst, a []float32) {
	var buf [256]float32
	for len(a) > 0 {
		c := min(len(a), len(buf))
		src := buf[:c]
		copy(src, a[:c])
		n := vectorPart(c)
		if n > 0 {
			rowVector.exp2(&dst[0], &src[0], n)
		}
		if n < c {
			// The rest, padded to a vector.
			var tail [16]float32
			copy(tail[:], src[n:])
			rowVector.exp2(&tail[0], &tail[0], rowVector.lanes)
			copy(dst[n:c], tail[:])
		}
		for i, x := range src {
			if !(x >= -126 && x <= 127) {
				dst[i] = float32(math.Exp2(float64(x)))
			}
		}
		dst, a = dst[c:], a[c:]
	}
}

// scaleRow multiplies every element of x by a.
func scaleRow(x []float32, a float32) {
	n := vectorPart(len(x))
	if n > 0 {
		rowVector.scale(&x[0], n, a)
	}
	for i := n; i < len(x); i++ {
		x[i] *= a
	}
}

// softmaxGrad sets each element p[j] of p to p[j]·(dp[j]-d), where dp has
// p's length.
func softmaxGrad(p, dp []float32, d float32) {
	dp = dp[:len(p)]
	n := vectorPart(len(p))
	if n > 0 {
		rowVector.softmaxGrad(&p[0], &dp[0], n, d)
	}
	for i := n; i < len(p); i++ {
		p[i] *= dp[i] - d
	}
}

// expf is e^x in float32, as the vector routines compute it: Cephes'
// expf, within 2 ulp. Arguments below the logarithm of the smallest
// normal float32 give 0 and arguments above 88 give e^88, which no
// softmax reaches: its arguments are at most 0.
func expf(x float32) float32 {
	const lowest, highest = -87.33654, 88.0
	if !(x >= lowest) { // and NaN
		if x != x {
			return x
		}
		return 0
	}
	x = min(x, highest)
	n := float32(math.RoundToEven(float64(x * math.Log2E)))
	r := x - n*0.693359375
	r -= n * -2.12194440e-4
	p := float32(1.9875691500e-4)
	p = p*r + 1.3981999507e-3
	p = p*r + 8.3334519073e-3
	p = p*r + 4.1665795894e-2
	p = p*r + 1.6666665459e-1
	p = p*r + 5.0000001201e-1
	y := p*(r*r) + r + 1
	return y * math.Float32frombits(uint32(int32(n)+127)<<23)
}
