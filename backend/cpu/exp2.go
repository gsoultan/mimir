package cpu

import "math"

// exp2Table holds 2^(j/64) for j < 64.
var exp2Table = func() (t [64]float64) {
	for j := range t {
		t[j] = math.Exp2(float64(j) / 64)
	}
	return t
}()

// exp2Coef are the Taylor coefficients ln(2)^k / k! of 2^f for k = 1..4.
var exp2Coef = [...]float64{
	6.931471805599453e-01,
	2.402265069591007e-01,
	5.5504108664821576e-02,
	9.618129107628477e-03,
}

// exp2f returns 2^x rounded to float32. It splits x into k/64, k an
// integer, and f in [-1/128, 1/128], and computes 2^(k/64) from a table
// and the exponent bits and 2^f from a polynomial, both in float64: the
// relative error, below 1e-13, rarely moves the rounding. That is several
// times faster than math.Exp2, with which it agrees to the float32 ulp.
func exp2f(x float32) float32 {
	switch {
	case x != x:
		return x
	case x >= 129:
		return float32(math.Inf(1))
	case x <= -151:
		return 0
	}
	// Adding 1.5·2⁵² rounds 64x, exact in float64, to the nearest integer
	// k, which then fills the low bits of the sum: no call to math.Floor.
	const shift = 0x1.8p52
	t := float64(x)*64 + shift
	f := float64(x) - (t-shift)/64
	k := int64(math.Float64bits(t)) - int64(math.Float64bits(shift))
	c := &exp2Coef
	p := 1 + f*(c[0]+f*(c[1]+f*(c[2]+f*c[3])))
	return float32(exp2Table[k&63] * p * math.Float64frombits(uint64(k>>6+1023)<<52))
}
