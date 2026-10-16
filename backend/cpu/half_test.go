package cpu

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// refF16Value returns the value of the binary16 h as IEEE 754 defines it:
// a sign, a biased exponent of 5 bits and a fraction of 10 bits, with
// subnormals below exponent 1. Exponent 31 is infinity, or a NaN with a
// non-zero fraction.
func refF16Value(h uint16) float64 {
	sign := 1.0
	if h&0x8000 != 0 {
		sign = -1
	}
	exp, frac := int(h>>10)&0x1f, float64(h&0x3ff)
	switch exp {
	case 0:
		return sign * math.Ldexp(frac, -24)
	case 31:
		if frac != 0 {
			return math.NaN()
		}
		return sign * math.Inf(1)
	default:
		return sign * math.Ldexp(1024+frac, exp-25)
	}
}

// f16Table lists the magnitudes of every non-negative binary16 in order of
// their encoding, so that encoding order is value order. Infinity stands
// at 2^16: the value the format's next element would have, which decides
// whether rounding overflows.
var f16Table = func() []float64 {
	t := make([]float64, 0x7c01)
	for h := range t {
		t[h] = refF16Value(uint16(h))
	}
	t[0x7c00] = math.Ldexp(1, 16)
	return t
}()

// refF32ToF16 rounds f to the nearest binary16 by searching the table of
// their values: the nearer of the two elements around |f|, the one with an
// even encoding at a tie.
func refF32ToF16(t *testing.T, f float32) uint16 {
	t.Helper()
	if f != f {
		t.Fatal("refF32ToF16 of a NaN")
	}
	sign := uint16(0)
	x := float64(f)
	if math.Signbit(x) {
		sign, x = 0x8000, -x
	}
	if x >= f16Table[0x7c00] {
		return sign | 0x7c00
	}
	hi := sort.SearchFloat64s(f16Table, x) // the first element >= x
	if f16Table[hi] == x {
		return sign | uint16(hi)
	}
	lo := hi - 1
	d1, d2 := x-f16Table[lo], f16Table[hi]-x
	switch {
	case d1 < d2:
		return sign | uint16(lo)
	case d1 > d2:
		return sign | uint16(hi)
	case lo&1 == 0:
		return sign | uint16(lo)
	default:
		return sign | uint16(hi)
	}
}

// refF32ToBF16 rounds f to the nearest bfloat16: the nearer of the top 16
// bits of f and the element after it, the even one at a tie. Infinity
// stands at 2^128 as the next element's value.
func refF32ToBF16(t *testing.T, f float32) uint16 {
	t.Helper()
	if f != f {
		t.Fatal("refF32ToBF16 of a NaN")
	}
	value := func(h uint32) float64 {
		if h&0x7fff == 0x7f80 {
			return math.Ldexp(1, 128)
		}
		return math.Abs(float64(math.Float32frombits(h << 16)))
	}
	b := math.Float32bits(f)
	lo := b >> 16
	x := math.Abs(float64(f))
	if value(lo) == x {
		return uint16(lo)
	}
	d1, d2 := x-value(lo), value(lo+1)-x
	switch {
	case d1 < d2:
		return uint16(lo)
	case d1 > d2:
		return uint16(lo + 1)
	case lo&1 == 0:
		return uint16(lo)
	default:
		return uint16(lo + 1)
	}
}

func isNaN16(h uint16) bool { return h&0x7c00 == 0x7c00 && h&0x3ff != 0 }
func isNaNBF(h uint16) bool { return h&0x7f80 == 0x7f80 && h&0x7f != 0 }

func TestHalfValuesRoundTrip(t *testing.T) {
	for h := range 1 << 16 {
		h := uint16(h)
		f := f16ToF32(h)
		if want := refF16Value(h); !(f != f && want != want) && float64(f) != want {
			t.Fatalf("f16ToF32(%#04x) = %v, want %v", h, f, want)
		}
		got, want := f32ToF16(f), h
		if isNaN16(h) {
			want |= 0x0200 // quiet
		}
		if got != want {
			t.Errorf("f32ToF16(f16ToF32(%#04x)) = %#04x, want %#04x", h, got, want)
		}
		got, want = f32ToBF16(bf16ToF32(h)), h
		if isNaNBF(h) {
			want |= 0x0040
		}
		if got != want {
			t.Errorf("f32ToBF16(bf16ToF32(%#04x)) = %#04x, want %#04x", h, got, want)
		}
	}
}

// halfInputs returns float32 values that exercise rounding: every binary16
// and bfloat16 value, the midpoints between neighbours and the float32s on
// either side of those midpoints, the edges of the formats and random bit
// patterns.
func halfInputs() []float32 {
	var in []float32
	add := func(f float32) {
		if f == f {
			in = append(in, f)
		}
	}
	around := func(f float32) {
		add(f)
		add(math.Nextafter32(f, math.MaxFloat32))
		add(math.Nextafter32(f, -math.MaxFloat32))
	}
	for h := range 1 << 16 {
		a := f16ToF32(uint16(h))
		around(a)
		if h&0x7fff < 0x7c00 {
			around(float32((float64(a) + float64(f16ToF32(uint16(h+1)))) / 2))
		}
		b := bf16ToF32(uint16(h))
		around(b)
		if h&0x7fff < 0x7f80 {
			around(float32((float64(b) + float64(bf16ToF32(uint16(h+1)))) / 2))
		}
	}
	for _, f := range []float32{65504, 65520, 65536, math.MaxFloat32, math.SmallestNonzeroFloat32, 1e-8, 1e-5, 0.1, 1.0 / 3} {
		around(f)
		around(-f)
	}
	for _, e := range []int{-30, -26, -25, -24, -15, -14, -13, 0, 15, 16, 127, 128} {
		around(float32(math.Ldexp(1, e)))
		around(float32(math.Ldexp(1.5, e)))
		around(float32(math.Ldexp(3, e-1)))
	}
	rng := rand.New(rand.NewPCG(3, 4))
	for range 1 << 21 {
		add(math.Float32frombits(rng.Uint32()))
	}
	return in
}

func TestFloat32ToHalfRoundsToNearestEven(t *testing.T) {
	for _, f := range halfInputs() {
		if got, want := f32ToF16(f), refF32ToF16(t, f); got != want {
			t.Fatalf("f32ToF16(%v = %#08x) = %#04x, want %#04x", f, math.Float32bits(f), got, want)
		}
		if got, want := f32ToBF16(f), refF32ToBF16(t, f); got != want {
			t.Fatalf("f32ToBF16(%v = %#08x) = %#04x, want %#04x", f, math.Float32bits(f), got, want)
		}
	}
}

func TestFloat32ToHalfEdges(t *testing.T) {
	ld := func(m float64, e int) float32 { return float32(math.Ldexp(m, e)) }
	cases := []struct {
		name string
		f    float32
		f16  uint16
		bf16 uint16
	}{
		{"zero", 0, 0, 0},
		{"negative zero", float32(math.Copysign(0, -1)), 0x8000, 0x8000},
		{"one", 1, 0x3c00, 0x3f80},
		{"minus two", -2, 0xc000, 0xc000},
		{"largest binary16", 65504, 0x7bff, 0x4780},
		{"below the overflow midpoint", math.Nextafter32(65520, 0), 0x7bff, 0x4780},
		{"overflow midpoint ties to even, infinity", 65520, 0x7c00, 0x4780},
		{"two to the sixteen", 65536, 0x7c00, 0x4780},
		{"largest float32", math.MaxFloat32, 0x7c00, 0x7f80},
		{"below the bfloat16 overflow midpoint", math.Float32frombits(0x7f7f7fff), 0x7c00, 0x7f7f},
		{"bfloat16 overflow midpoint ties to even, infinity", math.Float32frombits(0x7f7f8000), 0x7c00, 0x7f80},
		{"largest bfloat16", math.Float32frombits(0x7f7f0000), 0x7c00, 0x7f7f},
		{"infinity", float32(math.Inf(1)), 0x7c00, 0x7f80},
		{"negative infinity", float32(math.Inf(-1)), 0xfc00, 0xff80},
		{"smallest binary16 subnormal", ld(1, -24), 0x0001, 0x3380},
		{"half of it ties to even, zero", ld(1, -25), 0x0000, 0x3300},
		{"three halves of it rounds up", ld(1.5, -24), 0x0002, 0x33c0},
		{"below half of it", ld(1, -26), 0x0000, 0x3280},
		{"just above half of it", math.Nextafter32(ld(1, -25), 1), 0x0001, 0x3300},
		{"negative subnormal", -ld(3, -24), 0x8003, 0xb440},
		{"largest binary16 subnormal rounds up in bfloat16", ld(1023, -24), 0x03ff, 0x3880},
		{"subnormal to normal midpoint ties to even", ld(2047, -25), 0x0400, 0x3880},
		{"smallest binary16 normal", ld(1, -14), 0x0400, 0x3880},
		{"one third", 1.0 / 3, 0x3555, 0x3eab},
		{"tie in the normal range rounds to even", math.Float32frombits(0x3f801000), 0x3c00, 0x3f80},
		{"tie to even rounds up when odd", math.Float32frombits(0x3f803000), 0x3c02, 0x3f80},
		{"bfloat16 tie rounds to even", math.Float32frombits(0x3f808000), 0x3c04, 0x3f80},
		{"bfloat16 tie rounds up when odd", math.Float32frombits(0x3f818000), 0x3c0c, 0x3f82},
		{"smallest bfloat16 subnormal", ld(1, -133), 0, 0x0001},
		{"smallest float32 subnormal", math.SmallestNonzeroFloat32, 0, 0},
	}
	for _, c := range cases {
		if got := f32ToF16(c.f); got != c.f16 {
			t.Errorf("%s: f32ToF16(%v) = %#04x, want %#04x", c.name, c.f, got, c.f16)
		}
		if got := f32ToBF16(c.f); got != c.bf16 {
			t.Errorf("%s: f32ToBF16(%v) = %#04x, want %#04x", c.name, c.f, got, c.bf16)
		}
	}
}

func TestFloat32ToHalfKeepsNaNPayloadsQuiet(t *testing.T) {
	cases := []struct {
		name string
		bits uint32
		f16  uint16
		bf16 uint16
	}{
		{"quiet NaN", 0x7fc00000, 0x7e00, 0x7fc0},
		{"negative quiet NaN", 0xffc00000, 0xfe00, 0xffc0},
		{"signalling NaN with a payload", 0x7f812345, 0x7e09, 0x7fc1},
		{"signalling NaN with a low payload", 0x7f800001, 0x7e00, 0x7fc0},
		{"quiet NaN with a payload", 0x7fdead00, 0x7ef5, 0x7fde},
		{"negative signalling NaN", 0xff8f0000, 0xfe78, 0xffcf},
	}
	for _, c := range cases {
		f := math.Float32frombits(c.bits)
		if got := f32ToF16(f); got != c.f16 || !isNaN16(got) {
			t.Errorf("%s: f32ToF16(%#08x) = %#04x, want %#04x", c.name, c.bits, got, c.f16)
		}
		if got := f32ToBF16(f); got != c.bf16 || !isNaNBF(got) {
			t.Errorf("%s: f32ToBF16(%#08x) = %#04x, want %#04x", c.name, c.bits, got, c.bf16)
		}
	}
	for _, h := range []uint16{0x7e00, 0x7c01, 0xfdff} {
		if f := f16ToF32(h); f == f {
			t.Errorf("f16ToF32(%#04x) = %v, want a NaN", h, f)
		}
	}
	for _, h := range []uint16{0x7fc0, 0x7f81, 0xffff} {
		if f := bf16ToF32(h); f == f {
			t.Errorf("bf16ToF32(%#04x) = %v, want a NaN", h, f)
		}
	}
}

func TestElementHelpersRoundHalfTypes(t *testing.T) {
	if got := valueOf[bf16](1.0 / 3); got != bf16(0x3eab) {
		t.Errorf("valueOf[bf16](1/3) = %#04x, want 0x3eab", uint16(got))
	}
	if got := valueOf[f16](65520); got != f16(0x7c00) {
		t.Errorf("valueOf[f16](65520) = %#04x, want 0x7c00", uint16(got))
	}
	if got := valueOf[int32](2.9); got != 2 {
		t.Errorf("valueOf[int32](2.9) = %d, want 2", got)
	}
	if got := valueOf[float32](0.1); got != 0.1 {
		t.Errorf("valueOf[float32](0.1) = %v, want 0.1", got)
	}
	if got := conv[float32](bf16(0x3f80)); got != 1 {
		t.Errorf("conv[float32](bf16 1) = %v, want 1", got)
	}
	if got := conv[float64](f16(0x3c00)); got != 1 {
		t.Errorf("conv[float64](f16 1) = %v, want 1", got)
	}
	if got := conv[int32](int32(1 << 30)); got != 1<<30 {
		t.Errorf("conv[int32] = %d, want %d", got, 1<<30)
	}
	// 1 + 2^-8 is halfway between 1 and the next bfloat16, and rounds to
	// even: 1. In binary16 it is exact.
	if got := ofF32[bf16](1 + 1.0/256); got != 0x3f80 {
		t.Errorf("ofF32[bf16](1 + 2^-8) = %#04x, want 0x3f80", uint16(got))
	}
	if got := ofF32[f16](1 + 1.0/256); got != 0x3c04 {
		t.Errorf("ofF32[f16](1 + 2^-8) = %#04x, want 0x3c04", uint16(got))
	}
	if got := ofF32[float32](0.75); got != 0.75 {
		t.Errorf("ofF32[float32](0.75) = %v", got)
	}
	if got := ofF32[int32](-2.5); got != -2 {
		t.Errorf("ofF32[int32](-2.5) = %d, want -2", got)
	}
	v := []float32{1, 1.00390625, -0.1, math.MaxFloat32, float32(math.NaN())}
	out := make([]uint16, len(v))
	roundHalf(backend.BFloat16, v, out)
	if out[0] != 0x3f80 || out[1] != 0x3f80 || out[3] != 0x7f80 || !isNaNBF(out[4]) {
		t.Errorf("roundHalf stored %#04x", out)
	}
	if v[1] != 1 || v[2] != bf16ToF32(out[2]) || v[3] != float32(math.Inf(1)) {
		t.Errorf("roundHalf left %v", v)
	}
	dst := make([]float32, 3)
	loadHalf(backend.Float16, dst, []uint16{0x3c00, 0, 0x4000, 0, 0xc200}, 0, 2)
	if dst[0] != 1 || dst[1] != 2 || dst[2] != -3 {
		t.Errorf("loadHalf = %v, want [1 2 -3]", dst)
	}
}
