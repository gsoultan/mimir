package cpu

import (
	"math"
	"math/bits"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// bf16 is the bits of a backend.BFloat16 element and f16 those of a
// backend.Float16 element. Kernels convert them to float32 to compute and
// round float32 results to the nearest element, ties to even, with the
// conversions below. The generated Metal and CUDA kernels round the same
// way, bit for bit, so every device stores the same elements.
type (
	bf16 uint16
	f16  uint16
)

// bf16ToF32 returns the value of a bfloat16: its bits are the top half of
// the float32's.
func bf16ToF32(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

// f32ToBF16 rounds f to the nearest bfloat16, ties to even. Values beyond
// the largest finite bfloat16 round to infinity, and a NaN stays a NaN
// with its sign and the top of its payload, made quiet.
func f32ToBF16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7fffffff > 0x7f800000 {
		return uint16(b>>16) | 0x0040
	}
	// Adding one less than half the dropped range rounds up only above
	// the midpoint; the lowest kept bit then breaks ties towards even.
	b += 0x7fff + (b>>16)&1
	return uint16(b >> 16)
}

// f16ToF32 returns the value of an IEEE 754 binary16.
func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// A subnormal is mant × 2^-24: shift its leading bit to the
		// position of the float32's hidden bit and set the exponent of
		// 2^-14 one lower for each shift.
		shift := uint32(bits.LeadingZeros32(mant) - 21)
		mant = mant << shift & 0x3ff
		return math.Float32frombits(sign | (113-shift)<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
	}
}

// f32ToF16 rounds f to the nearest IEEE 754 binary16, ties to even, with
// the subnormals of that format. Values beyond the largest finite binary16
// round to infinity, and a NaN stays a NaN with its sign and the top of its
// payload, made quiet.
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	b &= 0x7fffffff
	if b > 0x7f800000 {
		return sign | 0x7e00 | uint16(b>>13)&0x1ff
	}
	exp := int(b>>23) - 127
	mant := b & 0x7fffff
	switch {
	case exp > 15: // infinity, or too large
		return sign | 0x7c00
	case exp >= -14: // normal: keep the top 10 fraction bits
		h := uint32(exp+15)<<10 | mant>>13
		return sign | uint16(roundHalfEven(h, mant, 13))
	case exp >= -25: // subnormal: a multiple of 2^-24
		m := mant | 0x800000 // f = m × 2^(exp-23)
		shift := uint(-exp - 1)
		return sign | uint16(roundHalfEven(m>>shift, m, shift))
	default: // too small
		return sign
	}
}

// roundHalfEven rounds up h, the top bits of m, by one if the low shift
// bits of m are above half their range, or at half and h is odd.
func roundHalfEven(h, m uint32, shift uint) uint32 {
	rest := m & (1<<shift - 1)
	half := uint32(1) << (shift - 1)
	if rest > half || rest == half && h&1 == 1 {
		h++
	}
	return h
}

// halfToF32 returns the value of the dt element h.
func halfToF32(dt backend.DType, h uint16) float32 {
	if dt == backend.BFloat16 {
		return bf16ToF32(h)
	}
	return f16ToF32(h)
}

// f32ToHalf returns the dt element nearest f.
func f32ToHalf(dt backend.DType, f float32) uint16 {
	if dt == backend.BFloat16 {
		return f32ToBF16(f)
	}
	return f32ToF16(f)
}

// loadHalf sets dst[i] to the value of the dt element src[off+i*stride].
func loadHalf(dt backend.DType, dst []float32, src []uint16, off, stride int) {
	if dt == backend.BFloat16 {
		for i := range dst {
			dst[i] = bf16ToF32(src[off])
			off += stride
		}
		return
	}
	for i := range dst {
		dst[i] = f16ToF32(src[off])
		off += stride
	}
}

// roundHalf rounds each v[i] to the nearest dt element and, if out is not
// nil, stores that element in out[i].
func roundHalf(dt backend.DType, v []float32, out []uint16) {
	if out != nil {
		out = out[:len(v)]
	}
	if dt == backend.BFloat16 {
		for i, x := range v {
			h := f32ToBF16(x)
			v[i] = bf16ToF32(h)
			if out != nil {
				out[i] = h
			}
		}
		return
	}
	for i, x := range v {
		h := f32ToF16(x)
		v[i] = f16ToF32(h)
		if out != nil {
			out[i] = h
		}
	}
}

// bitsOf returns the elements of a half buffer as their bits.
func bitsOf[T bf16 | f16](s []T) []uint16 {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(&s[0])), len(s))
}

// conv converts an element to a number type, a half element through its
// float32 value.
func conv[T float32 | int32 | float64, S elem](v S) T {
	switch v := any(v).(type) {
	case float32:
		return T(v)
	case int32:
		return T(v)
	case bf16:
		return T(bf16ToF32(uint16(v)))
	default:
		return T(f16ToF32(uint16(v.(f16))))
	}
}

// valueOf converts a kernel attribute to an element: a half element is the
// one nearest the float32 value.
func valueOf[T elem](v float64) T {
	var zero T
	switch any(zero).(type) {
	case float32, int32:
		return T(v)
	case bf16:
		return T(f32ToBF16(float32(v)))
	default:
		return T(f32ToF16(float32(v)))
	}
}

// ofF32 converts a float32 to an element: a half element is the nearest
// one.
func ofF32[T elem](f float32) T {
	var zero T
	switch any(zero).(type) {
	case float32, int32:
		return T(f)
	case bf16:
		return T(f32ToBF16(f))
	default:
		return T(f32ToF16(f))
	}
}
