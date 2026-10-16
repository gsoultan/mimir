package safetensors

import (
	"encoding/binary"
	"math"
	"math/bits"
	"slices"
)

// Float16Tensor returns an F16 tensor of shape holding v, each value
// rounded to the nearest IEEE 754 binary16, ties to even. Values beyond
// the largest finite binary16 become infinite; NaN and infinities are
// kept.
func Float16Tensor(shape []int, v []float32) Tensor {
	return halfTensor("F16", shape, v, f32ToF16)
}

// BFloat16Tensor returns a BF16 tensor of shape holding v, each value
// rounded to the nearest bfloat16, ties to even. Values beyond the
// largest finite bfloat16 become infinite; NaN and infinities are kept.
func BFloat16Tensor(shape []int, v []float32) Tensor {
	return halfTensor("BF16", shape, v, f32ToBF16)
}

func halfTensor(dtype string, shape []int, v []float32, round func(float32) uint16) Tensor {
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], round(x))
	}
	return Tensor{DType: dtype, Shape: slices.Clone(shape), Data: b}
}

// halfFloat32s decodes the elements of t, an F16 or BF16 tensor.
func (t Tensor) halfFloat32s() []float32 {
	conv := f16ToF32
	if t.DType == "BF16" {
		conv = bf16ToF32
	}
	out := make([]float32, len(t.Data)/2)
	for i := range out {
		out[i] = conv(binary.LittleEndian.Uint16(t.Data[2*i:]))
	}
	return out
}

// The conversions below are those of the backends, so a value written
// here reads back on every device as the element the device would have
// computed.

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
	case exp > 15:
		return sign | 0x7c00
	case exp >= -14:
		h := uint32(exp+15)<<10 | mant>>13
		return sign | uint16(roundHalfEven(h, mant, 13))
	case exp >= -25:
		m := mant | 0x800000
		shift := uint(-exp - 1)
		return sign | uint16(roundHalfEven(m>>shift, m, shift))
	default:
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
