// Package quanttest makes block-quantized data for tests and computes
// its values with a reference dequantizer, transcribed from ggml's
// dequantize_row functions (ggml-quants.c) independently of the backends'
// own.
package quanttest

import (
	"encoding/binary"
	"math"
	"math/rand/v2"

	"github.com/gsoultan/mimir/backend"
)

// Random returns n elements of dt, a block-quantized type, as random
// blocks: every quant, scale and min drawn uniformly from its range, and
// the binary16 scales from finite values between 2^-24 and 1/2 of either
// sign, subnormals and zeros among them.
func Random(rng *rand.Rand, dt backend.DType, n int) []byte {
	size, ok := dt.Bytes(n)
	if !ok {
		panic("quanttest: not whole blocks")
	}
	b := make([]byte, size)
	for i := range b {
		b[i] = byte(rng.Uint32())
	}
	for at := 0; at < size; at += dt.BlockBytes() {
		blk := b[at : at+dt.BlockBytes()]
		switch dt {
		case backend.Q8_0:
			binary.LittleEndian.PutUint16(blk, scale(rng))
		case backend.Q4_K:
			binary.LittleEndian.PutUint16(blk, scale(rng))
			binary.LittleEndian.PutUint16(blk[2:], scale(rng))
		case backend.Q6_K:
			binary.LittleEndian.PutUint16(blk[208:], scale(rng))
		}
	}
	return b
}

// scale returns the bits of a random finite binary16 below 1/2 in
// magnitude: one in eight a subnormal or zero.
func scale(rng *rand.Rand) uint16 {
	sign := uint16(rng.IntN(2)) << 15
	exp := uint16(1 + rng.IntN(13))
	if rng.IntN(8) == 0 {
		exp = 0
	}
	return sign | exp<<10 | uint16(rng.IntN(1024))
}

// Half returns the value of the binary16 bits h.
func Half(h uint16) float32 {
	sign := 1.0
	if h&0x8000 != 0 {
		sign = -1
	}
	exp, mant := int(h>>10&0x1f), float64(h&0x3ff)
	switch exp {
	case 0:
		return float32(sign * math.Ldexp(mant, -24))
	case 31:
		if mant != 0 {
			return float32(math.NaN())
		}
		return float32(sign * math.Inf(1))
	}
	return float32(sign * math.Ldexp(1+mant/1024, exp-15))
}

// Dequantize returns the values of the blocks of dt in data, as ggml's
// dequantize_row functions compute them in float, rounding after every
// operation.
func Dequantize(dt backend.DType, data []byte) []float32 {
	bb := dt.BlockBytes()
	var y []float32
	for at := 0; at+bb <= len(data); at += bb {
		x := data[at : at+bb]
		switch dt {
		case backend.Q8_0:
			d := Half(binary.LittleEndian.Uint16(x))
			for j := range 32 {
				y = append(y, float32(int8(x[2+j]))*d)
			}
		case backend.Q4_K:
			y = append(y, q4K(x)...)
		case backend.Q6_K:
			y = append(y, q6K(x)...)
		}
	}
	return y
}

// getScaleMinK4 is ggml's get_scale_min_k4.
func getScaleMinK4(j int, q []byte) (d, m byte) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)
}

// q4K is the loop of dequantize_row_q4_K for one block.
func q4K(x []byte) []float32 {
	y := make([]float32, 0, 256)
	q := x[16:]
	d := Half(binary.LittleEndian.Uint16(x[0:]))
	min := Half(binary.LittleEndian.Uint16(x[2:]))
	scales := x[4:16]
	is := 0
	for j := 0; j < 256; j += 64 {
		sc, m := getScaleMinK4(is+0, scales)
		d1 := d * float32(sc)
		m1 := float32(min * float32(m))
		sc, m = getScaleMinK4(is+1, scales)
		d2 := d * float32(sc)
		m2 := float32(min * float32(m))
		for l := range 32 {
			y = append(y, float32(d1*float32(q[l]&0xF))-m1)
		}
		for l := range 32 {
			y = append(y, float32(d2*float32(q[l]>>4))-m2)
		}
		q = q[32:]
		is += 2
	}
	return y
}

// q6K is the loop of dequantize_row_q6_K for one block.
func q6K(x []byte) []float32 {
	y := make([]float32, 256)
	d := Half(binary.LittleEndian.Uint16(x[208:]))
	ql, qh, sc := x[0:128], x[128:192], x[192:208]
	for n := 0; n < 256; n += 128 {
		for l := range 32 {
			is := l / 16
			q1 := int8((ql[l+0]&0xF)|(((qh[l]>>0)&3)<<4)) - 32
			q2 := int8((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4)) - 32
			q3 := int8((ql[l+0]>>4)|(((qh[l]>>4)&3)<<4)) - 32
			q4 := int8((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4)) - 32
			y[n+l+0] = float32(d*float32(int8(sc[is+0]))) * float32(q1)
			y[n+l+32] = float32(d*float32(int8(sc[is+2]))) * float32(q2)
			y[n+l+64] = float32(d*float32(int8(sc[is+4]))) * float32(q3)
			y[n+l+96] = float32(d*float32(int8(sc[is+6]))) * float32(q4)
		}
		ql, qh, sc = ql[64:], qh[32:], sc[8:]
	}
	return y
}

// QuantizeQ8_0 returns x, whose length is a multiple of 32, quantized to
// Q8_0 blocks as ggml's quantize_row_q8_0_ref does: each block's scale is
// its largest magnitude over 127, rounded to binary16, and its quants the
// values times the reciprocal of the unrounded scale, rounded half away
// from zero.
func QuantizeQ8_0(x []float32) []byte {
	out := make([]byte, 0, len(x)/32*34)
	for b := 0; b+32 <= len(x); b += 32 {
		var amax float32
		for _, v := range x[b : b+32] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		d := amax / 127
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		out = binary.LittleEndian.AppendUint16(out, ToHalf(d))
		for _, v := range x[b : b+32] {
			out = append(out, byte(int8(math.Round(float64(v*id)))))
		}
	}
	return out
}

// ToHalf rounds f to the nearest binary16, ties to even, as bits.
func ToHalf(f float32) uint16 {
	if f != f {
		return 0x7e00
	}
	var sign uint16
	if math.Signbit(float64(f)) {
		sign, f = 0x8000, -f
	}
	if f >= 65520 {
		return sign | 0x7c00
	}
	// Scale to the units of the last place of f's binade, which are 2^-24
	// for subnormals, and round there.
	exp := max(-14, math.Ilogb(float64(f)))
	units := math.RoundToEven(math.Ldexp(float64(f), 10-exp))
	if exp == -14 && units < 1024 {
		return sign | uint16(units)
	}
	if units == 2048 {
		units, exp = 1024, exp+1
	}
	return sign | uint16(exp+15)<<10 | uint16(units-1024)
}
