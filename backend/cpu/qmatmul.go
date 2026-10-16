package cpu

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// A product by quantized weights, x·Wᵀ, runs like llama.cpp's: each row of
// x is rounded to int8 first, in blocks with a float32 scale each, and the
// kernels take dot products of int8 activations with the weights' quants
// in integer arithmetic, applying the scales once per block. Weights are
// read once per row of the output for a matrix-vector product, which is
// what a model's decoding step computes, and once per tile of up to four
// rows of x otherwise, while those rows stay in cache.
//
// The activations of a Q8_0 product are blocks of 32 like the weights':
//
//	d float32 | q [32]int8                                  36 bytes
//
// and those of a Q4_K or Q6_K product super-blocks of 256 with the sums of
// each group of 16 quants, which apply the mins of Q4_K and the offset of
// Q6_K in one multiplication per group:
//
//	d float32 | q [256]int8 | sums [16]int16                292 bytes
//
// Every kernel computes the same float32 result from the same integers:
// integer sums are exact in any order, and the float operations, which
// apply the scales of each block, run in one order everywhere (see
// qdotQ8_0Go, qdotQ4KGo and qdotQ6KGo), so the assembly kernels and the
// portable ones agree bit for bit.

// A qdot computes the dot products of one row of nb blocks of weights at
// w with rows of quantized activations xstride bytes apart from x, and
// stores the product with row r at out[r*ldo]. Each dtype has kernels for
// one to four rows of activations; see qdots.
type qdot func(nb int, w, x *byte, xstride int, out *float32, ldo int)

// maxRows is the most rows of activations a qdot multiplies at once.
const maxRows = 4

// qdots holds the kernels of each block-quantized dtype for 1 to maxRows
// rows of activations: the portable ones, which the architecture's files
// replace at init with assembly where the CPU supports it.
var qdots = map[backend.DType]*[maxRows]qdot{
	backend.Q8_0: portable(backend.Q8_0, qdotQ8_0Go),
	backend.Q4_K: portable(backend.Q4_K, qdotQ4KGo),
	backend.Q6_K: portable(backend.Q6_K, qdotQ6KGo),
}

// qdotISA names the instructions of the kernels in qdots.
var qdotISA = "portable"

// portable returns the kernels of each number of rows that apply dot,
// the dot product of one row of weights of dt with one of activations,
// row by row.
func portable(dt backend.DType, dot func(nb int, w, x []byte) float32) *[maxRows]qdot {
	_, abb := actBlock(dt)
	var ks [maxRows]qdot
	for i := range ks {
		rows := i + 1
		ks[i] = func(nb int, w, x *byte, xstride int, out *float32, ldo int) {
			wb := unsafe.Slice(w, nb*dt.BlockBytes())
			o := unsafe.Slice(out, (rows-1)*ldo+1)
			for r := range rows {
				xb := unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(x), r*xstride)), nb*abb)
				o[r*ldo] = dot(nb, wb, xb)
			}
		}
	}
	return &ks
}

// actBlock returns the block size of the activations of a product by dt
// and the bytes of one of their blocks.
func actBlock(dt backend.DType) (size, bytes int) {
	if dt == backend.Q8_0 {
		return 32, 36
	}
	return 256, 292
}

// f32At reads the little-endian float32 at b.
func f32At(b []byte) float32 {
	return math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
}

// i16At reads the little-endian int16 at b.
func i16At(b []byte) int32 { return int32(int16(le16(b))) }

// qdotQ8_0Go returns the dot product of nb Q8_0 blocks of weights w with
// nb blocks of activations x. The products of each block are summed in
// four integer lanes, value p of the block into lane p/4 mod 4 as a
// vector of eight int32 folded in half sums them; each lane, times the
// product of the scales, is added to a float32 lane of its own; and the
// lanes are added (0+2)+(1+3).
func qdotQ8_0Go(nb int, w, x []byte) float32 {
	var acc [4]float32
	for b := range nb {
		wb, xb := w[34*b:34*b+34], x[36*b:36*b+36]
		var s [4]int32
		for p, q := range wb[2:] {
			s[p/4%4] += int32(int8(q)) * int32(int8(xb[4+p]))
		}
		dd := f16ToF32(le16(wb)) * f32At(xb)
		for l, v := range s {
			acc[l] += float32(dd * float32(v))
		}
	}
	return (acc[0] + acc[2]) + (acc[1] + acc[3])
}

// qdotQ4KGo returns the dot product of nb Q4_K super-blocks of weights w
// with nb super-blocks of activations x. Each super-block sums the
// products of the quants, times their sub-block's scale, into one int32,
// and the activations' sums, times the mins, into another; the first
// times d and the second times dmin, each also times the activations'
// scale, are added to two float32 sums, and the result is their
// difference.
func qdotQ4KGo(nb int, w, x []byte) float32 {
	var acc, accm float32
	for b := range nb {
		wb, xb := w[144*b:144*b+144], x[292*b:292*b+292]
		scales, xq, sums := wb[4:16], xb[4:260], xb[260:292]
		var sumi, summ int32
		for j := range 4 {
			q := wb[16+32*j : 48+32*j]
			var lo, hi int32
			for l, v := range q {
				lo += int32(v&0xf) * int32(int8(xq[64*j+l]))
				hi += int32(v>>4) * int32(int8(xq[64*j+32+l]))
			}
			sc0, m0 := scaleMinK4(2*j, scales)
			sc1, m1 := scaleMinK4(2*j+1, scales)
			sumi += int32(sc0)*lo + int32(sc1)*hi
			summ += int32(m0)*(i16At(sums[8*j:])+i16At(sums[8*j+2:])) + int32(m1)*(i16At(sums[8*j+4:])+i16At(sums[8*j+6:]))
		}
		xd := f32At(xb)
		dd, dm := f16ToF32(le16(wb))*xd, f16ToF32(le16(wb[2:]))*xd
		acc += float32(dd * float32(sumi))
		accm += float32(dm * float32(summ))
	}
	return acc - accm
}

// qdotQ6KGo returns the dot product of nb Q6_K super-blocks of weights w
// with nb super-blocks of activations x. Each super-block sums the
// products of the quants, 32 more than the values they stand for, times
// their group's scale into one int32, from which it takes 32 times the
// activations' sums times the scales; that times d and the activations'
// scale is added to a float32 sum.
func qdotQ6KGo(nb int, w, x []byte) float32 {
	var acc float32
	for b := range nb {
		wb, xb := w[210*b:210*b+210], x[292*b:292*b+292]
		xq, sums := xb[4:260], xb[260:292]
		var sumi int32
		for g := range 16 {
			n, k, l0 := g/8, g%8/2, 16*(g%2)
			ql, qh := wb[64*n:], wb[128+32*n:]
			var s int32
			for l := l0; l < l0+16; l++ {
				s += int32(q6(ql, qh, k, l)) * int32(int8(xq[128*n+32*k+l]))
			}
			sc := int32(int8(wb[192+g]))
			sumi += sc*s - 32*sc*i16At(sums[2*g:])
		}
		dd := f16ToF32(le16(wb[208:])) * f32At(xb)
		acc += float32(dd * float32(sumi))
	}
	return acc
}

// quantizeRow rounds the k values of x to blocks of activations of size
// bs at dst: each block's scale d is its largest magnitude over 127, and
// its quants the values over d rounded to the nearest integer, ties to
// even. A K-quant block also stores the sums of each 16 quants. A block
// holding a NaN or an infinity gets the scale NaN, which makes every
// product with it NaN.
func quantizeRow(dst []byte, x []float32, bs int, sums bool) {
	stride := 4 + bs
	if sums {
		stride += 32
	}
	for b := 0; b < len(x)/bs; b++ {
		v, out := x[b*bs:(b+1)*bs], dst[b*stride:(b+1)*stride]
		var amax float32
		finite := true
		for _, f := range v {
			a := float32(math.Abs(float64(f)))
			if !(a <= math.MaxFloat32) {
				finite = false
			}
			amax = max(amax, a)
		}
		q := out[4 : 4+bs]
		var d float32
		switch {
		case !finite:
			d = float32(math.NaN())
			clear(q)
		case amax == 0:
			clear(q)
		default:
			d = amax / 127
			id := 127 / amax
			for i, f := range v {
				r := math.RoundToEven(float64(f * id))
				q[i] = byte(int8(max(-127, min(127, r))))
			}
		}
		putF32(out, d)
		if !sums {
			continue
		}
		s := out[4+bs:]
		for g := range bs / 16 {
			var t int32
			for _, c := range q[16*g : 16*g+16] {
				t += int32(int8(c))
			}
			s[2*g], s[2*g+1] = byte(t), byte(t>>8)
		}
	}
}

// putF32 stores f little-endian at b.
func putF32(b []byte, f float32) {
	u := math.Float32bits(f)
	b[0], b[1], b[2], b[3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
}

// actScratch holds the quantized activations of products.
var actScratch = sync.Pool{New: func() any { return new([]byte) }}

// qmatmulRowWork is the least number of weights' multiply-adds worth a
// worker of its own.
const qmatmulRowWork = 1 << 16

// qmatmulCacheBytes bounds the quantized rows of x a tile walks while the
// row of weights it multiplies stays in L1: they stay in L2.
const qmatmulCacheBytes = 256 << 10

// runQMatMul computes a product by quantized weights, x·Wᵀ: k.In[0] reads
// x, [..., K], and k.In[1] the weights W, [N, K], contiguously.
func runQMatMul(ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, ins []backend.Buffer) error {
	out, err := data[float32](outBuf)
	if err != nil {
		return err
	}
	x, err := data[float32](ins[0])
	if err != nil {
		return err
	}
	w, err := quantized(ins[1])
	if err != nil {
		return err
	}
	vx, vw := k.In[0], k.In[1]
	r := len(vx.Shape)
	if k.DType != backend.Float32 || r == 0 || len(vw.Shape) != 2 || len(k.Shape) != r {
		return fmt.Errorf("product of %v by %s%v: %w", vx.Shape, w.dt, vw.Shape, backend.ErrInternal)
	}
	n, kk, bs := vw.Shape[0], vw.Shape[1], w.dt.BlockSize()
	if vx.Shape[r-1] != kk || k.Shape[r-1] != n || !slices.Equal(k.Shape[:r-1], vx.Shape[:r-1]) ||
		kk%bs != 0 || !vw.IsContiguous() || vw.Offset%bs != 0 {
		return fmt.Errorf("product of %v by %s%v into %v: %w", vx.Shape, w.dt, vw.Shape, k.Shape, backend.ErrInternal)
	}
	if len(out) == 0 {
		return nil
	}
	if kk == 0 {
		clear(out)
		return nil
	}
	m := len(out) / n

	// Quantize the rows of x.
	abs, abb := actBlock(w.dt)
	xstride := kk / abs * abb
	scratch := actScratch.Get().(*[]byte)
	defer actScratch.Put(scratch)
	*scratch = grow(*scratch, m*xstride)
	xq := *scratch
	lead := backend.View{Shape: vx.Shape[:r-1], Strides: vx.Strides[:r-1], Offset: vx.Offset}
	sxk := vx.Strides[r-1]
	err = b.parallelFor(ctx, m, max(1, elementGrain/kk), func(lo, hi int) error {
		c := newCursor(lead.Shape, []backend.View{lead}, lo)
		var row []float32
		if sxk != 1 {
			row = make([]float32, kk)
		}
		for i := lo; i < hi; i++ {
			off := c.offs[0]
			if sxk == 1 {
				row = x[off : off+kk]
			} else {
				for p := range row {
					row[p] = x[off+p*sxk]
				}
			}
			quantizeRow(xq[i*xstride:(i+1)*xstride], row, abs, abs == 256)
			c.next()
		}
		return nil
	})
	if err != nil {
		return err
	}

	// Multiply, splitting the rows of W between workers.
	kernels := qdots[w.dt]
	nb, wstride := kk/bs, kk/bs*w.dt.BlockBytes()
	wdata := w.data[vw.Offset/bs*w.dt.BlockBytes():]
	tile := max(maxRows, qmatmulCacheBytes/xstride/maxRows*maxRows)
	return b.parallelFor(ctx, n, max(1, qmatmulRowWork/(m*kk)), func(lo, hi int) error {
		// Bounds checks here leave the kernels none to do.
		_ = wdata[hi*wstride-1]
		_ = xq[m*xstride-1]
		for m0 := 0; m0 < m; m0 += tile {
			m1 := min(m, m0+tile)
			for j := lo; j < hi; j++ {
				wrow := &wdata[j*wstride]
				for i := m0; i < m1; i += maxRows {
					rows := min(maxRows, m1-i)
					kernels[rows-1](nb, wrow, &xq[i*xstride], xstride, &out[i*n+j], n)
				}
			}
		}
		return nil
	})
}
