package cpu

import (
	"context"
	"fmt"
	"runtime"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// blocks is a buffer of a block-quantized dtype: the bytes of whole
// blocks in ggml's layout, read in place. Kernels only read it, to
// dequantize its elements (see dequantize) or to multiply by it (see
// runQMatMul).
type blocks struct {
	dt   backend.DType
	n    int // elements
	data []byte
	free func() // as buffer.free
}

func (b *blocks) DType() backend.DType { return b.dt }
func (b *blocks) Len() int             { return b.n }

// release frees the buffer's memory once, as buffer.release does.
func (b *blocks) release() {
	if b.free == nil {
		return
	}
	b.free()
	b.free, b.data, b.n = nil, nil, 0
}

// allocBlocks returns a zeroed buffer of n elements of the block type dt,
// in an anonymous mapping when it is large and the platform maps memory,
// and on the heap otherwise. n must be a whole number of blocks.
func allocBlocks(dt backend.DType, n int) (buf backend.Buffer, err error) {
	size, ok := dt.Bytes(n)
	if !ok {
		return nil, fmt.Errorf("cpu: alloc %d elements of %s, blocks of %d: %w", n, dt, dt.BlockSize(), backend.ErrShape)
	}
	if offHeap && size >= offHeapMin {
		mem, unmap, err := mapAnon(size)
		if err != nil {
			return nil, fmt.Errorf("cpu: alloc %d x %s: %v: %w", n, dt, err, backend.ErrOutOfMemory)
		}
		offHeapBytes.Add(int64(size))
		paceGC(size)
		return newBlocks(dt, mem, func() {
			_ = unmap() // fails only for a range that is not mapped
			offHeapBytes.Add(-int64(size))
		}), nil
	}
	defer func() {
		if r := recover(); r != nil {
			buf, err = nil, fmt.Errorf("cpu: alloc %d x %s: %v: %w", n, dt, r, backend.ErrOutOfMemory)
		}
	}()
	return &blocks{dt: dt, n: n, data: make([]byte, size)}, nil
}

// wrapBlocks returns a buffer of the block type dt over data, which holds
// whole blocks at any address, that runs release, if any, once Free frees
// it or it is garbage, as wrapBuffer does.
func wrapBlocks(dt backend.DType, data []byte, release func()) (backend.Buffer, error) {
	if len(data)%dt.BlockBytes() != 0 {
		return nil, fmt.Errorf("cpu: wrap %d bytes as %s, blocks of %d bytes: %w", len(data), dt, dt.BlockBytes(), backend.ErrShape)
	}
	return newBlocks(dt, data, release), nil
}

// newBlocks returns a buffer over data that runs release, if not nil,
// once Free frees it or it is garbage.
func newBlocks(dt backend.DType, data []byte, release func()) *blocks {
	b := &blocks{dt: dt, n: len(data) / dt.BlockBytes() * dt.BlockSize(), data: data}
	if release != nil {
		cleanup := runtime.AddCleanup(b, func(release func()) { release() }, release)
		b.free = func() {
			cleanup.Stop()
			release()
		}
	}
	return b
}

// quantized returns the bytes of a block-quantized buffer.
func quantized(b backend.Buffer) (*blocks, error) {
	q, ok := b.(*blocks)
	if !ok {
		return nil, fmt.Errorf("cpu: got %T, want a block-quantized buffer: %w", b, backend.ErrDType)
	}
	return q, nil
}

// le16 reads a little-endian uint16.
func le16(b []byte) uint16 { return uint16(b[0]) | uint16(b[1])<<8 }

// The dequantizers below compute each value as ggml's dequantize_row
// functions do in C (ggml-quants.c), operation by operation: every
// product and difference is rounded to float32 on its own, never fused
// into a multiply-add, so that every device computes the same values bit
// for bit. The explicit float32 conversions are what keep Go from fusing.

// dequantQ8_0 sets dst[:32] to the values of the Q8_0 block b: d·q.
func dequantQ8_0(dst []float32, b []byte) {
	d := f16ToF32(le16(b))
	for i, q := range b[2:34] {
		dst[i] = float32(int8(q)) * d
	}
}

// scaleMinK4 returns the 6-bit scale and min of sub-block j of a Q4_K
// super-block from its 12 bytes of scales, as get_scale_min_k4 does.
func scaleMinK4(j int, s []byte) (sc, m uint8) {
	if j < 4 {
		return s[j] & 63, s[j+4] & 63
	}
	return s[j+4]&0xf | s[j-4]>>6<<4, s[j+4]>>4 | s[j]>>6<<4
}

// dequantQ4K sets dst[:256] to the values of the Q4_K super-block b. Each
// sub-block of 32 values has a scale d·sc and a min dmin·m, and value q
// of it is (d·sc)·q - dmin·m. Sub-blocks 2j and 2j+1 take the low and
// high nibbles of the same 32 bytes.
func dequantQ4K(dst []float32, b []byte) {
	d, dmin := f16ToF32(le16(b)), f16ToF32(le16(b[2:]))
	scales, qs := b[4:16], b[16:144]
	for j := range 4 {
		q, y := qs[32*j:32*j+32], dst[64*j:64*j+64]
		sc, m := scaleMinK4(2*j, scales)
		d1, m1 := d*float32(sc), float32(dmin*float32(m))
		sc, m = scaleMinK4(2*j+1, scales)
		d2, m2 := d*float32(sc), float32(dmin*float32(m))
		for l, v := range q {
			y[l] = float32(d1*float32(v&0xf)) - m1
			y[l+32] = float32(d2*float32(v>>4)) - m2
		}
	}
}

// q6 returns the quant of value l (of 32) in quarter k of a 128-value half
// of a Q6_K super-block, from that half's 64 low and 32 high bytes, with
// 32 added: 0 to 63.
func q6(ql, qh []byte, k, l int) int {
	lo := ql[l+32*(k&1)]
	if k >= 2 {
		lo >>= 4
	}
	return int(lo&0xf | (qh[l]>>(2*k)&3)<<4)
}

// dequantQ6K sets dst[:256] to the values of the Q6_K super-block b. Each
// group of 16 values has the scale d·sc, and value q of it is
// (d·sc)·(q - 32).
func dequantQ6K(dst []float32, b []byte) {
	d := f16ToF32(le16(b[208:]))
	for n := range 2 {
		ql, qh, sc, y := b[64*n:64*n+64], b[128+32*n:128+32*n+32], b[192+8*n:192+8*n+8], dst[128*n:128*n+128]
		for k := range 4 {
			for l := range 32 {
				s := d * float32(int8(sc[l/16+2*k]))
				y[32*k+l] = s * float32(q6(ql, qh, k, l)-32)
			}
		}
	}
}

// dequantBlock sets dst[:dt.BlockSize()] to the values of the block b of
// dt.
func dequantBlock(dt backend.DType, dst []float32, b []byte) {
	switch dt {
	case backend.Q8_0:
		dequantQ8_0(dst, b)
	case backend.Q4_K:
		dequantQ4K(dst, b)
	default:
		dequantQ6K(dst, b)
	}
}

// dequantAt returns element i of the block-quantized data of dt, computed
// as dequantBlock computes it.
func dequantAt(dt backend.DType, data []byte, i int) float32 {
	bs, bb := dt.BlockSize(), dt.BlockBytes()
	b, j := data[i/bs*bb:][:bb], i%bs
	switch dt {
	case backend.Q8_0:
		return float32(int8(b[2+j])) * f16ToF32(le16(b))
	case backend.Q4_K:
		d, dmin := f16ToF32(le16(b)), f16ToF32(le16(b[2:]))
		sub, l := j/32, j%32
		sc, m := scaleMinK4(sub, b[4:16])
		v := b[16+32*(sub/2)+l]
		if sub%2 == 1 {
			v >>= 4
		}
		return float32(d*float32(sc)*float32(v&0xf)) - float32(dmin*float32(m))
	default:
		n, k, l := j/128, j%128/32, j%32
		s := f16ToF32(le16(b[208:])) * float32(int8(b[192+8*n+l/16+2*k]))
		return s * float32(q6(b[64*n:], b[128+32*n:], k, l)-32)
	}
}

// dequantize computes a cast of the block-quantized input to the float
// dtype of k's output: each element's value, rounded to that dtype.
func dequantize(ctx context.Context, b *Backend, k *backend.Kernel, out backend.Buffer, in backend.Buffer) error {
	src, err := quantized(in)
	if err != nil {
		return err
	}
	dt, v := src.dt, k.In[0]
	n := out.Len()
	bs, bb := dt.BlockSize(), dt.BlockBytes()
	f32, isF32 := out.(*buffer[float32])
	whole := v.IsContiguous() && v.Offset%bs == 0 && n%bs == 0
	if !whole {
		vals := f32Out(out, n)
		if err := mapStrided(ctx, b, k.Shape, v, n, func(i, off int) { vals[i] = dequantAt(dt, src.data, off) }); err != nil {
			return err
		}
		return storeF32(out, vals)
	}
	// Whole blocks: each worker dequantizes its blocks straight into a
	// float32 output, or into a block of scratch that it rounds.
	data := src.data[v.Offset/bs*bb:]
	return b.parallelFor(ctx, n/bs, max(1, elementGrain/bs), func(lo, hi int) error {
		if isF32 {
			for i := lo; i < hi; i++ {
				dequantBlock(dt, f32.data[i*bs:(i+1)*bs], data[i*bb:])
			}
			return nil
		}
		scratch := make([]float32, bs)
		for i := lo; i < hi; i++ {
			dequantBlock(dt, scratch, data[i*bb:])
			storeHalf(out, scratch, i*bs)
		}
		return nil
	})
}

// f32Out returns where to compute n float32 values for out: its own
// elements if it is Float32, and new memory otherwise.
func f32Out(out backend.Buffer, n int) []float32 {
	if f, ok := out.(*buffer[float32]); ok {
		return f.data
	}
	return make([]float32, n)
}

// storeF32 stores vals, which f32Out returned for out, into out, rounding
// them to a half dtype.
func storeF32(out backend.Buffer, vals []float32) error {
	if _, ok := out.(*buffer[float32]); ok {
		return nil
	}
	return storeHalf(out, vals, 0)
}

// storeHalf rounds vals to the half dtype of out and stores them from
// element at.
func storeHalf(out backend.Buffer, vals []float32, at int) error {
	switch o := out.(type) {
	case *buffer[bf16]:
		for i, v := range vals {
			o.data[at+i] = bf16(f32ToBF16(v))
		}
	case *buffer[f16]:
		for i, v := range vals {
			o.data[at+i] = f16(f32ToF16(v))
		}
	default:
		return fmt.Errorf("cpu: dequantize into %s: %w", out.DType(), backend.ErrDType)
	}
	return nil
}

// gatherQuant gathers from a block-quantized input as gather does,
// dequantizing each element it reads into the float output.
func gatherQuant(ctx context.Context, b *Backend, k *backend.Kernel, out backend.Buffer, ins []backend.Buffer) error {
	src, err := quantized(ins[0])
	if err != nil {
		return err
	}
	idx, err := data[int32](ins[1])
	if err != nil {
		return err
	}
	axis := k.Attrs.Axis
	vs, vi := k.In[0], k.In[1]
	limit, step := vs.Shape[axis], vs.Strides[axis]
	base := backend.View{Shape: k.Shape, Strides: slices.Clone(vs.Strides), Offset: vs.Offset}
	base.Strides[axis] = 0
	vals := f32Out(out, out.Len())
	err = b.parallelFor(ctx, len(vals), elementGrain, func(lo, hi int) error {
		c := newCursor(k.Shape, []backend.View{vi, base}, lo)
		for i := lo; i < hi; i++ {
			j := int(idx[c.offs[0]])
			if j < 0 || j >= limit {
				return fmt.Errorf("index %d for axis %d of size %d: %w", j, axis, limit, backend.ErrIndex)
			}
			vals[i] = dequantAt(src.dt, src.data, c.offs[1]+j*step)
			c.next()
		}
		return nil
	})
	if err != nil {
		return err
	}
	return storeF32(out, vals)
}
