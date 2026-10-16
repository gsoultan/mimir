package cpu

import (
	"context"
	"encoding/binary"
	"math"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// A product of many rows of activations by K-quant weights, a prompt's
// rather than a decoding step's, runs in tiles of two rows of weights by
// sixteen rows of activations, as outer products: the kernels multiply
// four quants of a row of weights, broadcast, by four quants of each of
// sixteen rows of activations at once, so that every load of either
// serves many products and no sum needs folding across a register.
//
// Both sides are repacked for it, from the blocks of the weights and the
// rows of activations quantizeRow writes. A panel of rows of weights is
// unpacked, one super-block of each of two rows after the other, into
//
//	q [256]uint8 | scales [16]int32 | pairs [8][2]int16 | d, dmin float32 | pad    368 bytes
//
// the quants as unsigned bytes in the order of their values, the scale of
// each group of the super-block, and for each pair of groups of 16 values
// the two numbers their activations' sums are multiplied by: the group's
// min twice over for Q4_K, whose groups are 32 values, and the scales of
// the two groups for Q6_K. Sixteen rows of activations are interleaved,
// super-block by super-block, into
//
//	q [64][16][4]int8 | sums [8][16][2]int16 | d [16]float32                 4672 bytes
//
// four quants of each row after the other, then the sums of each row's
// pairs of groups of 16, then the rows' scales.
//
// The integer sums are the qdots' own, which are exact in any order, and
// each lane of the kernels applies the scales with the float operations
// of qdotQ4KGo or qdotQ6KGo in their order, so the tiles agree with the
// qdots bit for bit.

// A qtile multiplies two rows of nb super-blocks of unpacked weights at w
// by sixteen rows of interleaved activations at x. It adds the products'
// float32 sums for row t of the weights and row c of the activations to
// acc[16t+c] and, for Q4_K, the sums the mins apply to acc[32+16t+c].
type qtile func(nb int, w, x *byte, acc *float32)

const (
	qtileRows       = 16  // rows of activations a qtile multiplies
	qtileWeights    = 2   // rows of weights a qtile multiplies
	qtileRowBytes   = 368 // an unpacked super-block of one row of weights
	qtileBlockBytes = qtileWeights * qtileRowBytes
	qtileActBytes   = 4672 // a super-block of sixteen interleaved rows of activations

	// qtilePanel is the most rows of weights unpacked at a time, and
	// qtileSlab the super-blocks of them multiplied by each tile of
	// activations at a time: a slab of the panel, 64·4·368 bytes, stays
	// in L2 while the activations stream past it, and one of a tile of
	// activations, 4·4672 bytes, in L1 while it meets each pair of rows of
	// the panel.
	qtilePanel = 64
	qtileSlab  = 4

	// qtileMinRows is the fewest rows of activations a product runs in
	// tiles: below it, too few rows share the unpacking of each row of
	// weights for it to pay.
	qtileMinRows = 2 * qtileRows
)

// qtiles holds the tile kernels of the K-quants where the CPU has them;
// the architecture's init fills it, and products by other dtypes, or on
// other CPUs, run in qdots alone.
var qtiles = map[backend.DType]qtile{}

// qtileUnpacks holds the unpackings of rows of weights of each dtype of
// qtiles; the architecture's init replaces them with faster ones where
// it has them.
var qtileUnpacks = map[backend.DType]func(dst, w []byte, nb int){
	backend.Q4_K: unpackQ4KRow,
	backend.Q6_K: unpackQ6KRow,
}

// qtileSets holds the sets of tile kernels the CPU can run, by the name of
// their instructions, for the tests.
var qtileSets = map[string]map[backend.DType]qtile{}

// Scratch of products in tiles: the repacked activations, and each
// worker's unpacked panel and float32 sums.
var (
	qtileActScratch   = sync.Pool{New: func() any { return new([]byte) }}
	qtilePanelScratch = sync.Pool{New: func() any { return new(qtilePanelBuf) }}
)

// qtilePanelBuf is the scratch of one worker.
type qtilePanelBuf struct {
	panel []byte
	accs  []float32
}

// runQTiles computes the rows of out, [m, n], that hold whole tiles of
// activations, the first m/qtileRows·qtileRows, from the activations xq,
// quantized and xstride bytes apart, and the nb super-blocks of each of
// the n rows of weights at w; the qdots compute the rest. It reports the
// rows it computed.
//
// Workers take panels of rows of weights in turn, so that one slowed by
// another program takes fewer; each unpacks its panel and multiplies it
// by every tile of activations, a slab of super-blocks at a time, keeping
// the float32 sums of every tile until the last slab.
func runQTiles(ctx context.Context, b *Backend, tile qtile, dt backend.DType, w []byte, nb, n int, xq []byte, xstride int, out []float32) (int, error) {
	m := len(out) / n
	groups := m / qtileRows
	gbytes := nb * qtileActBytes
	scratch := qtileActScratch.Get().(*[]byte)
	defer qtileActScratch.Put(scratch)
	*scratch = grow(*scratch, groups*gbytes)
	xt := *scratch
	err := b.parallelFor(ctx, groups, 1, func(lo, hi int) error {
		for g := lo; g < hi; g++ {
			packActTile(xt[g*gbytes:(g+1)*gbytes], xq[g*qtileRows*xstride:], xstride, nb)
		}
		return nil
	})
	if err != nil {
		return 0, err
	}

	unpack := qtileUnpacks[dt]
	// Panels as large as leave each worker four or more.
	rows := min(qtilePanel, max(qtileWeights, n/(4*b.workers)/qtileWeights*qtileWeights))
	panels := (n + rows - 1) / rows
	wstride, pbytes := nb*dt.BlockBytes(), nb*qtileBlockBytes
	const accLen = 2 * qtileWeights * qtileRows
	err = b.parallelClaim(ctx, panels, 1, func(p, _ int) error {
		buf := qtilePanelScratch.Get().(*qtilePanelBuf)
		defer qtilePanelScratch.Put(buf)
		buf.panel = grow(buf.panel, rows/qtileWeights*pbytes)
		buf.accs = grow(buf.accs, rows/qtileWeights*groups*accLen)
		panel, accs := buf.panel, buf.accs
		j0 := p * rows
		prows := min(rows, n-j0)
		pairs := (prows + 1) / 2
		for r := range 2 * pairs {
			// An odd row out pairs with itself; its second
			// products are dropped.
			src := w[(j0+min(r, prows-1))*wstride:]
			unpack(panel[r/2*pbytes+r%2*qtileRowBytes:], src, nb)
		}
		clear(accs)
		for b0 := 0; b0 < nb; b0 += qtileSlab {
			sb := min(qtileSlab, nb-b0)
			for g := range groups {
				x := &xt[g*gbytes+b0*qtileActBytes]
				for pr := range pairs {
					tile(sb, &panel[pr*pbytes+b0*qtileBlockBytes], x, &accs[(pr*groups+g)*accLen])
				}
			}
		}
		// Row by row of out, so that its writes run along a row.
		for g := range groups {
			for c := range qtileRows {
				o := out[(g*qtileRows+c)*n+j0:][:prows]
				for pr := range pairs {
					acc := (*[accLen]float32)(accs[(pr*groups+g)*accLen:])
					v0, v1 := acc[c], acc[qtileRows+c]
					if dt == backend.Q4_K {
						v0 -= acc[2*qtileRows+c]
						v1 -= acc[3*qtileRows+c]
					}
					o[2*pr] = v0
					if 2*pr+1 < prows {
						o[2*pr+1] = v1
					}
				}
			}
		}
		return nil
	})
	return groups * qtileRows, err
}

// packActTile interleaves sixteen rows of nb quantized super-blocks of
// activations, xstride bytes apart from x, into dst.
func packActTile(dst, x []byte, xstride, nb int) {
	le := binary.LittleEndian
	for b := range nb {
		d := (*[qtileActBytes]byte)(dst[b*qtileActBytes:])
		for c := range qtileRows {
			src := (*[292]byte)(x[c*xstride+b*292:])
			for q := range 64 {
				le.PutUint32(d[64*q+4*c:], le.Uint32(src[4+4*q:]))
			}
			for p := range 8 {
				le.PutUint32(d[4096+64*p+4*c:], le.Uint32(src[260+4*p:]))
			}
			le.PutUint32(d[4608+4*c:], le.Uint32(src[:]))
		}
	}
}

// unpackQ4KRow unpacks the nb Q4_K super-blocks at w to dst, one every
// qtileBlockBytes.
func unpackQ4KRow(dst, w []byte, nb int) {
	le := binary.LittleEndian
	for b := range nb {
		wb := (*[144]byte)(w[144*b:])
		d := (*[qtileRowBytes]byte)(dst[b*qtileBlockBytes:])
		for i := 0; i < 128; i += 8 {
			v := le.Uint64(wb[16+i:])
			j := i / 32 * 64
			le.PutUint64(d[j+i%32:], v&0x0f0f0f0f0f0f0f0f)
			le.PutUint64(d[j+32+i%32:], v>>4&0x0f0f0f0f0f0f0f0f)
		}
		// The scales and mins of the eight groups, a byte each, as
		// ggml's dot product unpacks them from the 12 bytes of both.
		u0, u1, u2 := le.Uint32(wb[4:]), le.Uint32(wb[8:]), le.Uint32(wb[12:])
		sc := uint64(u0&0x3f3f3f3f) | uint64(u2&0x0f0f0f0f|(u0>>6&0x03030303)<<4)<<32
		mn := uint64(u1&0x3f3f3f3f) | uint64(u2>>4&0x0f0f0f0f|(u1>>6&0x03030303)<<4)<<32
		for g := range 8 {
			c, m := uint16(sc>>(8*g)&0xff), uint32(mn>>(8*g)&0xff)
			le.PutUint32(d[256+4*g:], uint32(c))
			le.PutUint32(d[320+4*g:], m|m<<16)
		}
		clear(d[288:320])
		le.PutUint32(d[352:], math.Float32bits(f16ToF32(le.Uint16(wb[0:]))))
		le.PutUint32(d[356:], math.Float32bits(f16ToF32(le.Uint16(wb[2:]))))
		clear(d[360:])
	}
}

// unpackQ6KRow unpacks the nb Q6_K super-blocks at w to dst, one every
// qtileBlockBytes.
func unpackQ6KRow(dst, w []byte, nb int) {
	le := binary.LittleEndian
	for b := range nb {
		wb := (*[210]byte)(w[210*b:])
		d := (*[qtileRowBytes]byte)(dst[b*qtileBlockBytes:])
		for n := range 2 {
			for l := 0; l < 32; l += 8 {
				l0, l1 := le.Uint64(wb[64*n+l:]), le.Uint64(wb[64*n+32+l:])
				h := le.Uint64(wb[128+32*n+l:])
				const lo, hi = 0x0f0f0f0f0f0f0f0f, 0x0303030303030303
				le.PutUint64(d[128*n+l:], l0&lo|(h&hi)<<4)
				le.PutUint64(d[128*n+32+l:], l1&lo|(h>>2&hi)<<4)
				le.PutUint64(d[128*n+64+l:], l0>>4&lo|(h>>4&hi)<<4)
				le.PutUint64(d[128*n+96+l:], l1>>4&lo|(h>>6&hi)<<4)
			}
		}
		for g := range 16 {
			sc := int32(int8(wb[192+g]))
			le.PutUint32(d[256+4*g:], uint32(sc))
			le.PutUint16(d[320+2*g:], uint16(sc))
		}
		le.PutUint32(d[352:], math.Float32bits(f16ToF32(le.Uint16(wb[208:]))))
		clear(d[356:])
	}
}
