package cpu

import (
	"context"
	"os"
	"sync"
	"sync/atomic"
)

// Float32 matrix products can also run on AMX, the tile matrix unit of
// recent Intel Xeons, when MatMulBFloat16 allows it: the operands are
// rounded to bfloat16 as they are packed, and each product of two of them,
// exact in float32, is summed in float32. A 4-core Xeon multiplies this way
// at several times the speed of float32 with AVX-512, at the precision
// that mixed-precision training uses, so it is off unless asked for.

// amxTile is tileAMX where the CPU and operating system support AMX, and
// nil elsewhere.
var amxTile func(a *uint16, lda int, b *uint16, c *float32, ldc int, k int, cfg *byte, accumulate bool)

// amxConvert is cvtBF16 where amxTile is set: it rounds n float32s, a
// positive multiple of 16, to bfloat16.
var amxConvert func(dst *uint16, src *float32, n int)

// toBF16 sets dst to src rounded to bfloat16, as f32ToBF16 rounds except
// that subnormals, which AMX reads as zero anyway, may become zero.
func toBF16(dst []uint16, src []float32) {
	src = src[:len(dst)]
	if n := len(dst) &^ 15; n > 0 {
		amxConvert(&dst[0], &src[0], n)
		dst, src = dst[n:], src[n:]
	}
	for i, v := range src {
		dst[i] = f32ToBF16(v)
	}
}

// amxOn says whether float32 products may round their operands to
// bfloat16; see SetMatMulBFloat16.
var amxOn atomic.Bool

func init() {
	if os.Getenv("MIMIR_MATMUL_PRECISION") == "bf16" {
		SetMatMulBFloat16(true)
	}
}

// SetMatMulBFloat16 sets whether float32 matrix products on the CPU may
// round their operands to bfloat16 and sum the products in float32, which
// runs several times faster on CPUs with AMX (Intel Xeon from Sapphire
// Rapids on). It reports whether this CPU has AMX; without it, products
// stay in float32 either way. The environment variable
// MIMIR_MATMUL_PRECISION=bf16 turns it on at start.
func SetMatMulBFloat16(on bool) bool {
	amxOn.Store(on)
	return amxTile != nil
}

// MatMulBFloat16 reports whether float32 matrix products round their
// operands to bfloat16: SetMatMulBFloat16 turned it on and the CPU has AMX.
func MatMulBFloat16() bool { return amxTile != nil && amxOn.Load() }

// amxConfig is the tile configuration tileAMX loads: palette 1, and eight
// tiles of 16 rows of 64 bytes.
var amxConfig = func() (c [64]byte) {
	c[0] = 1
	for t := range 8 {
		c[16+2*t] = 64
		c[48+t] = 16
	}
	return c
}()

const (
	// amxMinWork is the fewest multiply-adds per product worth packing for
	// AMX; smaller products stay on the float32 kernels.
	amxMinWork = 1 << 18
	// amxMaxPacked bounds the elements of the packed right operands of
	// one call, so a large batch of products stays on the float32 kernels.
	amxMaxPacked = 64 << 20
)

// amxWorkspace holds a worker's packed rows of the left operand, one block
// of 32 rows per product it sums, and the tile it computes at the ragged
// edges of the output.
type amxWorkspace struct {
	a    []uint16
	tmp  []uint16 // a column of the left operand, rounded
	tile [32 * 32]float32
}

var (
	amxWorkspaces = sync.Pool{New: func() any { return new(amxWorkspace) }}
	amxPacked     = sync.Pool{New: func() any { return new([]uint16) }}
)

// amxGemm computes out like sgemm, on AMX, and reports whether it did: it
// declines products too small to gain from it, and does nothing unless
// MatMulBFloat16 is on. The right operands are packed once, in panels of 32
// columns with each pair of rows interleaved, as the tiles read them; the
// output is split into blocks of 32 rows, each packing its rows of the
// left operands and computing every 32-column tile of its rows.
func (be *Backend) amxGemm(ctx context.Context, out []float32, a operand, offA []int, b operand, offB []int, m, n, k, sums int, epi *evaluator) (bool, error) {
	if !MatMulBFloat16() || m < 16 || n < 16 || k < 16 || m*n*k < amxMinWork || sums == 0 {
		return false, nil
	}
	items := len(offA)
	kp, np := roundUp(k, 32), roundUp(n, 32)
	if items*kp*np > amxMaxPacked {
		return false, nil
	}
	packed := amxPacked.Get().(*[]uint16)
	defer amxPacked.Put(packed)
	*packed = grow(*packed, items*kp*np)
	bp := *packed
	panel := kp * 32 // elements of one packed panel
	err := be.parallelFor(ctx, items*np/32, max(1, packGrain/panel), func(lo, hi int) error {
		var tmp []uint16
		for u := lo; u < hi; u++ {
			amxPackPanel(bp[u*panel:][:panel], b, offB[u/(np/32)], u%(np/32)*32, k, n, &tmp)
		}
		return nil
	})
	if err != nil {
		return true, err
	}
	outputs := items / sums
	blocks := (m + 31) / 32
	return true, be.parallelFor(ctx, outputs*blocks, 1, func(lo, hi int) error {
		ws := amxWorkspaces.Get().(*amxWorkspace)
		defer amxWorkspaces.Put(ws)
		ws.a = grow(ws.a, sums*32*kp)
		for u := lo; u < hi; u++ {
			o, i0 := u/blocks, u%blocks*32
			rows := min(32, m-i0)
			for s := range sums {
				amxPackRows(ws.a[s*32*kp:][:32*kp], kp, a, offA[o*sums+s], i0, rows, k, &ws.tmp)
			}
			c := out[o*m*n:][:m*n]
			for j0 := 0; j0 < n; j0 += 32 {
				cols := min(32, n-j0)
				full := rows == 32 && cols == 32
				dst, ldc := &ws.tile[0], 32*4
				if full {
					dst, ldc = &c[i0*n+j0], n*4
				}
				for s := range sums {
					t := o*sums + s
					amxTile(&ws.a[s*32*kp], kp*2, &bp[(t*np/32+j0/32)*panel], dst, ldc, kp, &amxConfig[0], s > 0)
				}
				if !full {
					for r := range rows {
						copy(c[(i0+r)*n+j0:][:cols], ws.tile[r*32:][:cols])
					}
				}
			}
			if err := epi.apply(o*m*n+i0*n, o*m*n+(i0+rows)*n); err != nil {
				return err
			}
		}
		return nil
	})
}

// amxPackRows sets dst, 32 rows of kp elements, to rows i0 to i0+rows of
// the m×k operand a starting at off, rounded to bfloat16, and zeros past
// them. It walks the operand along its smaller stride.
func amxPackRows(dst []uint16, kp int, a operand, off, i0, rows, k int, tmp *[]uint16) {
	clear(dst)
	off += i0 * a.rs
	switch {
	case a.cs == 1:
		for r := range rows {
			toBF16(dst[r*kp:][:k], a.data[off+r*a.rs:])
		}
	case a.rs == 1:
		// Each column of the block is contiguous: round it whole, then
		// spread it over the rows.
		*tmp = grow(*tmp, rows)
		col := *tmp
		for kk := range k {
			toBF16(col, a.data[off+kk*a.cs:])
			for r, v := range col {
				dst[r*kp+kk] = v
			}
		}
	default:
		for r := range rows {
			row, src := dst[r*kp:][:k], off+r*a.rs
			for kk := range row {
				row[kk] = f32ToBF16(a.data[src+kk*a.cs])
			}
		}
	}
}

// amxPackPanel sets dst, a packed panel, to columns j0 to j0+32 of the k×n
// operand b starting at off, rounded to bfloat16, each pair of rows
// interleaved: element (kk, j0+c) at (kk/2*32+c)*2 + kk%2, with zeros past
// the last row and column. It walks the operand along its smaller stride.
func amxPackPanel(dst []uint16, b operand, off, j0, k, n int, tmp *[]uint16) {
	clear(dst)
	cols := min(32, n-j0)
	off += j0 * b.cs
	switch {
	case b.cs == 1:
		*tmp = grow(*tmp, cols)
		row := *tmp
		for kk := range k {
			toBF16(row, b.data[off+kk*b.rs:])
			d := dst[kk/2*64+kk%2:]
			for c, v := range row {
				d[2*c] = v
			}
		}
	case b.rs == 1:
		// Each column is contiguous, and so are the pairs of rows that the
		// panel interleaves: round a column whole, then move its pairs.
		*tmp = grow(*tmp, roundUp(k, 2))
		col := *tmp
		col[len(col)-1] = 0
		for c := range cols {
			toBF16(col[:k], b.data[off+c*b.cs:])
			for r := range len(col) / 2 {
				dst[r*64+2*c], dst[r*64+2*c+1] = col[2*r], col[2*r+1]
			}
		}
	default:
		for kk := range k {
			src, d := off+kk*b.rs, dst[kk/2*64+kk%2:]
			for c := range cols {
				d[2*c] = f32ToBF16(b.data[src+c*b.cs])
			}
		}
	}
}
