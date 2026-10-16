package cpu

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/quanttest"
)

var quantTypes = []backend.DType{backend.Q8_0, backend.Q4_K, backend.Q6_K}

// qbuf returns a buffer of the block type dt over raw.
func qbuf(t *testing.T, dt backend.DType, raw []byte) backend.Buffer {
	t.Helper()
	b, err := Wrap(dt, raw)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestQuantizedBuffersHoldWholeBlocks(t *testing.T) {
	be := New()
	for _, dt := range quantTypes {
		bs, bb := dt.BlockSize(), dt.BlockBytes()
		for _, blocks := range []int{0, 3, offHeapMin/bb + 1} {
			b, err := be.Alloc(dt, blocks*bs)
			if err != nil {
				t.Fatal(err)
			}
			raw := quanttest.Random(rand.New(rand.NewPCG(1, uint64(blocks))), dt, blocks*bs)
			if b.Len() != blocks*bs || b.DType() != dt {
				t.Fatalf("Alloc(%s, %d) = %s[%d]", dt, blocks*bs, b.DType(), b.Len())
			}
			if err := be.Upload(t.Context(), b, raw); err != nil {
				t.Fatal(err)
			}
			got := make([]byte, len(raw))
			if err := be.Download(t.Context(), got, b); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, raw) {
				t.Fatalf("%s: download differs from upload", dt)
			}
			be.Free(b)
		}
		if _, err := be.Alloc(dt, bs+1); !errors.Is(err, backend.ErrShape) {
			t.Errorf("Alloc(%s, %d) = %v, want ErrShape", dt, bs+1, err)
		}
		if _, err := Wrap(dt, make([]byte, bb+1)); !errors.Is(err, backend.ErrShape) {
			t.Errorf("Wrap(%s) of %d bytes = %v, want ErrShape", dt, bb+1, err)
		}
		// Blocks need no alignment.
		raw := make([]byte, 2*bb+1)
		if b, err := Wrap(dt, raw[1:]); err != nil || b.Len() != 2*bs {
			t.Errorf("Wrap(%s) at an odd address = %v, %v", dt, b, err)
		}
	}
}

func TestDequantizeMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, dt := range quantTypes {
		bs := dt.BlockSize()
		n := 5 * bs
		raw := quanttest.Random(rng, dt, n)
		want := quanttest.Dequantize(dt, raw)
		in := qbuf(t, dt, raw)
		for _, out := range []backend.DType{backend.Float32, backend.BFloat16, backend.Float16} {
			views := map[string]backend.View{
				"contiguous": backend.Contiguous([]int{5, bs}),
				// Not whole blocks: element by element.
				"shrunk":     backend.Contiguous([]int{5, bs}).Shrink([][2]int{{1, 4}, {3, bs - 5}}),
				"transposed": backend.Contiguous([]int{5, bs}).Permute([]int{1, 0}),
			}
			for name, v := range views {
				t.Run(fmt.Sprintf("%s/%s/%s", dt, out, name), func(t *testing.T) {
					k := &backend.Kernel{Op: backend.OpCast, DType: out, Shape: v.Shape, In: []backend.View{v}}
					got := runFloats(t, 4, k, in)
					wantV := make([]float32, 0, len(got))
					c := newCursor(v.Shape, []backend.View{v}, 0)
					for range len(got) {
						wantV = append(wantV, roundTo(out, want[c.offs[0]]))
						c.next()
					}
					if !sameFloats(got, wantV) {
						for i := range got {
							if math.Float32bits(got[i]) != math.Float32bits(wantV[i]) {
								t.Fatalf("element %d = %v, want %v", i, got[i], wantV[i])
							}
						}
					}
				})
			}
		}
	}
}

func TestGatherFromQuantized(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, dt := range quantTypes {
		bs := dt.BlockSize()
		raw := quanttest.Random(rng, dt, 4*bs)
		table := quanttest.Dequantize(dt, raw)
		// Rows 2, 0 and 2 of a [4, bs] table, as an embedding reads them.
		idx := []int32{2, 0, 2}
		shape := []int{3, bs}
		vi := backend.View{Shape: shape, Strides: []int{1, 0}}
		k := &backend.Kernel{Op: backend.OpGather, DType: backend.Float32, Shape: shape,
			In: []backend.View{backend.Contiguous([]int{4, bs}), vi}}
		got, err := run[float32](t, k, qbuf(t, dt, raw), i32buf(idx...))
		if err != nil {
			t.Fatal(err)
		}
		for i, j := range idx {
			if !sameFloats(got[i*bs:(i+1)*bs], table[int(j)*bs:(int(j)+1)*bs]) {
				t.Fatalf("%s: row %d differs from row %d of the table", dt, i, j)
			}
		}
		idx[1] = 4
		if _, err := run[float32](t, k, qbuf(t, dt, raw), i32buf(idx...)); !errors.Is(err, backend.ErrIndex) {
			t.Errorf("%s: gather of row 4 = %v, want ErrIndex", dt, err)
		}
	}
}

// randomActivations returns rows of k values in [-1, 1) quantized for a
// product by dt, and xstride, the bytes of a row.
func randomActivations(rng *rand.Rand, dt backend.DType, rows, k int) ([]byte, int) {
	bs, bb := actBlock(dt)
	stride := k / bs * bb
	out := make([]byte, rows*stride)
	x := make([]float32, k)
	for r := range rows {
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		quantizeRow(out[r*stride:(r+1)*stride], x, bs, bs == 256)
	}
	return out, stride
}

// saturated returns k weights of dt and rows of activations at the
// extremes of the integers the kernels multiply: the largest scales, -128
// quants of Q8_0, 15 of Q4_K and 63 of Q6_K, and activations of -127, so
// that a sum that overflowed or saturated would show.
func saturated(dt backend.DType, rows, k int) (w, x []byte, xstride int) {
	w = make([]byte, k/dt.BlockSize()*dt.BlockBytes())
	for b := 0; b < len(w); b += dt.BlockBytes() {
		blk := w[b : b+dt.BlockBytes()]
		switch dt {
		case backend.Q8_0:
			for i := range blk {
				blk[i] = 0x80
			}
			blk[0], blk[1] = 0x00, 0x3c // 1
		case backend.Q4_K:
			for i := range blk {
				blk[i] = 0xff
			}
			blk[0], blk[1], blk[2], blk[3] = 0x00, 0x3c, 0x00, 0x38 // 1, 0.5
		case backend.Q6_K:
			for i := range blk {
				blk[i] = 0xff
			}
			for i := 192; i < 208; i++ {
				blk[i] = 0x80 // -128
			}
			blk[208], blk[209] = 0x00, 0x3c
		}
	}
	bs, bb := actBlock(dt)
	xstride = k / bs * bb
	x = make([]byte, rows*xstride)
	v := make([]float32, k)
	for i := range v {
		v[i] = -1
	}
	for r := range rows {
		quantizeRow(x[r*xstride:(r+1)*xstride], v, bs, bs == 256)
	}
	return w, x, xstride
}

func TestQDotKernelsMatchPortable(t *testing.T) {
	all := map[string]map[backend.DType]*[maxRows]qdot{qdotISA: qdots}
	maps.Copy(all, qdotSets)
	t.Logf("kernels %v, %s selected", slices.Sorted(maps.Keys(all)), qdotISA)
	rng := rand.New(rand.NewPCG(7, 8))
	portableKernels := map[backend.DType]func(int, []byte, []byte) float32{
		backend.Q8_0: qdotQ8_0Go, backend.Q4_K: qdotQ4KGo, backend.Q6_K: qdotQ6KGo,
	}
	for _, dt := range quantTypes {
		for _, nb := range []int{0, 1, 2, 7, 33} {
			k := nb * dt.BlockSize()
			w := quanttest.Random(rng, dt, k)
			x, xstride := randomActivations(rng, dt, maxRows, k)
			sw, sx, _ := saturated(dt, maxRows, k)
			for _, c := range []struct {
				name string
				w, x []byte
			}{{"random", w, x}, {"saturated", sw, sx}} {
				want := make([]float32, maxRows)
				for r := range want {
					want[r] = portableKernels[dt](nb, c.w, c.x[r*xstride:])
				}
				// The kernels read no byte of an empty row.
				wp, xp := new(byte), new(byte)
				if nb > 0 {
					wp, xp = &c.w[0], &c.x[0]
				}
				for isa, kernels := range all {
					for rows := 1; rows <= maxRows; rows++ {
						const ldo = 3
						got := make([]float32, maxRows*ldo)
						kernels[dt][rows-1](nb, wp, xp, xstride, &got[0], ldo)
						for r := range maxRows {
							g := got[r*ldo]
							if r >= rows {
								if g != 0 {
									t.Fatalf("%s %s, %s, %d blocks, %d rows: wrote row %d", isa, dt, c.name, nb, rows, r)
								}
								continue
							}
							if math.Float32bits(g) != math.Float32bits(want[r]) && !(g != g && want[r] != want[r]) {
								t.Fatalf("%s %s, %s, %d blocks, %d rows: row %d = %v, want %v", isa, dt, c.name, nb, rows, r, g, want[r])
							}
						}
					}
				}
			}
		}
	}
}

// refProduct returns x·Wᵀ for x [m, k] and the dequantized W [n, k] in
// float64, and for each element a bound on the error of a product that
// rounds x to int8 as quantizeRow does: half a step of each block's
// scale times the weight's magnitude, plus rounding.
func refProduct(dt backend.DType, x []float32, w []float32, m, n, k int) (want, bound []float64) {
	abs, _ := actBlock(dt)
	want, bound = make([]float64, m*n), make([]float64, m*n)
	for i := range m {
		row := x[i*k : (i+1)*k]
		for j := range n {
			var s, e, mag float64
			for p := range k {
				amax := 0.0
				for _, v := range row[p/abs*abs : (p/abs+1)*abs] {
					amax = math.Max(amax, math.Abs(float64(v)))
				}
				wv := float64(w[j*k+p])
				s += float64(row[p]) * wv
				e += math.Abs(wv) * amax / 127 / 2 * 1.0001
				mag += math.Abs(wv) * math.Abs(float64(row[p]))
			}
			want[i*n+j], bound[i*n+j] = s, e+1e-5*mag+1e-30
		}
	}
	return want, bound
}

func TestQMatMulIsCloseToTheDequantizedProduct(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	for _, dt := range quantTypes {
		for _, sz := range [][3]int{{1, 5, 1}, {3, 4, 2}, {9, 7, 3}, {13, 33, 1}} {
			m, n, k := sz[0], sz[1], sz[2]*dt.BlockSize()
			t.Run(fmt.Sprintf("%s/%dx%dx%d", dt, m, n, k), func(t *testing.T) {
				raw := quanttest.Random(rng, dt, n*k)
				w := quanttest.Dequantize(dt, raw)
				x := make([]float32, m*k)
				for i := range x {
					x[i] = rng.Float32()*2 - 1
				}
				want, bound := refProduct(dt, x, w, m, n, k)
				// x is read transposed from a [k, m] buffer, and the
				// weights from the second of two blocks of rows.
				xt := make([]float32, m*k)
				for i := range m {
					for p := range k {
						xt[p*m+i] = x[i*k+p]
					}
				}
				vx := backend.Contiguous([]int{k, m}).Permute([]int{1, 0})
				pad := quanttest.Random(rng, dt, k)
				vw := backend.Contiguous([]int{n, k})
				vw.Offset = k
				kern := &backend.Kernel{Op: backend.OpQMatMul, DType: backend.Float32, Shape: []int{m, n}, In: []backend.View{vx, vw}}
				got, err := run[float32](t, kern, f32buf(xt...), qbuf(t, dt, slices.Concat(pad, raw)))
				if err != nil {
					t.Fatal(err)
				}
				for i, g := range got {
					if math.Abs(float64(g)-want[i]) > bound[i] {
						t.Fatalf("element %d = %v, want %v ± %v", i, g, want[i], bound[i])
					}
				}
			})
		}
	}
}

func TestQMatMulOfBatchesMatchesRowByRow(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	dt := backend.Q4_K
	m, n, k := 2*3*5, 64, 2*dt.BlockSize()
	raw := quanttest.Random(rng, dt, n*k)
	x := make([]float32, m*k)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	kern := &backend.Kernel{Op: backend.OpQMatMul, DType: backend.Float32, Shape: []int{2, 3, 5, n},
		In: []backend.View{backend.Contiguous([]int{2, 3, 5, k}), backend.Contiguous([]int{n, k})}}
	all, err := run[float32](t, kern, f32buf(x...), qbuf(t, dt, raw))
	if err != nil {
		t.Fatal(err)
	}
	for i := range m {
		one := &backend.Kernel{Op: backend.OpQMatMul, DType: backend.Float32, Shape: []int{n},
			In: []backend.View{backend.Contiguous([]int{k}), backend.Contiguous([]int{n, k})}}
		got, err := run[float32](t, one, f32buf(x[i*k:(i+1)*k]...), qbuf(t, dt, raw))
		if err != nil {
			t.Fatal(err)
		}
		if !sameFloats(got, all[i*n:(i+1)*n]) {
			t.Fatalf("row %d of the batch differs from its product alone", i)
		}
	}
}

func TestQMatMulPropagatesNaN(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	for _, dt := range quantTypes {
		k := dt.BlockSize()
		raw := quanttest.Random(rng, dt, 2*k)
		x := make([]float32, 2*k)
		x[3], x[k+5] = float32(math.NaN()), float32(math.Inf(1))
		kern := &backend.Kernel{Op: backend.OpQMatMul, DType: backend.Float32, Shape: []int{2, 2},
			In: []backend.View{backend.Contiguous([]int{2, k}), backend.Contiguous([]int{2, k})}}
		got, err := run[float32](t, kern, f32buf(x...), qbuf(t, dt, raw))
		if err != nil {
			t.Fatal(err)
		}
		for i, g := range got {
			if !math.IsNaN(float64(g)) {
				t.Errorf("%s: element %d = %v, want NaN", dt, i, g)
			}
		}
	}
}

func TestQMatMulRejectsMalformedKernels(t *testing.T) {
	dt := backend.Q4_K
	k := dt.BlockSize()
	w := qbuf(t, dt, make([]byte, 4*dt.BlockBytes()))
	x := f32buf(make([]float32, 2*k)...)
	cases := map[string]*backend.Kernel{
		"inside a block": {Shape: []int{1, 2}, In: []backend.View{backend.Contiguous([]int{1, k / 2}), {Shape: []int{2, k / 2}, Strides: []int{k / 2, 1}}}},
		"misaligned":     {Shape: []int{1, 2}, In: []backend.View{backend.Contiguous([]int{1, k}), {Shape: []int{2, k}, Strides: []int{k, 1}, Offset: 16}}},
		"strided rows":   {Shape: []int{1, 2}, In: []backend.View{backend.Contiguous([]int{1, k}), {Shape: []int{2, k}, Strides: []int{2 * k, 1}}}},
		"wrong width":    {Shape: []int{1, 3}, In: []backend.View{backend.Contiguous([]int{1, k}), backend.Contiguous([]int{2, k})}},
	}
	for name, kern := range cases {
		kern.Op, kern.DType = backend.OpQMatMul, backend.Float32
		if _, err := run[float32](t, kern, x, w); !errors.Is(err, backend.ErrInternal) {
			t.Errorf("%s: %v, want ErrInternal", name, err)
		}
	}
}

// benchQMatMul measures a product of m rows by a weight of n×k of dt,
// reporting the bytes of weights read per second.
func benchQMatMul(b *testing.B, dt backend.DType, m, n, k int) {
	rng := rand.New(rand.NewPCG(1, 2))
	be := New()
	w, err := Wrap(dt, quanttest.Random(rng, dt, n*k))
	if err != nil {
		b.Fatal(err)
	}
	x := make([]float32, m*k)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	out, _ := be.Alloc(backend.Float32, m*n)
	kern := &backend.Kernel{Op: backend.OpQMatMul, DType: backend.Float32, Shape: []int{m, n},
		In: []backend.View{backend.Contiguous([]int{m, k}), backend.Contiguous([]int{n, k})}}
	ins := []backend.Buffer{f32buf(x...), w}
	bytes, _ := dt.Bytes(n * k)
	b.SetBytes(int64(bytes))
	for b.Loop() {
		if err := be.Run(b.Context(), kern, []backend.Buffer{out}, ins); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(2*float64(m)*float64(n)*float64(k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
}

func BenchmarkQMatMul(b *testing.B) {
	for _, dt := range quantTypes {
		for _, sz := range [][3]int{{1, 2048, 2048}, {1, 8192, 2048}, {1, 2048, 8192}, {64, 2048, 2048}, {256, 2048, 2048}} {
			b.Run(fmt.Sprintf("%s/%dx%dx%d", dt, sz[0], sz[1], sz[2]), func(b *testing.B) { benchQMatMul(b, dt, sz[0], sz[1], sz[2]) })
		}
	}
}

// BenchmarkMatVecFloat32 is the float32 matrix-vector product the
// quantized ones replace, reading the weights transposed as a linear
// layer does.
func BenchmarkMatVecFloat32(b *testing.B) {
	for _, sz := range [][2]int{{2048, 2048}, {8192, 2048}, {2048, 8192}} {
		n, k := sz[0], sz[1]
		b.Run(fmt.Sprintf("1x%dx%d", n, k), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))
			be := New()
			x, w := make([]float32, k), make([]float32, n*k)
			for i := range w {
				w[i] = rng.Float32()
			}
			out, _ := be.Alloc(backend.Float32, n)
			kern := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: []int{1, n},
				In: []backend.View{backend.Contiguous([]int{1, k}), backend.Contiguous([]int{n, k}).Permute([]int{1, 0})}}
			ins := []backend.Buffer{f32buf(x...), f32buf(w...)}
			b.SetBytes(int64(4 * n * k))
			for b.Loop() {
				if err := be.Run(b.Context(), kern, []backend.Buffer{out}, ins); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(2*float64(n)*float64(k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}

// BenchmarkMemoryBandwidth reads a buffer far larger than the caches on
// every worker, eight bytes at a time: the ceiling of a product that
// streams its weights from memory.
func BenchmarkMemoryBandwidth(b *testing.B) {
	be := New()
	const n = 1 << 30
	buf, err := be.Alloc(backend.Q8_0, n/34*32)
	if err != nil {
		b.Fatal(err)
	}
	defer be.Free(buf)
	raw := buf.(*blocks).data
	words := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[0])), len(raw)/8)
	for i := range words {
		words[i] = uint64(i)
	}
	sums := make([]uint64, be.workers*8)
	chunk := (len(words) + be.workers - 1) / be.workers
	b.SetBytes(int64(len(words) * 8))
	for b.Loop() {
		_ = be.parallelFor(b.Context(), len(words), chunk, func(lo, hi int) error {
			var s0, s1, s2, s3 uint64
			for i := lo; i+8 <= hi; i += 8 {
				v := words[i : i+8 : i+8]
				s0 += v[0] ^ v[4]
				s1 += v[1] ^ v[5]
				s2 += v[2] ^ v[6]
				s3 += v[3] ^ v[7]
			}
			sums[lo/chunk*8] = s0 + s1 + s2 + s3
			return nil
		})
	}
}
