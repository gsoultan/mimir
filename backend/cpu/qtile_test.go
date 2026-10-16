package cpu

import (
	"encoding/binary"
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

// withQTiles runs fn with the tile kernels set to tiles, nil for none.
func withQTiles(tiles map[backend.DType]qtile, fn func()) {
	saved := qtiles
	defer func() { qtiles = saved }()
	if tiles == nil {
		tiles = map[backend.DType]qtile{}
	}
	qtiles = tiles
	fn()
}

func TestQTilesMatchQDots(t *testing.T) {
	sets := map[string]map[backend.DType]qtile{
		"go": {backend.Q4_K: qtileGo(backend.Q4_K), backend.Q6_K: qtileGo(backend.Q6_K)},
	}
	maps.Copy(sets, qtileSets)
	t.Logf("tile kernels %v", slices.Sorted(maps.Keys(sets)))
	rng := rand.New(rand.NewPCG(15, 16))
	for _, dt := range []backend.DType{backend.Q4_K, backend.Q6_K} {
		for _, sz := range [][3]int{{32, 1, 1}, {32, 2, 2}, {33, 3, 1}, {50, 33, 3}, {48, 5, 9}, {64, 131, 1}} {
			m, n, k := sz[0], sz[1], sz[2]*dt.BlockSize()
			sw, _, _ := saturated(dt, 1, n*k)
			for _, c := range []struct {
				name string
				w    []byte
				x    func(i int) float32
			}{
				{"random", quanttest.Random(rng, dt, n*k), func(int) float32 { return rng.Float32()*2 - 1 }},
				// Rows of -1 and of 1, quantized to -127 and 127.
				{"saturated", sw, func(i int) float32 { return float32(1 - 2*(i/k%2)) }},
				{"special", quanttest.Random(rng, dt, n*k), func(i int) float32 {
					switch i {
					case 3:
						return float32(math.NaN())
					case 5*k + 7:
						return float32(math.Inf(-1))
					}
					return rng.Float32()*2 - 1
				}},
			} {
				x := make([]float32, m*k)
				for i := range x {
					x[i] = c.x(i)
				}
				kern := &backend.Kernel{Op: backend.OpQMatMul, DType: backend.Float32, Shape: []int{m, n},
					In: []backend.View{backend.Contiguous([]int{m, k}), backend.Contiguous([]int{n, k})}}
				var want []float32
				withQTiles(nil, func() {
					var err error
					if want, err = run[float32](t, kern, f32buf(x...), qbuf(t, dt, c.w)); err != nil {
						t.Fatal(err)
					}
				})
				for isa, tiles := range sets {
					withQTiles(tiles, func() {
						got, err := run[float32](t, kern, f32buf(x...), qbuf(t, dt, c.w))
						if err != nil {
							t.Fatal(err)
						}
						for i, g := range got {
							if math.Float32bits(g) != math.Float32bits(want[i]) && !(g != g && want[i] != want[i]) {
								t.Fatalf("%s %s %s %v: element %d (%d, %d) = %v, want %v",
									isa, dt, c.name, sz, i, i/n, i%n, g, want[i])
							}
						}
					})
				}
			}
		}
	}
}

// BenchmarkQTiles compares products in tiles with the same products by
// the qdots alone, at the shapes of a 7B model's projections, which
// llama.cpp's test-backend-ops measures too.
func BenchmarkQTiles(b *testing.B) {
	for _, dt := range quantTypes {
		for _, sz := range [][3]int{{1, 4096, 14336}, {512, 4096, 14336}, {512, 14336, 4096}} {
			for _, path := range []string{"qdots", "tiles"} {
				if path == "tiles" && (sz[0] < qtileMinRows || qtiles[dt] == nil) {
					continue
				}
				b.Run(fmt.Sprintf("%s/%s/%dx%dx%d", dt, path, sz[0], sz[1], sz[2]), func(b *testing.B) {
					tiles := qtiles
					if path == "qdots" {
						tiles = nil
					}
					withQTiles(tiles, func() { benchQMatMul(b, dt, sz[0], sz[1], sz[2]) })
				})
			}
		}
	}
}

// qtileGo is the portable tile kernel of dt, which the tests check the
// assembly against: qdotQ4KGo or qdotQ6KGo, lane by lane, over the
// repacked layouts.
func qtileGo(dt backend.DType) qtile {
	return func(nb int, w, x *byte, acc *float32) {
		wp := unsafe.Slice(w, nb*qtileBlockBytes)
		xp := unsafe.Slice(x, nb*qtileActBytes)
		a := unsafe.Slice(acc, 2*qtileWeights*qtileRows)
		for b := range nb {
			xb := xp[b*qtileActBytes:]
			for t := range qtileWeights {
				wb := wp[b*qtileBlockBytes+t*qtileRowBytes:]
				for c := range qtileRows {
					var sumi, summ int32
					for v := range 256 {
						s := v / 32
						if dt == backend.Q6_K {
							s = v / 16
						}
						sc := int32(binary.LittleEndian.Uint32(wb[256+4*s:]))
						sumi += sc * int32(wb[v]) * int32(int8(xb[64*(v/4)+4*c+v%4]))
					}
					for p := range 8 {
						for h := range 2 {
							summ += i16At(wb[320+4*p+2*h:]) * i16At(xb[4096+64*p+4*c+2*h:])
						}
					}
					xd := f32At(xb[4608+4*c:])
					dd := f32At(wb[352:]) * xd
					if dt == backend.Q6_K {
						a[qtileRows*t+c] += float32(dd * float32(sumi-32*summ))
						continue
					}
					dm := f32At(wb[356:]) * xd
					a[qtileRows*t+c] += float32(dd * float32(sumi))
					a[qtileWeights*qtileRows+qtileRows*t+c] += float32(dm * float32(summ))
				}
			}
		}
	}
}

func TestQTileUnpacksMatchGo(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 18))
	ref := map[backend.DType]func(dst, w []byte, nb int){backend.Q4_K: unpackQ4KRow, backend.Q6_K: unpackQ6KRow}
	// Scales of every kind: infinities, NaNs quiet and signaling, zeros
	// and subnormals of both signs.
	special := []uint16{0x7c00, 0xfc00, 0x7e00, 0x7d01, 0x0000, 0x8000, 0x0001, 0x83ff, 0x3c00}
	for dt, unpack := range qtileUnpacks {
		bb := dt.BlockBytes()
		dpos := []int{0, 2}
		if dt == backend.Q6_K {
			dpos = []int{208}
		}
		for _, nb := range []int{1, 2, 7, 16} {
			w := quanttest.Random(rng, dt, nb*256)
			if nb == 7 {
				// Every bit of the quants and scales set.
				for i := range w {
					w[i] = 0xff
				}
			}
			for b := range nb {
				for _, p := range dpos {
					binary.LittleEndian.PutUint16(w[b*bb+p:], special[rng.IntN(len(special))])
				}
			}
			got, want := make([]byte, nb*qtileBlockBytes), make([]byte, nb*qtileBlockBytes)
			for i := range got {
				got[i] = byte(rng.Uint32())
			}
			copy(want, got)
			unpack(got, w, nb)
			ref[dt](want, w, nb)
			for i := 0; i < len(got); i += 4 {
				g, v := binary.LittleEndian.Uint32(got[i:]), binary.LittleEndian.Uint32(want[i:])
				// A signaling NaN may come out quiet.
				inD := i%qtileRowBytes >= qtileRowBytes-16 && i%qtileRowBytes < qtileRowBytes-8
				if g != v && !(inD && math.IsNaN(float64(math.Float32frombits(g))) && math.IsNaN(float64(math.Float32frombits(v)))) {
					t.Fatalf("%s nb %d: bytes %d..%d = %#08x, want %#08x", dt, nb, i, i+4, g, v)
				}
			}
		}
	}
}
