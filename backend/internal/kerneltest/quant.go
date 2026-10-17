package kerneltest

import (
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/quanttest"
)

// quantTypes are the block-quantized dtypes.
var quantTypes = []backend.DType{backend.Q8_0, backend.Q4_K, backend.Q6_K}

// QuantBuffer returns a Host buffer of n random elements of the
// block-quantized dtype dt, with finite scales.
func QuantBuffer(t *testing.T, rng *rand.Rand, dt backend.DType, n int) backend.Buffer {
	t.Helper()
	b, err := Host.Alloc(dt, n)
	if err != nil {
		t.Fatal(err)
	}
	if err := Host.Upload(t.Context(), b, quanttest.Random(rng, dt, n)); err != nil {
		t.Fatal(err)
	}
	return b
}

// exactRows returns m rows of k float32s that the CPU backend rounds to
// int8 without error for a product by dt: in each block of the
// activations, 32 values for Q8_0 and 256 for the K-quants, integers
// over 128 of which one is ±127/128, so the block's scale is 1/128. A
// product by them on the CPU differs from one in float32 only in the
// order of its operations.
func exactRows(rng *rand.Rand, dt backend.DType, m, k int) []float32 {
	bs := 256
	if dt == backend.Q8_0 {
		bs = 32
	}
	x := make([]float32, m*k)
	for i := range x {
		x[i] = float32(rng.IntN(255)-127) / 128
	}
	for b := 0; b < len(x); b += bs {
		x[b+rng.IntN(bs)] = float32(2*rng.IntN(2)-1) * 127 / 128
	}
	return x
}

// floatBuffer returns a Host buffer holding v.
func floatBuffer(t *testing.T, v []float32) backend.Buffer {
	t.Helper()
	b, err := Host.Alloc(backend.Float32, len(v))
	if err != nil {
		t.Fatal(err)
	}
	if err := Host.Upload(t.Context(), b, Bytes(v)); err != nil {
		t.Fatal(err)
	}
	return b
}

// Quant returns cases of kernels that read block-quantized buffers:
// casts that dequantize them, which match the CPU backend bit for bit,
// gathers from them, and products by them.
func Quant(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(13, 14))
	f, i := backend.Float32, backend.Int32
	const n, k = 9, 1280 // 40 sub-blocks of 32 values a row
	var cases []Case
	for _, dt := range quantTypes {
		// Rows 1 to 9 of 10, so the weights start a row in.
		w := QuantBuffer(t, rng, dt, (n+1)*k)
		all := backend.Contiguous([]int{n + 1, k})
		rows := all.Shrink([][2]int{{1, n + 1}, {0, k}})
		for name, v := range map[string]backend.View{
			"contiguous": all,
			"shrunk":     rows,
			"transposed": all.Permute([]int{1, 0}),
		} {
			for _, to := range []backend.DType{f, backend.BFloat16, backend.Float16} {
				if to != f && name != "contiguous" {
					continue
				}
				cases = append(cases, Case{
					Name: fmt.Sprintf("dequantize/%s/%s/%s", dt, to, name),
					K:    backend.Kernel{Op: backend.OpCast, DType: to, Shape: v.Shape, In: []backend.View{v}},
					Ins:  []backend.Buffer{w},
				})
			}
		}
		// Rows 2, 0 and 9 of the table, as an embedding reads them, and
		// a row out of range.
		vi := backend.View{Shape: []int{3, k}, Strides: []int{1, 0}}
		for name, idx := range map[string][]int32{"rows": {2, 0, 9}, "out of range": {2, 10, 0}} {
			ib, err := Host.Alloc(i, 3)
			if err != nil {
				t.Fatal(err)
			}
			raw := make([]byte, 12)
			for j, x := range idx {
				binary.LittleEndian.PutUint32(raw[4*j:], uint32(x))
			}
			if err := Host.Upload(t.Context(), ib, raw); err != nil {
				t.Fatal(err)
			}
			cases = append(cases, Case{
				Name: fmt.Sprintf("gather/%s/%s", dt, name),
				K:    backend.Kernel{Op: backend.OpGather, DType: f, Shape: []int{3, k}, In: []backend.View{all, vi}, Attrs: backend.Attrs{Axis: 0}},
				Ins:  []backend.Buffer{w, ib},
			})
		}
		// Products by rows 1 to 9 of x of shapes [K], [1, K], [2, 3, K]
		// and [3, K] read transposed.
		for _, shape := range [][]int{{k}, {1, k}, {2, 3, k}} {
			m := 1
			for _, d := range shape[:len(shape)-1] {
				m *= d
			}
			out := slices.Clone(shape)
			out[len(out)-1] = n
			cases = append(cases, Case{
				Name: fmt.Sprintf("qmatmul/%s/%v", dt, shape),
				K:    backend.Kernel{Op: backend.OpQMatMul, DType: f, Shape: out, In: []backend.View{backend.Contiguous(shape), rows}},
				Ins:  []backend.Buffer{floatBuffer(t, exactRows(rng, dt, m, k)), w},
				Tol:  1e-4,
			})
		}
		x := exactRows(rng, dt, 3, k)
		xt := make([]float32, len(x))
		for r := range 3 {
			for p := range k {
				xt[p*3+r] = x[r*k+p]
			}
		}
		cases = append(cases, Case{
			Name: fmt.Sprintf("qmatmul/%s/transposed", dt),
			K: backend.Kernel{Op: backend.OpQMatMul, DType: f, Shape: []int{3, n},
				In: []backend.View{backend.Contiguous([]int{k, 3}).Permute([]int{1, 0}), rows}},
			Ins: []backend.Buffer{floatBuffer(t, xt), w},
			Tol: 1e-4,
		})
	}
	return cases
}
