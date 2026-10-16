package cpu

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

var halfTypes = []backend.DType{backend.BFloat16, backend.Float16}

// roundTo returns f rounded to dt.
func roundTo(dt backend.DType, f float32) float32 {
	if dt.Size() == 2 {
		return halfToF32(dt, f32ToHalf(dt, f))
	}
	return f
}

// halfBits returns the elements of a half buffer as their bits.
func halfBits(t *testing.T, b backend.Buffer) []uint16 {
	t.Helper()
	switch b := b.(type) {
	case *buffer[bf16]:
		return bitsOf(b.data)
	case *buffer[f16]:
		return bitsOf(b.data)
	}
	t.Fatalf("not a half buffer: %T", b)
	return nil
}

// sameFloats reports whether got and want hold the same bits, NaN for NaN.
func sameFloats(got, want []float32) bool {
	return slices.EqualFunc(got, want, func(g, w float32) bool {
		return math.Float32bits(g) == math.Float32bits(w) || g != g && w != w
	})
}

// TestHalfOpsComputeInFloat32 checks every elementwise op on half inputs
// against the op on their float32 values, rounded to the half type: the
// kernel loads half elements, computes in float32 and rounds on store.
func TestHalfOpsComputeInFloat32(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	shapes := [][]int{{7}, {3, 40}, {2, 3, 5, 4}, {5, 3}}
	unary := []backend.Op{backend.OpNeg, backend.OpExp2, backend.OpLog2, backend.OpSin, backend.OpSqrt, backend.OpRecip}
	binary := []backend.Op{backend.OpAdd, backend.OpMul, backend.OpMax, backend.OpCmpLT, backend.OpCmpEQ, backend.OpMod}
	for _, dt := range halfTypes {
		for _, shape := range shapes {
			size, _ := backend.NumElements(shape)
			views := stridedViews(shape)
			input := func() (backend.View, backend.Buffer, backend.Buffer) {
				sv := views[rng.IntN(len(views))]
				f := make([]float32, sv.size)
				for i := range f {
					f[i] = roundTo(dt, rng.Float32()*8-4)
				}
				if rng.IntN(5) == 0 {
					f[rng.IntN(len(f))] = float32(math.NaN())
				}
				return sv.view, typedBuf(dt, f, nil), f32buf(f...)
			}
			check := func(name string, k *backend.Kernel, half, float []backend.Buffer) {
				t.Helper()
				want := runFloats(t, 1+rng.IntN(3), k, float...)
				hk := *k
				if k.DType == backend.Float32 {
					hk.DType = dt
				}
				if hk.DType.Size() == 2 {
					for i, w := range want {
						want[i] = roundTo(dt, w)
					}
				}
				if got := runFloats(t, 1+rng.IntN(3), &hk, half...); !sameFloats(got, want) {
					t.Fatalf("%s %s %v: got %v, want %v", dt, name, shape, got, want)
				}
			}
			for _, op := range unary {
				v, h, f := input()
				k := &backend.Kernel{Op: op, DType: backend.Float32, Shape: shape, In: []backend.View{v}}
				check(op.String(), k, []backend.Buffer{h}, []backend.Buffer{f})
			}
			for _, op := range binary {
				v1, h1, f1 := input()
				v2, h2, f2 := input()
				k := &backend.Kernel{Op: op, DType: backend.Float32, Shape: shape, In: []backend.View{v1, v2}}
				check(op.String(), k, []backend.Buffer{h1, h2}, []backend.Buffer{f1, f2})
			}
			v1, h1, f1 := input()
			v2, h2, f2 := input()
			v3, h3, f3 := input()
			k := &backend.Kernel{Op: backend.OpWhere, DType: backend.Float32, Shape: shape, In: []backend.View{v1, v2, v3}}
			check("where", k, []backend.Buffer{h1, h2, h3}, []backend.Buffer{f1, f2, f3})
			v, h, f := input()
			k = &backend.Kernel{Op: backend.OpContiguous, DType: backend.Float32, Shape: shape, In: []backend.View{v}}
			check("contiguous", k, []backend.Buffer{h}, []backend.Buffer{f})

			// Casts from a half type to every type, and to it from every
			// type, as the op on the float32 values.
			for _, to := range allDTypes {
				v, h, f := input()
				k := &backend.Kernel{Op: backend.OpCast, DType: to, Shape: shape, In: []backend.View{v}}
				if to == backend.Int32 {
					want := runOn[int32](t, 1, k, f)
					if got := runOn[int32](t, 1, k, h); !slices.Equal(got, want) {
						t.Fatalf("%s to int32 %v: got %v, want %v", dt, shape, got, want)
					}
					continue
				}
				check("cast to "+to.String(), k, []backend.Buffer{h}, []backend.Buffer{f})
			}
			for _, from := range allDTypes {
				sv := views[rng.IntN(len(views))]
				f := make([]float32, sv.size)
				n := make([]int32, sv.size)
				for i := range f {
					f[i] = roundTo(from, rng.Float32()*8-4)
					n[i] = int32(rng.IntN(300) - 150)
				}
				if from == backend.Int32 {
					for i, x := range n {
						f[i] = float32(x)
					}
				}
				k := &backend.Kernel{Op: backend.OpCast, DType: dt, Shape: shape, In: []backend.View{sv.view}}
				want := make([]float32, 0, size)
				eachIndex(shape, func(idx []int) { want = append(want, roundTo(dt, f[offsetOf(sv.view, idx)])) })
				if got := runFloats(t, 1, k, typedBuf(from, f, n)); !sameFloats(got, want) {
					t.Fatalf("%s from %s %v: got %v, want %v", dt, from, shape, got, want)
				}
			}
		}
	}
}

// TestHalfLoadsReadBeforeTheirOutputIsWritten updates a half input in
// place with a program that reads it after writing a stored value over
// it, so the load is copied before the store.
func TestHalfLoadsReadBeforeTheirOutputIsWritten(t *testing.T) {
	for _, dt := range halfTypes {
		shape := []int{3, 1100}
		size, _ := backend.NumElements(shape)
		x := make([]float32, size)
		y := make([]float32, size)
		for i := range x {
			x[i] = roundTo(dt, float32(i%37)/8-2)
			y[i] = roundTo(dt, float32(i%11)/4-1)
		}
		xb, yb := typedBuf(dt, x, nil), typedBuf(dt, y, nil)
		prog := []backend.Instr{
			{Op: backend.OpLoad, DType: dt, Args: []int{0}},
			{Op: backend.OpLoad, DType: dt, Args: []int{1}},
			{Op: backend.OpMul, DType: dt, Args: []int{0, 1}},
			{Op: backend.OpAdd, DType: dt, Args: []int{2, 0}},
		}
		k := &backend.Kernel{Op: backend.OpFused, DType: dt, Shape: shape, In: []backend.View{backend.Contiguous(shape), backend.Contiguous(shape)}, Attrs: backend.Attrs{Program: prog, Stores: []int{2}}}
		// The product is stored over x, which the sum then reads.
		b := New()
		b.workers = 3
		out, err := b.Alloc(dt, size)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Run(t.Context(), k, []backend.Buffer{out, xb}, []backend.Buffer{xb, yb}); err != nil {
			t.Fatal(err)
		}
		for i := range x {
			p := roundTo(dt, x[i]*y[i])
			if got := floatsOf(t, xb)[i]; got != p {
				t.Fatalf("%s: stored product %d = %v, want %v", dt, i, got, p)
			}
			if got, want := floatsOf(t, out)[i], roundTo(dt, p+x[i]); got != want {
				t.Fatalf("%s: result %d = %v, want %v", dt, i, got, want)
			}
		}
	}
}

func TestHalfReductionsAccumulateInFloat32(t *testing.T) {
	shapes := [][]int{{7}, {3, 5}, {2, 3, 4}, {3, 2500}, {1100, 3}}
	for _, dt := range halfTypes {
		for _, shape := range shapes {
			for _, sv := range stridedViews(shape) {
				xf := make([]float32, sv.size)
				for i := range xf {
					// Small integers, so that sums are exact in any order.
					xf[i] = float32((i*7)%23 - 11)
				}
				x := typedBuf(dt, xf, nil)
				for _, axes := range subsets(len(shape)) {
					out := slices.Clone(shape)
					for _, a := range axes {
						out[a] = 1
					}
					name := fmt.Sprintf("%s %v %s axes %v", dt, shape, sv.name, axes)
					k := &backend.Kernel{Op: backend.OpReduceSum, DType: backend.Float32, Shape: out, In: []backend.View{sv.view}, Attrs: backend.Attrs{Axes: axes}}
					got := runOn[float32](t, 3, k, x)
					if want := refReduce(xf, sv.view, axes, func(a, b float32) float32 { return a + b }); !slices.Equal(got, want) {
						t.Fatalf("%s: sum %v, want %v", name, got, want)
					}
					k.Op = backend.OpReduceMax
					got = runOn[float32](t, 3, k, x)
					if want := refReduce(xf, sv.view, axes, maxOf[float32]); !slices.Equal(got, want) {
						t.Fatalf("%s: max %v, want %v", name, got, want)
					}
					// The output is Float32: a half output is an error.
					k.DType = dt
					if _, err := run[float32](t, k, x); !errors.Is(err, backend.ErrDType) {
						t.Fatalf("%s: half output: got %v, want ErrDType", name, err)
					}
				}
			}
		}
	}
	// 256 + 1 is not a bfloat16, nor 2048 + 1 a binary16, so a running
	// sum in the half type would drop every 1.
	for _, c := range []struct {
		dt   backend.DType
		big  float32
		want float32
	}{{backend.BFloat16, 256, 264}, {backend.Float16, 2048, 2056}} {
		x := slices.Concat([]float32{c.big}, slices.Repeat([]float32{1}, 8))
		k := &backend.Kernel{Op: backend.OpReduceSum, DType: backend.Float32, Shape: []int{1}, In: []backend.View{backend.Contiguous([]int{9})}, Attrs: backend.Attrs{Axes: []int{0}}}
		if got := runOn[float32](t, 1, k, typedBuf(c.dt, x, nil)); got[0] != c.want {
			t.Errorf("%s: sum = %v, want %v", c.dt, got, c.want)
		}
	}
}

// TestHalfMovesKeepBits checks that pad, gather, unfold and contiguous
// move half elements without converting them, whatever their bits, as
// the same ops on int32 elements of those bits do.
func TestHalfMovesKeepBits(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, dt := range halfTypes {
		src := []int{2, 3, 7, 6}
		views := stridedViews(src)
		for _, sv := range views {
			bits := make([]uint16, sv.size)
			ints := make([]int32, sv.size)
			for i := range bits {
				bits[i] = uint16(rng.Uint32()) // NaNs, infinities, subnormals and all
				ints[i] = int32(bits[i])
			}
			half := halfBuf(dt, bits)
			ks := map[string]*backend.Kernel{
				"contiguous": {Op: backend.OpContiguous, Shape: src, In: []backend.View{sv.view}},
				"pad":        {Op: backend.OpPad, Shape: []int{3, 4, 9, 6}, In: []backend.View{sv.view}, Attrs: backend.Attrs{Pads: [][2]int{{1, 0}, {0, 1}, {2, 0}, {0, 0}}}},
				"unfold":     {Op: backend.OpUnfold, Shape: []int{2, 3, 3, 2, 3, 3}, In: []backend.View{sv.view}, Attrs: window([]int{3, 2}, []int{2, 2}, []int{1, 2}, [2]int{1, 1}, [2]int{0, 1})},
			}
			idx := i32buf(5, 0, 3, 3, 1, 2, 4)
			ks["gather"] = &backend.Kernel{Op: backend.OpGather, Shape: []int{2, 3, 7, 1}, In: []backend.View{sv.view, backend.Contiguous([]int{1, 1, 7, 1}).Expand([]int{2, 3, 7, 1})}, Attrs: backend.Attrs{Axis: 3}}
			for name, k := range ks {
				k.DType = backend.Int32
				ins := []backend.Buffer{i32buf(ints...)}
				if name == "gather" {
					ins = append(ins, idx)
				}
				want := runOn[int32](t, 3, k, ins...)
				k.DType = dt
				ins[0] = half
				n, _ := backend.NumElements(k.Shape)
				b := New()
				out, err := b.Alloc(dt, n)
				if err != nil {
					t.Fatal(err)
				}
				if err := b.Run(t.Context(), k, []backend.Buffer{out}, ins); err != nil {
					t.Fatal(err)
				}
				got := halfBits(t, out)
				for i, w := range want {
					if got[i] != uint16(w) {
						t.Fatalf("%s %s %s: element %d = %#04x, want %#04x", dt, name, sv.name, i, got[i], uint16(w))
					}
				}
			}
		}
	}
}

// halfBuf returns a half buffer of dt holding bits.
func halfBuf(dt backend.DType, bits []uint16) backend.Buffer {
	if dt == backend.BFloat16 {
		b := make([]bf16, len(bits))
		for i, h := range bits {
			b[i] = bf16(h)
		}
		return &buffer[bf16]{data: b}
	}
	b := make([]f16, len(bits))
	for i, h := range bits {
		b[i] = f16(h)
	}
	return &buffer[f16]{data: b}
}

func TestHalfFillsRound(t *testing.T) {
	for _, dt := range halfTypes {
		third := roundTo(dt, 1.0/3)
		one := backend.Contiguous([]int{1})
		for name, k := range map[string]*backend.Kernel{
			"const":  {Op: backend.OpConst, DType: dt, Shape: []int{3}, Attrs: backend.Attrs{Value: 1.0 / 3}},
			"pad":    {Op: backend.OpPad, DType: dt, Shape: []int{3}, In: []backend.View{one}, Attrs: backend.Attrs{Pads: [][2]int{{1, 1}}, Value: 1.0 / 3}},
			"unfold": {Op: backend.OpUnfold, DType: dt, Shape: []int{3, 1}, In: []backend.View{one}, Attrs: backend.Attrs{Kernel: []int{3}, Stride: []int{1}, Dilation: []int{1}, Pads: [][2]int{{1, 1}}, Value: 1.0 / 3}},
		} {
			ins := []backend.Buffer{typedBuf(dt, []float32{5}, nil)}[:len(k.In)]
			got := runFloats(t, 1, k, ins...)
			want := []float32{third, 5, third}
			if name == "const" {
				want[1] = third
			}
			if !slices.Equal(got, want) {
				t.Errorf("%s %s: got %v, want %v", dt, name, got, want)
			}
		}
		k := &backend.Kernel{Op: backend.OpConst, DType: dt, Shape: []int{1}, Attrs: backend.Attrs{Value: 1e30}}
		want := roundTo(dt, 1e30) // infinity in binary16
		if got := runFloats(t, 1, k); got[0] != want {
			t.Errorf("%s const 1e30 = %v, want %v", dt, got, want)
		}
	}
}

// TestHalfSumsRoundOnce checks that fold and scatter_add sum half
// elements in float32 and round the sum, rather than each addition.
func TestHalfSumsRoundOnce(t *testing.T) {
	for _, dt := range halfTypes {
		// Three quarters of an ulp of 1 in three parts: each alone rounds
		// away, together they round up.
		ulp := float32(1.0 / 128)
		if dt == backend.Float16 {
			ulp = 1.0 / 1024
		}
		part := ulp / 4
		k := &backend.Kernel{
			Op: backend.OpScatterAdd, DType: dt, Shape: []int{2},
			In:    []backend.View{backend.Contiguous([]int{2}), backend.Contiguous([]int{3}), backend.Contiguous([]int{3})},
			Attrs: backend.Attrs{Axis: 0},
		}
		got := runFloats(t, 1, k, typedBuf(dt, []float32{1, 0}, nil), i32buf(0, 0, 0), typedBuf(dt, []float32{part, part, part}, nil))
		if want := []float32{1 + ulp, 0}; !slices.Equal(got, want) {
			t.Errorf("%s scatter_add: got %v, want %v", dt, got, want)
		}
		// Windows of 4 at stride 1 over 4 positions, with padding of 2:
		// kernel offset k at position o reads input o - 2 + k, so input
		// 1 receives window elements (0, 3), (1, 2), (2, 1) and (3, 0) of
		// the [K, O] windows.
		k = &backend.Kernel{
			Op: backend.OpFold, DType: dt, Shape: []int{3},
			In:    []backend.View{backend.Contiguous([]int{4, 4})},
			Attrs: backend.Attrs{Kernel: []int{4}, Stride: []int{1}, Dilation: []int{1}, Pads: [][2]int{{2, 2}}},
		}
		win := make([]float32, 16)
		win[3], win[6], win[9], win[12] = 1, part, part, part
		got = runFloats(t, 1, k, typedBuf(dt, win, nil))
		if want := []float32{0, 1 + ulp, 0}; !slices.Equal(got, want) {
			t.Errorf("%s fold: got %v, want %v", dt, got, want)
		}
	}
}

func TestHalfFoldMatchesFloat32(t *testing.T) {
	for _, dt := range halfTypes {
		for _, c := range windowCases {
			w := c.windows()
			for _, sv := range stridedViews(w) {
				y := make([]float32, sv.size)
				for i := range y {
					y[i] = roundTo(dt, float32(i%17-8)/3)
				}
				k := &backend.Kernel{Op: backend.OpFold, DType: backend.Float32, Shape: c.in, In: []backend.View{sv.view}, Attrs: c.attrs}
				want := runOn[float32](t, 3, k, f32buf(y...))
				for i, v := range want {
					want[i] = roundTo(dt, v)
				}
				k.DType = dt
				if got := runFloats(t, 3, k, typedBuf(dt, y, nil)); !slices.Equal(got, want) {
					t.Fatalf("%s %v %s:\ngot  %v\nwant %v", dt, c.in, sv.name, got, want)
				}
			}
		}
	}
}

func TestHalfProductsAndRandomNumbersAreRejected(t *testing.T) {
	for _, dt := range halfTypes {
		square := backend.Contiguous([]int{2, 2})
		x := typedBuf(dt, []float32{1, 2, 3, 4}, nil)
		k := &backend.Kernel{Op: backend.OpMatMul, DType: dt, Shape: []int{2, 2}, In: []backend.View{square, square}}
		if _, err := runWith[float32](t, New(), k, x, x); !errors.Is(err, backend.ErrDType) {
			t.Errorf("%s product: got %v, want ErrDType", dt, err)
		}
		k = &backend.Kernel{Op: backend.OpRandUniform, DType: dt, Shape: []int{4}}
		if _, err := runWith[float32](t, New(), k); !errors.Is(err, backend.ErrDType) {
			t.Errorf("%s random: got %v, want ErrDType", dt, err)
		}
	}
}

func TestHalfBuffersHoldTwoByteElements(t *testing.T) {
	b := New()
	for _, dt := range halfTypes {
		buf, err := b.Alloc(dt, 3)
		if err != nil {
			t.Fatal(err)
		}
		if buf.DType() != dt || buf.Len() != 3 {
			t.Fatalf("%s buffer is %s[%d]", dt, buf.DType(), buf.Len())
		}
		if err := b.Upload(t.Context(), buf, []byte{0x80, 0x3f, 0x00, 0x40, 0x00, 0xc0}); err != nil {
			t.Fatal(err)
		}
		if err := b.Upload(t.Context(), buf, make([]byte, 12)); !errors.Is(err, backend.ErrShape) {
			t.Fatalf("%s: upload of 12 bytes: %v", dt, err)
		}
		if got := halfBits(t, buf); !slices.Equal(got, []uint16{0x3f80, 0x4000, 0xc000}) {
			t.Fatalf("%s: uploaded %#04x", dt, got)
		}
		raw := make([]byte, 6)
		if err := b.Download(t.Context(), raw, buf); err != nil || !slices.Equal(raw, []byte{0x80, 0x3f, 0x00, 0x40, 0x00, 0xc0}) {
			t.Fatalf("%s: downloaded %x, %v", dt, raw, err)
		}
	}
}
