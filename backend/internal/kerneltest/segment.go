package kerneltest

import (
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// intBuffer returns a Host buffer holding the Int32s v.
func intBuffer(t *testing.T, v []int32) backend.Buffer {
	t.Helper()
	b, err := Host.Alloc(backend.Int32, len(v))
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(raw[4*i:], uint32(x))
	}
	if err := Host.Upload(t.Context(), b, raw); err != nil {
		t.Fatal(err)
	}
	return b
}

// listBuffer returns a Host buffer that the one-dimensional view v reads
// as the Int32s list.
func listBuffer(t *testing.T, list []int32, v View) backend.Buffer {
	t.Helper()
	raw := make([]int32, v.N)
	for j, x := range list {
		raw[v.V.Offset+j*v.V.Strides[0]] = x
	}
	return intBuffer(t, raw)
}

// segmentList returns the offsets of segs segments, some empty, into a
// row list of random rows below limit.
func segmentList(rng *rand.Rand, segs, limit int) (offsets, rows []int32) {
	offsets = []int32{0}
	for range segs {
		n := rng.IntN(5)
		if rng.IntN(4) == 0 {
			n = 0
		}
		for range n {
			rows = append(rows, int32(rng.IntN(limit)))
		}
		offsets = append(offsets, int32(len(rows)))
	}
	return offsets, rows
}

// Segment returns cases of segment sums and maxima: rows of one element
// and of several, read through every view, in every dtype, with empty
// segments, offsets and row lists read through strided views, and the
// bad offsets and rows that fail with ErrIndex.
func Segment(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(21, 22))
	i := backend.Int32
	var cases []Case
	add := func(name string, dt backend.DType, shape []int, x View, src backend.Buffer, offsets, rows []int32, vo, vr View) {
		out := slices.Clone(shape)
		out[0] = vo.V.Shape[0] - 1
		obuf, rbuf := listBuffer(t, offsets, vo), listBuffer(t, rows, vr)
		for _, c := range []backend.Combine{backend.CombineSum, backend.CombineMax} {
			cases = append(cases, Case{
				Name: fmt.Sprintf("%s/%s/%v/%s", c, dt, shape, name),
				K:    backend.Kernel{Op: backend.OpSegment, DType: dt, Shape: out, In: []backend.View{x.V, vo.V, vr.V}, Attrs: backend.Attrs{Combine: c}},
				Ins:  []backend.Buffer{src, obuf, rbuf},
			})
		}
	}
	for _, shape := range [][]int{{9}, {7, 33}, {6, 3, 5}} {
		offsets, rows := segmentList(rng, 11, shape[0])
		lists := func(name string) (View, View) {
			if name == "padded" {
				return Views([]int{len(offsets)})["padded"], Views([]int{len(rows)})["padded"]
			}
			return Views([]int{len(offsets)})["contiguous"], Views([]int{len(rows)})["contiguous"]
		}
		for name, x := range Views(shape) {
			vo, vr := lists(name)
			for _, dt := range append([]backend.DType{backend.Float32, i}, halfTypes...) {
				add(name, dt, shape, x, HostBuffer(t, rng, dt, x.N, -50, 50), offsets, rows, vo, vr)
			}
		}
	}
	// NaN and signed zeros in sums and maxima.
	shape := []int{40, 8}
	offsets, rows := segmentList(rng, 30, shape[0])
	vo, vr := Views([]int{len(offsets)})["contiguous"], Views([]int{len(rows)})["contiguous"]
	x := Views(shape)["contiguous"]
	add("specials", backend.Float32, shape, x, specials(t, rng, x.N), offsets, rows, vo, vr)

	bad := func(name string, offsets, rows []int32) {
		shape := []int{4, 3}
		x := Views(shape)["contiguous"]
		cases = append(cases, Case{
			Name: "bad/" + name,
			K: backend.Kernel{Op: backend.OpSegment, DType: backend.Float32, Shape: []int{len(offsets) - 1, 3},
				In: []backend.View{x.V, backend.Contiguous([]int{len(offsets)}), backend.Contiguous([]int{len(rows)})}, Attrs: backend.Attrs{Combine: backend.CombineSum}},
			Ins: []backend.Buffer{HostBuffer(t, rng, backend.Float32, x.N, 0, 0), intBuffer(t, offsets), intBuffer(t, rows)},
		})
	}
	bad("row", []int32{0, 2, 3}, []int32{1, 4, 0})
	bad("negative row", []int32{0, 1, 3}, []int32{1, -1, 0})
	bad("falling offsets", []int32{0, 2, 1}, []int32{1, 2, 0})
	bad("negative offset", []int32{-1, 2, 3}, []int32{1, 2, 0})
	bad("offset past the rows", []int32{0, 2, 4}, []int32{1, 2, 0})
	return cases
}
