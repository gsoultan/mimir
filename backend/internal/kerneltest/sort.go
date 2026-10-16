package kerneltest

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// Sort returns cases of argsorts: lines a thread sorts, lines a group
// sorts in shared memory, long lines sorted in the output, and top-k of
// long lines selected block by block, ascending and descending, with
// ties, NaN and -0.
func Sort(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(13, 14))
	f, i := backend.Float32, backend.Int32
	var cases []Case
	add := func(name string, shape []int, x View, in backend.Buffer, axis, k int) {
		out := slices.Clone(shape)
		out[axis] = k
		for _, desc := range []bool{false, true} {
			cases = append(cases, Case{
				Name: fmt.Sprintf("%s/%v/%d/k%d/%v/%s", in.DType(), shape, axis, k, desc, name),
				K: backend.Kernel{Op: backend.OpArgsort, DType: i, Shape: out, In: []backend.View{x.V},
					Attrs: backend.Attrs{Axis: axis, Descending: desc}},
				Ins: []backend.Buffer{in},
			})
		}
	}
	for _, s := range []struct {
		shape []int
		axis  int
		ks    []int // the axis' length if nil
		views []string
	}{
		{[]int{3, 5, 7}, 1, []int{5, 2}, nil},                        // a thread per line
		{[]int{2, 300}, 1, []int{300, 7}, []string{"contiguous"}},    // shared memory
		{[]int{2, 300, 3}, 1, nil, []string{"transposed", "padded"}}, // lines apart
		{[]int{3000}, 0, []int{3000, 100}, []string{"contiguous"}},   // the output; blocks of a top-k
		{[]int{2, 2500}, 1, []int{1024}, []string{"flipped"}},        // the longest top-k
	} {
		ks := s.ks
		if ks == nil {
			ks = []int{s.shape[s.axis]}
		}
		for name, x := range Views(s.shape) {
			if s.views != nil && !slices.Contains(s.views, name) {
				continue
			}
			ins := []backend.Buffer{HostBuffer(t, rng, i, x.N, -20, 20)}
			if s.shape[s.axis] < 1000 {
				ins = append(ins, HostBuffer(t, rng, f, x.N, 0, 0), specials(t, rng, x.N))
				for _, dt := range halfTypes {
					ins = append(ins, HostBuffer(t, rng, dt, x.N, 0, 0))
				}
			}
			for _, in := range ins {
				for _, k := range ks {
					add(name, s.shape, x, in, s.axis, k)
				}
			}
		}
	}
	return cases
}

// specials returns n floats of few values, with NaN, -0 and +0 among
// them.
func specials(t *testing.T, rng *rand.Rand, n int) backend.Buffer {
	t.Helper()
	vals := []float32{float32(math.NaN()), float32(math.Copysign(0, -1)), 0, 1, -1, float32(math.Inf(1)), float32(math.Inf(-1)), 0.5}
	raw := make([]byte, 4*n)
	for j := range n {
		binary.LittleEndian.PutUint32(raw[4*j:], math.Float32bits(vals[rng.IntN(len(vals))]))
	}
	b, err := Host.Alloc(backend.Float32, n)
	if err != nil {
		t.Fatal(err)
	}
	if err := Host.Upload(t.Context(), b, raw); err != nil {
		t.Fatal(err)
	}
	return b
}

// FFT returns cases of Fourier transforms of lengths that a group
// transforms in shared memory or in the output and of other lengths,
// which take the definition, forward and inverse.
func FFT(t *testing.T) []Case {
	rng := rand.New(rand.NewPCG(15, 16))
	f := backend.Float32
	var cases []Case
	for _, s := range []struct {
		shape []int
		axis  int
		views []string
	}{
		{[]int{3, 1, 2}, 1, nil},
		{[]int{3, 8, 2}, 1, nil},
		{[]int{6, 4, 2}, 0, []string{"contiguous", "transposed"}},
		{[]int{2, 12, 3, 2}, 1, []string{"contiguous", "padded"}},
		{[]int{2, 1024, 2}, 1, []string{"contiguous"}},
		{[]int{4096, 2}, 0, []string{"contiguous"}},
		{[]int{100, 2, 2}, 0, []string{"transposed"}},
	} {
		for name, x := range Views(s.shape) {
			if s.views != nil && !slices.Contains(s.views, name) {
				continue
			}
			in := HostBuffer(t, rng, f, x.N, 0, 0)
			for _, inverse := range []bool{false, true} {
				cases = append(cases, Case{
					Name: fmt.Sprintf("%v/%d/%v/%s", s.shape, s.axis, inverse, name),
					K: backend.Kernel{Op: backend.OpFFT, DType: f, Shape: s.shape, In: []backend.View{x.V},
						Attrs: backend.Attrs{Axis: s.axis, Inverse: inverse}},
					Ins: []backend.Buffer{in},
					Tol: 2e-5 * math.Sqrt(float64(s.shape[s.axis])),
				})
			}
		}
	}
	return cases
}
