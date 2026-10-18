// Package kerneltest checks that GPU backends compute what the CPU
// backend does. It holds kernel cases that cover every op and the ways
// kernels read their inputs, runs them on a Target and compares the
// outputs with the CPU backend's.
//
// Targets include GPUs and, with MIMIR_EMULATE=1, an emulator that
// compiles generated kernels with clang++ and runs them on the CPU, so the
// generated code is checked on any machine.
package kerneltest

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
)

// Host is the CPU backend that holds the cases' buffers and computes the
// results the targets must match.
var Host = cpu.New()

// Target runs kernels to compare with the CPU backend. Run returns the
// contents of the outputs and whether the kernel reported an error.
type Target interface {
	Run(t *testing.T, k *backend.Kernel, outs, ins []backend.Buffer) ([][]byte, bool)
}

// Case is a kernel and its inputs, which are Host buffers. Tol is the
// relative error allowed against the CPU backend for a float output of
// any dtype; 0 requires identical bits, but for a NaN, which matches
// any NaN.
type Case struct {
	Name string
	K    backend.Kernel
	Ins  []backend.Buffer
	Tol  float64
}

// Run checks every case on every target, each in its own subtest.
func Run(t *testing.T, cases []Case, targets map[string]Target) {
	t.Helper()
	if len(targets) == 0 {
		t.Skip("no GPU or kernel emulator")
	}
	for name, tg := range targets {
		t.Run(name, func(t *testing.T) {
			for _, c := range cases {
				t.Run(c.Name, func(t *testing.T) {
					t.Parallel()
					Check(t, tg, c)
				})
			}
		})
	}
}

// Check runs c on the CPU backend and on tg and compares them.
func Check(t *testing.T, tg Target, c Case) {
	t.Helper()
	n, _ := backend.NumElements(c.K.Shape)
	outs := make([]backend.Buffer, 1+len(c.K.Attrs.Stores))
	mine := make([]backend.Buffer, len(outs))
	want := make([][]byte, len(outs))
	for i := range outs {
		outs[i] = HostBuffer(t, nil, c.K.DType, n, 0, 0)
		mine[i] = HostBuffer(t, nil, c.K.DType, n, 0, 0)
	}
	cpuErr := Host.Run(t.Context(), &c.K, outs, c.Ins)
	for i := range outs {
		want[i] = Contents(t, outs[i])
	}
	got, failed := tg.Run(t, &c.K, mine, c.Ins)
	if cpuErr != nil {
		if !failed {
			t.Fatalf("CPU failed with %v, kernel reported no error", cpuErr)
		}
		return
	}
	if failed {
		t.Fatal("kernel reported an error, CPU succeeded")
	}
	dt := c.K.DType
	for o := range outs {
		if bytes.Equal(want[o], got[o]) || len(got[o]) != len(want[o]) {
			if len(got[o]) != len(want[o]) {
				t.Fatalf("output %d: got %d bytes, want %d", o, len(got[o]), len(want[o]))
			}
			continue
		}
		wv, gv := Values(t, dt, want[o]), Values(t, dt, got[o])
		for i, w := range wv {
			g := gv[i]
			size := dt.Size()
			if bytes.Equal(want[o][i*size:(i+1)*size], got[o][i*size:(i+1)*size]) {
				continue
			}
			// Arithmetic on a NaN may return any NaN: NVIDIA GPUs return
			// their canonical NaN where the CPU keeps the input's sign
			// and payload, so NaNs match whatever their bits.
			if dt.IsFloat() && (math.IsNaN(w) && math.IsNaN(g) || c.Tol > 0 && math.Abs(w-g) <= c.Tol*math.Max(1, math.Abs(w))) {
				continue
			}
			t.Fatalf("output %d element %d: got %v (%x), want %v (%x)", o, i, g, got[o][i*size:(i+1)*size], w, want[o][i*size:(i+1)*size])
		}
	}
}

// Values returns the elements of dt in raw as numbers: the values of
// half elements, or int32s.
func Values(t *testing.T, dt backend.DType, raw []byte) []float64 {
	t.Helper()
	n := len(raw) / dt.Size()
	out := make([]float64, n)
	switch {
	case dt == backend.Int32:
		for i := range out {
			out[i] = float64(int32(binary.LittleEndian.Uint32(raw[4*i:])))
		}
	case dt == backend.Float32:
		for i, f := range FromBytes(raw) {
			out[i] = float64(f)
		}
	default:
		b, err := Host.Alloc(dt, n)
		if err != nil {
			t.Fatal(err)
		}
		if err := Host.Upload(t.Context(), b, raw); err != nil {
			t.Fatal(err)
		}
		for i, f := range Floats(t, b) {
			out[i] = float64(f)
		}
	}
	return out
}

// byteSize returns the size of b in bytes.
func byteSize(b backend.Buffer) int {
	n, _ := b.DType().Bytes(b.Len())
	return n
}

// Contents returns the bytes of the Host buffer b.
func Contents(t *testing.T, b backend.Buffer) []byte {
	t.Helper()
	out := make([]byte, byteSize(b))
	if err := Host.Download(t.Context(), out, b); err != nil {
		t.Fatal(err)
	}
	return out
}

// cast returns a Host buffer holding the elements of the Host buffer b
// converted to dt by the CPU backend.
func cast(t *testing.T, b backend.Buffer, dt backend.DType) backend.Buffer {
	t.Helper()
	if b.DType() == dt {
		return b
	}
	out, err := Host.Alloc(dt, b.Len())
	if err != nil {
		t.Fatal(err)
	}
	shape := []int{b.Len()}
	k := &backend.Kernel{Op: backend.OpCast, DType: dt, Shape: shape, In: []backend.View{backend.Contiguous(shape)}}
	if err := Host.Run(t.Context(), k, []backend.Buffer{out}, []backend.Buffer{b}); err != nil {
		t.Fatal(err)
	}
	return out
}

// Floats returns the elements of the float Host buffer b, the values of
// half elements.
func Floats(t *testing.T, b backend.Buffer) []float32 {
	t.Helper()
	if b.DType().Size() == 2 {
		b = cast(t, b, backend.Float32)
	}
	return FromBytes(Contents(t, b))
}

// FromBytes returns the float32s in raw.
func FromBytes(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

// Bytes returns the bytes of the float32s in v.
func Bytes(v []float32) []byte {
	raw := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(f))
	}
	return raw
}

// HostBuffer returns a Host buffer of n random values: floats in [-2, 2),
// rounded to a half dtype, or ints in [lo, hi). Without rng, every
// element is a NaN payload that no kernel writes.
func HostBuffer(t *testing.T, rng *rand.Rand, dt backend.DType, n, lo, hi int) backend.Buffer {
	t.Helper()
	if rng != nil && dt.Size() == 2 {
		return cast(t, HostBuffer(t, rng, backend.Float32, n, lo, hi), dt)
	}
	b, err := Host.Alloc(dt, n)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, n*dt.Size())
	for i := range n {
		var w uint32
		switch {
		case rng == nil && dt.Size() == 2:
			w = 0x7fdd // a NaN in both half formats
		case rng == nil:
			w = 0x7fc0dead
		case dt == backend.Float32:
			w = math.Float32bits(rng.Float32()*4 - 2)
		default:
			w = uint32(int32(lo + rng.IntN(hi-lo)))
		}
		if dt.Size() == 2 {
			binary.LittleEndian.PutUint16(raw[2*i:], uint16(w))
		} else {
			binary.LittleEndian.PutUint32(raw[4*i:], w)
		}
	}
	if err := Host.Upload(t.Context(), b, raw); err != nil {
		t.Fatal(err)
	}
	return b
}

// View is a way to read a tensor and the size of the buffer it reads.
type View struct {
	V backend.View
	N int
}

// Views returns the ways a kernel reads a tensor of shape: row-major,
// transposed, inside a larger buffer, reversed and broadcast.
func Views(shape []int) map[string]View {
	r := len(shape)
	n, _ := backend.NumElements(shape)
	rev, reverse, all := slices.Clone(shape), make([]int, r), make([]int, r)
	slices.Reverse(rev)
	grown, inner := make([]int, r), make([][2]int, r)
	for d, s := range shape {
		reverse[d], all[d] = r-1-d, d
		grown[d], inner[d] = s+2, [2]int{1, s + 1}
	}
	g, _ := backend.NumElements(grown)
	out := map[string]View{
		"contiguous": {backend.Contiguous(shape), n},
		"transposed": {backend.Contiguous(rev).Permute(reverse), n},
		"padded":     {backend.Contiguous(grown).Shrink(inner), g},
		"flipped":    {backend.Contiguous(shape).Flip(all), n},
	}
	if r > 1 {
		one := slices.Clone(shape)
		one[0] = 1
		m, _ := backend.NumElements(one)
		out["broadcast"] = View{backend.Contiguous(one).Expand(shape), m}
	}
	return out
}
