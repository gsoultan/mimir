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
	"encoding/binary"
	"fmt"
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
// relative error allowed against the CPU backend; 0 requires identical
// bits.
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
	for o := range outs {
		for i := 0; i < len(want[o]); i += 4 {
			w := binary.LittleEndian.Uint32(want[o][i:])
			g := binary.LittleEndian.Uint32(got[o][i:])
			if w == g {
				continue
			}
			if c.K.DType == backend.Float32 && c.Tol > 0 {
				wf, gf := float64(math.Float32frombits(w)), float64(math.Float32frombits(g))
				if math.Abs(wf-gf) <= c.Tol*math.Max(1, math.Abs(wf)) || math.IsNaN(wf) && math.IsNaN(gf) {
					continue
				}
			}
			t.Fatalf("output %d element %d: got %s, want %s", o, i/4, show(c.K.DType, g), show(c.K.DType, w))
		}
	}
}

func show(dt backend.DType, w uint32) string {
	if dt == backend.Float32 {
		return fmt.Sprint(math.Float32frombits(w))
	}
	return fmt.Sprint(int32(w))
}

// Contents returns the bytes of the Host buffer b.
func Contents(t *testing.T, b backend.Buffer) []byte {
	t.Helper()
	out := make([]byte, 4*b.Len())
	if err := Host.Download(t.Context(), out, b); err != nil {
		t.Fatal(err)
	}
	return out
}

// Floats returns the elements of the float32 Host buffer b.
func Floats(t *testing.T, b backend.Buffer) []float32 {
	t.Helper()
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

// HostBuffer returns a Host buffer of n random values: floats in [-2, 2)
// or ints in [lo, hi). Without rng, every element is a NaN payload that no
// kernel writes.
func HostBuffer(t *testing.T, rng *rand.Rand, dt backend.DType, n, lo, hi int) backend.Buffer {
	t.Helper()
	b, err := Host.Alloc(dt, n)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 4*n)
	for i := range n {
		var w uint32
		switch {
		case rng == nil:
			w = 0x7fc0dead
		case dt == backend.Float32:
			w = math.Float32bits(rng.Float32()*4 - 2)
		default:
			w = uint32(int32(lo + rng.IntN(hi-lo)))
		}
		binary.LittleEndian.PutUint32(raw[4*i:], w)
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
