package metal

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
)

// With MIMIR_EMULATE=1, the tests in this file run the generated kernels
// on the CPU too: a C++
// compiler builds each kernel's source against shim, a header that
// stands in for metal_stdlib, and a driver runs its threadgroups one at a
// time, the threads of each in turn or, when they synchronize, at once.
// They check the generated code computes what the CPU backend does on any
// machine with clang++, while the Metal backend's own tests check it on a
// GPU.

const shim = `#pragma once
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#define kernel
#define device
#define constant const
#define threadgroup static
namespace metal {
using uint = unsigned int;
using atomic_uint = std::atomic<unsigned int>;
enum memory_order { memory_order_relaxed };
inline unsigned atomic_fetch_or_explicit(atomic_uint* a, unsigned v, memory_order) { return a->fetch_or(v); }
template <class T, class U> T as_type(U u) {
	static_assert(sizeof(T) == sizeof(U));
	T t;
	std::memcpy(&t, &u, sizeof t);
	return t;
}
using std::cos; using std::exp2; using std::fma; using std::fmod; using std::log;
using std::log2; using std::signbit; using std::sin; using std::sqrt;
inline float cospi(float x) { return float(std::cos(3.14159265358979323846 * x)); }
inline float sinpi(float x) { return float(std::sin(3.14159265358979323846 * x)); }
inline uint mulhi(uint a, uint b) { return uint((uint64_t(a) * b) >> 32); }
struct uint2 { uint x, y; uint2(uint a, uint b) : x(a), y(b) {} };
struct uint4 {
	uint x, y, z, w;
	uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
	uint& operator[](int i) { return (&x)[i]; }
};
enum class mem_flags { mem_threadgroup };
inline std::barrier<>* group_barrier;
inline void threadgroup_barrier(mem_flags) { group_barrier->arrive_and_wait(); }
}
`

// driver runs a kernel over buffers it reads from the file named by its
// first argument and writes them back to the file named by its second.
const driver = `
#include <cstdio>
#include <thread>
#include <vector>
int main(int argc, char** argv) {
	FILE* f = fopen(argv[1], "rb");
	int nb;
	fread(&nb, 4, 1, f);
	std::vector<std::vector<uint32_t>> bufs(nb);
	for (auto& b : bufs) {
		int n;
		fread(&n, 4, 1, f);
		b.resize(n + 1);
		fread(b.data(), 4, n, f);
	}
	int np, groups, threads, barriers;
	fread(&np, 4, 1, f);
	std::vector<int> p(np);
	fread(p.data(), 4, np, f);
	fread(&groups, 4, 1, f);
	fread(&threads, 4, 1, f);
	fread(&barriers, 4, 1, f);
	fclose(f);
	metal::atomic_uint err{0};
	auto run = [&](unsigned g, unsigned t) { k(ARGS, p.data(), &err, g * threads + t, t, g); };
	for (int g = 0; g < groups; g++) {
		if (!barriers) {
			for (int t = 0; t < threads; t++) run(g, t);
			continue;
		}
		std::barrier<> bar(threads);
		metal::group_barrier = &bar;
		std::vector<std::thread> ts;
		for (int t = 0; t < threads; t++) ts.emplace_back(run, g, t);
		for (auto& t : ts) t.join();
	}
	f = fopen(argv[2], "wb");
	for (auto& b : bufs) fwrite(b.data(), 4, b.size() - 1, f);
	unsigned e = err.load();
	fwrite(&e, 4, 1, f);
	fclose(f);
}
`

// emulator compiles and runs generated kernels.
type emulator struct {
	dir  string
	mu   sync.Mutex
	bins map[string]string // by source and argument types
}

var (
	emu     *emulator
	emuErr  error
	emuOnce sync.Once
)

// newEmulator returns the emulator, or nil unless MIMIR_EMULATE=1 and
// there is a clang++.
func newEmulator(t *testing.T) *emulator {
	t.Helper()
	if os.Getenv("MIMIR_EMULATE") != "1" {
		return nil
	}
	emuOnce.Do(func() {
		if _, err := exec.LookPath("clang++"); err != nil {
			emuErr = err
			return
		}
		dir, err := os.MkdirTemp("", "mimir-msl")
		if err != nil {
			emuErr = err
			return
		}
		emuErr = os.WriteFile(filepath.Join(dir, "metal_stdlib"), []byte(shim), 0o644)
		emu = &emulator{dir: dir, bins: map[string]string{}}
	})
	if emuErr != nil {
		t.Logf("no kernel emulator: %v", emuErr)
		return nil
	}
	return emu
}

func TestMain(m *testing.M) {
	code := m.Run()
	if emu != nil {
		os.RemoveAll(emu.dir)
	}
	os.Exit(code)
}

// binary returns the driver for l over buffers of the given types.
func (e *emulator) binary(t *testing.T, l *launch, types []backend.DType) string {
	t.Helper()
	var args []string
	for i, dt := range types {
		args = append(args, fmt.Sprintf("(%s*)bufs[%d].data()", typeName(dt), i))
	}
	src := strings.Replace(l.src, "kernel void k(", "void k(", 1) + strings.Replace(driver, "ARGS", strings.Join(args, ", "), 1)
	sum := sha256.Sum256([]byte(src))
	key := hex.EncodeToString(sum[:8])
	e.mu.Lock()
	defer e.mu.Unlock()
	if bin, ok := e.bins[key]; ok {
		return bin
	}
	cpp := filepath.Join(e.dir, key+".cpp")
	bin := filepath.Join(e.dir, key)
	if err := os.WriteFile(cpp, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("clang++", "-std=c++20", "-O0", "-w", "-pthread", "-I", e.dir, "-o", bin, cpp)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile kernel: %v\n%s\n%s", err, out, numberedSource(l.src))
	}
	e.bins[key] = bin
	return bin
}

// run executes k's kernel on the emulator and returns the contents of the
// outputs and the error word.
func (e *emulator) run(t *testing.T, k *backend.Kernel, outs, ins []backend.Buffer) ([][]byte, bool) {
	t.Helper()
	if err := backend.CheckRun(k, outs, ins); err != nil {
		t.Fatal(err)
	}
	l, err := plan(k, outs, ins)
	if err != nil {
		t.Fatal(err)
	}
	return e.exec(t, l, outs, ins)
}

// exec runs the launch l over outs and ins on the emulator and returns the
// contents of the outputs and the error word.
func (e *emulator) exec(t *testing.T, l *launch, outs, ins []backend.Buffer) ([][]byte, bool) {
	t.Helper()
	all := slices.Concat(outs, ins)
	if l == nil {
		res := make([][]byte, len(outs))
		for i := range outs {
			res[i] = make([]byte, 4*outs[i].Len())
		}
		return res, false
	}
	var types []backend.DType
	var in bytes.Buffer
	put := func(v any) { binary.Write(&in, binary.LittleEndian, v) }
	put(int32(len(all)))
	for _, b := range all {
		types = append(types, b.DType())
		put(int32(b.Len()))
		in.Write(contents(t, b))
	}
	put(int32(len(l.params)))
	put(l.params)
	put([]int32{int32(l.groups), int32(l.threads), int32(boolInt(l.barriers))})
	bin := e.binary(t, l, types)

	dir := t.TempDir()
	inPath, outPath := filepath.Join(dir, "in"), filepath.Join(dir, "out")
	if err := os.WriteFile(inPath, in.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(bin, inPath, outPath).CombinedOutput(); err != nil {
		t.Fatalf("run kernel: %v\n%s\n%s", err, out, numberedSource(l.src))
	}
	res, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	var got [][]byte
	for _, b := range all {
		got = append(got, res[:4*b.Len()])
		res = res[4*b.Len():]
	}
	return got[:len(outs)], binary.LittleEndian.Uint32(res) != 0
}

// emulatedMPS runs products that Metal Performance Shaders would run as
// the backend reads them for MPS, on the host, and their epilogues on the
// emulator. Other kernels run on the emulator.
type emulatedMPS struct{ *emulator }

func (e emulatedMPS) run(t *testing.T, k *backend.Kernel, outs, ins []backend.Buffer) ([][]byte, bool) {
	t.Helper()
	g, ok := planGEMM(k, []int{ins[0].Len(), ins[1].Len()})
	if !ok {
		return e.emulator.run(t, k, outs, ins)
	}
	x, y := floats(t, ins[0]), floats(t, ins[1])
	z := make([]float32, outs[0].Len())
	for b := range g.batch {
		for i := range g.m {
			for j := range g.n {
				var acc float32
				for p := range g.k {
					acc += x[g.a.at(b, i, p)] * y[g.b.at(b, p, j)]
				}
				z[b*g.resultSize+i*g.n+j] = acc
			}
		}
	}
	raw := make([]byte, 4*len(z))
	for i, v := range z {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	if k.Attrs.Program == nil {
		return [][]byte{raw}, false
	}
	if err := host.Upload(t.Context(), outs[0], raw); err != nil {
		t.Fatal(err)
	}
	l, err := planEpilogue(k, outs, ins)
	if err != nil {
		t.Fatal(err)
	}
	return e.exec(t, l, outs, ins)
}

func floats(t *testing.T, b backend.Buffer) []float32 {
	t.Helper()
	raw := contents(t, b)
	out := make([]float32, b.Len())
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

var host = cpu.New()

func contents(t *testing.T, b backend.Buffer) []byte {
	t.Helper()
	out := make([]byte, 4*b.Len())
	if err := host.Download(t.Context(), out, b); err != nil {
		t.Fatal(err)
	}
	return out
}

// hostBuffer returns a host buffer of n random values: floats in [-2, 2) or
// ints in [lo, hi). Without rng, every element is a NaN payload that no
// kernel writes.
func hostBuffer(t *testing.T, rng *rand.Rand, dt backend.DType, n, lo, hi int) backend.Buffer {
	t.Helper()
	b, err := host.Alloc(dt, n)
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
	if err := host.Upload(t.Context(), b, raw); err != nil {
		t.Fatal(err)
	}
	return b
}

// kernelCase is a kernel and its inputs. tol is the relative error allowed
// against the CPU backend; 0 requires identical bits.
type kernelCase struct {
	name string
	k    backend.Kernel
	ins  []backend.Buffer
	tol  float64
}

// check runs c on the CPU backend and the emulator and compares them.
// target runs kernels to compare with the CPU backend. run returns the
// contents of the outputs and whether the kernel reported an error.
type target interface {
	run(t *testing.T, k *backend.Kernel, outs, ins []backend.Buffer) ([][]byte, bool)
}

// check runs c on the CPU backend and on tg and compares them.
func check(t *testing.T, tg target, c kernelCase) {
	t.Helper()
	n, _ := backend.NumElements(c.k.Shape)
	outs := make([]backend.Buffer, 1+len(c.k.Attrs.Stores))
	mine := make([]backend.Buffer, len(outs))
	want := make([][]byte, len(outs))
	for i := range outs {
		outs[i] = hostBuffer(t, nil, c.k.DType, n, 0, 0)
		mine[i] = hostBuffer(t, nil, c.k.DType, n, 0, 0)
	}
	cpuErr := host.Run(t.Context(), &c.k, outs, c.ins)
	for i := range outs {
		want[i] = contents(t, outs[i])
	}
	got, failed := tg.run(t, &c.k, mine, c.ins)
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
			if c.k.DType == backend.Float32 && c.tol > 0 {
				wf, gf := float64(math.Float32frombits(w)), float64(math.Float32frombits(g))
				if math.Abs(wf-gf) <= c.tol*math.Max(1, math.Abs(wf)) || math.IsNaN(wf) && math.IsNaN(gf) {
					continue
				}
			}
			t.Fatalf("output %d element %d: got %s, want %s", o, i/4, show(c.k.DType, g), show(c.k.DType, w))
		}
	}
}

func show(dt backend.DType, w uint32) string {
	if dt == backend.Float32 {
		return fmt.Sprint(math.Float32frombits(w))
	}
	return fmt.Sprint(int32(w))
}

// run checks cases on the emulator and on the GPU, where they exist.
func run(t *testing.T, cases []kernelCase) { runOn(t, cases, false) }

// runOn checks cases like run and, if mps is set, also running every
// product it can through Metal Performance Shaders.
func runOn(t *testing.T, cases []kernelCase, mps bool) {
	targets := map[string]target{}
	if e := newEmulator(t); e != nil {
		targets["emulator"] = e
		if mps {
			targets["emulator-mps"] = emulatedMPS{e}
		}
	}
	if g := newGPU(t, -1); g != nil {
		targets["gpu"] = g
		if mps {
			targets["gpu-mps"] = newGPU(t, 0)
		}
	}
	if len(targets) == 0 {
		t.Skip("no Metal device or kernel emulator")
	}
	for name, tg := range targets {
		t.Run(name, func(t *testing.T) {
			for _, c := range cases {
				t.Run(c.name, func(t *testing.T) {
					t.Parallel()
					check(t, tg, c)
				})
			}
		})
	}
}

// views returns the ways a kernel reads a tensor of shape: row-major,
// transposed, inside a larger buffer, reversed and broadcast, each with
// the size of the buffer it reads.
func views(shape []int) map[string]struct {
	v backend.View
	n int
} {
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
	type vn = struct {
		v backend.View
		n int
	}
	out := map[string]vn{
		"contiguous": {backend.Contiguous(shape), n},
		"transposed": {backend.Contiguous(rev).Permute(reverse), n},
		"padded":     {backend.Contiguous(grown).Shrink(inner), g},
		"flipped":    {backend.Contiguous(shape).Flip(all), n},
	}
	if r > 1 {
		one := slices.Clone(shape)
		one[0] = 1
		m, _ := backend.NumElements(one)
		out["broadcast"] = vn{backend.Contiguous(one).Expand(shape), m}
	}
	return out
}

func TestElementwiseKernels(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	f, i := backend.Float32, backend.Int32
	shape := []int{3, 5, 7}
	var cases []kernelCase
	for name, x := range views(shape) {
		for _, op := range []backend.Op{backend.OpNeg, backend.OpExp2, backend.OpSin, backend.OpContiguous} {
			cases = append(cases, kernelCase{
				name: fmt.Sprintf("%s/%s", op, name),
				k:    backend.Kernel{Op: op, DType: f, Shape: shape, In: []backend.View{x.v}},
				ins:  []backend.Buffer{hostBuffer(t, rng, f, x.n, 0, 0)},
				tol:  1e-6,
			})
		}
		for _, op := range []backend.Op{backend.OpAdd, backend.OpMul, backend.OpMax, backend.OpCmpLT, backend.OpCmpEQ, backend.OpMod} {
			for _, dt := range []backend.DType{f, i} {
				y := backend.Contiguous(shape)
				cases = append(cases, kernelCase{
					name: fmt.Sprintf("%s/%s/%s", op, dt, name),
					k:    backend.Kernel{Op: op, DType: dt, Shape: shape, In: []backend.View{x.v, y}},
					ins:  []backend.Buffer{hostBuffer(t, rng, dt, x.n, -9, 9), hostBuffer(t, rng, dt, 105, 1, 5)},
				})
			}
		}
	}
	n := 105
	pos := hostBuffer(t, rng, f, n, 0, 0)
	cases = append(cases,
		kernelCase{name: "log2 sqrt recip", k: backend.Kernel{Op: backend.OpFused, DType: f, Shape: shape, In: []backend.View{backend.Contiguous(shape)},
			Attrs: backend.Attrs{Program: []backend.Instr{
				{Op: backend.OpLoad, DType: f, Args: []int{0}},
				{Op: backend.OpMul, DType: f, Args: []int{0, 0}},
				{Op: backend.OpLog2, DType: f, Args: []int{1}},
				{Op: backend.OpSqrt, DType: f, Args: []int{1}},
				{Op: backend.OpRecip, DType: f, Args: []int{3}},
				{Op: backend.OpAdd, DType: f, Args: []int{2, 4}},
			}}}, ins: []backend.Buffer{pos}, tol: 1e-6},
		kernelCase{name: "cast to int", k: backend.Kernel{Op: backend.OpCast, DType: i, Shape: shape, In: []backend.View{backend.Contiguous(shape)}},
			ins: []backend.Buffer{pos}},
		kernelCase{name: "cast to float", k: backend.Kernel{Op: backend.OpCast, DType: f, Shape: shape, In: []backend.View{backend.Contiguous(shape)}},
			ins: []backend.Buffer{hostBuffer(t, rng, i, n, -100, 100)}},
		kernelCase{name: "int mod by zero", k: backend.Kernel{Op: backend.OpMod, DType: i, Shape: shape, In: []backend.View{backend.Contiguous(shape), backend.Contiguous(shape)}},
			ins: []backend.Buffer{hostBuffer(t, rng, i, n, -9, 9), hostBuffer(t, rng, i, n, -1, 2)}},
		kernelCase{name: "where", k: backend.Kernel{Op: backend.OpWhere, DType: f, Shape: shape, In: []backend.View{backend.Contiguous(shape), backend.Contiguous(shape), backend.Contiguous([]int{1, 1, 1}).Expand(shape)}},
			ins: []backend.Buffer{hostBuffer(t, rng, i, n, 0, 2), pos, hostBuffer(t, rng, f, 1, 0, 0)}},
		kernelCase{name: "fused with stores and constants", k: backend.Kernel{Op: backend.OpFused, DType: f, Shape: shape,
			In: []backend.View{backend.Contiguous(shape), backend.Contiguous([]int{7, 5, 3}).Permute([]int{2, 1, 0})},
			Attrs: backend.Attrs{Program: []backend.Instr{
				{Op: backend.OpLoad, DType: f, Args: []int{0}},
				{Op: backend.OpLoad, DType: f, Args: []int{1}},
				{Op: backend.OpConst, DType: f, Value: 0.9},
				{Op: backend.OpMul, DType: f, Args: []int{0, 2}},
				{Op: backend.OpAdd, DType: f, Args: []int{3, 1}},
				{Op: backend.OpConst, DType: f, Value: math.Inf(-1)},
				{Op: backend.OpMax, DType: f, Args: []int{4, 5}},
				{Op: backend.OpCmpLT, DType: f, Args: []int{6, 2}},
				{Op: backend.OpWhere, DType: f, Args: []int{7, 3, 6}},
			}, Stores: []int{4, 6}}},
			ins: []backend.Buffer{pos, hostBuffer(t, rng, f, n, 0, 0)}},
		kernelCase{name: "const", k: backend.Kernel{Op: backend.OpConst, DType: i, Shape: shape, Attrs: backend.Attrs{Value: -7}}},
		kernelCase{name: "empty", k: backend.Kernel{Op: backend.OpConst, DType: f, Shape: []int{0, 3}, Attrs: backend.Attrs{Value: 1}}},
	)
	run(t, cases)
}

func TestRandomKernels(t *testing.T) {
	var cases []kernelCase
	for _, seed := range []uint64{0, 42, 1<<63 + 12345} {
		for _, n := range []int{1, 7, 1000} {
			cases = append(cases,
				kernelCase{name: fmt.Sprintf("uniform/%d/%d", seed, n), k: backend.Kernel{Op: backend.OpRandUniform, DType: backend.Float32, Shape: []int{n}, Attrs: backend.Attrs{Seed: seed}}},
				kernelCase{name: fmt.Sprintf("normal/%d/%d", seed, n), k: backend.Kernel{Op: backend.OpRandNormal, DType: backend.Float32, Shape: []int{n}, Attrs: backend.Attrs{Seed: seed}}, tol: 1e-5},
			)
		}
	}
	run(t, cases)
}

func TestReduceKernels(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	var cases []kernelCase
	for _, s := range []struct {
		shape []int
		axes  []int
	}{
		{[]int{4, 6, 5}, []int{1}},
		{[]int{4, 6, 5}, []int{0, 2}},
		{[]int{4, 6, 5}, []int{0, 1, 2}},
		{[]int{3, 700}, []int{1}},       // a threadgroup per output
		{[]int{2000, 3}, []int{0}},      // a threadgroup per output, strided
		{[]int{64, 3, 40}, []int{0, 2}}, // many outputs, many positions
	} {
		out := slices.Clone(s.shape)
		for _, a := range s.axes {
			out[a] = 1
		}
		for name, x := range views(s.shape) {
			for _, op := range []backend.Op{backend.OpReduceSum, backend.OpReduceMax} {
				for _, dt := range []backend.DType{backend.Float32, backend.Int32} {
					cases = append(cases, kernelCase{
						name: fmt.Sprintf("%s/%s/%v%v/%s", op, dt, s.shape, s.axes, name),
						k:    backend.Kernel{Op: op, DType: dt, Shape: out, In: []backend.View{x.v}, Attrs: backend.Attrs{Axes: s.axes}},
						ins:  []backend.Buffer{hostBuffer(t, rng, dt, x.n, -50, 50)},
						tol:  1e-5,
					})
				}
			}
		}
	}
	run(t, cases)
}

func TestMovementKernels(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	f, i := backend.Float32, backend.Int32
	var cases []kernelCase
	for name, x := range views([]int{3, 4, 5}) {
		cases = append(cases, kernelCase{
			name: "pad/" + name,
			k: backend.Kernel{Op: backend.OpPad, DType: f, Shape: []int{4, 4, 8}, In: []backend.View{x.v},
				Attrs: backend.Attrs{Pads: [][2]int{{1, 0}, {0, 0}, {2, 1}}, Value: -1.5}},
			ins: []backend.Buffer{hostBuffer(t, rng, f, x.n, 0, 0)},
		})
		// Gather along each axis with an index of shape [3, 2, 5] that
		// fits the source outside the axis.
		for axis := range 3 {
			idx := []int{3, 4, 5}
			idx[(axis+1)%3] = 2
			n, _ := backend.NumElements(idx)
			cases = append(cases, kernelCase{
				name: fmt.Sprintf("gather/%d/%s", axis, name),
				k:    backend.Kernel{Op: backend.OpGather, DType: f, Shape: idx, In: []backend.View{x.v, backend.Contiguous(idx)}, Attrs: backend.Attrs{Axis: axis}},
				ins:  []backend.Buffer{hostBuffer(t, rng, f, x.n, 0, 0), hostBuffer(t, rng, i, n, 0, []int{3, 4, 5}[axis])},
			})
			upd := slices.Clone(idx)
			upd[axis] = 7
			m, _ := backend.NumElements(upd)
			cases = append(cases, kernelCase{
				name: fmt.Sprintf("scatter add/%d/%s", axis, name),
				k: backend.Kernel{Op: backend.OpScatterAdd, DType: f, Shape: []int{3, 4, 5},
					In: []backend.View{x.v, backend.Contiguous(upd), backend.Contiguous(upd)}, Attrs: backend.Attrs{Axis: axis}},
				ins: []backend.Buffer{hostBuffer(t, rng, f, x.n, 0, 0), hostBuffer(t, rng, i, m, 0, []int{3, 4, 5}[axis]), hostBuffer(t, rng, f, m, 0, 0)},
			})
		}
	}
	cases = append(cases,
		kernelCase{name: "gather out of range", k: backend.Kernel{Op: backend.OpGather, DType: i, Shape: []int{4}, In: []backend.View{backend.Contiguous([]int{4}), backend.Contiguous([]int{4})}},
			ins: []backend.Buffer{hostBuffer(t, rng, i, 4, 0, 9), hostBuffer(t, rng, i, 4, 3, 5)}},
		kernelCase{name: "scatter add out of range", k: backend.Kernel{Op: backend.OpScatterAdd, DType: i, Shape: []int{4}, In: []backend.View{backend.Contiguous([]int{4}), backend.Contiguous([]int{6}), backend.Contiguous([]int{6})}},
			ins: []backend.Buffer{hostBuffer(t, rng, i, 4, 0, 9), hostBuffer(t, rng, i, 6, -1, 1), hostBuffer(t, rng, i, 6, 0, 9)}},
	)
	run(t, cases)
}

func TestWindowKernels(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	f := backend.Float32
	var cases []kernelCase
	for _, w := range []struct {
		name           string
		in             []int
		kernel, stride []int
		dilation       []int
		pads           [][2]int
		positions      []int
	}{
		{"1d", []int{2, 9}, []int{3}, []int{1}, []int{1}, [][2]int{{0, 0}}, []int{7}},
		{"2d strided padded", []int{2, 3, 7, 6}, []int{3, 2}, []int{2, 1}, []int{1, 2}, [][2]int{{1, 1}, {0, 2}}, []int{4, 6}},
		{"2d dilated", []int{1, 8, 8}, []int{2, 3}, []int{3, 2}, []int{3, 1}, [][2]int{{2, 0}, {1, 1}}, []int{4, 4}},
	} {
		attrs := backend.Attrs{Kernel: w.kernel, Stride: w.stride, Dilation: w.dilation, Pads: w.pads, Value: 0.5}
		s := len(w.kernel)
		lead := w.in[:len(w.in)-s]
		win := slices.Concat(lead, w.kernel, w.positions)
		for name, x := range views(w.in) {
			cases = append(cases, kernelCase{
				name: "unfold/" + w.name + "/" + name,
				k:    backend.Kernel{Op: backend.OpUnfold, DType: f, Shape: win, In: []backend.View{x.v}, Attrs: attrs},
				ins:  []backend.Buffer{hostBuffer(t, rng, f, x.n, 0, 0)},
			})
		}
		for name, x := range views(win) {
			cases = append(cases, kernelCase{
				name: "fold/" + w.name + "/" + name,
				k:    backend.Kernel{Op: backend.OpFold, DType: f, Shape: w.in, In: []backend.View{x.v}, Attrs: attrs},
				ins:  []backend.Buffer{hostBuffer(t, rng, f, x.n, 0, 0)},
			})
		}
	}
	run(t, cases)
}

func TestMatMulKernels(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	f, i := backend.Float32, backend.Int32
	var cases []kernelCase
	for _, s := range []struct {
		batch   []int
		m, k, n int
		axes    []int
	}{
		{nil, 5, 7, 3, nil},
		{[]int{2}, 4, 6, 5, nil},
		{[]int{3, 2}, 4, 6, 5, []int{0}},
		{[]int{3, 2}, 4, 6, 5, []int{0, 1}},
		{[]int{2}, 3, 0, 4, nil},
		{nil, 1, 9, 6, nil},
		{[]int{3}, 33, 70, 17, nil},
	} {
		xs := slices.Concat(s.batch, []int{s.m, s.k})
		ys := slices.Concat(s.batch, []int{s.k, s.n})
		out := slices.Concat(s.batch, []int{s.m, s.n})
		for _, a := range s.axes {
			out[a] = 1
		}
		for xn, x := range views(xs) {
			for yn, y := range views(ys) {
				// Every pair of views for float32, which MPS may run, and a
				// few for int32.
				dts := []backend.DType{f}
				if yn == "transposed" {
					dts = append(dts, i)
				}
				for _, dt := range dts {
					cases = append(cases, kernelCase{
						name: fmt.Sprintf("%s/%v x %v sum %v/%s/%s", dt, xs, ys, s.axes, xn, yn),
						k:    backend.Kernel{Op: backend.OpMatMul, DType: dt, Shape: out, In: []backend.View{x.v, y.v}, Attrs: backend.Attrs{Axes: s.axes}},
						ins:  []backend.Buffer{hostBuffer(t, rng, dt, x.n, -5, 5), hostBuffer(t, rng, dt, y.n, -5, 5)},
						tol:  1e-5,
					})
				}
			}
		}
	}
	// A bias and a ReLU as the epilogue of a product read as [2, 3, 4].
	bias := backend.Contiguous([]int{1, 1, 4}).Expand([]int{2, 3, 4})
	cases = append(cases, kernelCase{
		name: "epilogue",
		k: backend.Kernel{Op: backend.OpMatMul, DType: f, Shape: []int{6, 4},
			In: []backend.View{backend.Contiguous([]int{6, 5}), backend.Contiguous([]int{5, 4}), bias},
			Attrs: backend.Attrs{Program: []backend.Instr{
				{Op: backend.OpLoad, DType: f, Args: []int{0}},
				{Op: backend.OpLoad, DType: f, Args: []int{1}},
				{Op: backend.OpAdd, DType: f, Args: []int{0, 1}},
				{Op: backend.OpConst, DType: f},
				{Op: backend.OpMax, DType: f, Args: []int{2, 3}},
			}}},
		ins: []backend.Buffer{hostBuffer(t, rng, f, 30, 0, 0), hostBuffer(t, rng, f, 20, 0, 0), hostBuffer(t, rng, f, 4, 0, 0)},
		tol: 1e-5,
	})
	runOn(t, cases, true)
}
