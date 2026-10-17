package kerneltest

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// The emulator compiles a kernel's source with clang++ against a shim, a
// header that stands in for what the device's compiler provides, and a
// driver runs its groups one at a time: the threads of each in turn or,
// when they synchronize, at once.

// metalShim stands in for metal_stdlib.
const metalShim = `#pragma once
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
using ushort = unsigned short;
using uchar = unsigned char;
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
enum class mem_flags { mem_threadgroup, mem_device };
inline std::barrier<>* group_barrier;
inline void threadgroup_barrier(mem_flags) { group_barrier->arrive_and_wait(); }
}
`

// cudaShim stands in for the declarations NVRTC provides.
const cudaShim = `#pragma once
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#define __global__
#define __device__
#define __shared__ static
struct dim3_ { unsigned x; };
inline thread_local dim3_ blockIdx, threadIdx, blockDim;
inline std::barrier<>* group_barrier;
inline void __syncthreads() { group_barrier->arrive_and_wait(); }
template <class T, class U> T bit_cast_(U u) {
	static_assert(sizeof(T) == sizeof(U));
	T t;
	std::memcpy(&t, &u, sizeof t);
	return t;
}
inline float __int_as_float(int b) { return bit_cast_<float>(b); }
inline float __uint_as_float(unsigned b) { return bit_cast_<float>(b); }
inline unsigned __float_as_uint(float f) { return bit_cast_<unsigned>(f); }
inline unsigned __umulhi(unsigned a, unsigned b) { return unsigned((uint64_t(a) * b) >> 32); }
inline unsigned atomicOr(unsigned* a, unsigned v) { return std::atomic_ref<unsigned>(*a).fetch_or(v); }
inline float cospif(float x) { return float(std::cos(3.14159265358979323846 * x)); }
inline float sinpif(float x) { return float(std::sin(3.14159265358979323846 * x)); }
using std::exp2; using std::fma; using std::log; using std::log2; using std::sin; using std::sqrt;
`

// driver runs a kernel over buffers it reads from the file named by its
// first argument and writes them back to the file named by its second.
// SETUP declares what CALL, which runs thread t of group g, needs.
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
	SETUP
	auto run = [&](unsigned g, unsigned t) { CALL; };
	for (int g = 0; g < groups; g++) {
		if (!barriers) {
			for (int t = 0; t < threads; t++) run(g, t);
			continue;
		}
		std::barrier<> bar(threads);
		BARRIER = &bar;
		std::vector<std::thread> ts;
		for (int t = 0; t < threads; t++) ts.emplace_back(run, g, t);
		for (auto& t : ts) t.join();
	}
	f = fopen(argv[2], "wb");
	for (auto& b : bufs) fwrite(b.data(), 4, b.size() - 1, f);
	unsigned e = ERR;
	fwrite(&e, 4, 1, f);
	fclose(f);
}
`

// emulation is how the emulator builds and calls kernels of one dialect.
type emulation struct {
	shim, header string
	// source returns the program that runs src, calling the kernel with
	// the buffer arguments args.
	source func(src string, args []string) string
}

var emulations = map[*gen.Dialect]emulation{
	gen.Metal: {
		shim:   metalShim,
		header: "metal_stdlib",
		source: func(src string, args []string) string {
			return strings.Replace(src, "kernel void k(", "void k(", 1) + strings.NewReplacer(
				"SETUP", "metal::atomic_uint err{0};",
				"CALL", fmt.Sprintf("k(%s, p.data(), &err, g * threads + t, t, g)", strings.Join(args, ", ")),
				"BARRIER", "metal::group_barrier",
				"ERR", "err.load()",
			).Replace(driver)
		},
	},
	gen.CUDA: {
		shim:   cudaShim,
		header: "cuda_shim.h",
		source: func(src string, args []string) string {
			return "#include <cuda_shim.h>\n" + src + strings.NewReplacer(
				"SETUP", "Params pp;\n\tstd::memcpy(pp.v, p.data(), sizeof pp.v);\n\tunsigned err = 0;",
				"CALL", fmt.Sprintf("blockIdx.x = g; threadIdx.x = t; blockDim.x = threads; k(%s, pp, &err)", strings.Join(args, ", ")),
				"BARRIER", "group_barrier",
				"ERR", "err",
			).Replace(driver)
		},
	},
}

// Emulator compiles and runs kernels of one dialect on the CPU.
type Emulator struct {
	d  *gen.Dialect
	em emulation

	mu   sync.Mutex
	bins map[string]string // by source and argument types
}

var (
	emuDir     string
	emuErr     error
	emuOnce    sync.Once
	emuMu      sync.Mutex
	emulators  = map[*gen.Dialect]*Emulator{}
	clangFlags = []string{"-std=c++20", "-O0", "-w", "-pthread"}
)

// NewEmulator returns the emulator of kernels in dialect d, or nil unless
// MIMIR_EMULATE=1 and there is a clang++.
func NewEmulator(t *testing.T, d *gen.Dialect) *Emulator {
	t.Helper()
	if os.Getenv("MIMIR_EMULATE") != "1" {
		return nil
	}
	emuOnce.Do(func() {
		if _, err := exec.LookPath("clang++"); err != nil {
			emuErr = err
			return
		}
		emuDir, emuErr = os.MkdirTemp("", "mimir-emulator")
		for _, em := range emulations {
			if emuErr == nil {
				emuErr = os.WriteFile(filepath.Join(emuDir, em.header), []byte(em.shim), 0o644)
			}
		}
	})
	if emuErr != nil {
		t.Logf("no kernel emulator: %v", emuErr)
		return nil
	}
	emuMu.Lock()
	defer emuMu.Unlock()
	e := emulators[d]
	if e == nil {
		e = &Emulator{d: d, em: emulations[d], bins: map[string]string{}}
		emulators[d] = e
	}
	return e
}

// Cleanup removes the emulator's files. TestMain calls it after the tests.
func Cleanup() {
	if emuDir != "" {
		os.RemoveAll(emuDir)
	}
}

// binary returns the driver for l over buffers of the given types.
func (e *Emulator) binary(t *testing.T, l *gen.Launch, types []backend.DType) string {
	t.Helper()
	var args []string
	for i, dt := range types {
		name := "int"
		switch {
		case dt == backend.Float32:
			name = "float"
		case dt.Size() == 2:
			name = "unsigned short"
		case dt.IsQuantized():
			name = "unsigned char"
		}
		args = append(args, fmt.Sprintf("(%s*)bufs[%d].data()", name, i))
	}
	src := e.em.source(l.Src, args)
	sum := sha256.Sum256([]byte(src))
	key := hex.EncodeToString(sum[:8])
	e.mu.Lock()
	defer e.mu.Unlock()
	if bin, ok := e.bins[key]; ok {
		return bin
	}
	cpp := filepath.Join(emuDir, key+".cpp")
	bin := filepath.Join(emuDir, key)
	if err := os.WriteFile(cpp, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("clang++", append(clangFlags, "-I", emuDir, "-o", bin, cpp)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile kernel: %v\n%s\n%s", err, out, gen.Numbered(l.Src))
	}
	e.bins[key] = bin
	return bin
}

// Run executes k's kernel on the emulator and returns the contents of the
// outputs and the error word.
func (e *Emulator) Run(t *testing.T, k *backend.Kernel, outs, ins []backend.Buffer) ([][]byte, bool) {
	t.Helper()
	if err := backend.CheckRun(k, outs, ins); err != nil {
		t.Fatal(err)
	}
	l, err := gen.Plan(e.d, k, outs, ins)
	if err != nil {
		t.Fatal(err)
	}
	return e.Exec(t, l, outs, ins)
}

// Exec runs the launch l over the Host buffers outs and ins on the
// emulator and returns the contents of the outputs and the error word.
func (e *Emulator) Exec(t *testing.T, l *gen.Launch, outs, ins []backend.Buffer) ([][]byte, bool) {
	t.Helper()
	all := slices.Concat(outs, ins)
	if l == nil {
		res := make([][]byte, len(outs))
		for i := range outs {
			res[i] = make([]byte, byteSize(outs[i]))
		}
		return res, false
	}
	// The driver holds buffers as words of four bytes.
	words := func(b backend.Buffer) int { return (byteSize(b) + 3) / 4 }
	var types []backend.DType
	var in bytes.Buffer
	put := func(v any) { binary.Write(&in, binary.LittleEndian, v) }
	put(int32(len(all)))
	for _, b := range all {
		types = append(types, b.DType())
		put(int32(words(b)))
		in.Write(Contents(t, b))
		in.Write(make([]byte, 4*words(b)-byteSize(b)))
	}
	put(int32(len(l.Params)))
	put(l.Params)
	barriers := int32(0)
	if l.Barriers {
		barriers = 1
	}
	put([]int32{int32(l.Groups), int32(l.Threads), barriers})
	bin := e.binary(t, l, types)

	dir := t.TempDir()
	inPath, outPath := filepath.Join(dir, "in"), filepath.Join(dir, "out")
	if err := os.WriteFile(inPath, in.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(bin, inPath, outPath).CombinedOutput(); err != nil {
		t.Fatalf("run kernel: %v\n%s\n%s", err, out, gen.Numbered(l.Src))
	}
	res, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	var got [][]byte
	for _, b := range all {
		got = append(got, res[:byteSize(b)])
		res = res[4*words(b):]
	}
	return got[:len(outs)], binary.LittleEndian.Uint32(res) != 0
}

// EmulatedGEMM runs the products Plan accepts as a matrix library would,
// with Compute on the host, and their epilogues on the emulator. Other
// kernels run on the emulator.
type EmulatedGEMM struct {
	*Emulator
	// Plan returns how the library reads the product k of inputs whose
	// buffers have lens elements, or false if it does not run k.
	Plan func(k *backend.Kernel, lens []int) (gen.GEMM, bool)
	// Compute returns the result of g over the operand buffers x and y.
	Compute func(g gen.GEMM, x, y []float32) []float32
}

func (e EmulatedGEMM) Run(t *testing.T, k *backend.Kernel, outs, ins []backend.Buffer) ([][]byte, bool) {
	t.Helper()
	g, ok := e.Plan(k, []int{ins[0].Len(), ins[1].Len()})
	if !ok {
		return e.Emulator.Run(t, k, outs, ins)
	}
	raw := Bytes(e.Compute(g, Floats(t, ins[0]), Floats(t, ins[1])))
	if k.Attrs.Program == nil {
		return [][]byte{raw}, false
	}
	if err := Host.Upload(t.Context(), outs[0], raw); err != nil {
		t.Fatal(err)
	}
	l, err := gen.PlanEpilogue(e.d, k, outs, ins)
	if err != nil {
		t.Fatal(err)
	}
	return e.Exec(t, l, outs, ins)
}

// ColumnMajor computes the call s as a column-major BLAS does, over the
// operand buffers ins: for each batch, the column-major result
// C = op(A)·op(B).
func ColumnMajor(s gen.ColumnMajor, ins [2][]float32) []float32 {
	c := make([]float32, s.Batch*s.StrideC)
	for b := range s.Batch {
		for i := range s.M {
			for j := range s.N {
				var acc float32
				for p := range s.K {
					acc += ins[s.A.In][s.A.At(b, i, p)] * ins[s.B.In][s.B.At(b, p, j)]
				}
				c[b*s.StrideC+j*s.LDC+i] = acc
			}
		}
	}
	return c
}

// GEMM returns the result of g over the operand buffers x and y, reading
// them where g says.
func GEMM(g gen.GEMM, x, y []float32) []float32 {
	z := make([]float32, g.Batch*g.ResultSize)
	for b := range g.Batch {
		for i := range g.M {
			for j := range g.N {
				var acc float32
				for p := range g.K {
					acc += x[g.A.At(b, i, p)] * y[g.B.At(b, p, j)]
				}
				z[b*g.ResultSize+i*g.N+j] = acc
			}
		}
	}
	return z
}
