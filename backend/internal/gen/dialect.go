package gen

import (
	"fmt"
	"strings"

	"github.com/gsoultan/mimir/backend"
)

// Dialect is the language a device compiles kernels in. Kernels call the
// helpers its prelude defines, so the code generator writes the same body
// for every dialect:
//
//	fbits(b)        the float whose bits are the int or uint b
//	ubits(f)        the bits of the float f, as a uint
//	cos_pi(x)       cos(πx), and sin_pi(x) sin(πx)
//	mulhi(a, b)     the high 32 bits of the 64-bit product of uints
//	report(err, b)  ORs the bits b into the error word
//	mx(a, b)        the maximum, which propagates NaN and orders -0 below +0
//	md(a, b, err)   the truncated remainder; int division by zero reports
//	bf2f(h), h2f(h) the float of the bfloat16 or binary16 element h
//	f2bf(f), f2h(f) the nearest bfloat16 or binary16 element, ties to even
//
// and declares the variables gid, the thread's index in the launch, tid,
// its index in its group, and bid, its group's index. The half
// conversions are the common half of the preludes, in plain integer
// arithmetic that matches the CPU backend's bit for bit.
type Dialect struct {
	// Name is the dialect's name, such as "metal".
	Name string

	prelude string
	// param is the format of the expression reading parameter %d.
	param string
	// shared declares a variable its group shares, and sync waits for
	// every thread of the group to reach it.
	shared, sync string
	// signature writes the kernel's declaration and opening brace.
	signature func(s *strings.Builder, outs, ins []backend.DType, params int)
}

// Metal is Metal Shading Language. A kernel takes its outputs, inputs,
// parameters and error word as buffers 0, 1 and so on, in that order.
var Metal = &Dialect{
	Name:   "metal",
	param:  "p[%d]",
	shared: "threadgroup",
	sync:   "threadgroup_barrier(mem_flags::mem_threadgroup)",
	prelude: `#include <metal_stdlib>
using namespace metal;

static inline float fbits(int b) { return as_type<float>(b); }
static inline float fbits(uint b) { return as_type<float>(b); }
static inline uint ubits(float f) { return as_type<uint>(f); }
static inline float cos_pi(float x) { return cospi(x); }
static inline float sin_pi(float x) { return sinpi(x); }
static inline void report(device atomic_uint* err, uint bits) {
	atomic_fetch_or_explicit(err, bits, memory_order_relaxed);
}
static inline float mx(float a, float b) {
	if (a != a || b != b) return a + b;
	if (a == b) return signbit(a) ? b : a;
	return a > b ? a : b;
}
static inline int mx(int a, int b) { return a > b ? a : b; }
static inline int md(int a, int b, device atomic_uint* err) {
	if (b == 0) {
		report(err, 1u);
		return 0;
	}
	return b == -1 ? 0 : a % b;
}
static inline float md(float a, float b, device atomic_uint* err) { return fmod(a, b); }
` + halfPrelude("static inline"),
	signature: func(s *strings.Builder, outs, ins []backend.DType, params int) {
		var args []string
		for i, dt := range outs {
			args = append(args, fmt.Sprintf("device %s* o%d [[buffer(%d)]]", elemName(dt), i, len(args)))
		}
		for i, dt := range ins {
			args = append(args, fmt.Sprintf("const device %s* i%d [[buffer(%d)]]", elemName(dt), i, len(args)))
		}
		n := len(args)
		args = append(args,
			fmt.Sprintf("constant int* p [[buffer(%d)]]", n),
			fmt.Sprintf("device atomic_uint* err [[buffer(%d)]]", n+1),
			"uint gid [[thread_position_in_grid]]",
			"uint tid [[thread_position_in_threadgroup]]",
			"uint bid [[threadgroup_position_in_grid]]")
		fmt.Fprintf(s, "kernel void k(%s) {\n", strings.Join(args, ", "))
	},
}

// CUDA is CUDA C++ as NVRTC compiles it, with no headers. A kernel named
// k takes its outputs and inputs as pointers, its parameters by value as a
// struct Params of one int array, and the error word as a pointer, in
// that order.
var CUDA = &Dialect{
	Name:   "cuda",
	param:  "p.v[%d]",
	shared: "__shared__",
	sync:   "__syncthreads()",
	prelude: `typedef unsigned int uint;
typedef unsigned short ushort;

static __device__ inline float fbits(int b) { return __int_as_float(b); }
static __device__ inline float fbits(uint b) { return __uint_as_float(b); }
static __device__ inline uint ubits(float f) { return __float_as_uint(f); }
static __device__ inline float cos_pi(float x) { return cospif(x); }
static __device__ inline float sin_pi(float x) { return sinpif(x); }
static __device__ inline uint mulhi(uint a, uint b) { return __umulhi(a, b); }
static __device__ inline void report(uint* err, uint bits) { atomicOr(err, bits); }
static __device__ inline float mx(float a, float b) {
	if (a != a || b != b) return a + b;
	if (a == b) return (__float_as_uint(a) >> 31) ? b : a;
	return a > b ? a : b;
}
static __device__ inline int mx(int a, int b) { return a > b ? a : b; }
static __device__ inline int md(int a, int b, uint* err) {
	if (b == 0) {
		report(err, 1u);
		return 0;
	}
	return b == -1 ? 0 : a % b;
}
static __device__ inline float md(float a, float b, uint* err) { return fmodf(a, b); }
` + halfPrelude("static __device__ inline"),
	signature: func(s *strings.Builder, outs, ins []backend.DType, params int) {
		fmt.Fprintf(s, "struct Params { int v[%d]; };\n\n", params)
		var args []string
		for i, dt := range outs {
			args = append(args, fmt.Sprintf("%s* o%d", elemName(dt), i))
		}
		for i, dt := range ins {
			args = append(args, fmt.Sprintf("const %s* i%d", elemName(dt), i))
		}
		args = append(args, "Params p", "uint* err")
		fmt.Fprintf(s, "extern \"C\" __global__ void k(%s) {\n", strings.Join(args, ", "))
		s.WriteString("\tuint gid = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x, bid = blockIdx.x;\n")
	},
}

// halfPrelude returns the conversions between float and the half
// element types, declared with decl, as the CPU backend computes them: a
// bfloat16 is the top half of a float; a binary16 is converted field by
// field, with its subnormals, infinities and NaNs. Rounding is to nearest,
// ties to even, overflow to infinity, and a NaN keeps its sign and the
// top of its payload, made quiet.
func halfPrelude(decl string) string {
	return strings.ReplaceAll(`
DECL float bf2f(ushort h) { return fbits(uint(h) << 16); }
DECL ushort f2bf(float f) {
	uint b = ubits(f);
	if ((b & 0x7fffffffu) > 0x7f800000u) return ushort((b >> 16) | 0x40u);
	b += 0x7fffu + ((b >> 16) & 1u);
	return ushort(b >> 16);
}
DECL float h2f(ushort h) {
	uint sign = uint(h & 0x8000u) << 16;
	uint exp = (uint(h) >> 10) & 0x1fu;
	uint mant = uint(h & 0x3ffu);
	if (exp == 0u) {
		if (mant == 0u) return fbits(sign);
		uint shift = 0u;
		while ((mant & 0x400u) == 0u) { mant <<= 1; shift++; }
		return fbits(sign | ((113u - shift) << 23) | ((mant & 0x3ffu) << 13));
	}
	if (exp == 0x1fu) return fbits(sign | 0x7f800000u | (mant << 13));
	return fbits(sign | ((exp + 112u) << 23) | (mant << 13));
}
DECL uint rne(uint h, uint m, uint shift) {
	uint rest = m & ((1u << shift) - 1u);
	uint mid = 1u << (shift - 1u);
	if (rest > mid || (rest == mid && (h & 1u) == 1u)) h++;
	return h;
}
DECL ushort f2h(float f) {
	uint b = ubits(f);
	uint sign = (b >> 16) & 0x8000u;
	b &= 0x7fffffffu;
	if (b > 0x7f800000u) return ushort(sign | 0x7e00u | ((b >> 13) & 0x1ffu));
	int exp = int(b >> 23) - 127;
	uint mant = b & 0x7fffffu;
	if (exp > 15) return ushort(sign | 0x7c00u);
	if (exp >= -14) return ushort(sign | rne((uint(exp + 15) << 10) | (mant >> 13), mant, 13u));
	if (exp >= -25) {
		uint m = mant | 0x800000u;
		uint shift = uint(-exp - 1);
		return ushort(sign | rne(m >> shift, m, shift));
	}
	return ushort(sign);
}

`, "DECL ", decl+" ")
}
