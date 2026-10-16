// Command softmax generates the amd64 assembly of the row operations of
// backend/cpu's attention kernels: the maximum of a row, exponentials
// shifted by it and their sum, scaling a row, and the softmax gradient.
//
//	go run ./softmax -out ../../backend/cpu/softmax_amd64.s -stubs ../../backend/cpu/softmax_stub_amd64.go -pkg cpu
//
// Each routine takes rows of a whole number of vectors; the Go code that
// calls it handles the rest of a row.
package main

import (
	"fmt"
	"math"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// isa describes one vector extension.
type isa struct {
	name  string
	lanes int // float32 elements per register
	vec   func() VecVirtual
	// mask returns a register for a comparison's result, and below
	// zeroes the lanes of v where it is set.
	mask  func() Register
	below func(x, limit Op, m Register)
	clear func(v VecVirtual, m Register)
}

var avx512 = isa{
	name:  "AVX512",
	lanes: 16,
	vec:   func() VecVirtual { return ZMM() },
	mask:  func() Register { return K() },
	below: func(x, limit Op, m Register) { VCMPPS(U8(1), limit, x, m) },
	clear: func(v VecVirtual, m Register) {
		zero := ZMM()
		VPXORD(zero, zero, zero)
		VBLENDMPS(zero, v, m, v)
	},
}

var avx2 = isa{
	name:  "AVX2",
	lanes: 8,
	vec:   func() VecVirtual { return YMM() },
	mask:  func() Register { return YMM() },
	below: func(x, limit Op, m Register) { VCMPPS(U8(1), limit, x, m) },
	clear: func(v VecVirtual, m Register) { VANDNPS(v, m, v) },
}

func main() {
	ConstraintExpr("!purego")
	constants()
	for _, v := range []isa{avx512, avx2} {
		rowMax(v)
		expShift(v)
		scaleRow(v)
		softmaxGrad(v)
	}
	Generate()
}

// The exponential is Cephes' expf: e^x = 2^n · e^r with n = round(x/ln 2)
// and r = x - n·ln 2 in [-ln 2/2, ln 2/2], where ln 2 is split in two so
// that r is exact, and e^r = 1 + r + r²·p(r) with p of degree 5. It is
// within 2 ulp of e^x. Arguments below lowest give 0, and arguments are
// capped at highest, whose 2^n is finite.
const (
	lowest  = -87.33654 // ln of the smallest normal float32
	highest = 88.0
)

var coeffs = []float32{1.9875691500e-4, 1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1}

// consts holds the constants, each as 16 copies, so that every vector
// extension reads a whole register of it.
var consts = map[string]Mem{}

func constants() {
	f := map[string]float32{
		"log2e": float32(math.Log2E), "ln2hi": 0.693359375, "ln2lo": -2.12194440e-4,
		"one": 1, "lowest": lowest, "highest": highest,
	}
	for i, c := range coeffs {
		f[fmt.Sprintf("p%d", i)] = c
	}
	names := []string{"log2e", "ln2hi", "ln2lo", "one", "lowest", "highest", "p0", "p1", "p2", "p3", "p4", "p5"}
	for _, name := range names {
		m := GLOBL("exp_"+name, RODATA|NOPTR)
		for i := range 16 {
			DATA(4*i, U32(math.Float32bits(f[name])))
		}
		consts[name] = m
	}
	m := GLOBL("exp_bias", RODATA|NOPTR)
	for i := range 16 {
		DATA(4*i, U32(127))
	}
	consts["bias"] = m
}

// broadcast returns a register holding the float32 parameter name in
// every lane.
func broadcast(v isa, name string) VecVirtual {
	x := XMM()
	Load(Param(name), x)
	r := v.vec()
	VBROADCASTSS(x, r)
	return r
}

// horizontal reduces the lanes of r with op into lane 0 of an XMM
// register and returns it.
func horizontal(v isa, r VecVirtual, op func(...Op)) Register {
	var y Register = r
	if v.lanes == 16 {
		hi := YMM()
		VEXTRACTF64X4(U8(1), r, hi) // AVX512F, unlike VEXTRACTF32X8
		y = r.AsY()
		op(hi, y, y)
	}
	hi := XMM()
	VEXTRACTF128(U8(1), y, hi)
	x := y.(Vec).AsX()
	op(hi, x, x)
	t := XMM()
	VPERMILPS(U8(0x4e), x, t)
	op(t, x, x)
	VPERMILPS(U8(0xb1), x, t)
	op(t, x, x)
	return x
}

// loop emits a loop over the n elements at p, which are a whole number of
// vectors, calling body with the address of each vector.
func loop(v isa, name string, p, n Register, body func(at Mem)) {
	TESTQ(n, n)
	JZ(LabelRef(name + "_done"))
	Label(name + "_loop")
	body(Mem{Base: p})
	ADDQ(U32(4*v.lanes), p)
	SUBQ(U32(v.lanes), n)
	JNZ(LabelRef(name + "_loop"))
	Label(name + "_done")
}

func rowMax(v isa) {
	name := "rowMax" + v.name
	TEXT(name, NOSPLIT, "func(x *float32, n int) float32")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s returns the largest of the n elements at x, a positive multiple of %d.", name, v.lanes))
	p := Load(Param("x"), GP64())
	n := Load(Param("n"), GP64())
	m := v.vec()
	VMOVUPS(Mem{Base: p}, m)
	loop(v, name, p, n, func(at Mem) { VMAXPS(at, m, m) })
	Store(horizontal(v, m, VMAXPS), ReturnIndex(0))
	VZEROUPPER()
	RET()
}

func expShift(v isa) {
	name := "expShift" + v.name
	TEXT(name, NOSPLIT, "func(x *float32, n int, shift float32) float32")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s sets each of the n elements at x, a multiple of %d, to e^(x-shift) and returns their sum.", name, v.lanes))
	p := Load(Param("x"), GP64())
	n := Load(Param("n"), GP64())
	shift := broadcast(v, "shift")
	sum := v.vec()
	VXORPS(sum, sum, sum)
	c := func(name string) Mem { return consts[name] }
	loop(v, name, p, n, func(at Mem) {
		x := v.vec()
		VMOVUPS(at, x)
		VSUBPS(shift, x, x)
		low := v.mask()
		v.below(x, c("lowest"), low)
		VMAXPS(c("lowest"), x, x)
		VMINPS(c("highest"), x, x)
		// n = round(x·log2 e), as an int and a float.
		ni, nf := v.vec(), v.vec()
		VMULPS(c("log2e"), x, nf)
		VCVTPS2DQ(nf, ni)
		VCVTDQ2PS(ni, nf)
		// r = x - n·ln 2
		VFNMADD231PS(c("ln2hi"), nf, x)
		VFNMADD231PS(c("ln2lo"), nf, x)
		// e^r = 1 + r + r²·p(r)
		y := v.vec()
		VMOVUPS(c("p0"), y)
		for i := 1; i < len(coeffs); i++ {
			VFMADD213PS(c(fmt.Sprintf("p%d", i)), x, y)
		}
		r2 := v.vec()
		VMULPS(x, x, r2)
		VFMADD213PS(x, r2, y)
		VADDPS(c("one"), y, y)
		// · 2^n
		VPADDD(c("bias"), ni, ni)
		VPSLLD(U8(23), ni, ni)
		VMULPS(ni, y, y)
		v.clear(y, low)
		VMOVUPS(y, at)
		VADDPS(y, sum, sum)
	})
	Store(horizontal(v, sum, VADDPS), ReturnIndex(0))
	VZEROUPPER()
	RET()
}

func scaleRow(v isa) {
	name := "scaleRow" + v.name
	TEXT(name, NOSPLIT, "func(x *float32, n int, a float32)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s multiplies each of the n elements at x, a multiple of %d, by a.", name, v.lanes))
	p := Load(Param("x"), GP64())
	n := Load(Param("n"), GP64())
	a := broadcast(v, "a")
	loop(v, name, p, n, func(at Mem) {
		x := v.vec()
		VMULPS(at, a, x)
		VMOVUPS(x, at)
	})
	VZEROUPPER()
	RET()
}

func softmaxGrad(v isa) {
	name := "softmaxGrad" + v.name
	TEXT(name, NOSPLIT, "func(p *float32, dp *float32, n int, d float32)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s sets each of the n elements at p, a multiple of %d, to p·(dp-d), with dp the element at dp.", name, v.lanes))
	pp := Load(Param("p"), GP64())
	dp := Load(Param("dp"), GP64())
	n := Load(Param("n"), GP64())
	d := broadcast(v, "d")
	TESTQ(n, n)
	JZ(LabelRef(name + "_done"))
	Label(name + "_loop")
	x := v.vec()
	VMOVUPS(Mem{Base: dp}, x)
	VSUBPS(d, x, x)
	VMULPS(Mem{Base: pp}, x, x)
	VMOVUPS(x, Mem{Base: pp})
	ADDQ(U32(4*v.lanes), pp)
	ADDQ(U32(4*v.lanes), dp)
	SUBQ(U32(v.lanes), n)
	JNZ(LabelRef(name + "_loop"))
	Label(name + "_done")
	VZEROUPPER()
	RET()
}
