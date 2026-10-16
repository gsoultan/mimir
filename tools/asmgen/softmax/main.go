// Command softmax generates the amd64 assembly of the row operations of
// backend/cpu's attention kernels: the maximum of a row, exponentials
// shifted by it and their sum, scaling a row, and the softmax gradient;
// and of its elementwise kernels and products of few rows: sums,
// products, reciprocals and powers of 2 of vectors, and dot products.
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
		dot4(v)
		exp2(v)
		for _, op := range []string{"add", "mul"} {
			binaryVV(v, op)
			binaryVS(v, op)
		}
		recip(v)
	}
	Generate()
}

// exp2Coeffs are those of Cephes' exp2f: 2^f = 1 + f·q(f) for f in
// [-1/2, 1/2], with q of degree 5, within 2 ulp.
var exp2Coeffs = []float32{1.535336188319500e-4, 1.339887440266574e-3, 9.618437357674640e-3, 5.550332471162809e-2, 2.402264791363012e-1, 6.931472028550421e-1}

func exp2(v isa) {
	name := "exp2" + v.name
	TEXT(name, NOSPLIT, "func(dst *float32, src *float32, n int)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s sets the n elements at dst, a multiple of %d, to 2 to the power of those at src, which it clamps to [-126, 127].", name, v.lanes))
	pd := Load(Param("dst"), GP64())
	ps := Load(Param("src"), GP64())
	n := Load(Param("n"), GP64())
	c := func(name string) Mem { return consts[name] }
	TESTQ(n, n)
	JZ(LabelRef(name + "_done"))
	Label(name + "_loop")
	x := v.vec()
	VMOVUPS(Mem{Base: ps}, x)
	VMAXPS(c("exp2lo"), x, x)
	VMINPS(c("exp2hi"), x, x)
	ni, nf := v.vec(), v.vec()
	VCVTPS2DQ(x, ni)
	VCVTDQ2PS(ni, nf)
	VSUBPS(nf, x, x) // f = x - n, exactly
	y := v.vec()
	VMOVUPS(c("q0"), y)
	for i := 1; i < len(exp2Coeffs); i++ {
		VFMADD213PS(c(fmt.Sprintf("q%d", i)), x, y)
	}
	VFMADD213PS(c("one"), x, y) // 1 + f·q(f)
	VPADDD(c("bias"), ni, ni)
	VPSLLD(U8(23), ni, ni)
	VMULPS(ni, y, y)
	VMOVUPS(y, Mem{Base: pd})
	ADDQ(U32(4*v.lanes), pd)
	ADDQ(U32(4*v.lanes), ps)
	SUBQ(U32(v.lanes), n)
	JNZ(LabelRef(name + "_loop"))
	Label(name + "_done")
	VZEROUPPER()
	RET()
}

// arith emits the instruction of a binary operation: dst = x op y, with y
// a register or memory.
func arith(op string, y, x Op, dst VecVirtual) {
	switch op {
	case "add":
		VADDPS(y, x, dst)
	case "mul":
		VMULPS(y, x, dst)
	}
}

// unroll is how many vectors the elementwise loops handle at a time.
const unroll = 4

// elementLoop emits a loop over the n elements at the pointers in ptrs, a
// whole number of vectors, calling body with the displacement of each
// vector from the pointers, unroll vectors at a time while there are as
// many.
func elementLoop(v isa, name string, ptrs []Register, n Register, body func(disp int)) {
	step := unroll * v.lanes
	CMPQ(n, U32(step))
	JL(LabelRef(name + "_one"))
	Label(name + "_wide")
	for u := range unroll {
		body(4 * v.lanes * u)
	}
	for _, p := range ptrs {
		ADDQ(U32(4*step), p)
	}
	SUBQ(U32(step), n)
	CMPQ(n, U32(step))
	JGE(LabelRef(name + "_wide"))
	Label(name + "_one")
	TESTQ(n, n)
	JZ(LabelRef(name + "_done"))
	body(0)
	for _, p := range ptrs {
		ADDQ(U32(4*v.lanes), p)
	}
	SUBQ(U32(v.lanes), n)
	JMP(LabelRef(name + "_one"))
	Label(name + "_done")
	VZEROUPPER()
	RET()
}

func binaryVV(v isa, op string) {
	name := op + "VV" + v.name
	TEXT(name, NOSPLIT, "func(dst *float32, a *float32, b *float32, n int)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s sets the n elements at dst, a multiple of %d, to the %s of those at a and b.", name, v.lanes, map[string]string{"add": "sums", "mul": "products"}[op]))
	pd := Load(Param("dst"), GP64())
	pa := Load(Param("a"), GP64())
	pb := Load(Param("b"), GP64())
	n := Load(Param("n"), GP64())
	elementLoop(v, name, []Register{pd, pa, pb}, n, func(disp int) {
		x := v.vec()
		VMOVUPS(Mem{Base: pa, Disp: disp}, x)
		arith(op, Mem{Base: pb, Disp: disp}, x, x)
		VMOVUPS(x, Mem{Base: pd, Disp: disp})
	})
}

func binaryVS(v isa, op string) {
	name := op + "VS" + v.name
	TEXT(name, NOSPLIT, "func(dst *float32, a *float32, s float32, n int)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s sets the n elements at dst, a multiple of %d, to the %s of those at a and s.", name, v.lanes, map[string]string{"add": "sums", "mul": "products"}[op]))
	pd := Load(Param("dst"), GP64())
	pa := Load(Param("a"), GP64())
	sv := broadcast(v, "s")
	n := Load(Param("n"), GP64())
	elementLoop(v, name, []Register{pd, pa}, n, func(disp int) {
		x := v.vec()
		arith(op, Mem{Base: pa, Disp: disp}, sv, x)
		VMOVUPS(x, Mem{Base: pd, Disp: disp})
	})
}

func recip(v isa) {
	name := "recip" + v.name
	TEXT(name, NOSPLIT, "func(dst *float32, a *float32, n int)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s sets the n elements at dst, a multiple of %d, to the reciprocals of those at a, correctly rounded.", name, v.lanes))
	pd := Load(Param("dst"), GP64())
	pa := Load(Param("a"), GP64())
	n := Load(Param("n"), GP64())
	one := v.vec()
	VMOVUPS(consts["one"], one)
	elementLoop(v, name, []Register{pd, pa}, n, func(disp int) {
		x := v.vec()
		VDIVPS(Mem{Base: pa, Disp: disp}, one, x)
		VMOVUPS(x, Mem{Base: pd, Disp: disp})
	})
}

// dot4 computes the dot products of a vector with four rows at once, so
// that a product of few rows streams four rows of its other operand
// together. Each row's sum takes the same order whatever the other rows
// are, so a row repeated with stride 0 gets the sum it gets among others.
func dot4(v isa) {
	name := "dot4" + v.name
	TEXT(name, NOSPLIT, "func(x *float32, y *float32, stride int, n int, out *float32)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s sets out[r], for r < 4, to the sum of the products of the n elements at x, a multiple of %d, and those of row r of y, which starts stride elements after row r-1.", name, v.lanes))
	px := Load(Param("x"), GP64())
	stride := Load(Param("stride"), GP64())
	n := Load(Param("n"), GP64())
	SHLQ(U8(2), stride)
	var rows [4]Register
	rows[0] = Load(Param("y"), GP64())
	for r := 1; r < 4; r++ {
		rows[r] = GP64()
		LEAQ(Mem{Base: rows[r-1], Index: stride, Scale: 1}, rows[r])
	}
	// Two accumulators a row, over alternate vectors.
	var acc [4][2]VecVirtual
	for r := range acc {
		for i := range acc[r] {
			acc[r][i] = v.vec()
			VXORPS(acc[r][i], acc[r][i], acc[r][i])
		}
	}
	step := 2 * v.lanes
	CMPQ(n, U32(step))
	JL(LabelRef(name + "_tail"))
	Label(name + "_wide")
	x0, x1 := v.vec(), v.vec()
	VMOVUPS(Mem{Base: px}, x0)
	VMOVUPS(Mem{Base: px, Disp: 4 * v.lanes}, x1)
	for r := range rows {
		VFMADD231PS(Mem{Base: rows[r]}, x0, acc[r][0])
		VFMADD231PS(Mem{Base: rows[r], Disp: 4 * v.lanes}, x1, acc[r][1])
		ADDQ(U32(4*step), rows[r])
	}
	ADDQ(U32(4*step), px)
	SUBQ(U32(step), n)
	CMPQ(n, U32(step))
	JGE(LabelRef(name + "_wide"))
	Label(name + "_tail")
	TESTQ(n, n)
	JZ(LabelRef(name + "_done"))
	x := v.vec()
	VMOVUPS(Mem{Base: px}, x)
	for r := range rows {
		VFMADD231PS(Mem{Base: rows[r]}, x, acc[r][0])
	}
	Label(name + "_done")
	out := Load(Param("out"), GP64())
	for r := range acc {
		VADDPS(acc[r][1], acc[r][0], acc[r][0])
		MOVSS(horizontal(v, acc[r][0], VADDPS), Mem{Base: out, Disp: 4 * r})
	}
	VZEROUPPER()
	RET()
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
	f["exp2lo"], f["exp2hi"] = -126, 127
	for i, c := range exp2Coeffs {
		f[fmt.Sprintf("q%d", i)] = c
	}
	names := []string{"log2e", "ln2hi", "ln2lo", "one", "lowest", "highest", "p0", "p1", "p2", "p3", "p4", "p5",
		"exp2lo", "exp2hi", "q0", "q1", "q2", "q3", "q4", "q5"}
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
