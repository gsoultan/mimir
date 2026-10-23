// Command pool generates the amd64 assembly of backend/cpu's pooling
// kernels over 2×2 windows two apart: the maximum of each window and the
// routing of a gradient back to the first largest element of each.
//
//	go run ./pool -out ../../backend/cpu/pool_amd64.s -stubs ../../backend/cpu/pool_stub_amd64.go -pkg cpu
//
// Each routine takes pairs of rows w elements wide, w even, the rows of
// a pair w elements apart and the pairs 2w apart, as the rows of
// contiguous planes of even height are, and the w/2 outputs of each pair
// contiguous. Masks cover the ends of the rows, so a routine handles rows
// of any even width on its own.
package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// consts holds the permutation indices and the NaN the routines use.
var consts = map[string]Mem{}

func constants() {
	idx := map[string][]uint32{"even": {}, "odd": {}, "lo": {}, "hi": {}}
	for i := range uint32(16) {
		idx["even"] = append(idx["even"], 2*i)
		idx["odd"] = append(idx["odd"], 2*i+1)
	}
	// lo interleaves lanes 0-7 of two registers, hi lanes 8-15: index
	// 16+i reads lane i of the second register.
	for i := range uint32(8) {
		idx["lo"] = append(idx["lo"], i, 16+i)
		idx["hi"] = append(idx["hi"], 8+i, 24+i)
	}
	for _, name := range []string{"even", "odd", "lo", "hi"} {
		m := GLOBL("pool_"+name, RODATA|NOPTR)
		for i, v := range idx[name] {
			DATA(4*i, U32(v))
		}
		consts[name] = m
	}
	m := GLOBL("pool_nan", RODATA|NOPTR)
	for i := range 16 {
		DATA(4*i, U32(0x7fc00000))
	}
	consts["nan"] = m
}

func main() {
	ConstraintExpr("!purego")
	constants()
	poolMax()
	poolGrad()
	Generate()
}

// windows holds the lanes of a chunk of up to 16 windows of a pair of
// rows: the even and odd elements of each row, deinterleaved, and the
// mask bits of the chunk's outputs, m16, and of its 32 inputs, m32, in
// general registers, as AVX-512 has seven mask registers to share.
type windows struct {
	ea, oa, eb, ob VecVirtual
	m16, m32       Register
}

// masks sets m16 to the lanes below rem and m32 to the lanes below 2·rem.
var maskLabels int

func masks(rem Register) (m16, m32 Register) {
	maskLabels++
	full := fmt.Sprintf("full%d", maskLabels)
	m16, m32 = GP32(), GP32()
	MOVL(U32(0xffff), m16)
	MOVL(U32(0xffffffff), m32)
	CMPQ(rem, Imm(16))
	JGE(LabelRef(full))
	MOVL(U32(1), m16)
	MOVL(U32(1), m32)
	MOVQ(rem, RCX)
	SHLL(CL, m16)
	ADDQ(RCX, RCX)
	SHLL(CL, m32)
	DECL(m16)
	DECL(m32)
	Label(full)
	return m16, m32
}

// k returns a mask register holding the bits of m, or of its high half.
func k(m Register, high bool) Register {
	r := K()
	if high {
		t := GP32()
		MOVL(m, t)
		SHRL(U8(16), t)
		KMOVW(t, r)
		return r
	}
	KMOVW(m, r)
	return r
}

// load reads the chunk of windows at pa, the first row of a pair, whose
// second row is wBytes on, as deinterleaved lanes, with the masks of rem
// outputs. even and odd hold the permutation indices.
func load(pa, wBytes, rem Register, even, odd VecVirtual) windows {
	var w windows
	w.m16, w.m32 = masks(rem)
	kLo, kHi := k(w.m32, false), k(w.m32, true)
	split := func(at Mem) (e, o VecVirtual) {
		x0, x1 := ZMM(), ZMM()
		VMOVUPS_Z(at, kLo, x0)
		at.Disp += 64
		VMOVUPS_Z(at, kHi, x1)
		e, o = ZMM(), ZMM()
		VMOVAPS(x0, e)
		VPERMT2PS(x1, even, e)
		VMOVAPS(x0, o)
		VPERMT2PS(x1, odd, o)
		return e, o
	}
	w.ea, w.oa = split(Mem{Base: pa})
	w.eb, w.ob = split(Mem{Base: pa, Index: wBytes, Scale: 1})
	return w
}

// maxOf returns the largest lane of each window. VMAXPS returns its
// second source where either is NaN, so the lanes of unordered pairs are
// set to NaN after, as the reduction over the windows propagates it.
func maxOf(w windows, nan VecVirtual, propagateNaN bool) VecVirtual {
	m, m2 := ZMM(), ZMM()
	VMAXPS(w.oa, w.ea, m)
	VMAXPS(w.ob, w.eb, m2)
	VMAXPS(m2, m, m)
	if propagateNaN {
		ka, kb := K(), K()
		VCMPPS(U8(3), w.oa, w.ea, ka)
		VCMPPS(U8(3), w.ob, w.eb, kb)
		KORW(kb, ka, ka)
		VBLENDMPS(nan, m, ka, m)
	}
	return m
}

func poolMax() {
	name := "poolMax2x2AVX512"
	TEXT(name, NOSPLIT, "func(out *float32, x *float32, w int, pairs int)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s sets the w/2 elements at out of each of pairs pairs of rows w wide at x, w even, to the largest of the 2×2 window of each, NaN where a window holds one.", name))
	pOut := Load(Param("out"), GP64())
	pX := Load(Param("x"), GP64())
	w := Load(Param("w"), GP64())
	pairs := Load(Param("pairs"), GP64())
	even, odd, nan := ZMM(), ZMM(), ZMM()
	VMOVUPS(consts["even"], even)
	VMOVUPS(consts["odd"], odd)
	VMOVUPS(consts["nan"], nan)
	wBytes := GP64()
	MOVQ(w, wBytes)
	SHLQ(U8(2), wBytes)
	ow := GP64()
	MOVQ(w, ow)
	SHRQ(U8(1), ow)
	TESTQ(pairs, pairs)
	JZ(LabelRef(name + "_done"))
	Label(name + "_pair")
	pa, po, rem := GP64(), GP64(), GP64()
	MOVQ(pX, pa)
	MOVQ(pOut, po)
	MOVQ(ow, rem)
	Label(name + "_chunk")
	win := load(pa, wBytes, rem, even, odd)
	m := maxOf(win, nan, true)
	VMOVUPS(m, k(win.m16, false), Mem{Base: po})
	ADDQ(U32(128), pa)
	ADDQ(U32(64), po)
	SUBQ(U32(16), rem)
	JG(LabelRef(name + "_chunk"))
	ADDQ(wBytes, pX)
	ADDQ(wBytes, pX)
	LEAQ(Mem{Base: pOut, Index: ow, Scale: 4}, pOut)
	DECQ(pairs)
	JNZ(LabelRef(name + "_pair"))
	Label(name + "_done")
	VZEROUPPER()
	RET()
}

func poolGrad() {
	name := "poolGrad2x2AVX512"
	TEXT(name, NOSPLIT, "func(dx *float32, x *float32, grad *float32, w int, pairs int)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s writes to dx, shaped like the pairs pairs of rows w wide at x, w even, the w/2 gradients per pair at g, each to the first element in row-major order of its 2×2 window that holds the window's maximum, and zeros elsewhere.", name))
	pDX := Load(Param("dx"), GP64())
	pX := Load(Param("x"), GP64())
	pG := Load(Param("grad"), GP64())
	w := Load(Param("w"), GP64())
	pairs := Load(Param("pairs"), GP64())
	even, odd, lo, hi, nan := ZMM(), ZMM(), ZMM(), ZMM(), ZMM()
	VMOVUPS(consts["even"], even)
	VMOVUPS(consts["odd"], odd)
	VMOVUPS(consts["lo"], lo)
	VMOVUPS(consts["hi"], hi)
	VMOVUPS(consts["nan"], nan)
	wBytes := GP64()
	MOVQ(w, wBytes)
	SHLQ(U8(2), wBytes)
	ow := GP64()
	MOVQ(w, ow)
	SHRQ(U8(1), ow)
	TESTQ(pairs, pairs)
	JZ(LabelRef(name + "_done"))
	Label(name + "_pair")
	pa, pda, pg, rem := GP64(), GP64(), GP64(), GP64()
	MOVQ(pX, pa)
	MOVQ(pDX, pda)
	MOVQ(pG, pg)
	MOVQ(ow, rem)
	Label(name + "_chunk")
	win := load(pa, wBytes, rem, even, odd)
	m := maxOf(win, nan, true)
	// The first lane of each window equal to its maximum, in the order
	// ea, oa, eb, ob: each mask excludes the lanes an earlier one took.
	// A window with a NaN has NaN for its maximum, which nothing equals,
	// so its first NaN takes the gradient, as the Go loops' keys order it.
	hits := func(x VecVirtual, h Register) {
		u := K()
		VCMPPS(U8(0), m, x, h)
		VCMPPS(U8(3), x, x, u)
		KORW(u, h, h)
	}
	h0, h1, h2, h3, t, taken := K(), K(), K(), K(), K(), K()
	hits(win.ea, h0)
	hits(win.oa, t)
	KANDNW(t, h0, h1)
	KORW(h1, h0, taken)
	hits(win.eb, t)
	KANDNW(t, taken, h2)
	KORW(h2, taken, taken)
	hits(win.ob, t)
	KANDNW(t, taken, h3)
	gv := ZMM()
	VMOVUPS_Z(Mem{Base: pg}, k(win.m16, false), gv)
	// The gradients of a row's even and odd elements, interleaved back
	// into the row's order.
	row := func(he, ho Register, at Mem) {
		ge, go2 := ZMM(), ZMM()
		VMOVAPS_Z(gv, he, ge)
		VMOVAPS_Z(gv, ho, go2)
		l, h := ZMM(), ZMM()
		VMOVAPS(ge, l)
		VPERMT2PS(go2, lo, l)
		VMOVAPS(ge, h)
		VPERMT2PS(go2, hi, h)
		VMOVUPS(l, k(win.m32, false), at)
		at.Disp += 64
		VMOVUPS(h, k(win.m32, true), at)
	}
	row(h0, h1, Mem{Base: pda})
	row(h2, h3, Mem{Base: pda, Index: wBytes, Scale: 1})
	ADDQ(U32(128), pa)
	ADDQ(U32(128), pda)
	ADDQ(U32(64), pg)
	SUBQ(U32(16), rem)
	JG(LabelRef(name + "_chunk"))
	ADDQ(wBytes, pX)
	ADDQ(wBytes, pX)
	ADDQ(wBytes, pDX)
	ADDQ(wBytes, pDX)
	half := GP64()
	MOVQ(wBytes, half)
	SHRQ(U8(1), half)
	ADDQ(half, pG)
	DECQ(pairs)
	JNZ(LabelRef(name + "_pair"))
	Label(name + "_done")
	VZEROUPPER()
	RET()
}
