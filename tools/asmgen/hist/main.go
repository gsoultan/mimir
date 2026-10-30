// Command hist generates the amd64 assembly of package tree's histogram
// kernels: adding each row's gradients, hessians and count into its bin
// of every feature, with AVX2, and the CPUID helpers that choose them.
//
//	go run ./hist -out ../../tree/hist_amd64.s -stubs ../../tree/hist_stub_amd64.go -pkg tree
//
// Each row's gradients, hessians and count of one are kept side by side
// as one bin's worth of float32, four or eight of them, which a kernel
// widens to float64 once per row, so a bin is one or two vector adds
// where the Go loop makes one scalar add per number. Rows go two at a time, their
// features interleaved, so the second row's loads start while the first
// row's stores are in flight; within a feature the first row still adds
// before the second, so the sums come out bit for bit as the Go loop's.
package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

func main() {
	ConstraintExpr("!purego")
	addRows(4)
	addRows(8)

	TEXT("cpuid", NOSPLIT, "func(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)")
	Doc("cpuid executes CPUID with the given EAX and ECX.")
	Load(Param("eaxArg"), EAX)
	Load(Param("ecxArg"), ECX)
	CPUID()
	Store(EAX, Return("eax"))
	Store(EBX, Return("ebx"))
	Store(ECX, Return("ecx"))
	Store(EDX, Return("edx"))
	RET()

	TEXT("xgetbv", NOSPLIT, "func() (eax, edx uint32)")
	Doc("xgetbv reads extended control register 0.")
	XORL(ECX, ECX)
	XGETBV()
	Store(EAX, Return("eax"))
	Store(EDX, Return("edx"))
	RET()

	Generate()
}

// addRows writes the kernel for bins of s float64, s 4 or 8.
func addRows(s int) {
	name := fmt.Sprintf("addRows%dAVX2", s)
	TEXT(name, NOSPLIT, "func(hist *float64, bins *uint8, f int, feats *int, offs *int, m int, rows *int32, sums *float32, n int)")
	Pragma("noescape")
	Doc(name+" adds the "+fmt.Sprint(s)+" float32 at sums["+fmt.Sprint(s)+"*r] of each of the n rows r at rows, widened to float64,",
		"to the bin at hist["+fmt.Sprint(s)+"*(offs[j]+bins[r*f+feats[j]])] for each j below m.")
	shift, rowShift := U8(5), U8(4)
	if s == 8 {
		shift, rowShift = U8(6), U8(5)
	}
	hist, bins, feats, offs, rows, src, n := RDI, RSI, R9, R10, R12, R13, RCX
	Load(Param("hist"), hist)
	Load(Param("bins"), bins)
	Load(Param("feats"), feats)
	Load(Param("offs"), offs)
	Load(Param("rows"), rows)
	Load(Param("sums"), src)
	Load(Param("n"), n)
	f, m := NewParamAddr("f", 16), NewParamAddr("m", 40)
	a, b, j, t, ba, bb := RBX, RDX, R8, R11, RAX, R14
	// load sets the vector registers to row r's sums, widened; it
	// overwrites r.
	load := func(r Register, regs ...VecPhysical) {
		SHLQ(rowShift, r)
		ADDQ(src, r)
		for i, v := range regs {
			VCVTPS2PD(Mem{Base: r, Disp: 16 * i}, v)
		}
	}
	// add adds regs into the bin at p.
	add := func(p Register, regs ...VecPhysical) {
		for i, r := range regs {
			VADDPD(Mem{Base: p, Disp: 32 * i}, r, Y2)
			VMOVUPD(Y2, Mem{Base: p, Disp: 32 * i})
		}
	}
	rowA, rowB := []VecPhysical{Y0}, []VecPhysical{Y4}
	if s == 8 {
		rowA, rowB = []VecPhysical{Y0, Y1}, []VecPhysical{Y4, Y5}
	}

	CMPQ(n, Imm(2))
	JL(LabelRef(name + "_one"))
	Label(name + "_pair")
	MOVLQSX(Mem{Base: rows}, a)
	MOVQ(a, t)
	load(t, rowA...)
	IMULQ(f, a)
	ADDQ(bins, a)
	MOVLQSX(Mem{Base: rows, Disp: 4}, b)
	MOVQ(b, t)
	load(t, rowB...)
	IMULQ(f, b)
	ADDQ(bins, b)
	XORQ(j, j)
	Label(name + "_pairFeature")
	MOVQ(Mem{Base: feats, Index: j, Scale: 8}, t)
	MOVBQZX(Mem{Base: a, Index: t, Scale: 1}, ba)
	MOVBQZX(Mem{Base: b, Index: t, Scale: 1}, bb)
	MOVQ(Mem{Base: offs, Index: j, Scale: 8}, t)
	ADDQ(t, ba)
	ADDQ(t, bb)
	SHLQ(shift, ba)
	SHLQ(shift, bb)
	ADDQ(hist, ba)
	ADDQ(hist, bb)
	add(ba, rowA...)
	add(bb, rowB...)
	INCQ(j)
	CMPQ(j, m)
	JL(LabelRef(name + "_pairFeature"))
	ADDQ(Imm(8), rows)
	SUBQ(Imm(2), n)
	CMPQ(n, Imm(2))
	JGE(LabelRef(name + "_pair"))

	Label(name + "_one")
	TESTQ(n, n)
	JZ(LabelRef(name + "_done"))
	MOVLQSX(Mem{Base: rows}, a)
	MOVQ(a, t)
	load(t, rowA...)
	IMULQ(f, a)
	ADDQ(bins, a)
	XORQ(j, j)
	Label(name + "_oneFeature")
	MOVQ(Mem{Base: feats, Index: j, Scale: 8}, t)
	MOVBQZX(Mem{Base: a, Index: t, Scale: 1}, ba)
	ADDQ(Mem{Base: offs, Index: j, Scale: 8}, ba)
	SHLQ(shift, ba)
	ADDQ(hist, ba)
	add(ba, rowA...)
	INCQ(j)
	CMPQ(j, m)
	JL(LabelRef(name + "_oneFeature"))

	Label(name + "_done")
	VZEROUPPER()
	RET()
}
