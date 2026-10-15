// Command amd64 generates the amd64 assembly of backend/cpu: the GEMM
// micro-kernels, the transpose that packs their operands, and the CPUID
// helpers used to choose between them.
//
//	go run ./amd64 -out ../../backend/cpu/gemm_amd64.s -stubs ../../backend/cpu/gemm_stub_amd64.go -pkg cpu
package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

func main() {
	ConstraintExpr("!purego")

	tile("tileAVX512", 8, 2, avx512)
	tile("tileAVX2", 6, 2, avx2)
	transpose8()

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

// isa describes one vector extension.
type isa struct {
	lanes int                     // float32 elements per register
	reg   func(i int) VecPhysical // vector register i
	zero  func(r VecPhysical)
}

var avx512 = isa{
	lanes: 16,
	reg:   func(i int) VecPhysical { return zmm[i] },
	zero:  func(r VecPhysical) { VPXORD(r, r, r) },
}

var avx2 = isa{
	lanes: 8,
	reg:   func(i int) VecPhysical { return ymm[i] },
	zero:  func(r VecPhysical) { VXORPS(r, r, r) },
}

var zmm = []VecPhysical{
	Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15,
	Z16, Z17, Z18, Z19, Z20, Z21, Z22, Z23, Z24, Z25, Z26, Z27, Z28, Z29, Z30, Z31,
}

var ymm = []VecPhysical{
	Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14, Y15,
}

var xmm = []VecPhysical{
	X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15,
}

// tile emits a micro-kernel computing an mr×(nv*lanes) tile with the
// signature and contract of gemmKernel.tile. Each step of the k loop
// loads nv vectors of the packed B panel, then for each of the mr rows
// broadcasts one element of the packed A panel and multiplies it into the
// row's nv accumulators. Accumulators, B and the broadcast use
// mr*nv + nv + 1 registers.
func tile(name string, mr, nv int, v isa) {
	nr := nv * v.lanes
	width := 4 * v.lanes // bytes per vector
	TEXT(name, NOSPLIT, "func(kc int, a, b, c *float32, ldc int, accumulate bool)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s computes a %d×%d tile; see gemmKernel.tile.", name, mr, nr))

	acc := func(i, j int) VecPhysical { return v.reg(i*nv + j) }
	bv := func(j int) VecPhysical { return v.reg(mr*nv + j) }
	bcast := v.reg(mr*nv + nv)

	kc := Load(Param("kc"), GP64())
	a := Load(Param("a"), GP64())
	b := Load(Param("b"), GP64())
	c := Load(Param("c"), GP64())
	ldc := Load(Param("ldc"), GP64())
	SHLQ(U8(2), ldc)

	// Bring the output tile into cache while the products accumulate.
	row := GP64()
	MOVQ(c, row)
	for i := range mr {
		for off := 0; off < 4*nr; off += 64 {
			PREFETCHT0(Mem{Base: row, Disp: off})
		}
		if i < mr-1 {
			ADDQ(ldc, row)
		}
	}

	for i := range mr {
		for j := range nv {
			v.zero(acc(i, j))
		}
	}
	TESTQ(kc, kc)
	JZ(LabelRef(name + "_store"))

	Label(name + "_loop")
	for j := range nv {
		VMOVUPS(Mem{Base: b, Disp: j * width}, bv(j))
	}
	for i := range mr {
		VBROADCASTSS(Mem{Base: a, Disp: 4 * i}, bcast)
		for j := range nv {
			VFMADD231PS(bv(j), bcast, acc(i, j))
		}
	}
	ADDQ(U32(4*mr), a)
	ADDQ(U32(4*nr), b)
	DECQ(kc)
	JNZ(LabelRef(name + "_loop"))

	Label(name + "_store")
	accumulate := Load(Param("accumulate"), GP8())
	TESTB(accumulate, accumulate)
	JZ(LabelRef(name + "_overwrite"))
	for i := range mr {
		for j := range nv {
			VADDPS(Mem{Base: c, Disp: j * width}, acc(i, j), acc(i, j))
			VMOVUPS(acc(i, j), Mem{Base: c, Disp: j * width})
		}
		if i < mr-1 {
			ADDQ(ldc, c)
		}
	}
	VZEROUPPER()
	RET()

	Label(name + "_overwrite")
	for i := range mr {
		for j := range nv {
			VMOVUPS(acc(i, j), Mem{Base: c, Disp: j * width})
		}
		if i < mr-1 {
			ADDQ(ldc, c)
		}
	}
	VZEROUPPER()
	RET()
}

// transpose8 emits a routine that transposes 8×8 blocks with AVX: for
// each block q < blocks it reads rows r < 8, columns 8q..8q+7 of src and
// writes them as columns r of rows 8q..8q+7 of dst. Packing uses it to
// turn the rows of a matrix into the column-by-column layout of a panel.
func transpose8() {
	TEXT("transpose8", NOSPLIT, "func(dst *float32, ldd int, src *float32, lds int, blocks int)")
	Pragma("noescape")
	Doc("transpose8 sets dst[(8q+c)*ldd+r] = src[r*lds+8q+c] for r, c < 8 and q < blocks.")

	dst := Load(Param("dst"), GP64())
	ldd := Load(Param("ldd"), GP64())
	src := Load(Param("src"), GP64())
	lds := Load(Param("lds"), GP64())
	blocks := Load(Param("blocks"), GP64())
	SHLQ(U8(2), ldd)
	SHLQ(U8(2), lds)

	// Rows 0-3 are addressed from src and rows 4-7 from src4, each with an
	// index of 0, 1, 2 or 3 rows; likewise for dst.
	lds3, src4 := GP64(), GP64()
	LEAQ(Mem{Base: lds, Index: lds, Scale: 2}, lds3)
	LEAQ(Mem{Base: src, Index: lds, Scale: 4}, src4)
	ldd3, dst4 := GP64(), GP64()
	LEAQ(Mem{Base: ldd, Index: ldd, Scale: 2}, ldd3)
	LEAQ(Mem{Base: dst, Index: ldd, Scale: 4}, dst4)
	row := func(base, ld, ld3 Register, i int) Mem {
		switch i {
		case 0:
			return Mem{Base: base}
		case 1:
			return Mem{Base: base, Index: ld, Scale: 1}
		case 2:
			return Mem{Base: base, Index: ld, Scale: 2}
		default:
			return Mem{Base: base, Index: ld3, Scale: 1}
		}
	}
	srcRow := func(r int) Mem {
		if r < 4 {
			return row(src, lds, lds3, r)
		}
		return row(src4, lds, lds3, r-4)
	}
	dstRow := func(r int) Mem {
		if r < 4 {
			return row(dst, ldd, ldd3, r)
		}
		return row(dst4, ldd, ldd3, r-4)
	}
	step := GP64()
	MOVQ(ldd, step)
	SHLQ(U8(3), step) // 8 rows of dst

	TESTQ(blocks, blocks)
	JZ(LabelRef("transpose8_done"))
	Label("transpose8_loop")
	// Register h*4+i holds columns 4h..4h+3 of row i in its low half and
	// of row i+4 in its high half, so transposing each half as a 4×4
	// matrix leaves whole columns of the block in the registers.
	r, t := ymm[:8], ymm[8:]
	for h := range 2 {
		for i := range 4 {
			VMOVUPS(srcRow(i).Offset(16*h), xmm[4*h+i])
			VINSERTF128(U8(1), srcRow(i+4).Offset(16*h), r[4*h+i], r[4*h+i])
		}
	}
	for h := 0; h < 8; h += 4 {
		VUNPCKLPS(r[h+1], r[h], t[h])
		VUNPCKHPS(r[h+1], r[h], t[h+1])
		VUNPCKLPS(r[h+3], r[h+2], t[h+2])
		VUNPCKHPS(r[h+3], r[h+2], t[h+3])
		VSHUFPS(U8(0x44), t[h+2], t[h], r[h])
		VSHUFPS(U8(0xee), t[h+2], t[h], r[h+1])
		VSHUFPS(U8(0x44), t[h+3], t[h+1], r[h+2])
		VSHUFPS(U8(0xee), t[h+3], t[h+1], r[h+3])
	}
	for i := range 8 {
		VMOVUPS(r[i], dstRow(i))
	}
	ADDQ(U32(32), src)
	ADDQ(U32(32), src4)
	ADDQ(step, dst)
	ADDQ(step, dst4)
	DECQ(blocks)
	JNZ(LabelRef("transpose8_loop"))
	Label("transpose8_done")
	VZEROUPPER()
	RET()
}
