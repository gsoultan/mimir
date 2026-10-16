// Command linalg generates the amd64 assembly of the vector operations of
// backend/cpu's matrix factorizations that are not products of matrices:
// adding a multiple of one row to another.
//
//	go run ./linalg -out ../../backend/cpu/linalg_amd64.s -stubs ../../backend/cpu/linalg_stub_amd64.go -pkg cpu
//
// Each routine takes rows of a whole number of vectors; the Go code that
// calls it handles the rest of a row.
package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// isa describes one vector extension.
type isa struct {
	name  string
	lanes int // float32 elements per register
	vec   func() VecVirtual
}

var (
	avx512 = isa{name: "AVX512", lanes: 16, vec: func() VecVirtual { return ZMM() }}
	avx2   = isa{name: "AVX2", lanes: 8, vec: func() VecVirtual { return YMM() }}
)

func main() {
	ConstraintExpr("!purego")
	for _, v := range []isa{avx512, avx2} {
		axpy(v)
	}
	Generate()
}

func axpy(v isa) {
	name := "axpy" + v.name
	TEXT(name, NOSPLIT, "func(a float32, x *float32, y *float32, n int)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s adds a times each of the n elements at x, a multiple of %d, to those at y.", name, v.lanes))
	ax := XMM()
	Load(Param("a"), ax)
	a := v.vec()
	VBROADCASTSS(ax, a)
	px := Load(Param("x"), GP64())
	py := Load(Param("y"), GP64())
	n := Load(Param("n"), GP64())
	// Four vectors at a time keep several loads in flight.
	step := 4 * v.lanes
	CMPQ(n, U32(step))
	JL(LabelRef(name + "_tail"))
	Label(name + "_wide")
	for i := range 4 {
		t := v.vec()
		VMOVUPS(Mem{Base: py, Disp: 4 * v.lanes * i}, t)
		VFMADD231PS(Mem{Base: px, Disp: 4 * v.lanes * i}, a, t)
		VMOVUPS(t, Mem{Base: py, Disp: 4 * v.lanes * i})
	}
	ADDQ(U32(4*step), px)
	ADDQ(U32(4*step), py)
	SUBQ(U32(step), n)
	CMPQ(n, U32(step))
	JGE(LabelRef(name + "_wide"))
	Label(name + "_tail")
	TESTQ(n, n)
	JZ(LabelRef(name + "_done"))
	Label(name + "_loop")
	t := v.vec()
	VMOVUPS(Mem{Base: py}, t)
	VFMADD231PS(Mem{Base: px}, a, t)
	VMOVUPS(t, Mem{Base: py})
	ADDQ(U32(4*v.lanes), px)
	ADDQ(U32(4*v.lanes), py)
	SUBQ(U32(v.lanes), n)
	JNZ(LabelRef(name + "_loop"))
	Label(name + "_done")
	VZEROUPPER()
	RET()
}
