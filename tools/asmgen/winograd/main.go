// Command winograd generates the amd64 assembly of backend/cpu's
// transforms of Winograd's F(2×2, 3×3) convolution (see
// backend/cpu/winograd.go), each over the tiles of a row of tiles, 16 to
// a register:
//
//	go run ./winograd -out ../../backend/cpu/winograd_amd64.s -stubs ../../backend/cpu/winograd_stub_amd64.go -pkg cpu
//
// A row of tiles of a plane is read with the even and odd elements of
// its rows deinterleaved, and the transforms of its tiles, one register
// per element of the transformed tile, are stored with tiles in lanes.
// The output transform interleaves them back into rows. Masks cover the
// last tiles, so a routine handles rows of any number of tiles.
package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// consts holds the permutation indices the routines use.
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
	for i := range uint32(8) {
		idx["iota8"] = append(idx["iota8"], i)
	}
	for _, name := range []string{"even", "odd", "lo", "hi", "iota8"} {
		m := GLOBL("wino_"+name, RODATA|NOPTR)
		for i, v := range idx[name] {
			DATA(4*i, U32(v))
		}
		consts[name] = m
	}
	consts["half"] = GLOBL("wino_half", RODATA|NOPTR)
	DATA(0, F32(0.5))
}

func main() {
	ConstraintExpr("!purego")
	constants()
	input()
	grad()
	output()
	inputChannels()
	gradChannels()
	weights()
	Generate()
}

var maskLabels int

// maskBits sets m to the lowest n bits, n ≤ 32 or all 32 if more.
func maskBits(n Register) Register {
	maskLabels++
	full := fmt.Sprintf("full%d", maskLabels)
	m := GP32()
	MOVL(U32(0xffffffff), m)
	CMPQ(n, Imm(32))
	JGE(LabelRef(full))
	MOVL(U32(1), m)
	MOVQ(n, RCX)
	SHLL(CL, m)
	DECL(m)
	Label(full)
	return m
}

// k returns a mask register holding the low 16 bits of m, or the high.
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

// split loads the 2·rem elements at at, rem ≤ 16, of the masks lo and hi,
// as their even and odd elements.
func split(at Mem, lo, hi Register, even, odd VecVirtual) (e, o VecVirtual) {
	x0, x1 := ZMM(), ZMM()
	VMOVUPS_Z(at, lo, x0)
	at.Disp += 64
	VMOVUPS_Z(at, hi, x1)
	e, o = ZMM(), ZMM()
	VMOVAPS(x0, e)
	VPERMT2PS(x1, even, e)
	VMOVAPS(x0, o)
	VPERMT2PS(x1, odd, o)
	return e, o
}

// store writes the 16 registers v, element ξ of each tile, to dst+ξ·xs
// with the mask m.
func store(v [16]VecVirtual, dst, xsBytes Register, m Register) {
	p := GP64()
	MOVQ(dst, p)
	for xi := range 16 {
		VMOVUPS(v[xi], m, Mem{Base: p})
		if xi < 15 {
			ADDQ(xsBytes, p)
		}
	}
}

// loop is the loop of an input transform over chunks of 16 tiles: its
// parameters, the chunk's destination and source, the tiles left and the
// permutation indices.
type loop struct {
	name                        string
	dst, src, pw, step, xs, rem Register
	even, odd                   VecVirtual
}

// begin declares the transform name, with the parameters (dst, xs, src)
// first and tiles last, those of params between, and starts its loop.
// The parameters pw and step, if any, count elements.
func begin(name, params, doc string) loop {
	TEXT(name, NOSPLIT, "func(dst *float32, xs int, src *float32, "+params+", tiles int)")
	Pragma("noescape")
	Doc(doc)
	var l loop
	l.name = name
	l.dst = Load(Param("dst"), GP64())
	l.xs = Load(Param("xs"), GP64())
	l.src = Load(Param("src"), GP64())
	l.pw = Load(Param("pw"), GP64())
	SHLQ(U8(2), l.pw)
	if params != "pw int" {
		l.step = Load(Param("step"), GP64())
		SHLQ(U8(2), l.step)
	}
	l.rem = Load(Param("tiles"), GP64())
	SHLQ(U8(2), l.xs)
	l.even, l.odd = ZMM(), ZMM()
	VMOVUPS(consts["even"], l.even)
	VMOVUPS(consts["odd"], l.odd)
	TESTQ(l.rem, l.rem)
	JLE(LabelRef(name + "_done"))
	Label(name + "_chunk")
	return l
}

// masks returns the mask of the chunk's tiles and the low and high
// halves of that of twice as many elements.
func (l loop) masks() (m16, lo, hi Register) {
	n := GP64()
	MOVQ(l.rem, n)
	ADDQ(n, n)
	m32 := maskBits(n)
	return k(maskBits(l.rem), false), k(m32, false), k(m32, true)
}

func (l loop) end() {
	ADDQ(U32(128), l.src)
	ADDQ(U32(64), l.dst)
	SUBQ(U32(16), l.rem)
	JG(LabelRef(l.name + "_chunk"))
	Label(l.name + "_done")
	VZEROUPPER()
	RET()
}

// bt returns the four transforms of the rows d0, d1, d2, d3 by B^T:
// d0 - d2, d1 + d2, d2 - d1, d1 - d3.
func bt(d0, d1, d2, d3 VecVirtual) [4]VecVirtual {
	e := [4]VecVirtual{ZMM(), ZMM(), ZMM(), ZMM()}
	VSUBPS(d2, d0, e[0])
	VADDPS(d2, d1, e[1])
	VSUBPS(d1, d2, e[2])
	VSUBPS(d3, d1, e[3])
	return e
}

func input() {
	name := "winoInputAVX512"
	l := begin(name, "pw int, step int", name+" stores at dst+ξ·xs the element ξ of the transform B^T d B of each of tiles tiles of a row of 4×4 tiles d two apart, with tiles in lanes, whose rows are at src, src+pw, src+step and src+pw+step.")
	m16, lo, hi := l.masks()
	var d [4][4]VecVirtual // d[i][j]: row i, column j of each tile
	both := GP64()
	LEAQ(Mem{Base: l.pw, Index: l.step, Scale: 1}, both)
	for i := range 4 {
		row := Mem{Base: l.src}
		switch i {
		case 1:
			row = Mem{Base: l.src, Index: l.pw, Scale: 1}
		case 2:
			row = Mem{Base: l.src, Index: l.step, Scale: 1}
		case 3:
			row = Mem{Base: l.src, Index: both, Scale: 1}
		}
		d[i][0], d[i][1] = split(row, lo, hi, l.even, l.odd)
		row.Disp += 8
		d[i][2], d[i][3] = split(row, lo, hi, l.even, l.odd)
	}
	var e [4][4]VecVirtual // B^T d, by columns
	for j := range 4 {
		col := bt(d[0][j], d[1][j], d[2][j], d[3][j])
		for i := range 4 {
			e[i][j] = col[i]
		}
	}
	var v [16]VecVirtual
	for i := range 4 {
		r := bt(e[i][0], e[i][1], e[i][2], e[i][3])
		copy(v[4*i:], r[:])
	}
	store(v, l.dst, l.xs, m16)
	l.end()
}

// ay returns the four transforms of the rows y0, y1 by A: y0, y0 + y1,
// y0 - y1, -y1.
func ay(y0, y1, zero VecVirtual) [4]VecVirtual {
	e := [4]VecVirtual{y0, ZMM(), ZMM(), ZMM()}
	VADDPS(y1, y0, e[1])
	VSUBPS(y1, y0, e[2])
	VSUBPS(y1, zero, e[3])
	return e
}

func grad() {
	name := "winoGradAVX512"
	l := begin(name, "pw int", name+" stores at dst+ξ·xs the element ξ of the transform A y A^T of each of tiles tiles of a row of 2×2 tiles y, with tiles in lanes, of the two rows pw apart at src.")
	m16, lo, hi := l.masks()
	zero := ZMM()
	VXORPS(zero, zero, zero)
	var y [2][2]VecVirtual
	y[0][0], y[0][1] = split(Mem{Base: l.src}, lo, hi, l.even, l.odd)
	y[1][0], y[1][1] = split(Mem{Base: l.src, Index: l.pw, Scale: 1}, lo, hi, l.even, l.odd)
	var e [4][2]VecVirtual
	for j := range 2 {
		col := ay(y[0][j], y[1][j], zero)
		for i := range 4 {
			e[i][j] = col[i]
		}
	}
	var v [16]VecVirtual
	for i := range 4 {
		r := ay(e[i][0], e[i][1], zero)
		copy(v[4*i:], r[:])
	}
	store(v, l.dst, l.xs, m16)
	l.end()
}

func output() {
	name := "winoOutputAVX512"
	TEXT(name, NOSPLIT, "func(dst *float32, ow int, src *float32, xs int, tiles int, rows int)")
	Pragma("noescape")
	Doc(name + " writes the 2×2 output tiles A^T m A of tiles tiles of a row, from the elements ξ of each at src+ξ·xs, with tiles in lanes, to the first rows rows, 1 or 2, of the rows ow apart at dst, stopping each row after ow elements.")
	dst := Load(Param("dst"), GP64())
	ow := Load(Param("ow"), GP64())
	src := Load(Param("src"), GP64())
	xs := Load(Param("xs"), GP64())
	rem := Load(Param("tiles"), GP64())
	rows := Load(Param("rows"), GP64())
	owBytes := GP64()
	MOVQ(ow, owBytes)
	SHLQ(U8(2), owBytes)
	SHLQ(U8(2), xs)
	lo, hi := ZMM(), ZMM()
	VMOVUPS(consts["lo"], lo)
	VMOVUPS(consts["hi"], hi)
	TESTQ(rem, rem)
	JLE(LabelRef(name + "_done"))
	Label(name + "_chunk")
	m16 := k(maskBits(rem), false)
	mOut := maskBits(ow) // the chunk's elements of each row
	var m [16]VecVirtual
	p := GP64()
	MOVQ(src, p)
	for xi := range 16 {
		m[xi] = ZMM()
		VMOVUPS_Z(Mem{Base: p}, m16, m[xi])
		if xi < 15 {
			ADDQ(xs, p)
		}
	}
	// A^T m: rows m0 + m1 + m2 and m1 - m2 - m3, by columns; then the same
	// over the columns of each.
	at := func(a0, a1, a2, a3 VecVirtual) (VecVirtual, VecVirtual) {
		s, t := ZMM(), ZMM()
		VADDPS(a1, a0, s)
		VADDPS(a2, s, s)
		VSUBPS(a2, a1, t)
		VSUBPS(a3, t, t)
		return s, t
	}
	var e [2][4]VecVirtual
	for j := range 4 {
		e[0][j], e[1][j] = at(m[j], m[4+j], m[8+j], m[12+j])
	}
	row := func(i int, at Mem) {
		y0, y1 := at4(e[i])
		l, h := ZMM(), ZMM()
		VMOVAPS(y0, l)
		VPERMT2PS(y1, lo, l)
		VMOVAPS(y0, h)
		VPERMT2PS(y1, hi, h)
		VMOVUPS(l, k(mOut, false), at)
		at.Disp += 64
		VMOVUPS(h, k(mOut, true), at)
	}
	row(0, Mem{Base: dst})
	CMPQ(rows, Imm(2))
	JL(LabelRef(name + "_next"))
	row(1, Mem{Base: dst, Index: owBytes, Scale: 1})
	Label(name + "_next")
	ADDQ(U32(64), src)
	ADDQ(U32(128), dst)
	SUBQ(U32(32), ow)
	SUBQ(U32(16), rem)
	JG(LabelRef(name + "_chunk"))
	Label(name + "_done")
	VZEROUPPER()
	RET()
}

// at4 returns the transforms by A^T of the columns of a row: a0 + a1 + a2
// and a1 - a2 - a3.
func at4(a [4]VecVirtual) (VecVirtual, VecVirtual) {
	s, t := ZMM(), ZMM()
	VADDPS(a[1], a[0], s)
	VADDPS(a[2], s, s)
	VSUBPS(a[2], a[1], t)
	VSUBPS(a[3], t, t)
	return s, t
}

// The transforms of the weight gradient read planes with channels
// innermost, 16 channels to a register, and store the transforms of a
// row of tiles one tile after another, step elements apart, as the
// micro-kernel's panels lay them out.

// channels declares a transform over the tiles of a row with channels
// in lanes, whose element (i, j) of tile q is at src+i·rs+(2q+j)·cs, and
// starts its loop over the tiles.
type channels struct {
	name                       string
	dst, src, xs, rs, cs, step Register
	rem                        Register
	extra                      []Register
}

func beginChannels(name, params, doc string, extra ...string) channels {
	TEXT(name, NOSPLIT, "func(dst *float32, xs int, src *float32, rs int, cs int, tiles int, step int"+params+")")
	Pragma("noescape")
	Doc(doc)
	var l channels
	l.name = name
	l.dst = Load(Param("dst"), GP64())
	l.xs = Load(Param("xs"), GP64())
	l.src = Load(Param("src"), GP64())
	l.rs = Load(Param("rs"), GP64())
	l.cs = Load(Param("cs"), GP64())
	l.rem = Load(Param("tiles"), GP64())
	l.step = Load(Param("step"), GP64())
	for _, p := range extra {
		r := Load(Param(p), GP64())
		SHLQ(U8(2), r)
		l.extra = append(l.extra, r)
	}
	for _, r := range []Register{l.xs, l.rs, l.cs, l.step} {
		SHLQ(U8(2), r)
	}
	TESTQ(l.rem, l.rem)
	JLE(LabelRef(name + "_done"))
	Label(name + "_tile")
	return l
}

func (l channels) end() {
	ADDQ(l.cs, l.src)
	ADDQ(l.cs, l.src)
	ADDQ(l.step, l.dst)
	DECQ(l.rem)
	JNZ(LabelRef(l.name + "_tile"))
	Label(l.name + "_done")
	VZEROUPPER()
	RET()
}

// rows returns pointers to the first n rows of the tile.
func (l channels) rows(n int) []Register {
	r := []Register{l.src}
	for i := 1; i < n; i++ {
		p := GP64()
		LEAQ(Mem{Base: r[i-1], Index: l.rs, Scale: 1}, p)
		r = append(r, p)
	}
	return r
}

// load returns the register of the 16 channels of element j of the row
// at row, cs3 holding 3·cs.
func (l channels) load(row Register, j int, cs3 Register) VecVirtual {
	at := Mem{Base: row}
	switch j {
	case 1, 2:
		at = Mem{Base: row, Index: l.cs, Scale: uint8(j)}
	case 3:
		at = Mem{Base: row, Index: cs3, Scale: 1}
	}
	v := ZMM()
	VMOVUPS(at, v)
	return v
}

func inputChannels() {
	name := "winoInputChannelsAVX512"
	l := beginChannels(name, "", name+" stores at dst+ξ·xs, for each of tiles 4×4 tiles d, 16 channels in lanes, element (i, j) of tile q at src+i·rs+(2q+j)·cs, element ξ of the transform B^T d B, dst moving step on from tile to tile.")
	cs3 := GP64()
	LEAQ(Mem{Base: l.cs, Index: l.cs, Scale: 2}, cs3)
	r := l.rows(4)
	var e [4][4]VecVirtual // B^T d, by columns
	for j := range 4 {
		col := bt(l.load(r[0], j, cs3), l.load(r[1], j, cs3), l.load(r[2], j, cs3), l.load(r[3], j, cs3))
		for i := range 4 {
			e[i][j] = col[i]
		}
	}
	p := GP64()
	MOVQ(l.dst, p)
	for i := range 4 {
		r := bt(e[i][0], e[i][1], e[i][2], e[i][3])
		for j := range 4 {
			VMOVUPS(r[j], Mem{Base: p})
			if 4*i+j < 15 {
				ADDQ(l.xs, p)
			}
		}
	}
	l.end()
}

func gradChannels() {
	name := "winoGradChannelsAVX512"
	l := beginChannels(name, ", half int", name+" stores at dst+ξ·xs, for each of tiles 2×2 tiles y, 16 channels in lanes, element (i, j) of tile q at src+i·rs+(2q+j)·cs, element ξ of the transform A y A^T, the first 8 channels there and the others half on, dst moving step on from tile to tile.", "half")
	half := l.extra[0]
	zero := ZMM()
	VXORPS(zero, zero, zero)
	r := l.rows(2)
	var e [4][2]VecVirtual
	for j := range 2 {
		col := ay(l.load(r[0], j, nil), l.load(r[1], j, nil), zero)
		for i := range 4 {
			e[i][j] = col[i]
		}
	}
	p := GP64()
	MOVQ(l.dst, p)
	for i := range 4 {
		r := ay(e[i][0], e[i][1], zero)
		for j := range 4 {
			VMOVUPS(r[j].AsY(), Mem{Base: p})
			VEXTRACTF32X8(U8(1), r[j], Mem{Base: p, Index: half, Scale: 1})
			if 4*i+j < 15 {
				ADDQ(l.xs, p)
			}
		}
	}
	l.end()
}

// weights emits the weight transform of a panel of 8 output channels,
// one input channel at a time, with output channels in lanes: the 9
// weights of each window are gathered from their strides, and the 16
// elements of the transformed window stored 8 channels at a time.
func weights() {
	name := "winoWeightsAVX512"
	TEXT(name, NOSPLIT, "func(dst *float32, xs int, src *float32, so int, si int, sh int, sw int, cin int)")
	Pragma("noescape")
	Doc(name + " stores at dst+ξ·xs+8i+o, for the 8 output channels o and cin input channels i whose 3×3 windows g are at src+o·so+i·si, element (kh, kw) at kh·sh+kw·sw on, element ξ of the transform G g G^T of each window.")
	dst := Load(Param("dst"), GP64())
	xs := Load(Param("xs"), GP64())
	SHLQ(U8(2), xs)
	src := Load(Param("src"), GP64())
	so := Load(Param("so"), GP64())
	si := Load(Param("si"), GP64())
	SHLQ(U8(2), si)
	sh := Load(Param("sh"), GP64())
	SHLQ(U8(2), sh)
	sw := Load(Param("sw"), GP64())
	SHLQ(U8(2), sw)
	cin := Load(Param("cin"), GP64())
	// The lanes' offsets, o·so, and the windows' elements as 9 pointers.
	idx, sov := YMM(), YMM()
	VMOVUPS(consts["iota8"], idx)
	VPBROADCASTD(so.(GPVirtual).As32(), sov)
	VPMULLD(sov, idx, idx)
	half := YMM()
	VBROADCASTSS(consts["half"], half)
	var at [3][3]Register
	for kh := range 3 {
		for kw := range 3 {
			if kh == 0 && kw == 0 {
				at[kh][kw] = src
				continue
			}
			p := GP64()
			MOVQ(src, p)
			for range kh {
				ADDQ(sh, p)
			}
			for range kw {
				ADDQ(sw, p)
			}
			at[kh][kw] = p
		}
	}
	TESTQ(cin, cin)
	JLE(LabelRef(name + "_done"))
	Label(name + "_channel")
	var g [3][3]VecVirtual
	for kh := range 3 {
		for kw := range 3 {
			g[kh][kw] = YMM()
			KXNORW(K1, K1, K1) // a gather consumes its mask
			VGATHERDPS(Mem{Base: at[kh][kw], Index: idx, Scale: 4}, K1, g[kh][kw])
		}
	}
	// G g: rows g0, (g0+g1+g2)/2, (g0-g1+g2)/2, g2, by columns kw, the
	// sums associated as the Go transform's are.
	var t [4][3]VecVirtual
	for kw := range 3 {
		t[0][kw], t[1][kw], t[2][kw], t[3][kw] = gRows(g[0][kw], g[1][kw], g[2][kw], half)
	}
	// Then the same over the columns of each row, stored as ξ = 4r+c.
	p := GP64()
	MOVQ(dst, p)
	for r := range 4 {
		v0, v1, v2, v3 := gRows(t[r][0], t[r][1], t[r][2], half)
		for _, v := range []VecVirtual{v0, v1, v2, v3} {
			VMOVUPS(v, Mem{Base: p})
			ADDQ(xs, p)
		}
	}
	ADDQ(U32(32), dst)
	for kh := range 3 {
		for kw := range 3 {
			ADDQ(si, at[kh][kw])
		}
	}
	DECQ(cin)
	JNZ(LabelRef(name + "_channel"))
	Label(name + "_done")
	VZEROUPPER()
	RET()
}

// gRows returns the transform G of the rows g0, g1, g2: g0, (g0+g1+g2)/2,
// (g0-g1+g2)/2 and g2, summed left to right.
func gRows(g0, g1, g2, half VecVirtual) (VecVirtual, VecVirtual, VecVirtual, VecVirtual) {
	a, b := YMM(), YMM()
	VADDPS(g1, g0, a)
	VSUBPS(g1, g0, b)
	VADDPS(g2, a, a)
	VMULPS(half, a, a)
	VADDPS(g2, b, b)
	VMULPS(half, b, b)
	return g0, a, b, g2
}
