// Command qamd64 generates the amd64 assembly of backend/cpu's products by
// quantized weights: the AVX2 dot products of Q8_0, Q4_K and Q6_K weights
// with int8 activations, for one to four rows of activations, and the
// tiles of Q4_K and Q6_K weights, two rows of them by sixteen rows of
// activations, that products of many rows of activations run in.
//
//	go run ./qamd64 -out ../../backend/cpu/qdot_amd64.s -stubs ../../backend/cpu/qdot_stub_amd64.go -pkg cpu
//
// Each kernel has the contract of backend/cpu's qdot or qtile and
// computes what the portable kernel of its dtype computes, bit for bit:
// the integer dot products, which are exact in any order, then the
// float32 operations that apply the scales, in the order the portable
// kernel applies them, one rounding each, with no fused multiply-add.
package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

const maxRows = 4

var rowsOf = [...]string{1: "one row of activations", 2: "two rows of activations", 3: "three rows of activations", 4: "four rows of activations"}

func main() {
	ConstraintExpr("!purego")
	constants()
	for rows := 1; rows <= maxRows; rows++ {
		q8_0(rows)
	}
	for rows := 1; rows <= maxRows; rows++ {
		q4K(rows)
	}
	for rows := 1; rows <= maxRows; rows++ {
		q6K(rows)
	}
	qtile(false)
	qtile(true)
	Generate()
}

var (
	nibbles  Mem // 0x0f in every byte
	high2    Mem // 0x30 in every byte
	q4kScale Mem // VPSHUFB masks broadcasting word s of each lane, for s < 8
	q6kScale Mem // VPSHUFB masks broadcasting word 2k of the low lane and 2k+1 of the high
)

func constants() {
	nibbles = GLOBL("qNibbles", RODATA|NOPTR)
	for i := 0; i < 32; i += 8 {
		DATA(i, U64(0x0f0f0f0f0f0f0f0f))
	}
	high2 = GLOBL("qHigh2", RODATA|NOPTR)
	for i := 0; i < 32; i += 8 {
		DATA(i, U64(0x3030303030303030))
	}
	q4kScale = GLOBL("q4kScaleShuffle", RODATA|NOPTR)
	for s := range 8 {
		for i := range 16 {
			DATA(32*s+2*i, U16(2*s|(2*s+1)<<8))
		}
	}
	q6kScale = GLOBL("q6kScaleShuffle", RODATA|NOPTR)
	for k := range 4 {
		for i := range 16 {
			g := 2 * k
			if i >= 8 {
				g++
			}
			DATA(32*k+2*i, U16(2*g|(2*g+1)<<8))
		}
	}
}

// prologue loads the arguments of a qdot: the count of blocks, the
// weights and a pointer to each of rows rows of activations.
func prologue(name string, rows int, doc string) (nb, w GPVirtual, x []GPVirtual) {
	TEXT(name, NOSPLIT, "func(nb int, w, x *byte, xstride int, out *float32, ldo int)")
	Pragma("noescape")
	Doc(fmt.Sprintf("%s %s; see qdot.", name, doc))
	nb, w = GP64(), GP64()
	Load(Param("nb"), nb)
	Load(Param("w"), w)
	x = []GPVirtual{GP64()}
	Load(Param("x"), x[0])
	if rows > 1 {
		stride := GP64()
		Load(Param("xstride"), stride)
		for r := 1; r < rows; r++ {
			x = append(x, GP64())
			LEAQ(Mem{Base: x[r-1], Index: stride, Scale: 1}, x[r])
		}
	}
	return nb, w, x
}

// store writes the low float32 of each row's result to out, ldo apart.
func store(results []VecVirtual) {
	out := Load(Param("out"), GP64())
	ldo := Load(Param("ldo"), GP64())
	SHLQ(U8(2), ldo)
	for r, v := range results {
		VMOVSS(v, Mem{Base: out})
		if r < len(results)-1 {
			ADDQ(ldo, out)
		}
	}
	VZEROUPPER()
	RET()
}

// half converts the float16 at m to a float32 in the low lane of a new
// register.
func half(m Mem) VecVirtual {
	h, v := GP32(), XMM()
	MOVWLZX(m, h)
	VMOVD(h, v)
	VCVTPH2PS(v, v)
	return v
}

// q8_0 emits the kernel of Q8_0 weights for rows rows of activations,
// blocks of a float32 scale and 32 int8. The products of a block sum in
// four int32 lanes, those of values 4l..4l+3 and 16+4l..16+4l+3 in lane
// l, as qdotQ8_0Go sums them.
func q8_0(rows int) {
	name := fmt.Sprintf("qdotQ8_0AVX2x%d", rows)
	nb, w, x := prologue(name, rows, "multiplies Q8_0 weights by "+rowsOf[rows])
	acc := make([]VecVirtual, rows)
	for r := range acc {
		acc[r] = XMM()
		VXORPS(acc[r], acc[r], acc[r])
	}
	ones := YMM()
	VPCMPEQW(ones, ones, ones)
	VPSRLW(U8(15), ones, ones)
	TESTQ(nb, nb)
	JZ(LabelRef(name + "_done"))

	Label(name + "_loop")
	// |w| as unsigned bytes, and each row's activations with w's signs:
	// VPMADDUBSW multiplies the two, w·x, in pairs.
	q, aq := YMM(), YMM()
	VMOVDQU(Mem{Base: w, Disp: 2}, q)
	VPSIGNB(q, q, aq)
	dw := half(Mem{Base: w})
	for r := range rows {
		p := YMM()
		VMOVDQU(Mem{Base: x[r], Disp: 4}, p)
		VPSIGNB(q, p, p)
		VPMADDUBSW(p, aq, p)
		VPMADDWD(ones, p, p)
		hi, s := XMM(), XMM()
		VEXTRACTI128(U8(1), p, hi)
		VPADDD(hi, p.AsX(), s)
		VCVTDQ2PS(s, s)
		dd := XMM()
		VMULSS(Mem{Base: x[r]}, dw, dd)
		VBROADCASTSS(dd, dd)
		VMULPS(dd, s, s)
		VADDPS(s, acc[r], acc[r])
		ADDQ(U32(36), x[r])
	}
	ADDQ(U32(34), w)
	DECQ(nb)
	JNZ(LabelRef(name + "_loop"))

	// (a0+a2) + (a1+a3)
	Label(name + "_done")
	for r := range rows {
		t, u := XMM(), XMM()
		VMOVHLPS(acc[r], acc[r], t)
		VADDPS(t, acc[r], t)
		VMOVSHDUP(t, u)
		VADDSS(u, t, acc[r])
	}
	store(acc)
}

// q4K emits the kernel of Q4_K weights for rows rows of activations,
// super-blocks of a float32 scale, 256 int8 and the int16 sums of each
// 16. A row's accumulator holds the float32 sums of qdotQ4KGo, acc in
// lane 0 and accm in lane 1.
func q4K(rows int) {
	name := fmt.Sprintf("qdotQ4KAVX2x%d", rows)
	nb, w, x := prologue(name, rows, "multiplies Q4_K weights by "+rowsOf[rows])
	acc := make([]VecVirtual, rows)
	for r := range acc {
		acc[r] = XMM()
		VXORPS(acc[r], acc[r], acc[r])
	}
	TESTQ(nb, nb)
	JZ(LabelRef(name + "_done"))

	Label(name + "_loop")
	// Unpack the eight 6-bit scales and mins as ggml's dot product does:
	// with u0, u1, u2 the three words of the 12 bytes,
	//
	//	scales = u0 & 0x3f3f3f3f | (u2 & 0x0f0f0f0f | (u0>>6 & 0x03030303)<<4) << 32
	//	mins   = u1 & 0x3f3f3f3f | (u2>>4 & 0x0f0f0f0f | (u1>>6 & 0x03030303)<<4) << 32
	u0, u1, u2, t := GP64(), GP64(), GP64(), GP64()
	MOVL(Mem{Base: w, Disp: 4}, u0.As32())
	MOVL(Mem{Base: w, Disp: 8}, u1.As32())
	MOVL(Mem{Base: w, Disp: 12}, u2.As32())
	sc, mn := GP64(), GP64()
	MOVL(u0.As32(), t.As32())
	SHRL(U8(6), t.As32())
	ANDL(U32(0x03030303), t.As32())
	SHLL(U8(4), t.As32())
	MOVL(u2.As32(), sc.As32())
	ANDL(U32(0x0f0f0f0f), sc.As32())
	ORL(t.As32(), sc.As32())
	SHLQ(U8(32), sc)
	ANDL(U32(0x3f3f3f3f), u0.As32())
	ORQ(u0, sc)
	MOVL(u1.As32(), t.As32())
	SHRL(U8(6), t.As32())
	ANDL(U32(0x03030303), t.As32())
	SHLL(U8(4), t.As32())
	MOVL(u2.As32(), mn.As32())
	SHRL(U8(4), mn.As32())
	ANDL(U32(0x0f0f0f0f), mn.As32())
	ORL(t.As32(), mn.As32())
	SHLQ(U8(32), mn)
	ANDL(U32(0x3f3f3f3f), u1.As32())
	ORQ(u1, mn)

	// The scales as int16 in both lanes, for VPSHUFB to broadcast.
	scales := YMM()
	VMOVQ(sc, scales.AsX())
	VPMOVZXBW(scales.AsX(), scales.AsX())
	VINSERTI128(U8(1), scales.AsX(), scales, scales)

	isum := make([]VecVirtual, rows)
	for r := range isum {
		isum[r] = YMM()
		VPXOR(isum[r], isum[r], isum[r])
	}
	for j := range 4 {
		q, lo, hi := YMM(), YMM(), YMM()
		VMOVDQU(Mem{Base: w, Disp: 16 + 32*j}, q)
		VPAND(nibbles, q, lo)
		VPSRLW(U8(4), q, hi)
		VPAND(nibbles, hi, hi)
		sl, sh := YMM(), YMM()
		VPSHUFB(q4kScale.Offset(64*j), scales, sl)
		VPSHUFB(q4kScale.Offset(64*j+32), scales, sh)
		for r := range rows {
			p := YMM()
			VPMADDUBSW(Mem{Base: x[r], Disp: 4 + 64*j}, lo, p)
			VPMADDWD(sl, p, p)
			VPADDD(p, isum[r], isum[r])
			VPMADDUBSW(Mem{Base: x[r], Disp: 4 + 64*j + 32}, hi, p)
			VPMADDWD(sh, p, p)
			VPADDD(p, isum[r], isum[r])
		}
	}

	// The mins as int16, each twice, for the pairs of sums of 16.
	mins := YMM()
	VMOVQ(mn, mins.AsX())
	VPUNPCKLBW(mins.AsX(), mins.AsX(), mins.AsX())
	VPMOVZXBW(mins.AsX(), mins)
	d := XMM() // d and dmin, as two float32
	VMOVD(Mem{Base: w}, d)
	VCVTPH2PS(d, d)
	for r := range rows {
		m, h := YMM(), YMM()
		VPMADDWD(Mem{Base: x[r], Disp: 260}, mins, m)
		// [Σ isum, Σ m, ...]
		VPHADDD(m, isum[r], h)
		hx := XMM()
		VEXTRACTI128(U8(1), h, hx)
		VPADDD(hx, h.AsX(), hx)
		VPHADDD(hx, hx, hx)
		VCVTDQ2PS(hx, hx)
		dd := XMM()
		VBROADCASTSS(Mem{Base: x[r]}, dd)
		VMULPS(d, dd, dd)
		VMULPS(dd, hx, hx)
		VADDPS(hx, acc[r], acc[r])
		ADDQ(U32(292), x[r])
	}
	ADDQ(U32(144), w)
	DECQ(nb)
	JNZ(LabelRef(name + "_loop"))

	// acc - accm
	Label(name + "_done")
	for r := range rows {
		m := XMM()
		VMOVSHDUP(acc[r], m)
		VSUBSS(m, acc[r], acc[r])
	}
	store(acc)
}

// q6K emits the kernel of Q6_K weights for rows rows of activations,
// super-blocks of a float32 scale, 256 int8 and the int16 sums of each
// 16. The quants of 6 bits, 0 to 63, multiply as unsigned bytes, and 32
// times each group's sum, times its scale, is taken from the products,
// as qdotQ6KGo takes it.
func q6K(rows int) {
	name := fmt.Sprintf("qdotQ6KAVX2x%d", rows)
	nb, w, x := prologue(name, rows, "multiplies Q6_K weights by "+rowsOf[rows])
	acc := make([]VecVirtual, rows)
	for r := range acc {
		acc[r] = XMM()
		VXORPS(acc[r], acc[r], acc[r])
	}
	TESTQ(nb, nb)
	JZ(LabelRef(name + "_done"))

	Label(name + "_loop")
	isum := make([]VecVirtual, rows)
	for r := range isum {
		isum[r] = YMM()
		VPXOR(isum[r], isum[r], isum[r])
	}
	for n := range 2 {
		// The scales of the half's eight groups, as int16 in both lanes.
		scales := YMM()
		VPMOVSXBW(Mem{Base: w, Disp: 192 + 8*n}, scales.AsX())
		VINSERTI128(U8(1), scales.AsX(), scales, scales)
		for k := range 4 {
			// Values 128n+32k+l, l < 32: 4 bits of ql and 2 of qh.
			q, h := YMM(), YMM()
			VMOVDQU(Mem{Base: w, Disp: 64*n + 32*(k%2)}, q)
			if k >= 2 {
				VPSRLW(U8(4), q, q)
			}
			VPAND(nibbles, q, q)
			VMOVDQU(Mem{Base: w, Disp: 128 + 32*n}, h)
			switch k {
			case 0:
				VPSLLW(U8(4), h, h)
			case 1:
				VPSLLW(U8(2), h, h)
			case 3:
				VPSRLW(U8(2), h, h)
			}
			VPAND(high2, h, h)
			VPOR(h, q, q)
			sc := YMM()
			VPSHUFB(q6kScale.Offset(32*k), scales, sc)
			for r := range rows {
				p := YMM()
				VPMADDUBSW(Mem{Base: x[r], Disp: 4 + 128*n + 32*k}, q, p)
				VPMADDWD(sc, p, p)
				VPADDD(p, isum[r], isum[r])
			}
		}
	}
	all := YMM()
	VPMOVSXBW(Mem{Base: w, Disp: 192}, all)
	d := half(Mem{Base: w, Disp: 208})
	for r := range rows {
		m := YMM()
		VPMADDWD(Mem{Base: x[r], Disp: 260}, all, m)
		VPSLLD(U8(5), m, m)
		VPSUBD(m, isum[r], isum[r])
		s := XMM()
		VEXTRACTI128(U8(1), isum[r], s)
		VPADDD(s, isum[r].AsX(), s)
		VPHADDD(s, s, s)
		VPHADDD(s, s, s)
		VCVTDQ2PS(s, s)
		dd := XMM()
		VMULSS(Mem{Base: x[r]}, d, dd)
		VMULSS(s, dd, dd)
		VADDSS(dd, acc[r], acc[r])
		ADDQ(U32(292), x[r])
	}
	ADDQ(U32(210), w)
	DECQ(nb)
	JNZ(LabelRef(name + "_loop"))

	Label(name + "_done")
	store(acc)
}

// The layouts of backend/cpu's tiles: a super-block of a row of weights,
// unpacked, and one of sixteen rows of activations, interleaved.
const (
	tileRowBytes = 336
	tileScales   = 256 // [16]int16
	tilePairs    = 288 // [8][2]int16
	tileD        = 320 // float32, and dmin after it
	tileActBytes = 4672
	tileSums     = 4096 // [8][16][2]int16
	tileXD       = 4608 // [16]float32
)

// qtile emits the tile kernel of Q4_K weights, or of Q6_K weights if q6,
// for two rows of weights by sixteen of activations. A register holds a
// lane for each of eight rows of activations: VPMADDUBSW multiplies four
// quants of a row of weights, broadcast, by four of each row, and adds
// them in pairs into int16, which sum over as many products as cannot
// overflow, eight quads of quants of Q4_K, whose quants are at most 15,
// and two of Q6_K, whose quants are at most 63; VPMADDWD then applies the
// group's scale and adds the pairs into int32.
func qtile(q6 bool) {
	name, group, run := "qtileQ4KAVX2", 32, 8
	if q6 {
		name, group, run = "qtileQ6KAVX2", 16, 2
	}
	TEXT(name, NOSPLIT, "func(nb int, w, x *byte, acc *float32)")
	Pragma("noescape")
	Doc(name + " multiplies two rows of unpacked weights by sixteen rows of interleaved activations; see qtile.")
	nb, w, x, out := GP64(), GP64(), GP64(), GP64()
	Load(Param("nb"), nb)
	Load(Param("w"), w)
	Load(Param("x"), x)
	Load(Param("acc"), out)
	TESTQ(nb, nb)
	JZ(LabelRef(name + "_done"))

	Label(name + "_loop")
	var isum [2][2]VecVirtual
	for s := range 256 / group {
		var sc [2]VecVirtual
		for t := range 2 {
			sc[t] = YMM()
			VPBROADCASTW(Mem{Base: w, Disp: tileRowBytes*t + tileScales + 2*s}, sc[t])
		}
		for r := 0; r < group/4; r += run {
			var h [2][2]VecVirtual
			for q := r; q < r+run; q++ {
				quad := group/4*s + q
				var a [2]VecVirtual
				for c := range 2 {
					a[c] = YMM()
					VMOVDQU(Mem{Base: x, Disp: 64*quad + 32*c}, a[c])
				}
				for t := range 2 {
					wb := YMM()
					VPBROADCASTD(Mem{Base: w, Disp: tileRowBytes*t + 4*quad}, wb)
					for c := range 2 {
						if q == r {
							h[t][c] = YMM()
							VPMADDUBSW(a[c], wb, h[t][c])
							continue
						}
						p := YMM()
						VPMADDUBSW(a[c], wb, p)
						VPADDW(p, h[t][c], h[t][c])
					}
				}
			}
			for t := range 2 {
				for c := range 2 {
					VPMADDWD(sc[t], h[t][c], h[t][c])
					if s == 0 && r == 0 {
						isum[t][c] = h[t][c]
						continue
					}
					VPADDD(h[t][c], isum[t][c], isum[t][c])
				}
			}
		}
	}

	// The sums of the activations' pairs of groups of 16, times the
	// weights' mins or scales.
	for t := range 2 {
		var m [2]VecVirtual
		for p := range 8 {
			pb := YMM()
			VPBROADCASTD(Mem{Base: w, Disp: tileRowBytes*t + tilePairs + 4*p}, pb)
			for c := range 2 {
				if p == 0 {
					m[c] = YMM()
					VPMADDWD(Mem{Base: x, Disp: tileSums + 64*p + 32*c}, pb, m[c])
					continue
				}
				v := YMM()
				VPMADDWD(Mem{Base: x, Disp: tileSums + 64*p + 32*c}, pb, v)
				VPADDD(v, m[c], m[c])
			}
		}
		wd := YMM()
		VBROADCASTSS(Mem{Base: w, Disp: tileRowBytes*t + tileD}, wd)
		var wm VecVirtual
		if !q6 {
			wm = YMM()
			VBROADCASTSS(Mem{Base: w, Disp: tileRowBytes*t + tileD + 4}, wm)
		}
		for c := range 2 {
			xd := YMM()
			VMOVUPS(Mem{Base: x, Disp: tileXD + 32*c}, xd)
			if q6 {
				// sumi - 32·summ
				VPSLLD(U8(5), m[c], m[c])
				VPSUBD(m[c], isum[t][c], isum[t][c])
			}
			accumulate(wd, xd, isum[t][c], Mem{Base: out, Disp: 4 * (16*t + 8*c)})
			if !q6 {
				accumulate(wm, xd, m[c], Mem{Base: out, Disp: 4 * (32 + 16*t + 8*c)})
			}
		}
	}
	ADDQ(U32(2*tileRowBytes), w)
	ADDQ(U32(tileActBytes), x)
	DECQ(nb)
	JNZ(LabelRef(name + "_loop"))

	Label(name + "_done")
	VZEROUPPER()
	RET()
}

// accumulate adds float32(wd·xd)·float32(sum) to the eight float32 at m,
// rounding each product, as the portable kernels do.
func accumulate(wd, xd, sum VecVirtual, m Mem) {
	dd, f := YMM(), YMM()
	VMULPS(xd, wd, dd)
	VCVTDQ2PS(sum, f)
	VMULPS(dd, f, f)
	VADDPS(m, f, f)
	VMOVUPS(f, m)
}
