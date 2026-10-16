package gen

import (
	"fmt"
	"strings"

	"github.com/gsoultan/mimir/backend"
)

// Kernels read a block-quantized buffer as bytes, uchar, and dequantize
// each value as ggml does, rounding after every operation, so that a
// dequantizing cast computes what the CPU backend does bit for bit. A
// kernel that reads one gets the helpers of its dtype ahead of its
// body:
//
//	dq_T(q, e)                       value e of the buffer q
//	dot32_T(w, s, x, ox, sx, acc)    acc plus the products of the 32
//	                                 values of sub-block s of the row w
//	                                 with x[ox], x[ox+sx], ..., by fma
//
// where T names the dtype: q8_0, q4_k or q6_k.

// quantName returns the name of dt in the helpers.
func quantName(dt backend.DType) string { return strings.ToLower(dt.String()) }

// quantHelpers returns the helpers of dt in dialect d.
func quantHelpers(d *Dialect, dt backend.DType) string {
	common := `DECL float qh2f(const G uchar* q, int at) { return h2f(ushort(int(q[at]) | int(q[at + 1]) << 8)); }
DECL int qi8(int b) { return (b ^ 0x80) - 0x80; }
`
	var src string
	switch dt {
	case backend.Q8_0:
		// Blocks of 34 bytes: the scale d, a float16, then 32 int8.
		src = `DECL float dq_q8_0(const G uchar* q, int e) {
	const G uchar* b = q + e / 32 * 34;
	float d = qh2f(b, 0);
	return float(qi8(b[2 + e % 32])) * d;
}
DECL float dot32_q8_0(const G uchar* w, int s, const G float* x, int ox, int sx, float acc) {
	const G uchar* b = w + s * 34;
	float d = qh2f(b, 0);
	for (int l = 0; l < 32; l++) {
		float v = float(qi8(b[2 + l])) * d;
		acc = fma(x[ox + l * sx], v, acc);
	}
	return acc;
}
`
	case backend.Q4_K:
		// Super-blocks of 144 bytes: d and dmin, float16s, the 6-bit
		// scales and mins of eight sub-blocks of 32 in 12 bytes, then 128
		// bytes of 4-bit quants, sub-blocks 2j and 2j+1 in the low and
		// high nibbles of bytes 32j to 32j+31.
		src = `DECL int k4_scale(int j, const G uchar* q) {
	return j < 4 ? int(q[j]) & 63 : (int(q[j + 4]) & 15) | (int(q[j - 4]) >> 6) << 4;
}
DECL int k4_min(int j, const G uchar* q) {
	return j < 4 ? int(q[j + 4]) & 63 : int(q[j + 4]) >> 4 | (int(q[j]) >> 6) << 4;
}
DECL float dq_q4_k(const G uchar* q, int e) {
	const G uchar* b = q + e / 256 * 144;
	int r = e % 256, s = r / 32, l = r % 32;
	int qv = int(b[16 + 32 * (s / 2) + l]);
	qv = s % 2 == 0 ? qv & 15 : qv >> 4;
	float d1 = qh2f(b, 0) * float(k4_scale(s, b + 4));
	float m1 = qh2f(b, 2) * float(k4_min(s, b + 4));
	float v = d1 * float(qv);
	return v - m1;
}
DECL float dot32_q4_k(const G uchar* w, int s, const G float* x, int ox, int sx, float acc) {
	const G uchar* b = w + s / 8 * 144;
	int j = s % 8;
	float d1 = qh2f(b, 0) * float(k4_scale(j, b + 4));
	float m1 = qh2f(b, 2) * float(k4_min(j, b + 4));
	const G uchar* qs = b + 16 + 32 * (j / 2);
	int shift = 4 * (j % 2);
	for (int l = 0; l < 32; l++) {
		float v = d1 * float(int(qs[l]) >> shift & 15);
		v = v - m1;
		acc = fma(x[ox + l * sx], v, acc);
	}
	return acc;
}
`
	case backend.Q6_K:
		// Super-blocks of 210 bytes: the low 4 bits of 256 quants in 128
		// bytes, their high 2 bits in 64, 16 int8 scales of groups of 16,
		// then d, a float16. Value 128n+32k+l, k < 4, l < 32, takes the
		// low or high nibble of byte 64n+32(k%2)+l and bits 2k of byte
		// 128+32n+l, less 32, and the scale of group 8n+2k+l/16.
		src = `DECL int q6(const G uchar* b, int n, int k, int l) {
	int lo = int(b[64 * n + 32 * (k % 2) + l]) >> (4 * (k / 2)) & 15;
	int hi = int(b[128 + 32 * n + l]) >> (2 * k) & 3;
	return (lo | hi << 4) - 32;
}
DECL float dq_q6_k(const G uchar* q, int e) {
	const G uchar* b = q + e / 256 * 210;
	int r = e % 256, n = r / 128, k = r % 128 / 32, l = r % 32;
	float d = qh2f(b, 208) * float(qi8(b[192 + 8 * n + 2 * k + l / 16]));
	return d * float(q6(b, n, k, l));
}
DECL float dot32_q6_k(const G uchar* w, int s, const G float* x, int ox, int sx, float acc) {
	const G uchar* b = w + s / 8 * 210;
	int n = s % 8 / 4, k = s % 4;
	float d0 = qh2f(b, 208) * float(qi8(b[192 + 8 * n + 2 * k]));
	float d1 = qh2f(b, 208) * float(qi8(b[193 + 8 * n + 2 * k]));
	for (int l = 0; l < 32; l++) {
		float v = (l < 16 ? d0 : d1) * float(q6(b, n, k, l));
		acc = fma(x[ox + l * sx], v, acc);
	}
	return acc;
}
`
	}
	return strings.NewReplacer("DECL ", d.decl+" ", "G ", d.global).Replace(common + src)
}

// dequantizes reports whether k is a cast that dequantizes its input.
func dequantizes(k *backend.Kernel, ins []backend.DType) bool {
	return k.Op == backend.OpCast && len(ins) == 1 && ins[0].IsQuantized()
}

// dequantize casts a block-quantized input to the float output, element
// by element.
func (g *gen) dequantize(n int) (*Launch, error) {
	dt := g.ins[0]
	if !g.k.DType.IsFloat() || len(g.k.In) != 1 {
		return nil, fmt.Errorf("cast of %s to %s: %w", dt, g.k.DType, backend.ErrInternal)
	}
	g.helpers = quantHelpers(g.d, dt)
	shape, views := backend.Coalesce(g.k.Shape, g.k.In)
	off := g.offsets("i", shape, views)[0]
	g.line("o0[i] = %s;", store(g.k.DType, fmt.Sprintf("dq_%s(i0, %s)", quantName(dt), off)))
	return g.perElement(n)
}

// qmatmulRows is the number of outputs a group of a product by quantized
// weights computes: each output has 32 threads, which take the
// sub-blocks of 32 values of its row in turn and sum their results in a
// tree.
const qmatmulRows = GroupSize / 32

// qmatmul computes x·Wᵀ for x Float32 [..., K] and W block-quantized
// [N, K], contiguous from a whole block, into [..., N]. Each value of W
// is dequantized as by a cast and multiplied in float32, so the result
// differs from the CPU backend's, which rounds x to int8, by that
// rounding.
func (g *gen) qmatmul(n int) (*Launch, error) {
	k := g.k
	vx, vw := k.In[0], k.In[1]
	r := len(vx.Shape)
	if len(g.ins) != 2 {
		return nil, fmt.Errorf("product by quantized weights of %d inputs: %w", len(g.ins), backend.ErrInternal)
	}
	dt := g.ins[1]
	if k.DType != backend.Float32 || g.ins[0] != backend.Float32 || !dt.IsQuantized() || r == 0 || len(vw.Shape) != 2 || len(k.Shape) != r {
		return nil, fmt.Errorf("product of %s%v by %s%v: %w", g.ins[0], vx.Shape, dt, vw.Shape, backend.ErrInternal)
	}
	nn, kk, bs := vw.Shape[0], vw.Shape[1], dt.BlockSize()
	if vx.Shape[r-1] != kk || k.Shape[r-1] != nn || kk%bs != 0 || !vw.IsContiguous() || vw.Offset%bs != 0 {
		return nil, fmt.Errorf("product of %v by %s%v into %v: %w", vx.Shape, dt, vw.Shape, k.Shape, backend.ErrInternal)
	}
	g.helpers = quantHelpers(g.d, dt)
	name := quantName(dt)
	g.line("int sub = int(tid) / 32, lane = int(tid) %% 32;")
	g.line("int o = int(bid) * %d + sub;", qmatmulRows)
	g.line("bool live = o < %s;", g.param(n))
	g.line("int col = live ? o %% %s : 0;", g.param(nn))
	g.line("int row = live ? o / %s : 0;", g.param(nn))
	lead := backend.View{Shape: vx.Shape[:r-1], Strides: vx.Strides[:r-1], Offset: vx.Offset}
	shape, views := backend.Coalesce(lead.Shape, []backend.View{lead})
	ox := g.offsets("row", shape, views)[0]
	g.line("const %suchar* w = i1 + (%s + col * %s) * %d;", g.d.global, g.param(vw.Offset/bs), g.param(kk/bs), dt.BlockBytes())
	g.line("int sx = %s;", g.param(vx.Strides[r-1]))
	g.line("float acc = 0.0f;")
	g.line("for (int s = live ? lane : %s; s < %s; s += 32) {", g.param(kk/32), g.param(kk/32))
	g.line("\tacc = dot32_%s(w, s, i0, %s + s * 32 * sx, sx, acc);", name, ox)
	g.line("}")
	g.line("%s float part[%d];", g.d.shared, GroupSize)
	g.line("part[tid] = acc;")
	g.line("%s;", g.d.sync)
	g.line("for (int st = 16; st > 0; st >>= 1) {")
	g.line("\tif (lane < st) part[tid] = part[tid] + part[tid + st];")
	g.line("\t%s;", g.d.sync)
	g.line("}")
	g.line("if (live && lane == 0) o0[o] = part[tid];")
	return g.finish((n+qmatmulRows-1)/qmatmulRows, true)
}
