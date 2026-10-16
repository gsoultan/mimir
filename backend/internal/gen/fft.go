package gen

import (
	"fmt"
	"math/bits"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// FFTShared is the longest power-of-two transform a group computes in
// shared memory; longer ones are computed in the output.
const FFTShared = 2048

// planFFT transforms each line of complex elements, [..., 2], along
// k.Attrs.Axis. A group transforms a line whose length is a power of
// two by radix-2 butterflies over its elements in bit-reversed order, in
// shared memory or, for lines longer than FFTShared, in the output,
// with twiddles from cos_pi and sin_pi of exact fractions. Other lengths
// take the definition, a thread per output element summing the products
// in turn, which costs O(n²) per line: the tensor package builds longer
// transforms of such lengths from power-of-two ones (Bluestein's
// algorithm). Elements are computed in float, so results differ from the
// CPU's, which computes in float64, in their last bits.
func planFFT(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer, total int) (*Launch, error) {
	axis, v := k.Attrs.Axis, k.In[0]
	r := len(k.Shape)
	if k.DType != backend.Float32 || ins[0].DType() != backend.Float32 || r < 2 || k.Shape[r-1] != 2 || axis < 0 || axis >= r-1 || !slices.Equal(v.Shape, k.Shape) {
		return nil, fmt.Errorf("fft of %v along axis %d: %w", v.Shape, axis, backend.ErrInternal)
	}
	n := k.Shape[axis]
	lines := total / (2 * n)
	out := backend.ContiguousStrides(k.Shape)
	drop := func(s []int) []int { return slices.Delete(slices.Clone(s[:r-1]), axis, axis+1) }
	rest, views := backend.Coalesce(drop(k.Shape), []backend.View{
		{Strides: drop(v.Strides), Offset: v.Offset},
		{Strides: drop(out)},
	})
	sign := "-"
	if k.Attrs.Inverse {
		sign = ""
	}

	g := newGen(d, k, outs, ins)
	if n&(n-1) != 0 {
		g.line("if (gid >= uint(%s)) return;", g.param(lines*n))
		g.line("int line = int(gid) / %s, f = int(gid) %% %s;", g.param(n), g.param(n))
		offs := g.offsets("line", rest, views)
		g.line("int src = %s, dst = %s, si = %s, sl = %s, n = %s;", offs[0], offs[1], g.param(v.Strides[axis]), g.param(v.Strides[r-1]), g.param(n))
		g.line("float sr = 0.0f, sm = 0.0f;")
		g.line("int t = 0;")
		g.line("for (int j = 0; j < n; j++) {")
		g.line("\tfloat a = 2.0f * float(t) / float(n);")
		g.line("\tfloat c = cos_pi(a), s = %ssin_pi(a);", sign)
		g.line("\tfloat xr = i0[src + j * si], xi = i0[src + j * si + sl];")
		g.line("\tsr += xr * c - xi * s;")
		g.line("\tsm += xr * s + xi * c;")
		g.line("\tt += f;")
		g.line("\tif (t >= n) t -= n;")
		g.line("}")
		g.line("o0[dst + f * %s] = sr;", g.param(out[axis]))
		g.line("o0[dst + f * %s + 1] = sm;", g.param(out[axis]))
		return g.finish((lines*n+GroupSize-1)/GroupSize, false)
	}

	logN := bits.Len(uint(n)) - 1
	g.line("int line = int(bid);")
	offs := g.offsets("line", rest, views)
	g.line("int src = %s, dst = %s, si = %s, sl = %s, so = %s, n = %s;", offs[0], offs[1], g.param(v.Strides[axis]), g.param(v.Strides[r-1]), g.param(out[axis]), g.param(n))
	// Element e goes to the position of its bits reversed.
	reversed := func(e string) string {
		if logN == 0 {
			return "0"
		}
		return fmt.Sprintf("int(bitrev32(uint(%s)) >> %du)", e, 32-logN)
	}
	g.helpers = fmt.Sprintf(`%s uint bitrev32(uint x) {
	x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
	x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
	x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
	x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
	return (x >> 16) | (x << 16);
}

`, d.decl)
	re, im, sync := "re[%s]", "im[%s]", d.sync
	if n > FFTShared {
		re, im, sync = "o0[dst + (%s) * so]", "o0[dst + (%s) * so + 1]", d.syncDevice
	} else {
		g.line("%s float re[%d];", d.shared, FFTShared)
		g.line("%s float im[%d];", d.shared, FFTShared)
	}
	at := func(f, i string) string { return fmt.Sprintf(f, i) }
	g.line("for (int e = int(tid); e < n; e += %d) {", GroupSize)
	g.line("\tint p = %s;", reversed("e"))
	g.line("\t%s = i0[src + e * si];", at(re, "p"))
	g.line("\t%s = i0[src + e * si + sl];", at(im, "p"))
	g.line("}")
	g.line("%s;", sync)
	g.line("for (int len = 2; len <= n; len <<= 1) {")
	g.line("\tint hw = len >> 1;")
	g.line("\tfor (int c = int(tid); c < n / 2; c += %d) {", GroupSize)
	g.line("\t\tint off = c %% hw, a = (c / hw) * len + off, b = a + hw;")
	g.line("\t\tfloat x = 2.0f * float(off) / float(len);")
	g.line("\t\tfloat wr = cos_pi(x), wi = %ssin_pi(x);", sign)
	g.line("\t\tfloat br = %s, bi = %s;", at(re, "b"), at(im, "b"))
	g.line("\t\tfloat tr = br * wr - bi * wi, ti = br * wi + bi * wr;")
	g.line("\t\tfloat ar = %s, ai = %s;", at(re, "a"), at(im, "a"))
	g.line("\t\t%s = ar - tr;", at(re, "b"))
	g.line("\t\t%s = ai - ti;", at(im, "b"))
	g.line("\t\t%s = ar + tr;", at(re, "a"))
	g.line("\t\t%s = ai + ti;", at(im, "a"))
	g.line("\t}")
	g.line("\t%s;", sync)
	g.line("}")
	if n <= FFTShared {
		g.line("for (int e = int(tid); e < n; e += %d) {", GroupSize)
		g.line("\to0[dst + e * so] = re[e];")
		g.line("\to0[dst + e * so + 1] = im[e];")
		g.line("}")
	}
	return g.finish(lines, true)
}
