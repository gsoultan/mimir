package gen

import (
	"fmt"
	"math/bits"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// SortThread is the longest axis an argsort kernel sorts with one
// thread per line, by selection. SortShared is the longest a group sorts
// in shared memory; longer axes are sorted in the output, or, for at
// most backend.MaxTopK positions, selected block by block in shared
// memory.
const (
	SortThread = 32
	SortShared = 2048
)

// sortHelpers orders elements by unsigned keys as the CPU backend does:
// NaN is the largest float and -0 equals +0, both told by their bits so
// that fast math cannot fold them away.
func sortHelpers(decl string) string {
	return fmt.Sprintf(`%[1]s uint sortkey(float f) {
	uint b = ubits(f);
	if ((b & 0x7fffffffu) > 0x7f800000u) b = 0x7fc00000u;
	if ((b & 0x7fffffffu) == 0u) b = 0u;
	return (b >> 31) != 0u ? ~b : b | 0x80000000u;
}
%[1]s uint sortkey(int x) { return uint(x) ^ 0x80000000u; }

`, decl)
}

// planArgsort sorts each line of the input along k.Attrs.Axis by the
// pair of an element's key and its position, which makes the sort
// stable, and writes the positions of the first k.Shape[axis] of them.
// Descending order complements the keys. A thread selects the positions
// of each line of at most SortThread elements one after another; a group
// sorts each longer line with a bitonic network, whose comparators all
// put the smaller pair first: in shared memory for at most SortShared
// elements, padded with pairs larger than any, or, for longer lines, in
// the output, where it compares the elements at the positions it holds
// and skips the comparators past the end, whose padding would not move.
// A top-k of a longer line keeps the best k pairs at the start of shared
// memory and sorts them with each next block of the line.
func planArgsort(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer, total int) (*Launch, error) {
	axis, v := k.Attrs.Axis, k.In[0]
	dt := ins[0].DType()
	if k.DType != backend.Int32 || axis < 0 || axis >= len(k.Shape) || len(v.Shape) != len(k.Shape) || k.Shape[axis] > v.Shape[axis] || dt.Size() == 0 {
		return nil, fmt.Errorf("argsort of %s%v along axis %d to %v: %w", dt, v.Shape, axis, k.Shape, backend.ErrInternal)
	}
	n, kk := v.Shape[axis], k.Shape[axis]
	if kk < n && kk > backend.MaxTopK {
		return nil, fmt.Errorf("argsort of %d of %d positions: %w", kk, n, backend.ErrInternal)
	}
	lines := total / kk
	out := backend.ContiguousStrides(k.Shape)
	drop := func(s []int) []int { return slices.Delete(slices.Clone(s), axis, axis+1) }
	rest, views := backend.Coalesce(drop(k.Shape), []backend.View{
		{Strides: drop(v.Strides), Offset: v.Offset},
		{Strides: drop(out)},
	})

	g := newGen(d, k, outs, ins)
	g.helpers = sortHelpers(d.decl)
	key := func(pos string) string {
		x := fmt.Sprintf("sortkey(%s)", load(dt, fmt.Sprintf("i0[src + (%s) * si]", pos)))
		if k.Attrs.Descending {
			return "~" + x
		}
		return x
	}
	if n <= SortThread {
		g.line("if (gid >= uint(%s)) return;", g.param(lines))
		g.line("int line = int(gid);")
		offs := g.offsets("line", rest, views)
		g.line("int src = %s, dst = %s, si = %s, so = %s, n = %s;", offs[0], offs[1], g.param(v.Strides[axis]), g.param(out[axis]), g.param(n))
		g.line("uint pk = 0u;")
		g.line("int pidx = -1;")
		g.line("for (int j = 0; j < %s; j++) {", g.param(kk))
		g.line("\tuint bk = 0xffffffffu;")
		g.line("\tint bi = 0x7fffffff;")
		g.line("\tfor (int e = 0; e < n; e++) {")
		g.line("\t\tuint ke = %s;", key("e"))
		g.line("\t\tbool after = ke > pk || (ke == pk && e > pidx);")
		g.line("\t\tif (after && (ke < bk || (ke == bk && e < bi))) { bk = ke; bi = e; }")
		g.line("\t}")
		g.line("\to0[dst + j * so] = bi;")
		g.line("\tpk = bk;")
		g.line("\tpidx = bi;")
		g.line("}")
		return g.finish((lines+GroupSize-1)/GroupSize, false)
	}

	g.line("int line = int(bid);")
	offs := g.offsets("line", rest, views)
	g.line("int src = %s, dst = %s, si = %s, so = %s, n = %s, kk = %s;", offs[0], offs[1], g.param(v.Strides[axis]), g.param(out[axis]), g.param(n), g.param(kk))
	if n > SortShared && kk == n {
		// The whole line, sorted in the output.
		g.line("for (int e = int(tid); e < n; e += %d) o0[dst + e * so] = e;", GroupSize)
		g.line("%s;", d.syncDevice)
		size := 1 << bits.Len(uint(n-1))
		g.bitonic(size, "n", d.syncDevice, func(lo, hi string) {
			g.line("\t\t\tint xa = o0[dst + %s * so], xb = o0[dst + %s * so];", lo, hi)
			g.line("\t\t\tuint ka = %s, kb = %s;", key("xa"), key("xb"))
			g.line("\t\t\tif (kb < ka || (kb == ka && xb < xa)) { o0[dst + %s * so] = xb; o0[dst + %s * so] = xa; }", lo, hi)
		})
		return g.finish(lines, true)
	}

	g.line("%s uint sk[%d];", d.shared, SortShared)
	g.line("%s int sx[%d];", d.shared, SortShared)
	swap := func(lo, hi string) {
		g.line("\t\t\tuint ka = sk[%s], kb = sk[%s];", lo, hi)
		g.line("\t\t\tint xa = sx[%s], xb = sx[%s];", lo, hi)
		g.line("\t\t\tif (kb < ka || (kb == ka && xb < xa)) { sk[%s] = kb; sk[%s] = ka; sx[%s] = xb; sx[%s] = xa; }", lo, hi, lo, hi)
	}
	if n <= SortShared {
		size := 1 << bits.Len(uint(n-1))
		g.line("for (int e = int(tid); e < %d; e += %d) {", size, GroupSize)
		g.line("\tbool ok = e < n;")
		g.line("\tsk[e] = ok ? %s : 0xffffffffu;", key("ok ? e : 0"))
		g.line("\tsx[e] = ok ? e : 0x7fffffff;")
		g.line("}")
		g.line("%s;", d.sync)
		g.bitonic(size, fmt.Sprint(size), d.sync, swap)
	} else {
		// The best kk pairs so far stay in [0, kk); each block of the
		// line comes in after them.
		g.line("int block = %d - kk;", SortShared)
		g.line("for (int e = int(tid); e < kk; e += %d) { sk[e] = 0xffffffffu; sx[e] = 0x7fffffff; }", GroupSize)
		g.line("for (int c0 = 0; c0 < n; c0 += block) {")
		g.line("\tfor (int e = int(tid); e < block; e += %d) {", GroupSize)
		g.line("\t\tint j = c0 + e;")
		g.line("\t\tbool ok = j < n;")
		g.line("\t\tsk[kk + e] = ok ? %s : 0xffffffffu;", key("ok ? j : 0"))
		g.line("\t\tsx[kk + e] = ok ? j : 0x7fffffff;")
		g.line("\t}")
		g.line("\t%s;", d.sync)
		g.bitonic(SortShared, fmt.Sprint(SortShared), d.sync, swap)
		g.line("}")
	}
	g.line("for (int e = int(tid); e < kk; e += %d) o0[dst + e * so] = sx[e];", GroupSize)
	return g.finish(lines, true)
}

// bitonic emits a group's bitonic sorting network over size positions, a
// power of two, of which those below limit exist: every comparator
// orders positions lo < hi, emitted by swap, and sync follows each stage.
// The first stage of each merge compares mirrored positions, so that
// every comparator puts the smaller pair first.
func (g *gen) bitonic(size int, limit, sync string, swap func(lo, hi string)) {
	g.line("for (int size = 2; size <= %d; size <<= 1) {", size)
	g.line("\tfor (int hw = size >> 1; hw > 0; hw >>= 1) {")
	g.line("\t\tfor (int c = int(tid); c < %d; c += %d) {", size/2, GroupSize)
	g.line("\t\t\tint off = c %% hw, lo = (c / hw) * hw * 2 + off;")
	g.line("\t\t\tint hi = hw * 2 == size ? lo + size - 1 - 2 * off : lo + hw;")
	g.line("\t\t\tif (hi >= %s) continue;", limit)
	swap("lo", "hi")
	g.line("\t\t}")
	g.line("\t\t%s;", sync)
	g.line("\t}")
	g.line("}")
}
