package gen

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// ScanThread is the longest axis a scan kernel scans with one thread per
// line. A group scans each line of a longer axis.
const ScanThread = 64

// scanPer is the number of elements each thread of a group scanning a
// line combines in turn; the group scans scanPer*GroupSize elements at a
// time.
const scanPer = 4

// lsePrelude returns lse(a, b), log(exp(a) + exp(b)) in float, declared
// with decl: exactly the larger when the other is -Inf or the larger is
// +Inf, and NaN if either is, as the CPU backend's logAddExp.
func lsePrelude(decl string) string {
	return decl + ` float lse(float a, float b) {
	float hi = mx(a, b), lo = -mx(-a, -b);
	if (lo == fbits(0xff800000u) || hi == fbits(0x7f800000u)) return hi;
	return hi + log2(1.0f + exp2((lo - hi) * 1.44269504f)) * 0.693147181f;
}

`
}

// scanIdentity returns the identity of a scan's combine: the value that
// combines with any x into x itself. The identity of a float sum is -0,
// since -0 + 0 is 0 but 0 + -0 is 0 too.
func scanIdentity(c backend.Combine, dt backend.DType) string {
	switch {
	case c == backend.CombineSum && dt.IsFloat():
		return "fbits(0x80000000u)"
	case c == backend.CombineSum:
		return "0"
	case dt.IsFloat():
		return "fbits(0xff800000u)"
	default:
		return literal(math.MinInt32, dt)
	}
}

// scanStep returns the expression combining a with b.
func scanStep(c backend.Combine, a, b string) string {
	switch c {
	case backend.CombineSum:
		return fmt.Sprintf("%s + %s", a, b)
	case backend.CombineMax:
		return fmt.Sprintf("mx(%s, %s)", a, b)
	default:
		return fmt.Sprintf("lse(%s, %s)", a, b)
	}
}

// planScan scans the input along k.Attrs.Axis. When the axis has at most
// ScanThread elements a thread scans each line, in order like the CPU
// backend; threads with adjacent lines read adjacent elements unless the
// axis is innermost. A group scans each line of a longer axis,
// scanPer*GroupSize elements at a time: every thread combines scanPer
// consecutive elements, the group scans the threads' totals in shared
// memory and adds to each element the totals before its thread's and the
// carry of the elements before the block, so that the work is linear in
// the length of the line. Float elements are combined in float, so sums
// differ from the CPU's in their last bits.
func planScan(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer, n int) (*Launch, error) {
	axis, c, v := k.Attrs.Axis, k.Attrs.Combine, k.In[0]
	dt := ins[0].DType()
	if axis < 0 || axis >= len(k.Shape) || len(v.Shape) != len(k.Shape) || dt != k.DType {
		return nil, fmt.Errorf("scan of %s%v along axis %d to %s%v: %w", dt, v.Shape, axis, k.DType, k.Shape, backend.ErrInternal)
	}
	switch c {
	case backend.CombineSum, backend.CombineMax:
	case backend.CombineLogSumExp:
		if !dt.IsFloat() {
			return nil, fmt.Errorf("%s of %s: %w", c, dt, backend.ErrDType)
		}
	default:
		return nil, fmt.Errorf("no combine %s: %w", c, backend.ErrInternal)
	}
	length := k.Shape[axis]
	lines := n / length
	out := backend.ContiguousStrides(k.Shape)
	drop := func(s []int) []int { return slices.Delete(slices.Clone(s), axis, axis+1) }
	rest, views := backend.Coalesce(drop(k.Shape), []backend.View{
		{Strides: drop(v.Strides), Offset: v.Offset},
		{Strides: drop(out)},
	})

	g := newGen(d, k, outs, ins)
	t := typeName(dt)
	if length <= ScanThread {
		g.line("if (gid >= uint(%s)) return;", g.param(lines))
		g.line("int line = int(gid);")
		offs := g.offsets("line", rest, views)
		g.line("int src = %s, dst = %s;", offs[0], offs[1])
		g.line("%s acc = %s;", t, load(dt, "i0[src]"))
		g.line("o0[dst] = %s;", store(dt, "acc"))
		g.line("for (int j = 1; j < %s; j++) {", g.param(length))
		g.line("\tsrc += %s;", g.param(v.Strides[axis]))
		g.line("\tdst += %s;", g.param(out[axis]))
		g.line("\tacc = %s;", scanStep(c, "acc", load(dt, "i0[src]")))
		g.line("\to0[dst] = %s;", store(dt, "acc"))
		g.line("}")
		return g.finish((lines+GroupSize-1)/GroupSize, false)
	}

	block := scanPer * GroupSize
	ident := scanIdentity(c, dt)
	g.line("int line = int(bid);")
	offs := g.offsets("line", rest, views)
	g.line("int src = %s, dst = %s;", offs[0], offs[1])
	g.line("int len = %s, si = %s, so = %s;", g.param(length), g.param(v.Strides[axis]), g.param(out[axis]))
	g.line("%s %s buf[%d];", d.shared, t, block)
	g.line("%s %s part[%d];", d.shared, t, GroupSize)
	g.line("%s carry = %s;", t, ident)
	g.line("for (int c0 = 0; c0 < len; c0 += %d) {", block)
	// Read the block, adjacent threads adjacent elements.
	g.line("\tfor (int e = 0; e < %d; e++) {", scanPer)
	g.line("\t\tint j = c0 + e * %d + int(tid);", GroupSize)
	g.line("\t\tbuf[e * %d + int(tid)] = j < len ? %s : %s;", GroupSize, load(dt, "i0[src + j * si]"), ident)
	g.line("\t}")
	g.line("\t%s;", d.sync)
	// Each thread scans its elements.
	g.line("\t%s loc[%d];", t, scanPer)
	g.line("\tloc[0] = buf[int(tid) * %d];", scanPer)
	g.line("\tfor (int e = 1; e < %d; e++) loc[e] = %s;", scanPer, scanStep(c, "loc[e - 1]", fmt.Sprintf("buf[int(tid) * %d + e]", scanPer)))
	g.line("\tpart[tid] = loc[%d];", scanPer-1)
	g.line("\t%s;", d.sync)
	// The group scans the threads' totals.
	g.line("\tfor (uint s = 1; s < %du; s <<= 1) {", GroupSize)
	g.line("\t\t%s before = tid >= s ? part[tid - s] : %s;", t, ident)
	g.line("\t\t%s;", d.sync)
	g.line("\t\tif (tid >= s) part[tid] = %s;", scanStep(c, "before", "part[tid]"))
	g.line("\t\t%s;", d.sync)
	g.line("\t}")
	g.line("\t%s pre = tid > 0 ? %s : carry;", t, scanStep(c, "carry", "part[tid - 1]"))
	g.line("\tfor (int e = 0; e < %d; e++) buf[int(tid) * %d + e] = %s;", scanPer, scanPer, scanStep(c, "pre", "loc[e]"))
	g.line("\tcarry = %s;", scanStep(c, "carry", fmt.Sprintf("part[%d]", GroupSize-1)))
	g.line("\t%s;", d.sync)
	// Write the block, adjacent threads adjacent elements.
	g.line("\tfor (int e = 0; e < %d; e++) {", scanPer)
	g.line("\t\tint j = c0 + e * %d + int(tid);", GroupSize)
	g.line("\t\tif (j < len) o0[dst + j * so] = %s;", store(dt, fmt.Sprintf("buf[e * %d + int(tid)]", GroupSize)))
	g.line("\t}")
	g.line("\t%s;", d.sync)
	g.line("}")
	return g.finish(lines, true)
}

// ScatterLines is the number of updates per line above which a scatter
// add of 32-bit elements runs a thread per line instead of per element.
const ScatterLines = 32

// scatterAddLines copies dst into the output and adds the updates, one
// thread per line of the output: the elements that share their position
// outside axis. The thread copies its line, then adds the updates of the
// line in order along axis, as the CPU backend does, so that the work is
// linear in the number of updates instead of their product with the
// line's length. It suits 32-bit elements, which the output holds as the
// sums need them.
func (g *gen) scatterAddLines(n int) (*Launch, error) {
	axis := g.k.Attrs.Axis
	vd, vi, vu := g.k.In[0], g.k.In[1], g.k.In[2]
	shape := g.k.Shape
	length, updates := shape[axis], vi.Shape[axis]
	lines := n / length
	out := backend.ContiguousStrides(shape)
	drop := func(s []int) []int { return slices.Delete(slices.Clone(s), axis, axis+1) }
	rest := drop(shape)
	g.line("if (gid >= uint(%s)) return;", g.param(lines))
	g.line("int line = int(gid);")
	coords := g.index("line", rest)
	g.line("int dst = %s, o = 0, idx = %s, upd = %s;", g.param(vd.Offset), g.param(vi.Offset), g.param(vu.Offset))
	g.line("bool covered = true;")
	for r, c := range coords {
		d := r
		if r >= axis {
			d++
		}
		g.line("dst += %s * %s;", c, g.param(vd.Strides[d]))
		g.line("o += %s * %s;", c, g.param(out[d]))
		g.line("covered = covered && %s < %s;", c, g.param(vi.Shape[d]))
		g.line("idx += %s * %s;", c, g.param(vi.Strides[d]))
		g.line("upd += %s * %s;", c, g.param(vu.Strides[d]))
	}
	g.line("int so = %s, sd = %s;", g.param(out[axis]), g.param(vd.Strides[axis]))
	g.line("for (int j = 0; j < %s; j++) o0[o + j * so] = i0[dst + j * sd];", g.param(length))
	g.line("if (!covered) return;")
	g.line("int si = %s, su = %s;", g.param(vi.Strides[axis]), g.param(vu.Strides[axis]))
	g.line("for (int j = 0; j < %s; j++) {", g.param(updates))
	g.line("\tint to = i1[idx + j * si];")
	g.line("\tif (to < 0 || to >= %s) {", g.param(length))
	g.line("\t\treport(err, %du);", ErrIndex)
	g.line("\t\tcontinue;")
	g.line("\t}")
	g.line("\to0[o + to * so] = o0[o + to * so] + i2[upd + j * su];")
	g.line("}")
	return g.perElement(lines)
}
