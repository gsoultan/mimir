package gen

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// segment combines rows of the input by segment (see backend.OpSegment),
// a thread per output element: element d of segment s combines element d
// of the rows its segment lists, in order, as the CPU backend does. The
// threads of consecutive elements of a row read consecutive elements of
// each source row. A thread whose segment's offsets fall or leave the row
// list, or that meets a row outside the source, reports ErrIndex.
func (g *gen) segment(n int) (*Launch, error) {
	k := g.k
	vx, vo, vr := k.In[0], k.In[1], k.In[2]
	r := len(k.Shape)
	if r == 0 || len(vx.Shape) != r || len(vo.Shape) != 1 || len(vr.Shape) != 1 || vo.Shape[0] != k.Shape[0]+1 ||
		!slices.Equal(vx.Shape[1:], k.Shape[1:]) || g.ins[1] != backend.Int32 || g.ins[2] != backend.Int32 || g.ins[0] != k.DType {
		return nil, fmt.Errorf("segments %v of rows %v into %v: %w", vo.Shape, vx.Shape, k.Shape, backend.ErrInternal)
	}
	if c := k.Attrs.Combine; c != backend.CombineSum && c != backend.CombineMax {
		return nil, fmt.Errorf("segments combined by %s: %w", c, backend.ErrInternal)
	}
	width, _ := backend.NumElements(k.Shape[1:])
	g.line("int s = i / %s;", g.param(width))
	g.line("int d = i - s * %s;", g.param(width))
	shape, views := backend.Coalesce(k.Shape[1:], []backend.View{{Strides: vx.Strides[1:]}})
	inner := g.offsets("d", shape, views)[0]
	os := g.param(vo.Strides[0])
	g.line("int lo = i1[%s + s * %s], hi = i1[%s + (s + 1) * %s];", g.param(vo.Offset), os, g.param(vo.Offset), os)
	g.line("if (lo < 0 || hi < lo || hi > %s) {", g.param(vr.Shape[0]))
	g.line("\treport(err, %du);", ErrIndex)
	g.line("\to0[i] = %s;", store(k.DType, literal(0, k.DType)))
	g.line("\treturn;")
	g.line("}")
	t, dt := typeName(k.DType), k.DType
	g.line("%s acc = %s;", t, literal(0, dt))
	g.line("int rows = %s, ro = %s, rs = %s;", g.param(vx.Shape[0]), g.param(vr.Offset), g.param(vr.Strides[0]))
	g.line("int base = %s + %s, step = %s;", g.param(vx.Offset), inner, g.param(vx.Strides[0]))
	g.line("for (int j = lo; j < hi; j++) {")
	g.line("\tint row = i2[ro + j * rs];")
	g.line("\tif (row < 0 || row >= rows) {")
	g.line("\t\treport(err, %du);", ErrIndex)
	g.line("\t\tbreak;")
	g.line("\t}")
	g.line("\t%s x = %s;", t, load(dt, "i0[base + row * step]"))
	if k.Attrs.Combine == backend.CombineSum {
		g.line("\tacc = j == lo ? x : acc + x;")
	} else {
		g.line("\tacc = j == lo ? x : mx(acc, x);")
	}
	g.line("}")
	g.line("o0[i] = %s;", store(dt, "acc"))
	return g.perElement(n)
}
