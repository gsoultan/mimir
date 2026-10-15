package metal

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/gsoultan/mimir/backend"
)

// Every kernel runs as Metal Shading Language source that this file
// generates, one dispatch per Backend.Run. The source depends on the
// structure of a kernel: its op, dtypes, the rank of its coalesced shape,
// which inputs are read contiguously and which strides are zero. Sizes,
// strides and offsets are parameters, so kernels that differ only in them
// share one compiled pipeline.
//
// Each thread computes one output element, except for long reductions,
// where a threadgroup of groupSize threads computes one. Threads visit the
// inputs an element sums in the same order as the CPU backend, so sums
// that one thread computes match it bit for bit.

// groupSize is the number of threads in a threadgroup.
const groupSize = 256

// Error codes a kernel ORs into its error word.
const (
	errDivideByZero = 1 << iota
	errIndex
)

// launch is one compute dispatch: kernel k of src, run as groups
// threadgroups of threads threads each. The kernel binds the outputs, then
// the inputs, then params as constant int* p, then the error word.
type launch struct {
	src      string
	groups   int
	threads  int
	params   []int32
	barriers bool // threads of a threadgroup synchronize
}

// maxElems bounds the elements of a buffer and the offsets kernels
// compute, which are 32-bit ints.
const maxElems = math.MaxInt32

// gen builds the source of one kernel.
type gen struct {
	k      *backend.Kernel
	outs   []backend.DType
	ins    []backend.DType
	body   strings.Builder
	params []int32
	vars   int
	err    error
}

func newGen(k *backend.Kernel, outs, ins []backend.Buffer) *gen {
	g := &gen{k: k}
	for _, b := range outs {
		g.outs = append(g.outs, b.DType())
	}
	for _, b := range ins {
		g.ins = append(g.ins, b.DType())
	}
	return g
}

// param returns an expression for the value v, passed as a parameter.
func (g *gen) param(v int) string {
	if v < math.MinInt32 || v > math.MaxInt32 {
		g.err = fmt.Errorf("value %d does not fit a 32-bit kernel parameter: %w", v, backend.ErrShape)
	}
	g.params = append(g.params, int32(v))
	return fmt.Sprintf("p[%d]", len(g.params)-1)
}

// v returns a fresh variable name.
func (g *gen) v(prefix string) string {
	g.vars++
	return fmt.Sprintf("%s%d", prefix, g.vars)
}

func (g *gen) line(format string, args ...any) {
	fmt.Fprintf(&g.body, "\t"+format+"\n", args...)
}

// launch returns the dispatch of the generated kernel over n elements, one
// thread each.
func (g *gen) perElement(n int) (*launch, error) {
	return g.finish((n+groupSize-1)/groupSize, false)
}

func (g *gen) finish(groups int, barriers bool) (*launch, error) {
	if g.err != nil {
		return nil, g.err
	}
	var s strings.Builder
	s.WriteString(prelude)
	s.WriteString("kernel void k(")
	var args []string
	for i, dt := range g.outs {
		args = append(args, fmt.Sprintf("device %s* o%d [[buffer(%d)]]", typeName(dt), i, len(args)))
	}
	for i, dt := range g.ins {
		args = append(args, fmt.Sprintf("const device %s* i%d [[buffer(%d)]]", typeName(dt), i, len(args)))
	}
	n := len(args)
	args = append(args,
		fmt.Sprintf("constant int* p [[buffer(%d)]]", n),
		fmt.Sprintf("device atomic_uint* err [[buffer(%d)]]", n+1),
		"uint gid [[thread_position_in_grid]]",
		"uint tid [[thread_position_in_threadgroup]]",
		"uint bid [[threadgroup_position_in_grid]]")
	s.WriteString(strings.Join(args, ", "))
	s.WriteString(") {\n")
	s.WriteString(g.body.String())
	s.WriteString("}\n")
	if len(g.params) == 0 {
		g.params = []int32{0} // a constant buffer may not be empty
	}
	return &launch{src: s.String(), groups: groups, threads: groupSize, params: g.params, barriers: barriers}, nil
}

// prelude declares the helpers kernels share. mx is the maximum that
// propagates NaN and orders -0 below +0, like Go's max; md is the
// truncated remainder of int32s, which reports division by zero.
const prelude = `#include <metal_stdlib>
using namespace metal;

static inline float mx(float a, float b) {
	if (a != a || b != b) return a + b;
	if (a == b) return signbit(a) ? b : a;
	return a > b ? a : b;
}
static inline int mx(int a, int b) { return a > b ? a : b; }
static inline int md(int a, int b, device atomic_uint* err) {
	if (b == 0) {
		atomic_fetch_or_explicit(err, 1u, memory_order_relaxed);
		return 0;
	}
	return b == -1 ? 0 : a % b;
}
static inline float md(float a, float b, device atomic_uint* err) { return fmod(a, b); }

`

func typeName(dt backend.DType) string {
	if dt == backend.Float32 {
		return "float"
	}
	return "int"
}

// literal returns an exact MSL expression for v converted to dt, as the
// CPU backend converts it.
func literal(v float64, dt backend.DType) string {
	if dt == backend.Float32 {
		return fmt.Sprintf("as_type<float>(0x%08xu)", math.Float32bits(float32(v)))
	}
	return fmt.Sprintf("as_type<int>(0x%08xu)", uint32(int32(v)))
}

// offsets emits code that sets, for the element at linear index lin of
// shape, the offset each view reads it at, and returns the variables. The
// views walk shape, which should be coalesced.
func (g *gen) offsets(lin string, shape []int, views []backend.View) []string {
	names := make([]string, len(views))
	strided := false
	for i, v := range views {
		names[i] = g.v("off")
		switch {
		case len(shape) == 0:
			g.line("int %s = %s;", names[i], g.param(v.Offset))
		case len(shape) == 1 && v.Strides[0] == 1:
			g.line("int %s = %s + %s;", names[i], g.param(v.Offset), lin)
		default:
			g.line("int %s = %s;", names[i], g.param(v.Offset))
			strided = true
		}
	}
	if !strided {
		return names
	}
	rem := g.v("r")
	g.line("int %s = %s;", rem, lin)
	for d := len(shape) - 1; d >= 0; d-- {
		c := g.v("c")
		if d == 0 {
			g.line("int %s = %s;", c, rem)
		} else {
			size := g.param(shape[d])
			g.line("int %s = %s %% %s; %s /= %s;", c, rem, size, rem, size)
		}
		for i, v := range views {
			if len(shape) == 1 && v.Strides[0] == 1 || v.Strides[d] == 0 {
				continue
			}
			g.line("%s += %s * %s;", names[i], c, g.param(v.Strides[d]))
		}
	}
	return names
}

// index emits code that sets the coordinates of the element at linear
// index lin of shape, and returns their variables.
func (g *gen) index(lin string, shape []int) []string {
	coords := make([]string, len(shape))
	rem := g.v("r")
	g.line("int %s = %s;", rem, lin)
	for d := len(shape) - 1; d >= 0; d-- {
		coords[d] = g.v("c")
		if d == 0 {
			g.line("int %s = %s;", coords[d], rem)
			continue
		}
		size := g.param(shape[d])
		g.line("int %s = %s %% %s; %s /= %s;", coords[d], rem, size, rem, size)
	}
	return coords
}

// program emits the instructions of an elementwise program and returns the
// variable holding each value. load returns the expression of input a.
func (g *gen) program(prog []backend.Instr, load func(a int) string) []string {
	vals := make([]string, len(prog))
	for j, in := range prog {
		t := typeName(in.DType)
		arg := func(i int) string { return vals[in.Args[i]] }
		var e string
		switch op := in.Op; op {
		case backend.OpLoad:
			e = load(in.Args[0])
		case backend.OpConst:
			e = literal(in.Value, in.DType)
		case backend.OpNeg:
			e = "-" + arg(0)
		case backend.OpExp2:
			e = fmt.Sprintf("exp2(%s)", arg(0))
		case backend.OpLog2:
			e = fmt.Sprintf("log2(%s)", arg(0))
		case backend.OpSin:
			e = fmt.Sprintf("sin(%s)", arg(0))
		case backend.OpSqrt:
			e = fmt.Sprintf("sqrt(%s)", arg(0))
		case backend.OpRecip:
			e = fmt.Sprintf("1.0f / %s", arg(0))
		case backend.OpCast:
			e = fmt.Sprintf("%s(%s)", t, arg(0))
		case backend.OpAdd:
			e = fmt.Sprintf("%s + %s", arg(0), arg(1))
		case backend.OpMul:
			e = fmt.Sprintf("%s * %s", arg(0), arg(1))
		case backend.OpMax:
			e = fmt.Sprintf("mx(%s, %s)", arg(0), arg(1))
		case backend.OpCmpLT:
			e = fmt.Sprintf("%s(%s < %s)", t, arg(0), arg(1))
		case backend.OpCmpEQ:
			e = fmt.Sprintf("%s(%s == %s)", t, arg(0), arg(1))
		case backend.OpMod:
			e = fmt.Sprintf("md(%s, %s, err)", arg(0), arg(1))
		case backend.OpWhere:
			e = fmt.Sprintf("%s != 0 ? %s : %s", arg(0), arg(1), arg(2))
		default:
			g.err = fmt.Errorf("instruction %d: %s is not elementwise: %w", j, op, backend.ErrInternal)
			e = "0"
		}
		vals[j] = g.v("v")
		g.line("%s %s = %s;", t, vals[j], e)
	}
	return vals
}

// checkProgram verifies that prog is a well-formed program over ins that
// computes a value of dtype out.
func checkProgram(prog []backend.Instr, stores []int, out backend.DType, ins []backend.DType) error {
	if len(prog) == 0 || prog[len(prog)-1].DType != out {
		return fmt.Errorf("program of %d instructions for %s output: %w", len(prog), out, backend.ErrInternal)
	}
	for _, s := range stores {
		if s < 0 || s >= len(prog) || prog[s].DType != out {
			return fmt.Errorf("stored value %d: %w", s, backend.ErrInternal)
		}
	}
	for j, in := range prog {
		bad := func(what string) error {
			return fmt.Errorf("instruction %d (%s): %s: %w", j, in.Op, what, backend.ErrInternal)
		}
		if in.DType != backend.Float32 && in.DType != backend.Int32 {
			return backend.ErrDType
		}
		want := 0
		switch op := in.Op; {
		case op == backend.OpLoad:
			if len(in.Args) != 1 || in.Args[0] < 0 || in.Args[0] >= len(ins) {
				return bad("no such input")
			}
			if ins[in.Args[0]] != in.DType {
				return bad("input dtype")
			}
			continue
		case op == backend.OpConst:
		case op.IsUnary():
			want = 1
		case op.IsBinary():
			want = 2
		case op == backend.OpWhere:
			want = 3
		default:
			return bad("not elementwise")
		}
		if len(in.Args) != want {
			return bad("arguments")
		}
		for i, a := range in.Args {
			if a < 0 || a >= j {
				return bad("argument out of order")
			}
			same := in.Op != backend.OpCast && (in.Op != backend.OpWhere || i > 0)
			if same && prog[a].DType != in.DType {
				return bad("argument dtype")
			}
		}
		switch in.Op {
		case backend.OpExp2, backend.OpLog2, backend.OpSin, backend.OpSqrt, backend.OpRecip:
			if in.DType != backend.Float32 {
				return backend.ErrDType
			}
		}
	}
	return nil
}

// indexable reports an error if a kernel cannot index an output of n
// elements or one of the inputs in int.
func indexable(n int, ins []backend.Buffer) error {
	if n > maxElems {
		return fmt.Errorf("%d elements exceed the %d a Metal kernel indexes: %w", n, maxElems, backend.ErrShape)
	}
	for _, b := range ins {
		if b.Len() > maxElems {
			return fmt.Errorf("input of %d elements exceeds the %d a Metal kernel indexes: %w", b.Len(), maxElems, backend.ErrShape)
		}
	}
	return nil
}

// plan returns the dispatch that runs k, or nil if k has no elements.
func plan(k *backend.Kernel, outs, ins []backend.Buffer) (*launch, error) {
	n, _ := backend.NumElements(k.Shape)
	if err := indexable(n, ins); err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	g := newGen(k, outs, ins)
	g.line("if (gid >= uint(%s)) return;", g.param(n))
	g.line("int i = int(gid);")
	switch op := k.Op; {
	case op == backend.OpFused:
		return g.elementwise(n, k.Attrs.Program, k.Attrs.Stores)
	case op.RunsElementwise():
		prog := make([]backend.Instr, len(ins), len(ins)+1)
		args := make([]int, len(ins))
		for i, dt := range g.ins {
			prog[i] = backend.Instr{Op: backend.OpLoad, DType: dt, Args: []int{i}}
			args[i] = i
		}
		if op != backend.OpContiguous {
			prog = append(prog, backend.Instr{Op: op, DType: k.DType, Args: args})
		}
		return g.elementwise(n, prog, nil)
	case op == backend.OpConst:
		g.line("o0[i] = %s;", literal(k.Attrs.Value, k.DType))
		return g.perElement(n)
	case op == backend.OpRandUniform || op == backend.OpRandNormal:
		return g.random(n)
	case op.IsReduce():
		return planReduce(k, outs, ins, n)
	case op == backend.OpPad:
		return g.pad(n)
	case op == backend.OpGather:
		return g.gather(n)
	case op == backend.OpScatterAdd:
		return g.scatterAdd(n)
	case op == backend.OpUnfold:
		return g.unfold(n)
	case op == backend.OpFold:
		return g.fold(n)
	case op == backend.OpMatMul:
		return g.matmul(n)
	}
	return nil, fmt.Errorf("no metal kernel for %s: %w", k.Op, backend.ErrInternal)
}

// elementwise emits a program over inputs read through views of the
// output's shape, writing its value and the values it stores.
func (g *gen) elementwise(n int, prog []backend.Instr, stores []int) (*launch, error) {
	if err := checkProgram(prog, stores, g.k.DType, g.ins); err != nil {
		return nil, err
	}
	for i, v := range g.k.In {
		if !slices.Equal(v.Shape, g.k.Shape) {
			return nil, fmt.Errorf("input %d of shape %v for output %v: %w", i, v.Shape, g.k.Shape, backend.ErrInternal)
		}
	}
	shape, views := backend.Coalesce(g.k.Shape, g.k.In)
	offs := g.offsets("i", shape, views)
	vals := g.program(prog, func(a int) string { return fmt.Sprintf("i%d[%s]", a, offs[a]) })
	g.line("o0[i] = %s;", vals[len(vals)-1])
	for s, at := range stores {
		g.line("o%d[i] = %s;", s+1, vals[at])
	}
	return g.perElement(n)
}

// random fills the output from Philox4x32-10, as the CPU backend does:
// element i comes from word i%4 of block i/4.
func (g *gen) random(n int) (*launch, error) {
	if g.k.DType != backend.Float32 {
		return nil, backend.ErrDType
	}
	seed := g.k.Attrs.Seed
	g.line("uint4 w = uint4(uint(i / 4), 0u, 0u, 0u);")
	g.line("uint2 key = uint2(as_type<uint>(%s), as_type<uint>(%s));", g.param(int(int32(uint32(seed)))), g.param(int(int32(uint32(seed>>32)))))
	g.line("for (int round = 0; round < 10; round++) {")
	g.line("\tif (round > 0) { key.x += 0x9E3779B9u; key.y += 0xBB67AE85u; }")
	g.line("\tuint hi0 = mulhi(0xD2511F53u, w.x), lo0 = 0xD2511F53u * w.x;")
	g.line("\tuint hi1 = mulhi(0xCD9E8D57u, w.z), lo1 = 0xCD9E8D57u * w.z;")
	g.line("\tw = uint4(hi1 ^ w.y ^ key.x, lo1, hi0 ^ w.w ^ key.y, lo0);")
	g.line("}")
	g.line("int lane = i %% 4;")
	if g.k.Op == backend.OpRandUniform {
		g.line("o0[i] = float(w[lane] >> 8) * 5.9604644775390625e-8f;")
		return g.perElement(n)
	}
	g.line("int pair = lane & ~1;")
	g.line("float u1 = float((w[pair] >> 8) + 1u) * 5.9604644775390625e-8f;")
	g.line("float u2 = float(w[pair + 1] >> 8) * 5.9604644775390625e-8f;")
	g.line("float r = sqrt(-2.0f * log(u1));")
	// The angle is 2πu2. Doubling u2 is exact, while multiplying by 2π in
	// float would round the angle by as much as 2.4e-7.
	g.line("o0[i] = (lane & 1) == 0 ? r * cospi(2.0f * u2) : r * sinpi(2.0f * u2);")
	return g.perElement(n)
}

// reduceInit returns the identity of a reduction.
func reduceInit(op backend.Op, dt backend.DType) string {
	switch {
	case op == backend.OpReduceSum:
		return literal(0, dt)
	case dt == backend.Float32:
		return "-INFINITY"
	default:
		return "as_type<int>(0x80000000u)"
	}
}

// reduceStep returns the expression combining acc with x.
func reduceStep(op backend.Op, acc, x string) string {
	if op == backend.OpReduceSum {
		return fmt.Sprintf("%s + %s", acc, x)
	}
	return fmt.Sprintf("mx(%s, %s)", acc, x)
}

// planReduce reduces the input over k.Attrs.Axes. Each output element
// combines its reduced positions in row-major order, one thread per
// element, unless there are few elements that each reduce many positions:
// then a threadgroup computes each, and sums are no longer in the CPU's
// order.
func planReduce(k *backend.Kernel, outs, ins []backend.Buffer, n int) (*launch, error) {
	v := k.In[0]
	if len(v.Shape) != len(k.Shape) || ins[0].DType() != k.DType {
		return nil, fmt.Errorf("reduction of %v to %v: %w", v.Shape, k.Shape, backend.ErrInternal)
	}
	kept := backend.View{Shape: k.Shape, Strides: slices.Clone(v.Strides), Offset: v.Offset}
	var red backend.View
	for d, size := range v.Shape {
		if slices.Contains(k.Attrs.Axes, d) {
			kept.Strides[d] = 0
			red.Shape = append(red.Shape, size)
			red.Strides = append(red.Strides, v.Strides[d])
		} else if size != k.Shape[d] {
			return nil, fmt.Errorf("reduction of %v to %v: %w", v.Shape, k.Shape, backend.ErrInternal)
		}
	}
	rshape, rviews := backend.Coalesce(red.Shape, []backend.View{red})
	red = rviews[0]
	positions, _ := backend.NumElements(rshape)
	kshape, kviews := backend.Coalesce(k.Shape, []backend.View{kept})

	g := newGen(k, outs, ins)
	t := typeName(k.DType)
	init := reduceInit(k.Op, k.DType)
	if positions >= 1024 || positions >= 128 && n < 4096 {
		// A threadgroup per output element.
		g.line("int i = int(bid);")
		base := g.offsets("i", kshape, kviews)[0]
		g.line("%s acc = %s;", t, init)
		g.line("for (int j = int(tid); j < %s; j += %d) {", g.param(positions), groupSize)
		off := g.offsets("j", rshape, []backend.View{red})[0]
		g.line("\tacc = %s;", reduceStep(k.Op, "acc", fmt.Sprintf("i0[%s + %s]", base, off)))
		g.line("}")
		g.line("threadgroup %s part[%d];", t, groupSize)
		g.line("part[tid] = acc;")
		g.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
		g.line("for (uint s = %d; s > 0; s >>= 1) {", groupSize/2)
		g.line("\tif (tid < s) part[tid] = %s;", reduceStep(k.Op, "part[tid]", "part[tid + s]"))
		g.line("\tthreadgroup_barrier(mem_flags::mem_threadgroup);")
		g.line("}")
		g.line("if (tid == 0) o0[i] = part[0];")
		return g.finish(n, true)
	}
	g.line("if (gid >= uint(%s)) return;", g.param(n))
	g.line("int i = int(gid);")
	base := g.offsets("i", kshape, kviews)[0]
	g.line("%s acc = %s;", t, init)
	off := base
	for d, size := range rshape {
		a := g.v("a")
		g.line("for (int %s = 0; %s < %s; %s++) {", a, a, g.param(size), a)
		next := g.v("off")
		g.line("int %s = %s + %s * %s;", next, off, a, g.param(red.Strides[d]))
		off = next
	}
	g.line("acc = %s;", reduceStep(k.Op, "acc", fmt.Sprintf("i0[%s]", off)))
	for range rshape {
		g.line("}")
	}
	g.line("o0[i] = acc;")
	return g.perElement(n)
}

// pad copies the input into the middle of the output and fills the rest.
func (g *gen) pad(n int) (*launch, error) {
	v := g.k.In[0]
	if len(g.k.Attrs.Pads) != len(v.Shape) || len(g.k.Shape) != len(v.Shape) {
		return nil, fmt.Errorf("pad of %v by %v: %w", v.Shape, g.k.Attrs.Pads, backend.ErrInternal)
	}
	coords := g.index("i", g.k.Shape)
	g.line("int off = %s;", g.param(v.Offset))
	g.line("bool inside = true;")
	for d, c := range coords {
		s := g.v("s")
		g.line("int %s = %s - %s;", s, c, g.param(g.k.Attrs.Pads[d][0]))
		g.line("inside = inside && %s >= 0 && %s < %s;", s, s, g.param(v.Shape[d]))
		g.line("off += %s * %s;", s, g.param(v.Strides[d]))
	}
	g.line("o0[i] = inside ? i0[off] : %s;", literal(g.k.Attrs.Value, g.k.DType))
	return g.perElement(n)
}

// gather reads out[m] = src[m with m[axis] = idx[m]].
func (g *gen) gather(n int) (*launch, error) {
	axis := g.k.Attrs.Axis
	vs, vi := g.k.In[0], g.k.In[1]
	if g.ins[1] != backend.Int32 || axis < 0 || axis >= len(g.k.Shape) || len(vs.Shape) != len(g.k.Shape) {
		return nil, fmt.Errorf("gather along axis %d: %w", axis, backend.ErrInternal)
	}
	base := backend.View{Shape: g.k.Shape, Strides: slices.Clone(vs.Strides), Offset: vs.Offset}
	base.Strides[axis] = 0
	shape, views := backend.Coalesce(g.k.Shape, []backend.View{vi, base})
	offs := g.offsets("i", shape, views)
	g.line("int j = i1[%s];", offs[0])
	g.line("if (j < 0 || j >= %s) {", g.param(vs.Shape[axis]))
	g.line("\tatomic_fetch_or_explicit(err, %du, memory_order_relaxed);", errIndex)
	g.line("\to0[i] = 0;")
	g.line("\treturn;")
	g.line("}")
	g.line("o0[i] = i0[%s + j * %s];", offs[1], g.param(vs.Strides[axis]))
	return g.perElement(n)
}

// scatterAdd copies dst into the output and adds each update into
// out[m with m[axis] = idx[m]]. Each output element adds the updates that
// land on it in the order of their position along axis, as the CPU
// backend does, and the threads at position 0 along axis check the
// indices of their column.
func (g *gen) scatterAdd(n int) (*launch, error) {
	axis := g.k.Attrs.Axis
	vd, vi, vu := g.k.In[0], g.k.In[1], g.k.In[2]
	r := len(g.k.Shape)
	if g.ins[1] != backend.Int32 || axis < 0 || axis >= r || len(vd.Shape) != r || len(vi.Shape) != r || !slices.Equal(vi.Shape, vu.Shape) {
		return nil, fmt.Errorf("scatter add along axis %d: %w", axis, backend.ErrInternal)
	}
	coords := g.index("i", g.k.Shape)
	g.line("int dst = %s;", g.param(vd.Offset))
	g.line("int idx = %s;", g.param(vi.Offset))
	g.line("int upd = %s;", g.param(vu.Offset))
	g.line("bool covered = true;")
	for d, c := range coords {
		g.line("dst += %s * %s;", c, g.param(vd.Strides[d]))
		if d == axis {
			continue
		}
		g.line("covered = covered && %s < %s;", c, g.param(vi.Shape[d]))
		g.line("idx += %s * %s;", c, g.param(vi.Strides[d]))
		g.line("upd += %s * %s;", c, g.param(vu.Strides[d]))
	}
	t := typeName(g.k.DType)
	g.line("%s acc = i0[dst];", t)
	g.line("if (covered) {")
	g.line("\tint at = %s;", coords[axis])
	g.line("\tint limit = %s;", g.param(g.k.Shape[axis]))
	g.line("\tint si = %s, su = %s;", g.param(vi.Strides[axis]), g.param(vu.Strides[axis]))
	g.line("\tfor (int j = 0; j < %s; j++) {", g.param(vi.Shape[axis]))
	g.line("\t\tint to = i1[idx + j * si];")
	g.line("\t\tif (at == 0 && (to < 0 || to >= limit)) atomic_fetch_or_explicit(err, %du, memory_order_relaxed);", errIndex)
	g.line("\t\tif (to == at) acc = acc + i2[upd + j * su];")
	g.line("\t}")
	g.line("}")
	g.line("o0[i] = acc;")
	return g.perElement(n)
}

// windows checks the attributes of a sliding-window kernel over s windowed
// dimensions.
func (g *gen) windows() (s int, err error) {
	a := g.k.Attrs
	s = len(a.Kernel)
	if s == 0 || len(a.Stride) != s || len(a.Dilation) != s || len(a.Pads) != s {
		return 0, fmt.Errorf("window %+v: %w", a, backend.ErrInternal)
	}
	return s, nil
}

// unfold copies the sliding windows of the input: element [b, k, o] reads
// input [b, o*stride - pad + k*dilation], or the fill value in the padding.
func (g *gen) unfold(n int) (*launch, error) {
	s, err := g.windows()
	if err != nil {
		return nil, err
	}
	a, v := g.k.Attrs, g.k.In[0]
	r := len(v.Shape)
	if len(g.k.Shape) != r+s || r < s {
		return nil, fmt.Errorf("unfold of %v to %v: %w", v.Shape, g.k.Shape, backend.ErrInternal)
	}
	lead := r - s
	coords := g.index("i", g.k.Shape)
	g.line("int off = %s;", g.param(v.Offset))
	g.line("bool inside = true;")
	for d := range lead {
		g.line("off += %s * %s;", coords[d], g.param(v.Strides[d]))
	}
	for d := range s {
		k, o := coords[lead+d], coords[lead+s+d]
		x := g.v("x")
		g.line("int %s = %s * %s - %s + %s * %s;", x, o, g.param(a.Stride[d]), g.param(a.Pads[d][0]), k, g.param(a.Dilation[d]))
		g.line("inside = inside && %s >= 0 && %s < %s;", x, x, g.param(v.Shape[lead+d]))
		g.line("off += %s * %s;", x, g.param(v.Strides[lead+d]))
	}
	g.line("o0[i] = inside ? i0[off] : %s;", literal(a.Value, g.k.DType))
	return g.perElement(n)
}

// fold sums the windows of the input into the output, dropping the
// padding: output [b, x] adds input [b, k, o] for every kernel offset k,
// in row-major order, whose window position o reads x.
func (g *gen) fold(n int) (*launch, error) {
	s, err := g.windows()
	if err != nil {
		return nil, err
	}
	a, v := g.k.Attrs, g.k.In[0]
	r := len(g.k.Shape)
	if len(v.Shape) != r+s || r < s {
		return nil, fmt.Errorf("fold of %v to %v: %w", v.Shape, g.k.Shape, backend.ErrInternal)
	}
	lead := r - s
	coords := g.index("i", g.k.Shape)
	off := g.v("off")
	g.line("int %s = %s;", off, g.param(v.Offset))
	for d := range lead {
		g.line("%s += %s * %s;", off, coords[d], g.param(v.Strides[d]))
	}
	t := typeName(g.k.DType)
	g.line("%s acc = %s;", t, literal(0, g.k.DType))
	for d := range s {
		k, tt, o := g.v("k"), g.v("t"), g.v("o")
		g.line("for (int %s = 0; %s < %s; %s++) {", k, k, g.param(a.Kernel[d]), k)
		g.line("int %s = %s + %s - %s * %s;", tt, coords[lead+d], g.param(a.Pads[d][0]), k, g.param(a.Dilation[d]))
		stride := g.param(a.Stride[d])
		g.line("if (%s < 0 || %s %% %s != 0) continue;", tt, tt, stride)
		g.line("int %s = %s / %s;", o, tt, stride)
		g.line("if (%s >= %s) continue;", o, g.param(v.Shape[r+d]))
		next := g.v("off")
		g.line("int %s = %s + %s * %s + %s * %s;", next, off, k, g.param(v.Strides[lead+d]), o, g.param(v.Strides[r+d]))
		off = next
	}
	g.line("acc = acc + i0[%s];", off)
	for range s {
		g.line("}")
	}
	g.line("o0[i] = acc;")
	return g.perElement(n)
}

// matmul computes [..., M, K] x [..., K, N] -> [..., M, N], summing the
// products over the batch axes k.Attrs.Axes, one thread per output
// element, and applies the epilogue program, if any, to the result.
func (g *gen) matmul(n int) (*launch, error) {
	k := g.k
	vx, vy := k.In[0], k.In[1]
	r := len(k.Shape)
	if r < 2 || len(vx.Shape) != r || len(vy.Shape) != r || g.ins[0] != k.DType || g.ins[1] != k.DType {
		return nil, fmt.Errorf("matmul of %v and %v: %w", vx.Shape, vy.Shape, backend.ErrInternal)
	}
	m, nn, kk := k.Shape[r-2], k.Shape[r-1], vx.Shape[r-1]
	var keep, sum []int
	for d := range r - 2 {
		if slices.Contains(k.Attrs.Axes, d) {
			sum = append(sum, d)
		} else {
			keep = append(keep, d)
		}
	}
	g.line("int col = i %% %s;", g.param(nn))
	g.line("int row = (i / %s) %% %s;", g.param(nn), g.param(m))
	g.line("int ox = %s + row * %s, oy = %s + col * %s;", g.param(vx.Offset), g.param(vx.Strides[r-2]), g.param(vy.Offset), g.param(vy.Strides[r-1]))
	if len(keep) > 0 {
		shape := make([]int, len(keep))
		for j, d := range keep {
			shape[j] = k.Shape[d]
		}
		coords := g.index(fmt.Sprintf("i / %s", g.param(m*nn)), shape)
		for j, d := range keep {
			g.line("ox += %s * %s; oy += %s * %s;", coords[j], g.param(vx.Strides[d]), coords[j], g.param(vy.Strides[d]))
		}
	}
	t := typeName(k.DType)
	g.line("%s acc = %s;", t, literal(0, k.DType))
	x, y := "ox", "oy"
	for _, d := range sum {
		a, nx, ny := g.v("a"), g.v("x"), g.v("y")
		g.line("for (int %s = 0; %s < %s; %s++) {", a, a, g.param(vx.Shape[d]), a)
		g.line("int %s = %s + %s * %s, %s = %s + %s * %s;", nx, x, a, g.param(vx.Strides[d]), ny, y, a, g.param(vy.Strides[d]))
		x, y = nx, ny
	}
	sxk, syk := g.param(vx.Strides[r-1]), g.param(vy.Strides[r-2])
	g.line("for (int p_ = 0; p_ < %s; p_++) {", g.param(kk))
	if k.DType == backend.Float32 {
		g.line("\tacc = fma(i0[%s + p_ * %s], i1[%s + p_ * %s], acc);", x, sxk, y, syk)
	} else {
		g.line("\tacc = acc + i0[%s + p_ * %s] * i1[%s + p_ * %s];", x, sxk, y, syk)
	}
	g.line("}")
	for range sum {
		g.line("}")
	}
	return g.epilogue(n, "acc")
}

// epilogue writes the product acc, after the epilogue program of the
// product g.k if it has one. The epilogue reads the product as input 0 and
// inputs 2 on as inputs 1 on, through views of a shape that lists the
// output's elements in row-major order.
func (g *gen) epilogue(n int, acc string) (*launch, error) {
	k := g.k
	if k.Attrs.Program == nil {
		g.line("o0[i] = %s;", acc)
		return g.perElement(n)
	}
	// The epilogue's inputs: the product, then ins[2:].
	shape := k.Shape
	if len(k.In) > 2 {
		shape = k.In[2].Shape
	}
	views := k.In[2:]
	for _, v := range views {
		if !slices.Equal(v.Shape, shape) {
			return nil, fmt.Errorf("epilogue inputs of shapes %v and %v: %w", shape, v.Shape, backend.ErrInternal)
		}
	}
	if e, _ := backend.NumElements(shape); e != n {
		return nil, fmt.Errorf("epilogue of shape %v for %d elements: %w", shape, n, backend.ErrInternal)
	}
	types := append([]backend.DType{k.DType}, g.ins[2:]...)
	if err := checkProgram(k.Attrs.Program, nil, k.DType, types); err != nil {
		return nil, err
	}
	cshape, cviews := backend.Coalesce(shape, views)
	offs := g.offsets("i", cshape, cviews)
	vals := g.program(k.Attrs.Program, func(a int) string {
		if a == 0 {
			return acc
		}
		return fmt.Sprintf("i%d[%s]", a+1, offs[a-1])
	})
	g.line("o0[i] = %s;", vals[len(vals)-1])
	return g.perElement(n)
}
