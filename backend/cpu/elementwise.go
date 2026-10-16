package cpu

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

// elementGrain is the smallest chunk of elementwise work worth a goroutine.
const elementGrain = 1 << 15

// Elementwise kernels, fused (backend.OpFused) or single ops, run as
// programs. The output is split into rows along its innermost coalesced
// dimension, and rows into tiles of at most tileLen elements. Each
// instruction runs over a whole tile in a loop of its own, so interpreting
// the program costs a few dispatches per tile, not per element.
//
// Within a row each value is a vector, one element per index, or a scalar,
// the same at every index: a constant, an input broadcast along the row, or
// a value computed from scalars only. Inputs read with stride 1 are used in
// place. Other vectors are computed into scratch slots, which are reused
// once their value is dead, so a program needs only as many tiles of
// scratch as it has vectors live at once. Values the kernel writes are
// computed straight into their output.
//
// Rows shorter than wideRow make for tiles too short to be worth their
// dispatches, so tiles then span rows. An input read at one stride across
// the whole output is still a vector at that stride, or a scalar; any
// other input is gathered into scratch for each tile first. The trailing
// dimensions that fit in a tile make up a block, whose elements every
// gathered input reads at the same offsets from the block's first.
//
// An input may also be an output, as in an in-place update. Each element
// is then read before it is written as long as nothing reads the input
// after the instruction that writes the output; otherwise the input is
// copied into scratch at the start of each tile.
//
// Values of a half type live in float32 lanes: a load converts the input
// into scratch, every other instruction computes in float32 and rounds
// its result to the nearest element of its type, which a store writes.

// tileLen is the number of elements an instruction processes at a time.
const tileLen = 1024

// wideRow is the row length below which tiles span rows.
const wideRow = 64

// program is a fused program compiled for one kernel.
type program struct {
	steps          []step
	order          []int       // the steps in the order they run
	slotsF, slotsI int         // scratch slots of each lane type
	inF            [][]float32 // the inputs by kernel input, in their dtype
	inI            [][]int32
	inH            [][]uint16 // the bits of half inputs
}

// step is one compiled instruction.
type step struct {
	backend.Instr
	slot  int // the scratch slot of a vector value, or -1
	out   int // the output it writes: 0 for the result, i+1 for Stores[i], or -1
	store int // the half output it writes, as out; computed into slot first
}

// lane is a value over one tile: a vector, or a scalar when vec is nil.
type lane[T number] struct {
	vec []T
	s   T
}

// scratch is the memory of one worker running a program.
type scratch struct {
	f      []lane[float32] // values by instruction; each uses one lane
	i      []lane[int32]
	slotsF [][]float32
	slotsI [][]int32
	offs   []int
	cur    cursor
	inF    [][]float32 // the inputs, or their gathered tiles
	inI    [][]int32
	inH    [][]uint16
	wideF  [][]float32 // the memory of inF, inI and inH for wide tiles
	wideI  [][]int32
	wideH  [][]uint16
	tileF  [][]float32 // the gathered tiles of inputs, by input
	tileI  [][]int32
	tileH  [][]uint16
}

// laneOf returns the type of the lanes that hold values of dt: Int32, or
// Float32 for the float types.
func laneOf(dt backend.DType) backend.DType {
	if dt == backend.Int32 {
		return backend.Int32
	}
	return backend.Float32
}

var scratchPool = sync.Pool{New: func() any { return new(scratch) }}

// floatOnly reports whether op is defined for Float32 only.
func floatOnly(op backend.Op) bool {
	switch op {
	case backend.OpExp2, backend.OpLog2, backend.OpSin, backend.OpSqrt, backend.OpRecip:
		return true
	}
	return false
}

// singleOp returns the program of a kernel that applies one elementwise op
// to its inputs, or copies its input for OpContiguous.
func singleOp(k *backend.Kernel, ins []backend.Buffer) []backend.Instr {
	prog := make([]backend.Instr, len(ins), len(ins)+1)
	args := make([]int, len(ins))
	for i, buf := range ins {
		prog[i] = backend.Instr{Op: backend.OpLoad, DType: buf.DType(), Args: []int{i}}
		args[i] = i
	}
	if k.Op == backend.OpContiguous {
		return prog
	}
	return append(prog, backend.Instr{Op: k.Op, DType: k.DType, Args: args})
}

// compile checks prog and stores against its inputs and output dtype and
// assigns scratch slots. strides holds the innermost stride of each input
// after coalescing; an input with stride 0 is a scalar in every row.
// aliasOf holds, for each input, the output it is too, or -1.
func compile(out backend.DType, prog []backend.Instr, stores []int, ins []backend.Buffer, aliasOf, strides []int) (*program, error) {
	if len(prog) == 0 || prog[len(prog)-1].DType != out {
		return nil, fmt.Errorf("program of %d instructions for %s output: %w", len(prog), out, backend.ErrInternal)
	}
	last := len(prog) - 1
	outOf := make([]int, len(prog))
	for j := range outOf {
		outOf[j] = -1
	}
	outOf[last] = 0
	writer := []int{last} // the instruction that writes each output
	for i, j := range stores {
		// A stored load would write its output wherever the load runs,
		// which may be before other inputs of that output are read.
		if j < 0 || j >= last || outOf[j] >= 0 || prog[j].DType != out || prog[j].Op == backend.OpLoad {
			return nil, fmt.Errorf("store %d of instruction %d: %w", i, j, backend.ErrInternal)
		}
		outOf[j] = i + 1
		writer = append(writer, j)
	}
	p := &program{steps: make([]step, len(prog)), inF: make([][]float32, len(ins)), inI: make([][]int32, len(ins)), inH: make([][]uint16, len(ins))}
	for i, buf := range ins {
		switch buf := buf.(type) {
		case *buffer[float32]:
			p.inF[i] = buf.data
		case *buffer[int32]:
			p.inI[i] = buf.data
		case *buffer[bf16]:
			p.inH[i] = bitsOf(buf.data)
		case *buffer[f16]:
			p.inH[i] = bitsOf(buf.data)
		default:
			return nil, fmt.Errorf("foreign buffer %T: %w", buf, backend.ErrDType)
		}
	}
	scalar := make([]bool, len(prog))
	lastUse := make([]int, len(prog))
	for j, in := range prog {
		lastUse[j] = j // a value nothing reads dies where it is made
		if err := check(in, j, prog, ins); err != nil {
			return nil, err
		}
		switch in.Op {
		case backend.OpLoad:
			scalar[j] = strides[in.Args[0]] == 0
		case backend.OpConst:
			scalar[j] = true
		default:
			scalar[j] = true
			for _, a := range in.Args {
				scalar[j] = scalar[j] && scalar[a]
				lastUse[a] = j
			}
		}
	}
	// Run first the loads of inputs that are read after their output is
	// written, copying them; then the rest in program order. Last uses
	// become positions in that order.
	hoist := make([]bool, len(prog))
	for j, in := range prog {
		if in.Op == backend.OpLoad {
			o := aliasOf[in.Args[0]]
			hoist[j] = o >= 0 && lastUse[j] > writer[o]
		}
	}
	p.order = make([]int, 0, len(prog))
	for _, first := range []bool{true, false} {
		for j := range prog {
			if hoist[j] == first {
				p.order = append(p.order, j)
			}
		}
	}
	pos := make([]int, len(prog))
	for at, j := range p.order {
		pos[j] = at
	}
	for j := range lastUse {
		lastUse[j] = pos[lastUse[j]]
	}

	// Free the slots of arguments at their last use before taking one for
	// the result, which may then overwrite an argument in place: every loop
	// reads index i of its arguments before writing index i of its result.
	free := map[backend.DType][]int{} // free slots by lane type
	slot := func(dt backend.DType) int {
		dt = laneOf(dt)
		if f := free[dt]; len(f) > 0 {
			free[dt] = f[:len(f)-1]
			return f[len(f)-1]
		}
		if dt == backend.Float32 {
			p.slotsF++
			return p.slotsF - 1
		}
		p.slotsI++
		return p.slotsI - 1
	}
	released := make([]bool, len(prog))
	release := func(j int) {
		if s := p.steps[j].slot; s >= 0 && !released[j] {
			dt := laneOf(prog[j].DType)
			free[dt] = append(free[dt], s)
			released[j] = true
		}
	}
	for at, j := range p.order {
		in := prog[j]
		for _, a := range in.Args {
			if in.Op != backend.OpLoad && lastUse[a] == at {
				release(a)
			}
		}
		p.steps[j] = step{Instr: in, slot: -1, out: outOf[j], store: -1}
		half := in.DType.Size() == 2
		if half {
			// A half value is computed in a slot and stored from there;
			// a half load converts into its slot, or writes its output.
			p.steps[j].out, p.steps[j].store = -1, outOf[j]
			if in.Op == backend.OpLoad && outOf[j] >= 0 || scalar[j] {
				p.steps[j].slot = -1
			} else {
				p.steps[j].slot = slot(in.DType)
			}
		} else {
			inPlace := in.Op == backend.OpLoad && strides[in.Args[0]] == 1 && !hoist[j]
			if outOf[j] < 0 && !scalar[j] && !inPlace {
				p.steps[j].slot = slot(in.DType)
			}
		}
		if lastUse[j] == at {
			release(j)
		}
	}
	return p, nil
}

// check verifies that instruction j of prog is well formed.
func check(in backend.Instr, j int, prog []backend.Instr, ins []backend.Buffer) error {
	bad := func(what string) error {
		return fmt.Errorf("instruction %d (%s): %s: %w", j, in.Op, what, backend.ErrInternal)
	}
	switch in.DType {
	case backend.Float32, backend.Int32, backend.BFloat16, backend.Float16:
	default:
		return backend.ErrDType
	}
	if in.Op == backend.OpLoad {
		if len(in.Args) != 1 || in.Args[0] < 0 || in.Args[0] >= len(ins) {
			return bad("no such input")
		}
		if ins[in.Args[0]].DType() != in.DType {
			return bad("input dtype")
		}
		return nil
	}
	want := 0
	switch op := in.Op; {
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
	if floatOnly(in.Op) && !in.DType.IsFloat() {
		return backend.ErrDType
	}
	return nil
}

// runProgram computes the elementwise program prog into outs[0], and the
// values it stores into the other outputs.
func runProgram(ctx context.Context, b *Backend, k *backend.Kernel, prog []backend.Instr, stores []int, outs, ins []backend.Buffer) error {
	e, err := newEvaluator(k.Shape, k.In, prog, stores, outs, ins)
	if err != nil || e == nil {
		return err
	}
	total, _ := backend.NumElements(k.Shape)
	return b.parallelFor(ctx, total, elementGrain, func(lo, hi int) error {
		s := scratchPool.Get().(*scratch)
		defer scratchPool.Put(s)
		return e.run(s, lo, hi)
	})
}

// evaluator computes an elementwise program over ranges of its output.
type evaluator struct {
	p       *program
	shape   []int          // the coalesced index space, of rank 1 or more
	outer   []backend.View // the views without their innermost dimension
	inner   []int          // the innermost stride of each view
	strides []int          // the stride of each input in a tile: inner, or for wide tiles see wide
	outF    [][]float32    // the outputs, in their dtype
	outI    [][]int32
	outH    [][]uint16 // the bits of half outputs

	// Wide tiles span rows. An input read at one stride across the index
	// space starts at base; the others are gathered, block by block, from
	// the offsets in pat.
	wide   bool
	block  int     // the elements of a block
	base   []int   // by input
	gather []int   // the gathered inputs
	pat    [][]int // by gathered input
}

// newEvaluator prepares prog to compute outs[0], indexed by shape, and the
// values stores lists into the other outputs, from ins read through views
// of shape. It returns nil if shape is empty.
func newEvaluator(shape []int, views []backend.View, prog []backend.Instr, stores []int, outs, ins []backend.Buffer) (*evaluator, error) {
	shape, views = backend.Coalesce(shape, views)
	total, _ := backend.NumElements(shape)
	if total == 0 {
		return nil, nil
	}
	if len(shape) == 0 {
		shape = []int{1}
		for v := range views {
			views[v].Strides = []int{0}
		}
	}
	r := len(shape)
	e := &evaluator{shape: shape, outer: make([]backend.View, len(views)), inner: make([]int, len(views)), wide: r > 1 && shape[r-1] < wideRow}
	outer := r - 1 // the dimensions the cursor walks
	if e.wide {
		e.block = 1
		for outer = r; outer > 0 && e.block*shape[outer-1] <= tileLen; outer-- {
			e.block *= shape[outer-1]
		}
	}
	for v, view := range views {
		e.outer[v] = backend.View{Strides: view.Strides[:outer], Offset: view.Offset}
		e.inner[v] = view.Strides[r-1]
	}
	e.strides = e.inner
	if e.wide {
		e.strides, e.base = make([]int, len(views)), make([]int, len(views))
		for v, view := range views {
			if linear(shape, view.Strides) {
				e.strides[v], e.base[v] = view.Strides[r-1], view.Offset
				continue
			}
			e.strides[v] = 1
			e.gather = append(e.gather, v)
			pat := make([]int, 0, e.block)
			walk(backend.View{Shape: shape[outer:], Strides: view.Strides[outer:]}, func(off int) { pat = append(pat, off) })
			e.pat = append(e.pat, pat)
		}
	}
	aliasOf := make([]int, len(ins))
	for i, in := range ins {
		aliasOf[i] = slices.Index(outs, in)
	}
	var err error
	if e.p, err = compile(outs[0].DType(), prog, stores, ins, aliasOf, e.strides); err != nil {
		return nil, err
	}
	for _, out := range outs {
		switch out := out.(type) {
		case *buffer[float32]:
			e.outF = append(e.outF, out.data)
		case *buffer[int32]:
			e.outI = append(e.outI, out.data)
		case *buffer[bf16]:
			e.outH = append(e.outH, bitsOf(out.data))
		case *buffer[f16]:
			e.outH = append(e.outH, bitsOf(out.data))
		default:
			return nil, fmt.Errorf("foreign buffer %T: %w", out, backend.ErrDType)
		}
	}
	return e, nil
}

// apply computes output elements [lo, hi) with pooled scratch memory. It
// does nothing if e is nil, for kernels without an epilogue.
func (e *evaluator) apply(lo, hi int) error {
	if e == nil || lo == hi {
		return nil
	}
	s := scratchPool.Get().(*scratch)
	defer scratchPool.Put(s)
	return e.run(s, lo, hi)
}

// run computes output elements [lo, hi) with the scratch memory s.
func (e *evaluator) run(s *scratch, lo, hi int) error {
	s.size(e.p, len(e.strides))
	s.inF, s.inI, s.inH = e.p.inF, e.p.inI, e.p.inH
	if e.wide {
		return e.runWide(s, lo, hi)
	}
	r := len(e.shape)
	inner := e.shape[r-1]
	c := &s.cur
	c.reset(e.shape[:r-1], e.outer, lo/inner)
	for i := lo; i < hi; c.next() {
		col := i % inner
		n := min(inner-col, hi-i)
		for t := 0; t < n; t += tileLen {
			m := min(tileLen, n-t)
			for v := range s.offs {
				s.offs[v] = c.offs[v] + (col+t)*e.strides[v]
			}
			if err := e.p.tile(s, e.strides, m, e, i+t); err != nil {
				return err
			}
		}
		i += n
	}
	return nil
}

// runWide computes output elements [lo, hi) in tiles that span rows.
func (e *evaluator) runWide(s *scratch, lo, hi int) error {
	s.wideF = append(s.wideF[:0], e.p.inF...)
	s.wideI = append(s.wideI[:0], e.p.inI...)
	s.wideH = append(s.wideH[:0], e.p.inH...)
	s.inF, s.inI, s.inH = s.wideF, s.wideI, s.wideH
	for _, v := range e.gather {
		switch {
		case e.p.inF[v] != nil:
			s.tileF = grow(s.tileF, v+1)
			s.tileF[v] = grow(s.tileF[v], tileLen)
			s.inF[v] = s.tileF[v]
		case e.p.inH[v] != nil:
			s.tileH = grow(s.tileH, v+1)
			s.tileH[v] = grow(s.tileH[v], tileLen)
			s.inH[v] = s.tileH[v]
		default:
			s.tileI = grow(s.tileI, v+1)
			s.tileI[v] = grow(s.tileI[v], tileLen)
			s.inI[v] = s.tileI[v]
		}
	}
	c := &s.cur
	c.reset(e.shape[:len(e.outer[0].Strides)], e.outer, lo/e.block)
	j := lo % e.block // the position in the block
	for i := lo; i < hi; {
		m := min(tileLen, hi-i)
		for k := 0; k < m; {
			n := min(e.block-j, m-k)
			for g, v := range e.gather {
				pat := e.pat[g][j : j+n]
				switch {
				case e.p.inF[v] != nil:
					gatherAt(s.inF[v][k:k+n], e.p.inF[v], c.offs[v], pat)
				case e.p.inH[v] != nil:
					gatherAt(s.inH[v][k:k+n], e.p.inH[v], c.offs[v], pat)
				default:
					gatherAt(s.inI[v][k:k+n], e.p.inI[v], c.offs[v], pat)
				}
			}
			k += n
			if j += n; j == e.block {
				j = 0
				c.next()
			}
		}
		for v, b := range e.base {
			s.offs[v] = b + i*e.strides[v]
		}
		for _, v := range e.gather {
			s.offs[v] = 0
		}
		if err := e.p.tile(s, e.strides, m, e, i); err != nil {
			return err
		}
		i += m
	}
	return nil
}

// linear reports whether strides read the elements of shape at one
// stride, that of the innermost dimension.
func linear(shape, strides []int) bool {
	for d := len(shape) - 2; d >= 0; d-- {
		if strides[d] != strides[d+1]*shape[d+1] {
			return false
		}
	}
	return true
}

// gatherAt sets dst to the elements of src at off plus each offset in pat.
func gatherAt[T any](dst, src []T, off int, pat []int) {
	for i, p := range pat {
		dst[i] = src[off+p]
	}
}

// size makes s hold the values and slots of p over inputs inputs.
func (s *scratch) size(p *program, inputs int) {
	s.f = grow(s.f, len(p.steps))
	s.i = grow(s.i, len(p.steps))
	s.offs = grow(s.offs, inputs)
	for len(s.slotsF) < p.slotsF {
		s.slotsF = append(s.slotsF, make([]float32, tileLen))
	}
	for len(s.slotsI) < p.slotsI {
		s.slotsI = append(s.slotsI, make([]int32, tileLen))
	}
}

// tile runs the program over the n elements at output index at, whose
// inputs start at s.offs in s.inF, s.inI and s.inH. Whichever of e.outF,
// e.outI and e.outH holds the outputs is written.
func (p *program) tile(s *scratch, strides []int, n int, e *evaluator, at int) error {
	for _, j := range p.order {
		var err error
		switch dt := p.steps[j].DType; {
		case dt.Size() == 2:
			err = p.runHalf(s, j, strides, n, e.outH, at)
		case dt == backend.Float32:
			err = run1(p, j, s.f, s.i, s.inF, s.slotsF, s.offs, strides, n, e.outF, at)
		default:
			err = run1(p, j, s.i, s.f, s.inI, s.slotsI, s.offs, strides, n, e.outI, at)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// runHalf computes value j, of a half type, into the float32 lane
// s.f[j], over the n elements at output index at: a load converts the
// input, or moves it to the output it writes; any other instruction
// computes in float32 and rounds its result to the nearest element of its
// type, which a store writes to its output in outs.
func (p *program) runHalf(s *scratch, j int, strides []int, n int, outs [][]uint16, at int) error {
	st := &p.steps[j]
	dst := &s.f[j]
	var buf []float32
	if st.slot >= 0 {
		buf = s.slotsF[st.slot][:n]
	}
	var out []uint16
	if st.store >= 0 {
		out = outs[st.store][at : at+n]
	}
	if st.Op == backend.OpLoad {
		a := st.Args[0]
		src, off, stride := s.inH[a], s.offs[a], strides[a]
		switch {
		case out != nil:
			for i := range out {
				out[i] = src[off]
				off += stride
			}
		case stride == 0:
			*dst = lane[float32]{s: halfToF32(st.DType, src[off])}
		default:
			loadHalf(st.DType, buf, src, off, stride)
			*dst = lane[float32]{vec: buf}
		}
		return nil
	}
	if err := run1(p, j, s.f, s.i, s.inF, s.slotsF, s.offs, strides, n, nil, at); err != nil {
		return err
	}
	if dst.vec == nil {
		h := f32ToHalf(st.DType, dst.s)
		dst.s = halfToF32(st.DType, h)
		for i := range out {
			out[i] = h
		}
		return nil
	}
	roundHalf(st.DType, dst.vec, out)
	return nil
}

// run1 computes value j, of lane type T, into lanes[j], over the n
// elements at output index at. other holds the values of the other lane
// type, which casts and conditions may read. A value the kernel writes
// goes to its output in outs.
func run1[T, U number](p *program, j int, lanes []lane[T], other []lane[U], in [][]T, slots [][]T, offs, strides []int, n int, outs [][]T, at int) error {
	st := &p.steps[j]
	var buf []T // where a vector result goes
	switch {
	case st.out >= 0:
		buf = outs[st.out][at : at+n]
	case st.slot >= 0:
		buf = slots[st.slot][:n]
	}
	dst := &lanes[j]
	switch op := st.Op; {
	case op == backend.OpLoad:
		a := st.Args[0]
		src, off, stride := in[a], offs[a], strides[a]
		switch stride {
		case 0:
			*dst = lane[T]{s: src[off]}
		case 1:
			if buf == nil {
				*dst = lane[T]{vec: src[off : off+n]}
				break
			}
			copy(buf, src[off:off+n])
			*dst = lane[T]{vec: buf}
		default:
			for i := range buf {
				buf[i] = src[off]
				off += stride
			}
			*dst = lane[T]{vec: buf}
		}
	case op == backend.OpConst:
		*dst = lane[T]{s: T(st.Value)}
	case op == backend.OpCast:
		a := st.Args[0]
		if laneOf(p.steps[a].DType) == laneOf(st.DType) {
			copyLane(dst, buf, lanes[a])
		} else {
			castLane(dst, buf, other[a])
		}
	case op.IsUnary():
		x := lanes[st.Args[0]]
		if x.vec == nil {
			var r, a [1]T
			a[0] = x.s
			unaryLoop(op, r[:], a[:])
			*dst = lane[T]{s: r[0]}
			break
		}
		unaryLoop(op, buf, x.vec)
		*dst = lane[T]{vec: buf}
	case op.IsBinary():
		if err := binaryLane(op, dst, buf, lanes[st.Args[0]], lanes[st.Args[1]]); err != nil {
			return err
		}
	default: // OpWhere
		c, x, y := st.Args[0], lanes[st.Args[1]], lanes[st.Args[2]]
		if laneOf(p.steps[c].DType) == laneOf(st.DType) {
			whereLane(dst, buf, lanes[c], x, y)
		} else {
			whereLane(dst, buf, other[c], x, y)
		}
	}
	if st.out >= 0 {
		switch {
		case dst.vec == nil:
			for i := range buf {
				buf[i] = dst.s
			}
		case &dst.vec[0] != &buf[0]:
			copy(buf, dst.vec)
		}
	}
	return nil
}

// copyLane sets dst to x, copying a vector into buf.
func copyLane[T number](dst *lane[T], buf []T, x lane[T]) {
	if x.vec == nil {
		*dst = x
		return
	}
	copy(buf, x.vec)
	*dst = lane[T]{vec: buf}
}

// castLane sets dst to x converted to T, like a Go conversion.
func castLane[T, U number](dst *lane[T], buf []T, x lane[U]) {
	if x.vec == nil {
		*dst = lane[T]{s: T(x.s)}
		return
	}
	for i, v := range x.vec[:len(buf)] {
		buf[i] = T(v)
	}
	*dst = lane[T]{vec: buf}
}

// whereLane sets dst to x where c is non-zero and to y elsewhere.
func whereLane[C, T number](dst *lane[T], buf []T, c lane[C], x, y lane[T]) {
	if c.vec == nil {
		if c.s != 0 {
			copyLane(dst, buf, x)
		} else {
			copyLane(dst, buf, y)
		}
		return
	}
	cv := c.vec[:len(buf)]
	switch {
	case x.vec != nil && y.vec != nil:
		xv, yv := x.vec[:len(buf)], y.vec[:len(buf)]
		for i, ci := range cv {
			v := yv[i]
			if ci != 0 {
				v = xv[i]
			}
			buf[i] = v
		}
	case x.vec != nil:
		xv := x.vec[:len(buf)]
		for i, ci := range cv {
			v := y.s
			if ci != 0 {
				v = xv[i]
			}
			buf[i] = v
		}
	case y.vec != nil:
		yv := y.vec[:len(buf)]
		for i, ci := range cv {
			v := yv[i]
			if ci != 0 {
				v = x.s
			}
			buf[i] = v
		}
	default:
		for i, ci := range cv {
			v := y.s
			if ci != 0 {
				v = x.s
			}
			buf[i] = v
		}
	}
	*dst = lane[T]{vec: buf}
}

// unaryLoop sets dst[i] = op(a[i]).
func unaryLoop[T number](op backend.Op, dst, a []T) {
	a = a[:len(dst)]
	switch op {
	case backend.OpNeg:
		for i, x := range a {
			dst[i] = -x
		}
	case backend.OpRecip:
		for i, x := range a {
			dst[i] = 1 / x
		}
	case backend.OpSqrt:
		for i, x := range a {
			dst[i] = T(math.Sqrt(float64(x)))
		}
	case backend.OpExp2:
		for i, x := range a {
			dst[i] = T(math.Exp2(float64(x)))
		}
	case backend.OpLog2:
		for i, x := range a {
			dst[i] = T(math.Log2(float64(x)))
		}
	case backend.OpSin:
		for i, x := range a {
			dst[i] = T(math.Sin(float64(x)))
		}
	}
}

// binaryLane sets dst to op applied to x and y, a vector in buf unless both
// are scalars.
func binaryLane[T number](op backend.Op, dst *lane[T], buf []T, x, y lane[T]) error {
	var err error
	switch {
	case x.vec == nil && y.vec == nil:
		var r, a, b [1]T
		a[0], b[0] = x.s, y.s
		err = binaryVV(op, r[:], a[:], b[:])
		*dst = lane[T]{s: r[0]}
		return err
	case x.vec == nil:
		err = binarySV(op, buf, x.s, y.vec)
	case y.vec == nil:
		err = binaryVS(op, buf, x.vec, y.s)
	default:
		err = binaryVV(op, buf, x.vec, y.vec)
	}
	*dst = lane[T]{vec: buf}
	return err
}

// binaryVV sets dst[i] = a[i] op b[i].
func binaryVV[T number](op backend.Op, dst, a, b []T) error {
	a, b = a[:len(dst)], b[:len(dst)]
	switch op {
	case backend.OpAdd:
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	case backend.OpMul:
		for i := range dst {
			dst[i] = a[i] * b[i]
		}
	case backend.OpMax:
		for i := range dst {
			dst[i] = maxOf(a[i], b[i])
		}
	case backend.OpCmpLT:
		for i := range dst {
			dst[i] = boolTo[T](a[i] < b[i])
		}
	case backend.OpCmpEQ:
		for i := range dst {
			dst[i] = boolTo[T](a[i] == b[i])
		}
	case backend.OpMod:
		for i := range dst {
			r, err := mod(a[i], b[i])
			if err != nil {
				return err
			}
			dst[i] = r
		}
	}
	return nil
}

// binaryVS sets dst[i] = a[i] op s.
func binaryVS[T number](op backend.Op, dst, a []T, s T) error {
	a = a[:len(dst)]
	switch op {
	case backend.OpAdd:
		for i, x := range a {
			dst[i] = x + s
		}
	case backend.OpMul:
		for i, x := range a {
			dst[i] = x * s
		}
	case backend.OpMax:
		for i, x := range a {
			dst[i] = maxOf(x, s)
		}
	case backend.OpCmpLT:
		for i, x := range a {
			dst[i] = boolTo[T](x < s)
		}
	case backend.OpCmpEQ:
		for i, x := range a {
			dst[i] = boolTo[T](x == s)
		}
	case backend.OpMod:
		for i, x := range a {
			r, err := mod(x, s)
			if err != nil {
				return err
			}
			dst[i] = r
		}
	}
	return nil
}

// binarySV sets dst[i] = s op b[i].
func binarySV[T number](op backend.Op, dst []T, s T, b []T) error {
	switch op {
	case backend.OpCmpLT:
		b = b[:len(dst)]
		for i, x := range b {
			dst[i] = boolTo[T](s < x)
		}
	case backend.OpMod:
		b = b[:len(dst)]
		for i, x := range b {
			r, err := mod(s, x)
			if err != nil {
				return err
			}
			dst[i] = r
		}
	default: // commutative
		return binaryVS(op, dst, b, s)
	}
	return nil
}

// mod returns the truncated remainder of a/b with the sign of a, like C
// fmod. Float modulo by zero is NaN; integer modulo by zero is an error.
func mod[T number](a, b T) (T, error) {
	if dtypeOf[T]() == backend.Float32 {
		return T(math.Mod(float64(a), float64(b))), nil
	}
	if b == 0 {
		return 0, backend.ErrDivideByZero
	}
	return T(int32(a) % int32(b)), nil
}

// maxOf returns the larger of a and b, propagating NaN like PyTorch. The
// builtin compiles to branch-free code, which matters when the comparison
// is unpredictable, as in ReLU and max pooling.
func maxOf[T number](a, b T) T { return max(a, b) }

// boolTo returns 1 for true and 0 for false. Reading the bool's byte
// avoids a branch per element, which mispredicts on random data.
func boolTo[T number](v bool) T { return T(*(*uint8)(unsafe.Pointer(&v))) }
