package cpu

import (
	"context"
	"fmt"
	"math"
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
// scratch as it has vectors live at once. Stored values are computed
// straight into their part of the output.

// tileLen is the number of elements an instruction processes at a time.
const tileLen = 1024

// program is a fused program compiled for one kernel.
type program struct {
	steps          []step
	slotsF, slotsI int         // scratch slots of each dtype
	inF            [][]float32 // the inputs by kernel input, in their dtype
	inI            [][]int32
}

// step is one compiled instruction.
type step struct {
	backend.Instr
	slot int // the scratch slot of a vector value, or -1
	part int // the part of the output it writes: 0 for the result, i+1 for Stores[i], or -1
}

// lane is a value over one tile: a vector, or a scalar when vec is nil.
type lane[T elem] struct {
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
func compile(out backend.DType, prog []backend.Instr, stores []int, ins []backend.Buffer, strides []int) (*program, error) {
	if len(prog) == 0 || prog[len(prog)-1].DType != out {
		return nil, fmt.Errorf("program of %d instructions for %s output: %w", len(prog), out, backend.ErrInternal)
	}
	last := len(prog) - 1
	part := make([]int, len(prog))
	for j := range part {
		part[j] = -1
	}
	part[last] = 0
	for i, j := range stores {
		if j < 0 || j >= last || part[j] >= 0 || prog[j].DType != out {
			return nil, fmt.Errorf("store %d of instruction %d: %w", i, j, backend.ErrInternal)
		}
		part[j] = i + 1
	}
	p := &program{steps: make([]step, len(prog)), inF: make([][]float32, len(ins)), inI: make([][]int32, len(ins))}
	for i, buf := range ins {
		switch buf := buf.(type) {
		case *buffer[float32]:
			p.inF[i] = buf.data
		case *buffer[int32]:
			p.inI[i] = buf.data
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
	// Free the slots of arguments at their last use before taking one for
	// the result, which may then overwrite an argument in place: every loop
	// reads index i of its arguments before writing index i of its result.
	free := map[backend.DType][]int{} // free slots by dtype
	slot := func(dt backend.DType) int {
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
			free[prog[j].DType] = append(free[prog[j].DType], s)
			released[j] = true
		}
	}
	for j, in := range prog {
		for _, a := range in.Args {
			if in.Op != backend.OpLoad && lastUse[a] == j {
				release(a)
			}
		}
		p.steps[j] = step{Instr: in, slot: -1, part: part[j]}
		inPlace := in.Op == backend.OpLoad && strides[in.Args[0]] == 1
		if part[j] < 0 && !scalar[j] && !inPlace {
			p.steps[j].slot = slot(in.DType)
		}
		if lastUse[j] == j {
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
	if in.DType != backend.Float32 && in.DType != backend.Int32 {
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
	if floatOnly(in.Op) && in.DType != backend.Float32 {
		return backend.ErrDType
	}
	return nil
}

// runProgram computes the elementwise program prog, and the values it
// stores, into out.
func runProgram(ctx context.Context, b *Backend, k *backend.Kernel, prog []backend.Instr, stores []int, out backend.Buffer, ins []backend.Buffer) error {
	e, err := newEvaluator(k.Shape, k.In, prog, stores, out, ins)
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
	strides []int          // the innermost stride of each view
	size    int            // the number of elements in shape
	outF    []float32      // the output, in its dtype, with its stored values
	outI    []int32
}

// newEvaluator prepares prog to compute out, indexed by shape, and the
// values stores lists after it, from ins read through views of shape. It
// returns nil if shape is empty.
func newEvaluator(shape []int, views []backend.View, prog []backend.Instr, stores []int, out backend.Buffer, ins []backend.Buffer) (*evaluator, error) {
	shape, views = coalesce(shape, views)
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
	e := &evaluator{shape: shape, outer: make([]backend.View, len(views)), strides: make([]int, len(views)), size: total}
	for v, view := range views {
		e.outer[v] = backend.View{Strides: view.Strides[:r-1], Offset: view.Offset}
		e.strides[v] = view.Strides[r-1]
	}
	var err error
	if e.p, err = compile(out.DType(), prog, stores, ins, e.strides); err != nil {
		return nil, err
	}
	switch out := out.(type) {
	case *buffer[float32]:
		e.outF = out.data
	case *buffer[int32]:
		e.outI = out.data
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
			if err := e.p.tile(s, e.strides, m, e.outF, e.outI, i+t, e.size); err != nil {
				return err
			}
		}
		i += n
	}
	return nil
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
// inputs start at s.offs. Whichever of outF and outI is the output holds
// parts of size elements: the result, then the stored values.
func (p *program) tile(s *scratch, strides []int, n int, outF []float32, outI []int32, at, size int) error {
	for j := range p.steps {
		var err error
		if p.steps[j].DType == backend.Float32 {
			err = run1(p, j, s.f, s.i, p.inF, s.slotsF, s.offs, strides, n, outF, at, size)
		} else {
			err = run1(p, j, s.i, s.f, p.inI, s.slotsI, s.offs, strides, n, outI, at, size)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// run1 computes value j, of type T, into lanes[j], over the n elements
// at output index at. other holds the values of the other dtype, which
// casts and conditions may read. A value the kernel writes goes to its
// part of out, each part holding size elements.
func run1[T, U elem](p *program, j int, lanes []lane[T], other []lane[U], in [][]T, slots [][]T, offs, strides []int, n int, out []T, at, size int) error {
	st := &p.steps[j]
	var buf []T // where a vector result goes
	switch {
	case st.part >= 0:
		at += st.part * size
		buf = out[at : at+n]
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
			*dst = lane[T]{vec: src[off : off+n]}
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
		if p.steps[a].DType == st.DType {
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
		if p.steps[c].DType == st.DType {
			whereLane(dst, buf, lanes[c], x, y)
		} else {
			whereLane(dst, buf, other[c], x, y)
		}
	}
	if st.part >= 0 {
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
func copyLane[T elem](dst *lane[T], buf []T, x lane[T]) {
	if x.vec == nil {
		*dst = x
		return
	}
	copy(buf, x.vec)
	*dst = lane[T]{vec: buf}
}

// castLane sets dst to x converted to T, like a Go conversion.
func castLane[T, U elem](dst *lane[T], buf []T, x lane[U]) {
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
func whereLane[C, T elem](dst *lane[T], buf []T, c lane[C], x, y lane[T]) {
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
func unaryLoop[T elem](op backend.Op, dst, a []T) {
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
func binaryLane[T elem](op backend.Op, dst *lane[T], buf []T, x, y lane[T]) error {
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
func binaryVV[T elem](op backend.Op, dst, a, b []T) error {
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
func binaryVS[T elem](op backend.Op, dst, a []T, s T) error {
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
func binarySV[T elem](op backend.Op, dst []T, s T, b []T) error {
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
func mod[T elem](a, b T) (T, error) {
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
func maxOf[T elem](a, b T) T { return max(a, b) }

// boolTo returns 1 for true and 0 for false. Reading the bool's byte
// avoids a branch per element, which mispredicts on random data.
func boolTo[T elem](v bool) T { return T(*(*uint8)(unsafe.Pointer(&v))) }
