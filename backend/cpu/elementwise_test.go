package cpu

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// refScalar is one element of a value in the reference evaluator.
type refScalar struct {
	f float32
	i int32
}

// refEval evaluates instruction in at one index, given the values of the
// earlier instructions and of the inputs at that index.
func refEval(in backend.Instr, prog []backend.Instr, vals []refScalar, inputs []refScalar) refScalar {
	float := in.DType == backend.Float32
	arg := func(i int) refScalar { return vals[in.Args[i]] }
	b2f := func(v bool) refScalar {
		if v {
			return refScalar{1, 1}
		}
		return refScalar{}
	}
	switch op := in.Op; op {
	case backend.OpLoad:
		return inputs[in.Args[0]]
	case backend.OpConst:
		return refScalar{float32(in.Value), int32(in.Value)}
	case backend.OpCast:
		x := arg(0)
		if prog[in.Args[0]].DType == backend.Float32 {
			return refScalar{x.f, int32(x.f)}
		}
		return refScalar{float32(x.i), x.i}
	case backend.OpWhere:
		c := arg(0)
		if (prog[in.Args[0]].DType == backend.Float32 && c.f != 0) || (prog[in.Args[0]].DType == backend.Int32 && c.i != 0) {
			return arg(1)
		}
		return arg(2)
	case backend.OpNeg:
		return refScalar{-arg(0).f, -arg(0).i}
	case backend.OpExp2:
		return refScalar{f: float32(math.Exp2(float64(arg(0).f)))}
	case backend.OpLog2:
		return refScalar{f: float32(math.Log2(float64(arg(0).f)))}
	case backend.OpSin:
		return refScalar{f: float32(math.Sin(float64(arg(0).f)))}
	case backend.OpSqrt:
		return refScalar{f: float32(math.Sqrt(float64(arg(0).f)))}
	case backend.OpRecip:
		return refScalar{f: 1 / arg(0).f}
	}
	a, b := arg(0), arg(1)
	switch in.Op {
	case backend.OpAdd:
		return refScalar{a.f + b.f, a.i + b.i}
	case backend.OpMul:
		return refScalar{a.f * b.f, a.i * b.i}
	case backend.OpMax:
		if float {
			if a.f != a.f || b.f > a.f {
				if a.f != a.f {
					return a
				}
				return b
			}
			if b.f != b.f {
				return b
			}
			return a
		}
		return refScalar{i: max(a.i, b.i)}
	case backend.OpCmpLT:
		if float {
			return b2f(a.f < b.f)
		}
		return b2f(a.i < b.i)
	case backend.OpCmpEQ:
		if float {
			return b2f(a.f == b.f)
		}
		return b2f(a.i == b.i)
	default: // OpMod
		if float {
			return refScalar{f: float32(math.Mod(float64(a.f), float64(b.f)))}
		}
		return refScalar{i: a.i % b.i}
	}
}

// randomProgram returns a program of about n instructions over inputs of
// the given dtypes whose last instruction has dtype out.
func randomProgram(rng *rand.Rand, dtypes []backend.DType, n int, out backend.DType) []backend.Instr {
	var prog []backend.Instr
	for i, dt := range dtypes {
		prog = append(prog, backend.Instr{Op: backend.OpLoad, DType: dt, Args: []int{i}})
	}
	prog = append(prog,
		backend.Instr{Op: backend.OpConst, DType: backend.Float32, Value: 0.5},
		backend.Instr{Op: backend.OpConst, DType: backend.Int32, Value: 3},
	)
	of := func(dt backend.DType) int {
		for {
			if i := rng.IntN(len(prog)); prog[i].DType == dt {
				return i
			}
		}
	}
	unary := []backend.Op{backend.OpNeg, backend.OpExp2, backend.OpLog2, backend.OpSin, backend.OpSqrt, backend.OpRecip}
	binary := []backend.Op{backend.OpAdd, backend.OpMul, backend.OpMax, backend.OpCmpLT, backend.OpCmpEQ, backend.OpMod}
	for len(prog) < n {
		dt := backend.Float32
		if rng.IntN(4) == 0 {
			dt = backend.Int32
		}
		var in backend.Instr
		switch rng.IntN(5) {
		case 0:
			op := unary[rng.IntN(len(unary))]
			if dt == backend.Int32 {
				op = backend.OpNeg
			}
			in = backend.Instr{Op: op, DType: dt, Args: []int{of(dt)}}
		case 1, 2:
			op := binary[rng.IntN(len(binary))]
			if dt == backend.Int32 && op == backend.OpMod {
				op = backend.OpAdd // integer modulo by zero is an error
			}
			in = backend.Instr{Op: op, DType: dt, Args: []int{of(dt), of(dt)}}
		case 3:
			from := backend.Float32
			if rng.IntN(2) == 0 {
				from = backend.Int32
			}
			in = backend.Instr{Op: backend.OpCast, DType: dt, Args: []int{of(from)}}
		default:
			cond := backend.Float32
			if rng.IntN(2) == 0 {
				cond = backend.Int32
			}
			in = backend.Instr{Op: backend.OpWhere, DType: dt, Args: []int{of(cond), of(dt), of(dt)}}
		}
		prog = append(prog, in)
	}
	return append(prog, backend.Instr{Op: backend.OpCast, DType: out, Args: []int{of(backend.Float32)}})
}

func TestFusedProgramsMatchReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	// Shapes with short rows run in tiles that span rows.
	shapes := [][]int{{7}, {3, 1100}, {2, 3, 5, 4}, {2500}, {37, 3, 20}, {2, 700, 3}}
	for trial := range 60 {
		shape := shapes[trial%len(shapes)]
		size, _ := backend.NumElements(shape)
		views := stridedViews(shape)
		views = append(views, stridedView{"scalar", backend.View{Shape: shape, Strides: make([]int, len(shape)), Offset: 0}, 1})
		var (
			dtypes []backend.DType
			in     []backend.View
			bufs   []backend.Buffer
			data   [][]refScalar // input elements in row-major order
		)
		for i := range 4 {
			sv := views[rng.IntN(len(views))]
			dt := backend.Float32
			if i == 3 {
				dt = backend.Int32
			}
			f := make([]float32, sv.size)
			n := make([]int32, sv.size)
			for j := range f {
				f[j] = float32(rng.IntN(13)-6) / 2
				n[j] = int32(rng.IntN(9) - 4)
			}
			var buf backend.Buffer = f32buf(f...)
			if dt == backend.Int32 {
				buf = i32buf(n...)
			}
			vals := make([]refScalar, 0, size)
			eachIndex(shape, func(idx []int) {
				off := offsetOf(sv.view, idx)
				vals = append(vals, refScalar{f[off], n[off]})
			})
			dtypes, in, bufs, data = append(dtypes, dt), append(in, sv.view), append(bufs, buf), append(data, vals)
		}
		out := backend.Float32
		if trial%2 == 1 {
			out = backend.Int32
		}
		prog := randomProgram(rng, dtypes, 6+trial, out)
		// Store up to two earlier values of the output's dtype as well.
		var stores []int
		for j := len(prog) - 2; j >= 0 && len(stores) < 2; j-- {
			if prog[j].DType == out && prog[j].Op != backend.OpLoad && rng.IntN(3) == 0 {
				stores = append(stores, j)
			}
		}

		// want holds the result, then each stored value.
		want := make([]refScalar, size*(1+len(stores)))
		vals := make([]refScalar, len(prog))
		inputs := make([]refScalar, len(data))
		for e := range size {
			for i := range data {
				inputs[i] = data[i][e]
			}
			for j, instr := range prog {
				vals[j] = refEval(instr, prog, vals, inputs)
			}
			want[e] = vals[len(prog)-1]
			for i, j := range stores {
				want[(i+1)*size+e] = vals[j]
			}
		}
		for _, workers := range []int{1, 3} {
			name := fmt.Sprintf("trial %d, %v, %d workers, stores %v", trial, shape, workers, stores)
			k := &backend.Kernel{Op: backend.OpFused, DType: out, Shape: shape, In: in, Attrs: backend.Attrs{Program: prog, Stores: stores}}
			if out == backend.Float32 {
				got := runOn[float32](t, workers, k, bufs...)
				for e, w := range want {
					if g := got[e]; math.Float32bits(g) != math.Float32bits(w.f) && !(g != g && w.f != w.f) {
						t.Fatalf("%s: element %d = %v, want %v\nprogram %v", name, e, g, w.f, prog)
					}
				}
				continue
			}
			got := runOn[int32](t, workers, k, bufs...)
			for e, w := range want {
				if got[e] != w.i {
					t.Fatalf("%s: element %d = %v, want %v\nprogram %v", name, e, got[e], w.i, prog)
				}
			}
		}
	}
}

func TestFusedProgramRejectsMalformedInstructions(t *testing.T) {
	load := backend.Instr{Op: backend.OpLoad, DType: backend.Float32, Args: []int{0}}
	for name, prog := range map[string][]backend.Instr{
		"empty":            {},
		"missing input":    {{Op: backend.OpLoad, DType: backend.Float32, Args: []int{1}}},
		"input dtype":      {{Op: backend.OpLoad, DType: backend.Int32, Args: []int{0}}, {Op: backend.OpCast, DType: backend.Float32, Args: []int{0}}},
		"forward argument": {load, {Op: backend.OpAdd, DType: backend.Float32, Args: []int{0, 2}}, load},
		"argument count":   {load, {Op: backend.OpAdd, DType: backend.Float32, Args: []int{0}}},
		"not elementwise":  {load, {Op: backend.OpMatMul, DType: backend.Float32, Args: []int{0, 0}}},
		"output dtype":     {load, {Op: backend.OpCast, DType: backend.Int32, Args: []int{0}}},
	} {
		k := &backend.Kernel{Op: backend.OpFused, DType: backend.Float32, Shape: []int{2}, In: []backend.View{backend.Contiguous([]int{2})}, Attrs: backend.Attrs{Program: prog}}
		if _, err := run[float32](t, k, f32buf(1, 2)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	neg := backend.Instr{Op: backend.OpNeg, DType: backend.Float32, Args: []int{0}}
	cast := backend.Instr{Op: backend.OpCast, DType: backend.Int32, Args: []int{0}}
	for name, c := range map[string]struct {
		prog   []backend.Instr
		stores []int
	}{
		"store of the result": {[]backend.Instr{load, neg}, []int{1}},
		"store out of range":  {[]backend.Instr{load, neg}, []int{-1}},
		"store twice":         {[]backend.Instr{load, neg, neg}, []int{1, 1}},
		"store dtype":         {[]backend.Instr{load, cast, neg}, []int{1}},
		"store of a load":     {[]backend.Instr{load, neg}, []int{0}},
	} {
		k := &backend.Kernel{Op: backend.OpFused, DType: backend.Float32, Shape: []int{2}, In: []backend.View{backend.Contiguous([]int{2})}, Attrs: backend.Attrs{Program: c.prog, Stores: c.stores}}
		if _, err := run[float32](t, k, f32buf(1, 2)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	k := &backend.Kernel{Op: backend.OpNeg, DType: backend.Float32, Shape: []int{2}, In: []backend.View{backend.Contiguous([]int{2})}, Attrs: backend.Attrs{Stores: []int{0}}}
	if _, err := run[float32](t, k, f32buf(1, 2)); err == nil {
		t.Error("stores on a single op: no error")
	}
}

func TestFusedIntegerModByZeroIsAnError(t *testing.T) {
	prog := []backend.Instr{
		{Op: backend.OpLoad, DType: backend.Int32, Args: []int{0}},
		{Op: backend.OpConst, DType: backend.Int32, Value: 0},
		{Op: backend.OpMod, DType: backend.Int32, Args: []int{0, 1}},
	}
	k := &backend.Kernel{Op: backend.OpFused, DType: backend.Int32, Shape: []int{3}, In: []backend.View{backend.Contiguous([]int{3})}, Attrs: backend.Attrs{Program: prog}}
	if _, err := run[int32](t, k, i32buf(1, 2, 3)); !errors.Is(err, backend.ErrDivideByZero) {
		t.Fatalf("got %v, want ErrDivideByZero", err)
	}
}

func TestFusedProgramsUpdateTheirInputsInPlace(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	dtypes := []backend.DType{backend.Float32, backend.Float32, backend.Float32, backend.Int32}
	for trial := range 60 {
		// In odd trials the integer input is broadcast along short rows,
		// which tiles then span. One in three pairs of trials is big enough
		// for the workers below to split it, mid-row.
		shapes := [][]int{{3, 1500}, {150, 30}, {3, 29999}, {3001, 30}}
		shape := shapes[trial%2+trial%6/4*2]
		size, _ := backend.NumElements(shape)
		prog := randomProgram(rng, dtypes, 6+trial%10, backend.Float32)
		var stores []int
		for j := len(prog) - 2; j >= 0 && len(stores) < 2; j-- {
			if prog[j].DType == backend.Float32 && prog[j].Op != backend.OpLoad && rng.IntN(3) == 0 {
				stores = append(stores, j)
			}
		}
		in := make([]backend.View, len(dtypes))
		bufs := make([]backend.Buffer, len(dtypes))
		elems := make([][]refScalar, len(dtypes))
		for i, dt := range dtypes {
			in[i] = backend.Contiguous(shape)
			f := make([]float32, size)
			n := make([]int32, size)
			elems[i] = make([]refScalar, size)
			for j := range f {
				f[j] = float32(rng.IntN(13)-6) / 2
				n[j] = int32(rng.IntN(9) - 4)
				elems[i][j] = refScalar{f[j], n[j]}
			}
			if i == 3 && trial%2 == 1 {
				in[i] = backend.Contiguous([]int{shape[0], 1}).Expand(shape)
				for j := range elems[i] {
					elems[i][j] = refScalar{f[j/shape[1]], n[j/shape[1]]}
				}
			}
			bufs[i] = f32buf(f...)
			if dt == backend.Int32 {
				bufs[i] = i32buf(n...)
			}
		}
		want := make([][]refScalar, 1+len(stores))
		for o := range want {
			want[o] = make([]refScalar, size)
		}
		vals := make([]refScalar, len(prog))
		inputs := make([]refScalar, len(elems))
		for e := range size {
			for i := range elems {
				inputs[i] = elems[i][e]
			}
			for j, instr := range prog {
				vals[j] = refEval(instr, prog, vals, inputs)
			}
			want[0][e] = vals[len(prog)-1]
			for i, j := range stores {
				want[i+1][e] = vals[j]
			}
		}
		// Each output goes over a different float input, or into a new buffer.
		b := New()
		b.workers = 1 + trial%3
		outs := make([]backend.Buffer, 1+len(stores))
		for o, i := range rng.Perm(4)[:len(outs)] {
			if i < 3 {
				outs[o] = bufs[i]
				continue
			}
			var err error
			if outs[o], err = b.Alloc(backend.Float32, size); err != nil {
				t.Fatal(err)
			}
		}
		k := &backend.Kernel{Op: backend.OpFused, DType: backend.Float32, Shape: shape, In: in, Attrs: backend.Attrs{Program: prog, Stores: stores}}
		if err := b.Run(t.Context(), k, outs, bufs); err != nil {
			t.Fatal(err)
		}
		for o, out := range outs {
			got, _ := data[float32](out)
			for e, w := range want[o] {
				if g := got[e]; math.Float32bits(g) != math.Float32bits(w.f) && !(g != g && w.f != w.f) {
					t.Fatalf("trial %d: output %d element %d = %v, want %v\nprogram %v, stores %v", trial, o, e, g, w.f, prog, stores)
				}
			}
		}
	}
}

func TestOnlyElementwiseKernelsUpdateInputsInPlace(t *testing.T) {
	b := New()
	x := f32buf(1, 2, 3, 4)
	square := backend.Contiguous([]int{2, 2})
	for name, k := range map[string]*backend.Kernel{
		"product":    {Op: backend.OpMatMul, DType: backend.Float32, Shape: []int{2, 2}, In: []backend.View{square, square}},
		"transposed": {Op: backend.OpNeg, DType: backend.Float32, Shape: []int{2, 2}, In: []backend.View{square.Permute([]int{1, 0})}},
		"reduction":  {Op: backend.OpReduceSum, DType: backend.Float32, Shape: []int{2, 1}, In: []backend.View{backend.Contiguous([]int{2, 1})}, Attrs: backend.Attrs{Axes: []int{1}}},
	} {
		ins := []backend.Buffer{x, x}[:len(k.In)]
		out := x
		if k.Op == backend.OpReduceSum {
			out = f32buf(0, 0)
			ins = []backend.Buffer{out}
		}
		if err := b.Run(t.Context(), k, []backend.Buffer{out}, ins); !errors.Is(err, backend.ErrInternal) {
			t.Errorf("%s: got %v, want ErrInternal", name, err)
		}
	}
	neg := &backend.Kernel{Op: backend.OpNeg, DType: backend.Float32, Shape: []int{2, 2}, In: []backend.View{square}}
	if err := b.Run(t.Context(), neg, []backend.Buffer{x}, []backend.Buffer{x}); err != nil {
		t.Fatal(err)
	}
	if got, _ := data[float32](x); !slices.Equal(got, []float32{-1, -2, -3, -4}) {
		t.Errorf("negated in place: %v", got)
	}
}
