package cpu

import (
	"runtime"
	"slices"
	"sync"
	"weak"

	"github.com/gsoultan/mimir/backend"
)

// A program that a schedule replays runs the same kernels over and over,
// each time with new buffers and perhaps new constants. Preparing an
// evaluator, coalescing the index space and compiling the program, costs
// about as much as running a small kernel, so the evaluator of each
// kernel is kept, without its buffers, for as long as the kernel lives,
// and each run binds its buffers to a copy. A kept evaluator is reused
// only while what it was made from still holds: the program's
// instructions but for their constants, which each run takes anew, the
// shape and views, the input dtypes and which inputs are outputs too.

// evalKey names a kept evaluator: its kernel, and which of the kernel's
// evaluators it is, as a product's epilogue is one of its own.
type evalKey struct {
	k    weak.Pointer[backend.Kernel]
	kind uint8
}

const (
	evalProgram uint8 = iota
	evalEpilogue
)

// kept is an evaluator without its buffers and what it was made from.
type kept struct {
	e      *evaluator
	shape  []int
	views  []backend.View
	prog   []backend.Instr
	stores []int
	dtypes []backend.DType
	alias  []int
	outs   []backend.DType
}

var evals sync.Map // evalKey → *kept

// cachedEvaluator is newEvaluator for the kernel k, reusing the evaluator
// made the last time k ran when it still fits.
func cachedEvaluator(k *backend.Kernel, kind uint8, shape []int, views []backend.View, prog []backend.Instr, stores []int, outs, ins []backend.Buffer) (*evaluator, error) {
	key := evalKey{weak.Make(k), kind}
	alias := make([]int, len(ins))
	for i, in := range ins {
		alias[i] = slices.Index(outs, in)
	}
	if v, ok := evals.Load(key); ok {
		if c := v.(*kept); c.fits(shape, views, prog, stores, outs, ins, alias) {
			return c.bind(prog, outs, ins)
		}
	}
	e, err := newEvaluator(shape, views, prog, stores, outs, ins)
	if err != nil || e == nil {
		return e, err
	}
	c := &kept{e: e, shape: slices.Clone(shape), views: cloneViews(views), prog: cloneProg(prog), stores: slices.Clone(stores), alias: alias}
	for _, in := range ins {
		c.dtypes = append(c.dtypes, in.DType())
	}
	for _, out := range outs {
		c.outs = append(c.outs, out.DType())
	}
	if _, loaded := evals.Swap(key, c); !loaded {
		runtime.AddCleanup(k, func(key evalKey) { evals.Delete(key) }, key)
	}
	return e, nil
}

// fits reports whether c was made from what the arguments hold, but for
// buffers and constants.
func (c *kept) fits(shape []int, views []backend.View, prog []backend.Instr, stores []int, outs, ins []backend.Buffer, alias []int) bool {
	if !slices.Equal(c.shape, shape) || !slices.Equal(c.stores, stores) || !slices.Equal(c.alias, alias) ||
		len(c.views) != len(views) || len(c.prog) != len(prog) || len(c.dtypes) != len(ins) || len(c.outs) != len(outs) {
		return false
	}
	for i, v := range views {
		w := c.views[i]
		if w.Offset != v.Offset || !slices.Equal(w.Shape, v.Shape) || !slices.Equal(w.Strides, v.Strides) {
			return false
		}
	}
	for i, in := range prog {
		p := c.prog[i]
		if p.Op != in.Op || p.DType != in.DType || !slices.Equal(p.Args, in.Args) {
			return false
		}
	}
	for i, in := range ins {
		if c.dtypes[i] != in.DType() {
			return false
		}
	}
	for i, out := range outs {
		if c.outs[i] != out.DType() {
			return false
		}
	}
	return true
}

// bind returns a copy of c's evaluator that computes over outs from ins,
// with the constants of prog.
func (c *kept) bind(prog []backend.Instr, outs, ins []backend.Buffer) (*evaluator, error) {
	e := *c.e
	p := *e.p
	for j, in := range prog {
		if in.Op == backend.OpConst && p.steps[j].Value != in.Value {
			if &p.steps[0] == &c.e.p.steps[0] {
				p.steps = slices.Clone(p.steps)
			}
			p.steps[j].Value = in.Value
		}
	}
	if err := p.bindInputs(ins); err != nil {
		return nil, err
	}
	e.p = &p
	e.outF, e.outI, e.outH = nil, nil, nil
	if err := e.bindOutputs(outs); err != nil {
		return nil, err
	}
	return &e, nil
}

func cloneViews(vs []backend.View) []backend.View {
	out := make([]backend.View, len(vs))
	for i, v := range vs {
		out[i] = backend.View{Shape: slices.Clone(v.Shape), Strides: slices.Clone(v.Strides), Offset: v.Offset}
	}
	return out
}

func cloneProg(prog []backend.Instr) []backend.Instr {
	out := slices.Clone(prog)
	for i := range out {
		out[i].Args = slices.Clone(out[i].Args)
	}
	return out
}
