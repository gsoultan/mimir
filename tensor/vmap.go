package tensor

import (
	"errors"
	"fmt"
	"slices"
	"sync/atomic"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// ErrNotBatchable reports a computation that VMap cannot vectorize: one
// that reads the values of a tensor inside the function, as Data, Item,
// Realize and Backward do, or an operation without a batching rule.
var ErrNotBatchable = errors.New("tensor: not batchable")

// A VMapOption configures VMap.
type VMapOption func(*vmapOptions)

type vmapOptions struct {
	in, out []int
}

// InAxes sets the axis of each argument that VMap maps over, one per
// argument; it is 0 for every argument by default. Negative axes count
// from the end.
func InAxes(axes ...int) VMapOption {
	return func(o *vmapOptions) { o.in = slices.Clone(axes) }
}

// OutAxes sets the axis of each result along which VMap stacks the
// examples' results, one per result; it is 0 for every result by
// default. Negative axes count from the end of the stacked result.
func OutAxes(axes ...int) VMapOption {
	return func(o *vmapOptions) { o.out = slices.Clone(axes) }
}

// VMap returns f vectorized over a batch axis, like jax.vmap: a function
// that applies f to the slices of its arguments along their batch axes,
// which must have one length, and stacks the results along theirs.
//
// f runs once, on tensors that stand for one example each: they have the
// example's shape, without the batch axis, and every operation on them
// computes the whole batch at once, with the batch as one more axis.
// Elementwise operations, reductions and movements run on the batch;
// products become batched products, or one larger product when only one
// operand is batched; gathers, scatters, convolutions, scans and
// attention take the batch as a leading axis. Nothing loops over the
// examples.
//
// f may read tensors other than its arguments, such as parameters: they
// are the same for every example. It may also differentiate, which gives
// derivatives for each example: the gradient of the loss of one example
// with respect to the parameters (see Gradients) is the per-example
// gradient. Random tensors inside f are the same for every example.
// Reading values inside f, with Data, Item, Realize or Backward, fails
// with ErrNotBatchable, as does an operation that has no batching rule.
//
// VMap composes with itself, for several batch axes, and with the other
// transformations; its results are lazy and differentiable with respect
// to the arguments and to what f reads.
func VMap(f func(x ...*Tensor) []*Tensor, opts ...VMapOption) func(x ...*Tensor) []*Tensor {
	var o vmapOptions
	for _, opt := range opts {
		opt(&o)
	}
	return func(x ...*Tensor) []*Tensor {
		args, n, err := mapArgs(x, o.in)
		if err != nil {
			return failedCall(f, len(x), err)
		}
		level := args[0].batch().level
		ys := f(args...)
		if o.out != nil && len(o.out) != len(ys) {
			return failAll(len(ys), fmt.Errorf("tensor: VMap with %d out axes for %d results: %w", len(o.out), len(ys), ErrShape))
		}
		out := make([]*Tensor, len(ys))
		for j, y := range ys {
			if err := firstErr(y); err != nil {
				out[j] = fail(err)
				continue
			}
			p := y.physical(level, n)
			axis := 0
			if o.out != nil {
				if axis, err = normAxis(o.out[j], len(p.node.Shape)); err != nil {
					out[j] = fail(err)
					continue
				}
			}
			out[j] = p.moveAxis(0, axis)
		}
		return out
	}
}

// levels counts the calls of VMap functions. A call's count is its
// level: those of nested calls are greater than those of the calls they
// run in.
var levels atomic.Uint64

// mapArgs returns the tensors that stand for one example of each of x,
// mapped over the axes in, or 0, at a new level, and the length of the
// batch.
func mapArgs(x []*Tensor, in []int) ([]*Tensor, int, error) {
	if err := firstErr(x...); err != nil {
		return nil, 0, err
	}
	switch {
	case len(x) == 0:
		return nil, 0, fmt.Errorf("tensor: VMap of no arguments: %w", ErrShape)
	case in != nil && len(in) != len(x):
		return nil, 0, fmt.Errorf("tensor: VMap with %d in axes for %d arguments: %w", len(in), len(x), ErrShape)
	}
	level := levels.Add(1)
	args := make([]*Tensor, len(x))
	n := -1
	for i, xi := range x {
		axis := 0
		if in != nil {
			axis = in[i]
		}
		a, err := normAxis(axis, len(xi.node.Shape))
		if err != nil {
			return nil, 0, err
		}
		if m := xi.node.Shape[a]; n < 0 {
			n = m
		} else if m != n {
			return nil, 0, fmt.Errorf("tensor: VMap over axes of lengths %d and %d: %w", n, m, ErrShape)
		}
		args[i] = newBatched(xi.moveAxis(a, 0), level)
	}
	return args, n, nil
}

// moveAxis returns t with axis from moved to to, as a view.
func (t *Tensor) moveAxis(from, to int) *Tensor {
	if from == to || firstErr(t) != nil {
		return t
	}
	perm := make([]int, 0, len(t.node.Shape))
	for d := range t.node.Shape {
		if d != from {
			perm = append(perm, d)
		}
	}
	return t.permute(slices.Insert(perm, to, from))
}

// batched is a tensor inside VMap: the value of one example, which the
// function sees, standing for the values of the whole batch, which the
// operations on it compute.
type batched struct {
	// node is the example's: a placeholder of an invalid op with its dtype
	// and shape, which operations read to check their arguments and which
	// no graph holds.
	// It is the first field, so that the tensor's node leads back to the
	// batch (see batch).
	node  graph.Node
	phys  *Tensor // the batch: the example's shape after a first batch axis
	level uint64  // of the VMap call the batch axis belongs to
}

// newBatched returns the tensor that stands for one example of phys,
// whose first axis is the batch axis of level.
func newBatched(phys *Tensor, level uint64) *Tensor {
	if err := firstErr(phys); err != nil {
		return fail(err)
	}
	b := &batched{phys: phys, level: level}
	b.node = graph.Node{Op: backend.OpInvalid, DType: phys.node.DType, Placeholder: true, Shape: phys.node.Shape[1:], Dev: sched.Device(phys.node)}
	return &Tensor{node: &b.node}
}

// batch returns the batch t stands for, or nil if t is not inside VMap.
// Only a batched tensor has a placeholder node, which is the first field
// of the batched struct holding it. Unlike its Op, a node's Placeholder
// never changes, so reading it is safe while other goroutines realize the
// node.
func (t *Tensor) batch() *batched {
	if t == nil || t.node == nil || !t.node.Placeholder {
		return nil
	}
	return (*batched)(unsafe.Pointer(t.node))
}

// computable reports ErrNotBatchable for a tensor inside VMap, whose
// values cannot be computed on their own.
func computable(ts ...*Tensor) error {
	for _, t := range ts {
		if t.batch() != nil {
			return fmt.Errorf("tensor: the values of a tensor inside VMap: %w", ErrNotBatchable)
		}
	}
	return nil
}

// severed returns out, which a rule of its own level alone computed from
// in, with a tape at the level of each batch it stands for that in
// requires gradients at, whose derivatives fail with ErrNotBatchable.
func severed(out *Tensor, in []*Tensor) *Tensor {
	b := out.batch()
	if b == nil || !anyTracked(in...) {
		return out
	}
	n := b.phys.node.Shape[0]
	phys := make([]*Tensor, len(in))
	for i, x := range in {
		phys[i] = x.physical(b.level, n)
	}
	p := record(severed(b.phys, phys), failing(fmt.Errorf("tensor: derivative outside VMap through a Func applied inside: %w", ErrNotBatchable)), phys...)
	return newBatched(p, b.level)
}

// stopGradient returns t without gradient tracking at any level: inside
// VMap, the batch it stands for is detached too.
func (t *Tensor) stopGradient() *Tensor {
	if b := t.batch(); b != nil {
		return newBatched(b.phys.stopGradient(), b.level)
	}
	return t.detached()
}

// batchLevel returns the level of the innermost VMap call that any of ts,
// which may hold nils, belongs to, or 0 if none is inside VMap.
func batchLevel(ts ...*Tensor) uint64 {
	var l uint64
	for _, t := range ts {
		if b := t.batch(); b != nil && b.level > l {
			l = b.level
		}
	}
	return l
}

// inBatch reports whether t is batched at level l.
func (t *Tensor) inBatch(l uint64) bool {
	b := t.batch()
	return b != nil && b.level == l
}

// batchSize returns the length of the batch axis of level l, which one of
// ts is batched at.
func batchSize(l uint64, ts ...*Tensor) int {
	for _, t := range ts {
		if t.inBatch(l) {
			return t.batch().phys.node.Shape[0]
		}
	}
	return 0
}

// physical returns the batch that t stands for at level l, with the batch
// axis first: its own if it is batched at l, and otherwise t repeated n
// times, as a view, since it is the same for every example. A nil t
// stays nil.
func (t *Tensor) physical(l uint64, n int) *Tensor {
	if t == nil {
		return nil
	}
	if t.inBatch(l) {
		return t.batch().phys
	}
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	return t.reshape(slices.Insert(slices.Clone(t.node.Shape), 0, 1)).expand(slices.Insert(slices.Clone(t.node.Shape), 0, n))
}

// shifted returns axes moved one place on, past a new first axis.
func shifted(axes []int) []int {
	out := make([]int, len(axes))
	for i, a := range axes {
		out[i] = a + 1
	}
	return out
}

// The batching rules below compute a primitive operation on inputs of
// which one at least is batched at level l: they apply the operation to
// the batches, with their batch axes first, and return the tensor that
// stands for one example of the result. Each primitive's function calls
// its rule in place of building its node, and records its gradient rule
// as usual, for the derivatives inside VMap; the operations on the
// batches record theirs, for the derivatives around it.

func batchUnary(l uint64, op backend.Op, t *Tensor) *Tensor {
	return newBatched(t.batch().phys.unary(op), l)
}

func batchCast(l uint64, t *Tensor, dt DType) *Tensor {
	return newBatched(t.batch().phys.Cast(dt), l)
}

func batchClone(l uint64, t *Tensor) *Tensor {
	return newBatched(t.batch().phys.Clone(), l)
}

func batchBinary(l uint64, op backend.Op, x, y *Tensor) *Tensor {
	n := batchSize(l, x, y)
	return newBatched(binaryPrim(op, x.physical(l, n), y.physical(l, n)), l)
}

func batchWhere(l uint64, c, x, y *Tensor) *Tensor {
	n := batchSize(l, c, x, y)
	return newBatched(wherePrim(c.physical(l, n), x.physical(l, n), y.physical(l, n)), l)
}

func batchReduce(l uint64, op backend.Op, t *Tensor, axes []int) *Tensor {
	return newBatched(t.batch().phys.reduce(op, shifted(axes)), l)
}

func batchReshape(l uint64, t *Tensor, shape []int) *Tensor {
	p := t.batch().phys
	return newBatched(p.reshape(slices.Insert(slices.Clone(shape), 0, p.node.Shape[0])), l)
}

func batchExpand(l uint64, t *Tensor, shape []int) *Tensor {
	p := t.batch().phys
	return newBatched(p.expand(slices.Insert(slices.Clone(shape), 0, p.node.Shape[0])), l)
}

func batchPermute(l uint64, t *Tensor, perm []int) *Tensor {
	return newBatched(t.batch().phys.permute(slices.Insert(shifted(perm), 0, 0)), l)
}

func batchPad(l uint64, t *Tensor, pads [][2]int, value float64) *Tensor {
	return newBatched(t.batch().phys.pad(slices.Insert(slices.Clone(pads), 0, [2]int{}), value), l)
}

func batchShrink(l uint64, t *Tensor, ranges [][2]int) *Tensor {
	p := t.batch().phys
	return newBatched(p.shrink(slices.Insert(slices.Clone(ranges), 0, [2]int{0, p.node.Shape[0]})), l)
}

func batchFlip(l uint64, t *Tensor, axes []int) *Tensor {
	return newBatched(t.batch().phys.flip(shifted(axes)), l)
}

// batchMatMul multiplies matrices with equal batch dimensions d. When
// both are batched, the batch is one more batch dimension. When one is,
// the batch joins its rows or columns, so that the other operand, often
// a weight, is read once rather than repeated for every example.
func batchMatMul(l uint64, x, y *Tensor) *Tensor {
	switch {
	case x.inBatch(l) && y.inBatch(l):
		return newBatched(matmulPrim(x.batch().phys, y.batch().phys), l)
	case x.inBatch(l):
		// [n, d..., M, K] -> [d..., n*M, K], times [d..., K, N].
		p := x.batch().phys
		s := p.node.Shape
		r := len(s)
		d, n, m := s[1:r-2], s[0], s[r-2]
		rows := p.moveAxis(0, r-3).reshape(slices.Concat(d, []int{n * m, s[r-1]}))
		prod := matmulPrim(rows, y)
		if prod.err != nil {
			return prod
		}
		nc := prod.node.Shape[len(prod.node.Shape)-1]
		return newBatched(prod.reshape(slices.Concat(d, []int{n, m, nc})).moveAxis(r-3, 0), l)
	default:
		// [d..., M, K] times [n, d..., K, N] -> [d..., K, n*N].
		p := y.batch().phys
		s := p.node.Shape
		r := len(s)
		d, n, k, nc := s[1:r-2], s[0], s[r-2], s[r-1]
		cols := p.moveAxis(0, r-2).reshape(slices.Concat(d, []int{k, n * nc}))
		prod := matmulPrim(x, cols)
		if prod.err != nil {
			return prod
		}
		m := prod.node.Shape[len(prod.node.Shape)-2]
		return newBatched(prod.reshape(slices.Concat(d, []int{m, n, nc})).moveAxis(r-2, 0), l)
	}
}

func batchGather(l uint64, t *Tensor, axis int, idx *Tensor) *Tensor {
	n := batchSize(l, t, idx)
	return newBatched(t.physical(l, n).Gather(axis+1, idx.physical(l, n)), l)
}

func batchScatterAdd(l uint64, t *Tensor, axis int, idx, src *Tensor) *Tensor {
	n := batchSize(l, t, idx, src)
	return newBatched(t.physical(l, n).ScatterAdd(axis+1, idx.physical(l, n), src.physical(l, n)), l)
}

func batchCat(l uint64, axis int, ts []*Tensor) *Tensor {
	n := batchSize(l, ts...)
	ps := make([]*Tensor, len(ts))
	for i, t := range ts {
		ps[i] = t.physical(l, n)
	}
	return newBatched(cat(axis+1, ps), l)
}

func batchSliceScatter(l uint64, t, src *Tensor, ranges [][2]int) *Tensor {
	n := batchSize(l, t, src)
	return newBatched(t.physical(l, n).SliceScatter(src.physical(l, n), slices.Insert(slices.Clone(ranges), 0, [2]int{0, n})), l)
}

func batchUnfold(l uint64, t *Tensor, w graph.Window, value float64) *Tensor {
	return newBatched(t.batch().phys.unfoldPrim(w, value), l)
}

func batchFold(l uint64, t *Tensor, spatial []int, w graph.Window) *Tensor {
	return newBatched(t.batch().phys.fold(spatial, w), l)
}

func batchScan(l uint64, t *Tensor, axis int, c backend.Combine) *Tensor {
	return newBatched(t.batch().phys.scan(axis+1, c), l)
}

func batchQMatMul(l uint64, t, w *Tensor) *Tensor {
	if w.inBatch(l) {
		return fail(fmt.Errorf("tensor: product with a batch of quantized matrices: %w", ErrNotBatchable))
	}
	return newBatched(t.batch().phys.MatMulT(w), l)
}

func batchAttention(l uint64, q, k, v, mask *Tensor, a graph.AttentionAttrs) (out, lse *Tensor) {
	n := batchSize(l, q, k, v, mask)
	o, s := attentionPrim(q.physical(l, n), k.physical(l, n), v.physical(l, n), mask.physical(l, n), a)
	return newBatched(o, l), newBatched(s, l)
}

func batchAttentionGrad(l uint64, q, k, v, mask, out, lse, g *Tensor, a graph.AttentionAttrs) []*Tensor {
	n := batchSize(l, q, k, v, mask, out, lse, g)
	p := func(t *Tensor) *Tensor { return t.physical(l, n) }
	gs := attentionGrad(p(q), p(k), p(v), p(mask), p(out), p(lse), p(g), a)
	for i, x := range gs {
		gs[i] = newBatched(x, l)
	}
	return gs
}

func batchTo(l uint64, t *Tensor, d Device) *Tensor {
	return newBatched(t.batch().phys.To(d), l)
}

func batchCumSum(l uint64, t *Tensor, axis int) *Tensor {
	return newBatched(t.batch().phys.cumsum(axis+1), l)
}

func batchLogCumSumExp(l uint64, t *Tensor, axis int) *Tensor {
	return newBatched(t.batch().phys.logCumSumExp(axis+1), l)
}

func batchCumMax(l uint64, t *Tensor, axis int) (values, indices *Tensor) {
	v, i := t.batch().phys.CumMax(axis + 1)
	return newBatched(v, l), newBatched(i, l)
}

// batchReLU applies ReLU to the batch, so that its own gradient applies
// around VMap too.
func batchReLU(l uint64, t *Tensor) *Tensor {
	return newBatched(t.batch().phys.ReLU(), l)
}

func batchFFT(l uint64, t *Tensor, axis int, inverse bool) *Tensor {
	return newBatched(t.batch().phys.dft(axis+1, inverse), l)
}

func batchArgsort(l uint64, t *Tensor, axis, k int, descending bool) *Tensor {
	return newBatched(t.batch().phys.argsort(axis+1, k, descending), l)
}
