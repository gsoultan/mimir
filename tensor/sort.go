package tensor

import (
	"fmt"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// Sort returns t's elements sorted along axis, ascending or descending,
// and the Int32 positions along axis they come from, like
// torch.sort(stable=True): equal elements keep their order. NaN sorts as
// the largest value, last ascending and first descending, and -0 equals
// +0. The values' gradient flows back to the positions they come from.
func (t *Tensor) Sort(axis int, descending bool) (values, indices *Tensor) {
	x, a, back := t.scanAxis(axis)
	if x.err != nil {
		return x, x
	}
	idx := x.argsort(a, x.node.Shape[a], descending)
	return back(x.Gather(a, idx)), back(idx)
}

// Argsort returns the Int32 positions along axis of the elements of
// Sort(axis, descending), like torch.argsort(stable=True).
func (t *Tensor) Argsort(axis int, descending bool) *Tensor {
	x, a, back := t.scanAxis(axis)
	if x.err != nil {
		return x
	}
	return back(x.argsort(a, x.node.Shape[a], descending))
}

// TopK returns the k largest elements of t along axis, or the k smallest,
// in order, and their Int32 positions along axis, like
// torch.topk(sorted=True). Equal elements come in the order of their
// positions, and NaN is the largest value. The values' gradient flows
// back to their positions. k must be within the length of axis.
func (t *Tensor) TopK(k, axis int, largest bool) (values, indices *Tensor) {
	x, a, back := t.scanAxis(axis)
	if x.err != nil {
		return x, x
	}
	if n := x.node.Shape[a]; k < 0 || k > n {
		err := fail(fmt.Errorf("tensor: top %d of %d elements along axis %d: %w", k, n, axis, ErrShape))
		return err, err
	}
	idx := x.argsort(a, k, largest)
	return back(x.Gather(a, idx)), back(idx)
}

// argsort is the positions of the first k elements of the sort of t
// along the normalized axis. Kernels select up to backend.MaxTopK of a
// longer axis; more are cut from the whole sort.
func (t *Tensor) argsort(axis, k int, descending bool) *Tensor {
	if n := t.node.Shape[axis]; k < n && k > backend.MaxTopK {
		return t.argsort(axis, n, descending).shrink(axisRange(t.node.Shape, axis, 0, k))
	}
	if l := batchLevel(t); l != 0 {
		return batchArgsort(l, t, axis, k, descending)
	}
	return wrap(graph.Argsort(t.node, axis, k, descending))
}
