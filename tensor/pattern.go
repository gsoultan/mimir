package tensor

import (
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// A Pattern is a sparse matrix whose nonzero entries all are one, such
// as the adjacency matrix of a graph or the matrix that sums the
// messages of a graph's edges into its nodes. Multiplying a dense tensor
// by it combines rows of the tensor without a dense matrix: row i of
// p.MatMul(x) sums the rows j of x with an entry (i, j), as
// torch.sparse.mm and the scatter operations of PyTorch Geometric do.
//
// A Pattern is built once, on the host, and holds its entries sorted
// both by row and by column, so that a product and its gradient, the
// product by the transpose, each run as one kernel that reads the rows
// it combines where they lie and writes every output row once, in a
// fixed order. Results are the same on every device and for any number
// of workers. The entries move to a device the first time a product
// there needs them, and stay there. A Pattern is safe for concurrent use.
type Pattern struct {
	rows, cols int
	by, tr     *entries // sorted by row, and the transpose's
}

// entries holds the entries of a pattern in row-major order, as
// compressed sparse rows: the entries of row i are those from
// offsets[i] up to offsets[i+1].
type entries struct {
	offsets []int32 // [rows+1]
	cols    []int32 // the column of each entry
	rows    []int32 // the row of each entry
	// byCol lists the entries, by their position here, grouped by
	// column in order and in row order within a column: the order of
	// the transpose's entries.
	byCol []int32

	mu  sync.Mutex
	dev map[backend.Backend]*[numLists]*Tensor
}

// The lists of entries a kernel reads.
const (
	listOffsets = iota
	listCols
	listRows
	listByCol
	listAll     // 0, 1, ..., n: each entry a segment of its own
	listDegrees // the Float32 number of entries of each row
	numLists
)

// NewPattern returns the rows × cols pattern with an entry at (row[k],
// col[k]) for each k, like the coordinate lists of a torch.sparse_coo_tensor.
// An entry listed twice counts twice. Entries out of range fail with
// ErrIndex, and lists of different lengths with ErrShape.
func NewPattern(rows, cols int, row, col []int32) (*Pattern, error) {
	if rows < 0 || cols < 0 || rows >= math.MaxInt32 || cols >= math.MaxInt32 {
		return nil, fmt.Errorf("tensor: pattern of %d × %d: %w", rows, cols, ErrShape)
	}
	if len(row) != len(col) || len(row) >= math.MaxInt32 {
		return nil, fmt.Errorf("tensor: pattern of %d rows and %d columns of entries: %w", len(row), len(col), ErrShape)
	}
	for k := range row {
		if r, c := row[k], col[k]; r < 0 || int(r) >= rows || c < 0 || int(c) >= cols {
			return nil, fmt.Errorf("tensor: entry (%d, %d) of a pattern of %d × %d: %w", r, c, rows, cols, ErrIndex)
		}
	}
	// Sorting by row and then by column, each stably by counting, puts
	// the entries in row-major order and their transpose's in
	// column-major order, with ties in the order given.
	order := countingSort(row, rows)
	by := &entries{offsets: segmentOffsets(row, rows), cols: make([]int32, len(row)), rows: make([]int32, len(row))}
	for k, e := range order {
		by.cols[k], by.rows[k] = col[e], row[e]
	}
	by.byCol = countingSort(by.cols, cols)
	tr := &entries{offsets: segmentOffsets(col, cols), cols: make([]int32, len(row)), rows: make([]int32, len(row)), byCol: make([]int32, len(row))}
	for q, k := range by.byCol {
		tr.cols[q], tr.rows[q] = by.rows[k], by.cols[k]
		tr.byCol[k] = int32(q)
	}
	return &Pattern{rows: rows, cols: cols, by: by, tr: tr}, nil
}

// countingSort returns the positions of keys, each in [0, n), in the
// stable order of their keys.
func countingSort(keys []int32, n int) []int32 {
	start := segmentOffsets(keys, n)
	order := make([]int32, len(keys))
	for k, key := range keys {
		order[start[key]] = int32(k)
		start[key]++
	}
	return order
}

// segmentOffsets returns where the run of each key in [0, n) starts in
// keys sorted, followed by len(keys).
func segmentOffsets(keys []int32, n int) []int32 {
	offsets := make([]int32, n+1)
	for _, key := range keys {
		offsets[key+1]++
	}
	for i := range n {
		offsets[i+1] += offsets[i]
	}
	return offsets
}

// Shape returns the numbers of rows and columns of p.
func (p *Pattern) Shape() (rows, cols int) { return p.rows, p.cols }

// Entries returns the number of entries of p.
func (p *Pattern) Entries() int { return len(p.by.cols) }

// T returns the transpose of p, which shares its entries.
func (p *Pattern) T() *Pattern { return &Pattern{rows: p.cols, cols: p.rows, by: p.tr, tr: p.by} }

// Degrees returns the number of entries in each row of p, as a Float32
// tensor of p's rows on the default device.
func (p *Pattern) Degrees() *Tensor { return p.by.list(defaultDevice().be, listDegrees) }

// list returns list i of e on the device be.
func (e *entries) list(be backend.Backend, i int) *Tensor {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.dev == nil {
		e.dev = map[backend.Backend]*[numLists]*Tensor{}
	}
	lists := e.dev[be]
	if lists == nil {
		lists = new([numLists]*Tensor)
		e.dev[be] = lists
	}
	if lists[i] == nil {
		var v []int32
		switch i {
		case listOffsets:
			v = e.offsets
		case listCols:
			v = e.cols
		case listRows:
			v = e.rows
		case listByCol:
			v = e.byCol
		case listAll:
			v = make([]int32, len(e.cols)+1)
			for k := range v {
				v[k] = int32(k)
			}
		case listDegrees:
			d := make([]float32, len(e.offsets)-1)
			for k := range d {
				d[k] = float32(e.offsets[k+1] - e.offsets[k])
			}
			t := upload(be, Float32, asBytes(d), []int{len(d)})
			if t.err != nil {
				return t
			}
			lists[i] = t
			return t
		}
		t := upload(be, Int32, asBytes(v), []int{len(v)})
		if t.err != nil {
			return t // not kept, so that a later call tries again
		}
		lists[i] = t
	}
	return lists[i]
}

// check returns an error unless x is a tensor of p's columns of rows.
func (p *Pattern) check(x *Tensor, what string) error {
	if err := firstErr(x); err != nil {
		return err
	}
	if p == nil {
		return fmt.Errorf("tensor: %s by a nil pattern: %w", what, ErrNilTensor)
	}
	if len(x.node.Shape) == 0 || x.node.Shape[0] != p.cols {
		return fmt.Errorf("tensor: %s of %d × %d pattern by %v: %w", what, p.rows, p.cols, x.node.Shape, ErrShape)
	}
	return nil
}

// MatMul returns the product of p and x, a tensor of p's columns of
// rows, [cols, D...]: row i of the result, [rows, D...], sums the rows j
// of x with an entry (i, j) of p, in the order of j, and is zero if row
// i has no entries. Half x is summed in float32 and rounded once.
func (p *Pattern) MatMul(x *Tensor) *Tensor {
	if err := p.check(x, "product"); err != nil {
		return fail(err)
	}
	return p.sum(x, p.by, listOffsets, p.by, listCols, p.MatMul, p.T().MatMul)
}

// Mean returns the product of p and x divided, row by row, by the
// number of entries of the row: the mean of the rows j of x with an
// entry (i, j), or zero for a row without any. x must be a float tensor.
func (p *Pattern) Mean(x *Tensor) *Tensor {
	if err := p.check(x, "mean"); err != nil {
		return fail(err)
	}
	shape := slices.Repeat([]int{1}, len(x.node.Shape))
	shape[0] = p.rows
	d := p.by.list(backendOf(x.node), listDegrees).Maximum(Scalar[float32](1)).Reshape(shape...)
	return p.MatMul(x).Div(d)
}

// Max returns the elementwise maximum of the rows j of x, [cols, D...],
// with an entry (i, j) of p, as row i of the result, [rows, D...], or
// zero for a row without entries, like the max aggregation of
// torch_scatter. NaN propagates. The gradient of an element is shared
// evenly between the rows that tie for it, as Max shares it.
func (p *Pattern) Max(x *Tensor) *Tensor {
	if err := p.check(x, "maximum"); err != nil {
		return fail(err)
	}
	be := backendOf(x.node)
	out := segment(x, p.by.list(be, listOffsets), p.by.list(be, listCols), backend.CombineMax)
	return record(out, rule{
		vjp: func(g *Tensor, in []*Tensor, out *Tensor) []*Tensor {
			hit, n := p.maxHits(in[0], out)
			return grads(p.colSum(p.atRows(g.Div(n)).Mul(hit)))
		},
		jvp: func(t, in []*Tensor, out *Tensor) *Tensor {
			hit, n := p.maxHits(in[0], out)
			return p.rowSum(p.atCols(t[0]).Mul(hit)).Div(n)
		},
	}, x)
}

// maxHits returns, for the maximum out of x by p, whether each entry
// holds its row's maximum, as 1 or 0 in an [entries, D...] tensor, and
// how many entries of each row do, at least 1, [rows, D...].
func (p *Pattern) maxHits(x, out *Tensor) (hit, n *Tensor) {
	hit = p.atCols(x.stopGradient()).Equal(p.atRows(out.stopGradient()))
	// A row without entries has no hits; counting it as one keeps its
	// derivatives zero rather than 0/0.
	return hit, p.rowSum(hit).Maximum(Scalar[float32](1))
}

// The maps below take a tensor with a row per column or row of p to one
// with a row per entry of p, [entries, D...] in row-major order, and
// back. atCols and colSum are each other's transpose, as are atRows and
// rowSum, which gives their derivatives.

// atCols returns the row of x, [cols, D...], at the column of each entry.
func (p *Pattern) atCols(x *Tensor) *Tensor {
	return p.sum(x, p.by, listAll, p.by, listCols, p.atCols, p.colSum)
}

// colSum sums v over the entries of each column, in row order: [cols, D...].
func (p *Pattern) colSum(v *Tensor) *Tensor {
	return p.sum(v, p.tr, listOffsets, p.by, listByCol, p.colSum, p.atCols)
}

// atRows returns the row of y, [rows, D...], at the row of each entry.
func (p *Pattern) atRows(y *Tensor) *Tensor {
	return p.sum(y, p.by, listAll, p.by, listRows, p.atRows, p.rowSum)
}

// rowSum sums v over the entries of each row: [rows, D...].
func (p *Pattern) rowSum(v *Tensor) *Tensor {
	return p.sum(v, p.by, listOffsets, p.by, listAll, p.rowSum, p.atRows)
}

// sum sums rows of x in segments: list off of seg holds the offsets of
// the segments into list rows of from, which names the rows of x. self
// is the linear map sum computes, for its forward derivative, and
// adjoint its transpose, for its gradient.
func (p *Pattern) sum(x *Tensor, seg *entries, off int, from *entries, rows int, self, adjoint func(*Tensor) *Tensor) *Tensor {
	if err := firstErr(x); err != nil {
		return fail(err)
	}
	be := backendOf(x.node)
	return record(segment(x, seg.list(be, off), from.list(be, rows), backend.CombineSum), rule{
		vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(adjoint(g)) },
		jvp: func(t, _ []*Tensor, _ *Tensor) *Tensor { return self(t[0]) },
	}, x)
}

// segment is the segment primitive: rows of x combined by c in the
// segments that offsets cuts list into (see backend.OpSegment).
func segment(x, offsets, list *Tensor, c backend.Combine) *Tensor {
	if err := firstErr(x, offsets, list); err != nil {
		return fail(err)
	}
	if l := batchLevel(x); l != 0 {
		return batchSegment(l, x, offsets, list, c)
	}
	return wrap(graph.Segment(x.node, offsets.node, list.node, c))
}

// batchSegment combines the rows of each example of x: the batch axis
// moves behind the rows, among the dimensions a segment keeps.
func batchSegment(l uint64, x, offsets, list *Tensor, c backend.Combine) *Tensor {
	phys := x.batch().phys.moveAxis(0, 1)
	return newBatched(segment(phys, offsets, list, c).moveAxis(1, 0), l)
}
