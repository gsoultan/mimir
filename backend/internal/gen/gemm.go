package gen

import (
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// Matrix is how a matrix library reads an operand: matrices of Rows ×
// Cols elements in row-major order, rows LD elements apart and matrices
// Stride elements apart, from element Offset of the buffer. A transposed
// operand is stored as the transpose of the matrix it stands for.
type Matrix struct {
	Offset, Rows, Cols, LD, Stride int
	Trans                          bool
}

// GEMM is a product of Batch pairs of an M × K and a K × N matrix into the
// contiguous output, as matrix libraries such as Metal Performance Shaders
// and cuBLAS compute it.
type GEMM struct {
	A, B       Matrix
	M, N, K    int
	Batch      int
	ResultSize int // elements per result matrix
}

// PlanGEMM returns the product k as a matrix library computes it, or
// false if it reads its operands in a way the library cannot: not
// float32, a sum over batches that does not fold into the inner
// dimension, a reversed operand, more than one batch dimension that
// cannot be folded away, or a matrix whose rows are not equally spaced.
// lens are the lengths of the input buffers.
//
// An operand that every batch shares has a Stride of 0, unless it folds
// into the rows of one product. A sum over batch axes folds into the
// inner dimension when both operands lay out each summed axis outside
// the inner one, as the gradient of a layer's weights over a batch of
// sequences, [B, T, F]ᵀ times [B, T, O] summed over B, reads them: the
// sum is then one product over B·T.
func PlanGEMM(k *backend.Kernel, lens []int) (GEMM, bool) {
	r := len(k.Shape)
	if k.Op != backend.OpMatMul || k.DType != backend.Float32 || r < 2 || len(k.In) < 2 {
		return GEMM{}, false
	}
	vx, vy := k.In[0], k.In[1]
	if len(vx.Shape) != r || len(vy.Shape) != r {
		return GEMM{}, false
	}
	if len(k.Attrs.Axes) > 0 {
		var ok bool
		if vx, vy, ok = foldSums(vx, vy, k.Attrs.Axes); !ok {
			return GEMM{}, false
		}
	}
	m, n, kk := k.Shape[r-2], k.Shape[r-1], vx.Shape[r-1]
	if m == 0 || n == 0 || kk == 0 {
		return GEMM{}, false
	}
	batchShape := k.Shape[:r-2]
	bx := backend.View{Shape: batchShape, Strides: vx.Strides[:r-2]}
	by := backend.View{Shape: batchShape, Strides: vy.Strides[:r-2]}
	bshape, bviews := backend.Coalesce(batchShape, []backend.View{bx, by})
	g := GEMM{M: m, N: n, K: kk, Batch: 1}
	switch {
	case len(bshape) == 0:
	case len(bshape) == 1 && bviews[1].Strides[0] == 0:
		// A right operand shared by every batch: the batches of the left
		// operand are rows of one product, if they are spaced as rows.
		rows := backend.View{Shape: []int{bshape[0], m}, Strides: []int{bviews[0].Strides[0], vx.Strides[r-2]}}
		folded, fv := backend.Coalesce(rows.Shape, []backend.View{rows})
		if len(folded) != 1 {
			return GEMM{}, false
		}
		g.M = folded[0]
		vx = backend.View{
			Shape:   []int{g.M, kk},
			Strides: []int{fv[0].Strides[0], vx.Strides[r-1]},
			Offset:  vx.Offset,
		}
		vy = backend.View{Shape: []int{kk, n}, Strides: vy.Strides[r-2:], Offset: vy.Offset}
		r = 2
	case len(bshape) == 1:
		g.Batch = bshape[0]
	default:
		return GEMM{}, false
	}
	var ok bool
	if g.A, ok = layout(g.M, kk, vx.Strides[r-2], vx.Strides[r-1], vx.Offset); !ok {
		return GEMM{}, false
	}
	if g.B, ok = layout(kk, n, vy.Strides[r-2], vy.Strides[r-1], vy.Offset); !ok {
		return GEMM{}, false
	}
	g.ResultSize = g.M * n
	if g.Batch > 1 {
		g.A.Stride, g.B.Stride = bviews[0].Strides[0], bviews[1].Strides[0]
	}
	// Libraries read whole rows of whole matrices, padding included, so
	// they must lie inside the buffers, and not overlap unless they are
	// one matrix that every batch shares.
	for i, mat := range []*Matrix{&g.A, &g.B} {
		size := mat.Rows * mat.LD
		if g.Batch == 1 {
			mat.Stride = size
		}
		switch {
		case mat.Stride == 0 && mat.Offset+size <= lens[i]:
		case mat.Stride < size || mat.Offset+g.Batch*mat.Stride > lens[i]:
			return GEMM{}, false
		}
	}
	return g, true
}

// foldSums returns the operands of a product summed over the batch axes
// axes with those axes folded into the inner dimension, or false if they
// cannot be: unless each summed axis of each operand is spaced as the
// inner dimension's elements times its length, outermost first, the
// product's terms are not those of one longer product.
func foldSums(vx, vy backend.View, axes []int) (backend.View, backend.View, bool) {
	r := len(vx.Shape)
	var shape, sx, sy []int
	for d := range r - 2 {
		if !slices.Contains(axes, d) {
			continue
		}
		if vx.Shape[d] != vy.Shape[d] {
			return vx, vy, false
		}
		shape = append(shape, vx.Shape[d])
		sx, sy = append(sx, vx.Strides[d]), append(sy, vy.Strides[d])
	}
	inner := vx.Shape[r-1]
	shape = append(shape, inner)
	sx, sy = append(sx, vx.Strides[r-1]), append(sy, vy.Strides[r-2])
	folded, views := backend.Coalesce(shape, []backend.View{{Shape: shape, Strides: sx}, {Shape: shape, Strides: sy}})
	if len(folded) > 1 {
		return vx, vy, false
	}
	vx = backend.View{Shape: slices.Clone(vx.Shape), Strides: slices.Clone(vx.Strides), Offset: vx.Offset}
	vy = backend.View{Shape: slices.Clone(vy.Shape), Strides: slices.Clone(vy.Strides), Offset: vy.Offset}
	for _, d := range axes {
		vx.Shape[d], vx.Strides[d], vy.Shape[d], vy.Strides[d] = 1, 0, 1, 0
	}
	n, _ := backend.NumElements(shape)
	vx.Shape[r-1], vy.Shape[r-2] = n, n
	if len(folded) == 1 {
		vx.Strides[r-1], vy.Strides[r-2] = views[0].Strides[0], views[1].Strides[0]
	}
	return vx, vy, true
}

// layout returns how a library reads a rows × cols matrix with the given
// strides and offset: row-major with equally spaced rows, or the transpose
// of one.
func layout(rows, cols, sr, sc, offset int) (Matrix, bool) {
	if (cols == 1 || sc == 1) && sr >= 0 {
		ld := sr
		if rows == 1 {
			ld = cols
		}
		if ld >= cols {
			return Matrix{Offset: offset, Rows: rows, Cols: cols, LD: ld}, true
		}
	}
	if (rows == 1 || sr == 1) && sc >= 0 {
		ld := sc
		if cols == 1 {
			ld = rows
		}
		if ld >= rows {
			return Matrix{Offset: offset, Rows: cols, Cols: rows, LD: ld, Trans: true}, true
		}
	}
	return Matrix{}, false
}

// At returns the element of the buffer that holds element (i, j) of
// matrix b of the operand mat stands for.
func (mat Matrix) At(b, i, j int) int {
	if mat.Trans {
		i, j = j, i
	}
	return mat.Offset + b*mat.Stride + i*mat.LD + j
}

// PlanEpilogue returns the kernel, in dialect d, that applies the epilogue
// program of the product k to its result, already in outs[0], in place.
func PlanEpilogue(d *Dialect, k *backend.Kernel, outs, ins []backend.Buffer) (*Launch, error) {
	n, _ := backend.NumElements(k.Shape)
	if err := indexable(n, ins); err != nil {
		return nil, err
	}
	g := newGen(d, k, outs, ins)
	g.line("if (gid >= uint(%s)) return;", g.param(n))
	g.line("int i = int(gid);")
	g.line("float acc = o0[i];")
	return g.epilogue(n, "acc")
}
