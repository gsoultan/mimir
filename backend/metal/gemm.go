package metal

import "github.com/gsoultan/mimir/backend"

// matrix is how Metal Performance Shaders reads an operand: matrices of
// rows × cols elements in row-major order, rows ld elements apart and
// matrices stride elements apart, from element offset of the buffer. A
// transposed operand is stored as the transpose of the matrix it stands
// for.
type matrix struct {
	offset, rows, cols, ld, stride int
	trans                          bool
}

// gemm is a product of batch pairs of an m × k and a k × n matrix into the
// contiguous output, as Metal Performance Shaders computes it.
type gemm struct {
	a, b       matrix
	m, n, k    int
	batch      int
	resultSize int // elements per result matrix
}

// planGEMM returns the product k as Metal Performance Shaders computes it,
// or false if it reads its operands in a way MPS cannot: not float32, a
// sum over batches, a broadcast or reversed operand that cannot be folded
// away, or a matrix whose rows are not equally spaced. lens are the
// lengths of the input buffers.
func planGEMM(k *backend.Kernel, lens []int) (gemm, bool) {
	r := len(k.Shape)
	if k.Op != backend.OpMatMul || k.DType != backend.Float32 || r < 2 || len(k.In) < 2 {
		return gemm{}, false
	}
	vx, vy := k.In[0], k.In[1]
	if len(vx.Shape) != r || len(vy.Shape) != r {
		return gemm{}, false
	}
	for _, d := range k.Attrs.Axes {
		if vx.Shape[d] != 1 {
			return gemm{}, false
		}
	}
	m, n, kk := k.Shape[r-2], k.Shape[r-1], vx.Shape[r-1]
	if m == 0 || n == 0 || kk == 0 {
		return gemm{}, false
	}
	batchShape := k.Shape[:r-2]
	bx := backend.View{Shape: batchShape, Strides: vx.Strides[:r-2]}
	by := backend.View{Shape: batchShape, Strides: vy.Strides[:r-2]}
	bshape, bviews := backend.Coalesce(batchShape, []backend.View{bx, by})
	g := gemm{m: m, n: n, k: kk, batch: 1}
	switch {
	case len(bshape) == 0:
	case len(bshape) == 1 && bviews[1].Strides[0] == 0:
		// A right operand shared by every batch: the batches of the left
		// operand are rows of one product, if they are spaced as rows.
		rows := backend.View{Shape: []int{bshape[0], m}, Strides: []int{bviews[0].Strides[0], vx.Strides[r-2]}}
		folded, fv := backend.Coalesce(rows.Shape, []backend.View{rows})
		if len(folded) != 1 {
			return gemm{}, false
		}
		g.m = folded[0]
		vx = backend.View{
			Shape:   []int{g.m, kk},
			Strides: []int{fv[0].Strides[0], vx.Strides[r-1]},
			Offset:  vx.Offset,
		}
		vy = backend.View{Shape: []int{kk, n}, Strides: vy.Strides[r-2:], Offset: vy.Offset}
		r = 2
	case len(bshape) == 1:
		g.batch = bshape[0]
	default:
		return gemm{}, false
	}
	var ok bool
	if g.a, ok = layout(g.m, kk, vx.Strides[r-2], vx.Strides[r-1], vx.Offset); !ok {
		return gemm{}, false
	}
	if g.b, ok = layout(kk, n, vy.Strides[r-2], vy.Strides[r-1], vy.Offset); !ok {
		return gemm{}, false
	}
	g.resultSize = g.m * n
	if g.batch > 1 {
		g.a.stride, g.b.stride = bviews[0].Strides[0], bviews[1].Strides[0]
	}
	// MPS reads whole rows of whole matrices, padding included, so they
	// must lie inside the buffers and not overlap.
	for i, mat := range []*matrix{&g.a, &g.b} {
		size := mat.rows * mat.ld
		if g.batch == 1 {
			mat.stride = size
		}
		if mat.stride < size || mat.offset+g.batch*mat.stride > lens[i] {
			return gemm{}, false
		}
	}
	return g, true
}

// packed reports whether each operand's matrices lie one after another,
// rows × ld elements apart, or there is one matrix. Metal Performance
// Shaders on an M5 under macOS 26 reads batched matrices that far apart
// whatever the descriptor's matrix stride says, so a product of batches
// spaced further apart runs elsewhere.
func (g gemm) packed() bool {
	return g.batch == 1 || g.a.stride == g.a.rows*g.a.ld && g.b.stride == g.b.rows*g.b.ld
}

// layout returns how MPS reads a rows × cols matrix with the given strides
// and offset: row-major with equally spaced rows, or the transpose of one.
func layout(rows, cols, sr, sc, offset int) (matrix, bool) {
	if (cols == 1 || sc == 1) && sr >= 0 {
		ld := sr
		if rows == 1 {
			ld = cols
		}
		if ld >= cols {
			return matrix{offset: offset, rows: rows, cols: cols, ld: ld}, true
		}
	}
	if (rows == 1 || sr == 1) && sc >= 0 {
		ld := sc
		if cols == 1 {
			ld = rows
		}
		if ld >= rows {
			return matrix{offset: offset, rows: cols, cols: rows, ld: ld, trans: true}, true
		}
	}
	return matrix{}, false
}

// at returns the element of the buffer that holds element (i, j) of
// matrix b of the operand mat stands for.
func (mat matrix) at(b, i, j int) int {
	if mat.trans {
		i, j = j, i
	}
	return mat.offset + b*mat.stride + i*mat.ld + j
}

// planEpilogue returns the kernel that applies the epilogue program of the
// product k to its result, already in outs[0], in place.
func planEpilogue(k *backend.Kernel, outs, ins []backend.Buffer) (*launch, error) {
	n, _ := backend.NumElements(k.Shape)
	if err := indexable(n, ins); err != nil {
		return nil, err
	}
	g := newGen(k, outs, ins)
	g.line("if (gid >= uint(%s)) return;", g.param(n))
	g.line("int i = int(gid);")
	g.line("float acc = o0[i];")
	return g.epilogue(n, "acc")
}
