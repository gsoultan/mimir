package tensor

import (
	"fmt"

	"github.com/gsoultan/mimir/internal/graph"
)

// MatMulT returns t·wᵀ, the product with the transpose of the matrix w,
// [N, K], like torch.nn.functional.linear without a bias: t is [..., K]
// and the result [..., N]. A float w gives t.MatMul(w.Transpose(-2, -1)).
//
// A block-quantized w is multiplied without dequantizing it, as
// llama.cpp does: on the CPU each block of a row of t is rounded to int8
// with a float32 scale, and the product of the quants is summed in
// integers, so the result differs from the product with the dequantized
// weights by about the weights' own quantization error. A half t is
// multiplied in Float32 and the product returned in its dtype. The
// gradient with respect to t is that of the product with the dequantized
// weights; w has none.
func (t *Tensor) MatMulT(w *Tensor) *Tensor {
	if err := firstErr(t, w); err != nil {
		return fail(err)
	}
	if !w.node.DType.IsQuantized() {
		if len(w.node.Shape) < 2 {
			return fail(fmt.Errorf("tensor: product with the transpose of %v: %w", w.node.Shape, ErrShape))
		}
		return t.MatMul(w.Transpose(-2, -1))
	}
	dt := t.node.DType
	if dt.Size() == 2 {
		return t.Cast(Float32).MatMulT(w).Cast(dt)
	}
	var out *Tensor
	if l := batchLevel(t, w); l != 0 {
		out = batchQMatMul(l, t, w)
	} else {
		out = wrap(graph.QMatMul(t.node, w.node))
	}
	return record(out, qmatmulRule(), t, w)
}

// qmatmulRule differentiates the product as if by the dequantized weights.
func qmatmulRule() rule {
	return rule{
		vjp: func(g *Tensor, in []*Tensor, _ *Tensor) []*Tensor { return grads(g.MatMul(in[1].Cast(Float32)), nil) },
		jvp: func(t, in []*Tensor, _ *Tensor) *Tensor { return t[0].MatMul(in[1].Cast(Float32).Transpose(-2, -1)) },
	}
}
