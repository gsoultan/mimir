package tensor

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// AttentionOption configures ScaledDotProductAttention.
type AttentionOption func(*attentionOptions)

type attentionOptions struct {
	causal   bool
	offset   int
	mask     *Tensor
	boolMask bool
	scale    float64
	scaled   bool
}

// Causal lets query i attend only to keys j ≤ i, like the is_causal
// argument of torch.nn.functional.scaled_dot_product_attention: the first
// query sees the first key alone.
func Causal() AttentionOption { return CausalOffset(0) }

// CausalOffset lets query i attend only to keys j ≤ i + offset. With an
// offset of K-Q, for Q queries and K keys, the last query sees every key,
// as the queries of a decoding step that extends a cache of K-Q keys do.
func CausalOffset(offset int) AttentionOption {
	return func(o *attentionOptions) { o.causal, o.offset = true, offset }
}

// AdditiveMask adds m, a float tensor that broadcasts to the scores'
// shape [..., H, Q, K], to the scores before the softmax, like a float
// attn_mask: -Inf hides a key. m is a constant: no gradient flows into
// it.
func AdditiveMask(m *Tensor) AttentionOption {
	return func(o *attentionOptions) { o.mask, o.boolMask = m, false }
}

// BoolMask hides the keys where m, which broadcasts to the scores' shape
// [..., H, Q, K], is zero, like a boolean attn_mask: query i attends to
// key j where m is non-zero.
func BoolMask(m *Tensor) AttentionOption {
	return func(o *attentionOptions) { o.mask, o.boolMask = m, true }
}

// AttentionScale sets the factor applied to the scores q·k, 1/√D by
// default.
func AttentionScale(s float64) AttentionOption {
	return func(o *attentionOptions) { o.scale, o.scaled = s, true }
}

// ScaledDotProductAttention returns softmax(q·kᵀ·scale + mask)·v, like
// torch.nn.functional.scaled_dot_product_attention, for queries q of
// shape [..., H, Q, D], keys k of shape [..., HK, K, D] and values v of
// shape [..., HK, K, DV], whose leading dimensions broadcast. With fewer
// key heads than query heads, HK dividing H, each group of H/HK
// consecutive query heads reads one key head, which is grouped-query
// attention. Inputs of rank 2 have no head dimension. The result,
// [..., H, Q, DV], has the dtype of the inputs, or Float32 when their
// float dtypes differ; half inputs are computed in float32.
//
// The attention runs as one kernel that never holds the Q×K scores at
// once, and its gradient as another that recomputes them, so its memory
// grows with Q and K rather than their product. A query that may attend
// to no key gets zeros, and so does its gradient.
func ScaledDotProductAttention(q, k, v *Tensor, opts ...AttentionOption) *Tensor {
	var o attentionOptions
	for _, opt := range opts {
		opt(&o)
	}
	if err := firstErr(q, k, v); err != nil {
		return fail(err)
	}
	if o.mask != nil {
		if err := firstErr(o.mask); err != nil {
			return fail(err)
		}
	}
	r := len(q.node.Shape)
	if r < 2 || len(k.node.Shape) != r || len(v.node.Shape) != r {
		return fail(fmt.Errorf("tensor: attention of %v, %v and %v: %w", q.node.Shape, k.node.Shape, v.node.Shape, ErrShape))
	}
	if r == 2 {
		out := ScaledDotProductAttention(q.Unsqueeze(0), k.Unsqueeze(0), v.Unsqueeze(0), opts...)
		return out.Squeeze(0)
	}
	lead, err := broadcastShape(q.node.Shape[:r-3], k.node.Shape[:r-3], v.node.Shape[:r-3])
	if err != nil {
		return fail(fmt.Errorf("tensor: attention of %v, %v and %v: %w", q.node.Shape, k.node.Shape, v.node.Shape, ErrShape))
	}
	q = q.broadcastTo(slices.Concat(lead, q.node.Shape[r-3:]))
	k = k.broadcastTo(slices.Concat(lead, k.node.Shape[r-3:]))
	v = v.broadcastTo(slices.Concat(lead, v.node.Shape[r-3:]))
	if err := firstErr(q, k, v); err != nil {
		return fail(err)
	}
	if !o.scaled {
		o.scale = 1 / math.Sqrt(float64(max(q.node.Shape[r-1], 1)))
	}
	attrs := graph.AttentionAttrs{Scale: o.scale, Causal: o.causal, Offset: o.offset}

	var mask *Tensor
	if o.mask != nil {
		mask = o.mask.detached()
		if o.boolMask {
			mask = Where(mask.Equal(mask.scalar(0)), Scalar(float32(math.Inf(-1))), Scalar[float32](0))
		} else if !mask.node.DType.IsFloat() {
			return fail(fmt.Errorf("tensor: attention mask of %s: %w", mask.node.DType, ErrDType))
		}
		scores := slices.Concat(lead, []int{q.node.Shape[r-3], q.node.Shape[r-2], k.node.Shape[r-2]})
		if mask = mask.broadcastTo(scores); mask.err != nil {
			return fail(fmt.Errorf("tensor: attention mask of %v for scores %v: %w", o.mask.node.Shape, scores, ErrShape))
		}
	}
	var maskNode *graph.Node
	if mask != nil {
		maskNode = mask.node
	}
	flat := wrap(graph.Attention(q.node, k.node, v.node, maskNode, attrs))
	if flat.err != nil {
		return flat
	}
	s, err := backend.NewAttentionShape(q.node.Shape, k.node.Shape, v.node.Shape, nil)
	if err != nil {
		return fail(err)
	}
	no := s.Size() - s.Batches()*s.Heads*s.Q
	out := flat.shrink([][2]int{{0, no}}).reshape(s.OutShape())
	lse := flat.shrink([][2]int{{no, s.Size()}}).reshape(s.LSEShape())
	if tracked(q, k, v) {
		qd, kd, vd, od := q.detached(), k.detached(), v.detached(), out
		out = track(out, func(g *Tensor) []*Tensor {
			gr := wrap(graph.AttentionGrad(qd.node, kd.node, vd.node, maskNode, od.node, lse.node, g.node, attrs))
			nq, nk := s.Batches()*s.Heads*s.Q*s.D, s.Batches()*s.KVHeads*s.K*s.D
			part := func(lo, hi int, shape []int, dt DType) *Tensor {
				return gr.shrink([][2]int{{lo, hi}}).reshape(shape).Cast(dt)
			}
			return grads(
				part(0, nq, s.QShape(), qd.node.DType),
				part(nq, nq+nk, s.KShape(), kd.node.DType),
				part(nq+nk, s.GradSize(), s.VShape(), vd.node.DType))
		}, q, k, v)
	}
	if dt := q.node.DType; dt == k.node.DType && dt == v.node.DType {
		return out.Cast(dt)
	}
	return out
}
