package backend

import (
	"fmt"
	"slices"
)

// AttentionShape is the geometry of an attention kernel (see OpAttention):
// queries [B..., H, Q, D], keys [B..., HK, K, D] and values
// [B..., HK, K, DV].
type AttentionShape struct {
	Batch   []int // the leading dimensions, B...
	Heads   int   // query heads, H
	KVHeads int   // key and value heads, HK, which divide H
	Q, K    int   // query and key positions
	D, DV   int   // features of a query or key, and of a value
}

// NewAttentionShape checks the shapes of the queries, keys and values of
// an attention, and of its mask unless mask is nil, and returns the
// attention's geometry.
func NewAttentionShape(q, k, v, mask []int) (AttentionShape, error) {
	r := len(q)
	bad := func() (AttentionShape, error) {
		return AttentionShape{}, fmt.Errorf("attention of queries %v, keys %v, values %v and mask %v: %w", q, k, v, mask, ErrShape)
	}
	if r < 3 || len(k) != r || len(v) != r || !slices.Equal(q[:r-3], k[:r-3]) || !slices.Equal(k[:r-1], v[:r-1]) {
		return bad()
	}
	s := AttentionShape{
		Batch: slices.Clone(q[:r-3]), Heads: q[r-3], KVHeads: k[r-3],
		Q: q[r-2], K: k[r-2], D: q[r-1], DV: v[r-1],
	}
	if k[r-1] != s.D || s.KVHeads < 1 || s.Heads%s.KVHeads != 0 {
		return bad()
	}
	if mask != nil && !slices.Equal(mask, slices.Concat(s.Batch, []int{s.Heads, s.Q, s.K})) {
		return bad()
	}
	// The sizes of the output and of the gradients must be addressable.
	for _, shape := range [][]int{{s.Size()}, {s.GradSize()}, q, k, v} {
		if _, ok := NumElements(shape); !ok {
			return bad()
		}
	}
	return s, nil
}

// Group returns H/HK, the number of query heads that read each key head.
func (s AttentionShape) Group() int { return s.Heads / s.KVHeads }

// Batches returns the number of items in the leading dimensions.
func (s AttentionShape) Batches() int {
	n, _ := NumElements(s.Batch)
	return n
}

// OutShape returns the shape of the result, [B..., H, Q, DV].
func (s AttentionShape) OutShape() []int {
	return slices.Concat(s.Batch, []int{s.Heads, s.Q, s.DV})
}

// LSEShape returns the shape of the logsumexp, [B..., H, Q].
func (s AttentionShape) LSEShape() []int { return slices.Concat(s.Batch, []int{s.Heads, s.Q}) }

// QShape, KShape and VShape return the shapes of the queries, keys and
// values.
func (s AttentionShape) QShape() []int { return slices.Concat(s.Batch, []int{s.Heads, s.Q, s.D}) }
func (s AttentionShape) KShape() []int { return slices.Concat(s.Batch, []int{s.KVHeads, s.K, s.D}) }
func (s AttentionShape) VShape() []int { return slices.Concat(s.Batch, []int{s.KVHeads, s.K, s.DV}) }

// size returns the number of elements of shape, or -1 if it overflows.
func size(shape []int) int {
	n, ok := NumElements(shape)
	if !ok {
		return -1
	}
	return n
}

// Size returns the number of elements of an OpAttention's output: the
// result followed by the logsumexp, or -1 if it overflows.
func (s AttentionShape) Size() int {
	o, l := size(s.OutShape()), size(s.LSEShape())
	if o < 0 || l < 0 || o > maxInt-l {
		return -1
	}
	return o + l
}

// GradSize returns the number of elements of an OpAttentionGrad's output:
// the gradients of the queries, keys and values, or -1 if it overflows.
func (s AttentionShape) GradSize() int {
	n := 0
	for _, shape := range [][]int{s.QShape(), s.KShape(), s.VShape()} {
		m := size(shape)
		if m < 0 || n > maxInt-m {
			return -1
		}
		n += m
	}
	return n
}

// SplitHeads returns v, a view of [B..., heads, X, Y] whose heads are the
// query heads H or the key heads HK, as a view of [B..., HK, G, X, Y],
// where G = H/HK: query head hk·G + g and key head hk are both element
// [..., hk, g, ...], for every g of a key head. Kernels walk the leading
// dimensions of these views together, one key head at a time.
func (s AttentionShape) SplitHeads(v View) View {
	r := len(v.Shape)
	g := s.Group()
	out := View{Offset: v.Offset}
	out.Shape = slices.Concat(v.Shape[:r-3], []int{s.KVHeads, g}, v.Shape[r-2:])
	stride := v.Strides[r-3]
	if v.Shape[r-3] == s.Heads {
		out.Strides = slices.Concat(v.Strides[:r-3], []int{g * stride, stride}, v.Strides[r-2:])
	} else {
		out.Strides = slices.Concat(v.Strides[:r-3], []int{stride, 0}, v.Strides[r-2:])
	}
	return out
}
