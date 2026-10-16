package nn

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/tensor"
)

// RMSNorm divides the last dimension by its root mean square, then scales
// it, like torch.nn.RMSNorm. It normalizes in float32 and rounds the
// result to the input's dtype before the scale, as Llama's normalization does.
type RMSNorm struct {
	Weight *tensor.Param // [dim], or nil
	Eps    float64
}

// NewRMSNorm returns an RMSNorm over a last dimension of size dim, with
// scale 1 and Eps the machine epsilon of float32, PyTorch's default for
// float32 inputs; Llama models set Eps to 1e-5 or 1e-6. NoBias leaves out
// the scale.
func NewRMSNorm(dim int, opts ...Option) *RMSNorm {
	o := resolve(opts, options{eps: 0x1p-23})
	r := &RMSNorm{Eps: o.eps}
	if !o.noBias {
		r.Weight = tensor.NewParam(tensor.Ones(tensor.Float32, dim))
	}
	return r
}

// Forward normalizes x over its last dimension.
func (r *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x.Err() != nil {
		return x
	}
	f := x.Cast(tensor.Float32)
	ms := f.Mul(f).Mean(-1).Unsqueeze(-1)
	y := f.Mul(ms.Add(scalar(r.Eps)).Sqrt().Recip()).Cast(x.DType())
	if r.Weight != nil {
		y = y.Mul(r.Weight.Value())
	}
	return y
}

// RotaryEmbedding rotates pairs of features of queries and keys by angles
// proportional to their positions, so that their products depend on the
// distance between positions: rotary position embedding (RoPE). Pair i
// turns by position·θ^(-2i/Dim).
type RotaryEmbedding struct {
	// Dim is the number of leading features of a head that rotate, which
	// is even; the rest pass through, as in GPT-NeoX's partial rotary.
	Dim int
	// Base is θ: 10000 in Llama 2, 500000 in Llama 3.
	Base float64
	// Interleaved rotates the pairs (2i, 2i+1), as GPT-J and Meta's Llama
	// code do. Otherwise feature i turns with feature i + Dim/2, as in
	// GPT-NeoX and Hugging Face's Llama, whose checkpoints permute the
	// query and key weights to match.
	Interleaved bool
	// Scaling, if set, maps the inverse frequency of each pair to the one
	// used, to extend the context: see LinearRoPEScaling and
	// Llama3RoPEScaling.
	Scaling func(invFreq float64) float64

	mu     sync.Mutex
	tables map[tensor.Device]*ropeTable `nn:"-"`
}

// ropeTable holds the cosines and sines of the angles of positions
// [0, n) for one device, each [n, Dim/2].
type ropeTable struct {
	n        int
	cos, sin *tensor.Tensor
}

// NewRotaryEmbedding returns rotary embeddings of the first dim features
// of each head with base θ, pairing feature i with feature i + dim/2.
func NewRotaryEmbedding(dim int, base float64) *RotaryEmbedding {
	return &RotaryEmbedding{Dim: dim, Base: base}
}

// LinearRoPEScaling divides every frequency by factor, which stretches
// positions by it: position interpolation.
func LinearRoPEScaling(factor float64) func(float64) float64 {
	return func(f float64) float64 { return f / factor }
}

// Llama3RoPEScaling scales frequencies as Llama 3.1 does: wavelengths
// shorter than original/high stay, those longer than original/low are
// divided by factor, and those between are interpolated smoothly. Llama
// 3.1 uses factor 8, low 1, high 4 and an original context of 8192.
func Llama3RoPEScaling(factor, low, high float64, original int) func(float64) float64 {
	lowWave, highWave := float64(original)/low, float64(original)/high
	return func(f float64) float64 {
		wave := 2 * math.Pi / f
		switch {
		case wave < highWave:
			return f
		case wave > lowWave:
			return f / factor
		}
		smooth := (float64(original)/wave - low) / (high - low)
		return (1-smooth)*f/factor + smooth*f
	}
}

// table returns the cosines and sines of positions [0, n) on d, growing
// the table as needed.
func (r *RotaryEmbedding) table(n int, d tensor.Device) (cos, sin *tensor.Tensor) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if t := r.tables[d]; t != nil && t.n >= n {
		return t.cos, t.sin
	}
	if r.tables == nil {
		r.tables = map[tensor.Device]*ropeTable{}
	}
	n = max(n, 2*tableLen(r.tables[d]), 64)
	half := r.Dim / 2
	inv := make([]float64, half)
	for i := range inv {
		inv[i] = math.Pow(r.Base, -float64(2*i)/float64(r.Dim))
		if r.Scaling != nil {
			inv[i] = r.Scaling(inv[i])
		}
	}
	c, s := make([]float32, n*half), make([]float32, n*half)
	for p := range n {
		for i, f := range inv {
			sn, cs := math.Sincos(float64(p) * f)
			c[p*half+i], s[p*half+i] = float32(cs), float32(sn)
		}
	}
	t := &ropeTable{n: n, cos: tensor.FromSlice(c, n, half).To(d), sin: tensor.FromSlice(s, n, half).To(d)}
	r.tables[d] = t
	return t.cos, t.sin
}

func tableLen(t *ropeTable) int {
	if t == nil {
		return 0
	}
	return t.n
}

// Apply rotates x, of shape [..., S, head features], whose positions
// start at offset.
func (r *RotaryEmbedding) Apply(x *tensor.Tensor, offset int) *tensor.Tensor {
	if x.Err() != nil {
		return x
	}
	shape := x.Shape()
	rank := len(shape)
	if rank < 2 || r.Dim <= 0 || r.Dim%2 != 0 || r.Dim > shape[rank-1] || offset < 0 || r.Base <= 0 {
		return tensor.Failed(fmt.Errorf("nn: rotary embedding of %d features with base %v at offset %d for %v: %w", r.Dim, r.Base, offset, shape, tensor.ErrShape))
	}
	s, half := shape[rank-2], r.Dim/2
	cos, sin := r.table(offset+s, x.Device())
	cos = cos.Shrink([][2]int{{offset, offset + s}, {0, half}})
	sin = sin.Shrink([][2]int{{offset, offset + s}, {0, half}})
	if dt := x.DType(); dt != tensor.Float32 {
		cos, sin = cos.Cast(dt), sin.Cast(dt)
	}
	parts := x.Split(-1, r.Dim, shape[rank-1]-r.Dim)
	rot, rest := parts[0], parts[1]
	var out *tensor.Tensor
	if r.Interleaved {
		pairs := rot.Reshape(slices.Concat(shape[:rank-1], []int{half, 2})...).Split(-1, 1, 1)
		a, b := pairs[0].Squeeze(-1), pairs[1].Squeeze(-1)
		out = tensor.Stack(-1, a.Mul(cos).Sub(b.Mul(sin)), a.Mul(sin).Add(b.Mul(cos))).Reshape(rot.Shape()...)
	} else {
		halves := rot.Split(-1, half, half)
		a, b := halves[0], halves[1]
		out = tensor.Cat(-1, a.Mul(cos).Sub(b.Mul(sin)), b.Mul(cos).Add(a.Mul(sin)))
	}
	if r.Dim == shape[rank-1] {
		return out
	}
	return tensor.Cat(-1, out, rest)
}

// splitHeads maps [B, S, heads·dim] to [B, heads, S, dim].
func splitHeads(x *tensor.Tensor, heads, dim int) *tensor.Tensor {
	s := x.Shape()
	return x.Reshape(s[0], s[1], heads, dim).Transpose(1, 2)
}

// mergeHeads maps [B, heads, S, dim] to [B, S, heads·dim].
func mergeHeads(x *tensor.Tensor) *tensor.Tensor {
	s := x.Shape()
	return x.Transpose(1, 2).Reshape(s[0], s[2], s[1]*s[3])
}

// MultiheadAttention is attention with learned projections of the query,
// key and value into Heads heads and of the heads' results back, like
// torch.nn.MultiheadAttention with batch_first=True and no dropout.
type MultiheadAttention struct {
	InProjWeight *tensor.Param // [3·embed, embed]: the query, key and value projections
	InProjBias   *tensor.Param // [3·embed], or nil
	OutProj      *Linear       // embed → embed
	Heads        int
}

// NewMultiheadAttention returns attention over embed features in heads
// heads, which divide embed, initialized as PyTorch does: the input
// projections Xavier-uniform, the output projection like a Linear, and
// the biases zero. NoBias leaves out the biases.
func NewMultiheadAttention(embed, heads int, opts ...Option) *MultiheadAttention {
	o := resolve(opts, options{})
	bound := math.Sqrt(6 / float64(max(4*embed, 1)))
	m := &MultiheadAttention{InProjWeight: uniform(o, bound, 3*embed, embed), OutProj: NewLinear(embed, embed, opts...), Heads: heads}
	if !o.noBias {
		m.InProjBias = tensor.NewParam(tensor.Zeros(tensor.Float32, 3*embed))
		m.OutProj.Bias = tensor.NewParam(tensor.Zeros(tensor.Float32, embed))
	}
	return m
}

// Forward returns the self-attention of x, [B, S, embed] or [S, embed].
func (m *MultiheadAttention) Forward(x *tensor.Tensor) *tensor.Tensor { return m.Attend(x, x, x) }

// Attend returns the attention of query, [B, L, embed], to key and
// value, [B, S, embed], or of their unbatched forms, [L, embed] and
// [S, embed]. Its options, such as tensor.Causal or tensor.BoolMask, are
// those of tensor.ScaledDotProductAttention: a boolean mask is non-zero
// where a query may attend, the opposite of the boolean attn_mask of
// torch.nn.MultiheadAttention.
func (m *MultiheadAttention) Attend(query, key, value *tensor.Tensor, opts ...tensor.AttentionOption) *tensor.Tensor {
	for _, x := range []*tensor.Tensor{query, key, value} {
		if x.Err() != nil {
			return x
		}
	}
	w := m.InProjWeight.Value()
	if w.Err() != nil {
		return w
	}
	embed := w.Shape()[1]
	r := query.Rank()
	if r == 2 && key.Rank() == 2 && value.Rank() == 2 {
		return m.Attend(query.Unsqueeze(0), key.Unsqueeze(0), value.Unsqueeze(0), opts...).Squeeze(0)
	}
	if m.Heads <= 0 || embed%m.Heads != 0 || r != 3 || key.Rank() != 3 || value.Rank() != 3 {
		return tensor.Failed(fmt.Errorf("nn: attention of %v, %v and %v in %d heads of %d features: %w",
			query.Shape(), key.Shape(), value.Shape(), m.Heads, embed, tensor.ErrShape))
	}
	project := func(x *tensor.Tensor, part int) *tensor.Tensor {
		rows := [][2]int{{part * embed, (part + 1) * embed}, {0, embed}}
		y := x.MatMul(w.Shrink(rows).Transpose(0, 1))
		if m.InProjBias != nil {
			y = y.Add(m.InProjBias.Value().Shrink(rows[:1]))
		}
		return splitHeads(y, m.Heads, embed/m.Heads)
	}
	out := tensor.ScaledDotProductAttention(project(query, 0), project(key, 1), project(value, 2), opts...)
	return m.OutProj.Forward(mergeHeads(out))
}

// KVCache holds the keys and values of the positions a decoder has seen,
// so that each decoding step computes those of its new positions alone.
// Its memory, for a fixed number of positions, is allocated once: each
// step writes its keys and values into it in place.
type KVCache struct {
	keys, values *tensor.Tensor // [B, KV heads, capacity, dim]
	shape        [4]int         // B, KV heads, capacity, dim
	dt           tensor.DType
	n            int
	err          error
}

// NewKVCache returns an empty cache of capacity positions of batch
// sequences, with kvHeads heads of dim features of dtype dt.
func NewKVCache(batch, kvHeads, capacity, dim int, dt tensor.DType) *KVCache {
	c := &KVCache{shape: [4]int{batch, kvHeads, capacity, dim}, dt: dt}
	c.alloc()
	return c
}

func (c *KVCache) alloc() {
	// A clone has memory of its own, which the appends write over.
	c.keys = tensor.Zeros(c.dt, c.shape[:]...).Clone()
	c.values = tensor.Zeros(c.dt, c.shape[:]...).Clone()
	c.err = nil
}

// Len returns the number of positions the cache holds.
func (c *KVCache) Len() int { return c.n }

// Cap returns the number of positions the cache can hold.
func (c *KVCache) Cap() int { return c.shape[2] }

// Reset empties the cache for a new sequence.
func (c *KVCache) Reset() { c.Truncate(0) }

// Truncate keeps the first n positions the cache holds and drops the
// rest, as rejecting speculated tokens or reusing a prompt's prefix does.
func (c *KVCache) Truncate(n int) {
	c.n = min(max(n, 0), c.n)
	if c.err != nil {
		c.n = 0
		c.alloc()
	}
}

// Append computes k and v, [B, KV heads, S, dim], writes them into the
// cache after the positions it holds, and returns the keys and values of
// every position held, [B, KV heads, Len()+S, dim]. The returned tensors
// read the cache's memory, which the next Append writes over: compute
// what reads them, such as a decoding step's logits, before appending
// again. k and v are cast to the cache's dtype; no gradient flows through
// the cache. A failed Append leaves the cache unusable until Reset.
func (c *KVCache) Append(ctx context.Context, k, v *tensor.Tensor) (keys, values *tensor.Tensor, err error) {
	if c.err != nil {
		return nil, nil, c.err
	}
	if err := errors.Join(k.Err(), v.Err()); err != nil {
		return nil, nil, err
	}
	ks, vs := k.Shape(), v.Shape()
	if len(ks) != 4 || len(vs) != 4 || ks[0] != c.shape[0] || ks[1] != c.shape[1] || ks[3] != c.shape[3] || !slices.Equal(ks, vs) {
		return nil, nil, fmt.Errorf("nn: append keys %v and values %v to a cache of %v: %w", ks, vs, c.shape, tensor.ErrShape)
	}
	s := ks[2]
	if c.n+s > c.shape[2] {
		return nil, nil, fmt.Errorf("nn: append %d positions to a cache holding %d of %d: %w", s, c.n, c.shape[2], tensor.ErrShape)
	}
	block := [][2]int{{0, c.shape[0]}, {0, c.shape[1]}, {c.n, c.n + s}, {0, c.shape[3]}}
	nk := c.keys.SliceScatter(k.Detach().Cast(c.dt), block)
	nv := c.values.SliceScatter(v.Detach().Cast(c.dt), block)
	if err := tensor.RealizeInto(ctx, []*tensor.Tensor{c.keys, c.values}, []*tensor.Tensor{nk, nv}); err != nil {
		c.err = fmt.Errorf("nn: cache append failed: %w", err)
		return nil, nil, err
	}
	c.keys, c.values = nk, nv
	c.n += s
	held := [][2]int{{0, c.shape[0]}, {0, c.shape[1]}, {0, c.n}, {0, c.shape[3]}}
	return c.keys.Shrink(held), c.values.Shrink(held), nil
}

// SelfAttention is causal multi-head self-attention as decoders use it,
// with grouped-query attention when KVHeads < Heads and rotary position
// embeddings when Rotary is set: the attention of GPT-2 and of Llama.
type SelfAttention struct {
	// QKV projects the input to the queries of every head, then the keys
	// and the values of every key head: embed → (Heads + 2·KVHeads)·HeadDim.
	QKV *Linear
	// Out projects the heads' results back: Heads·HeadDim → embed.
	Out            *Linear
	Heads, KVHeads int
	HeadDim        int
	Rotary         *RotaryEmbedding // or nil
	NotCausal      bool             // attend to every position
}

// NewSelfAttention returns causal self-attention over embed features with
// heads query heads of embed/heads features and kvHeads key and value
// heads, which divide heads. NoBias leaves out the projections' biases.
func NewSelfAttention(embed, heads, kvHeads int, opts ...Option) *SelfAttention {
	dim := embed / max(heads, 1)
	return &SelfAttention{
		QKV:   NewLinear(embed, (heads+2*kvHeads)*dim, opts...),
		Out:   NewLinear(heads*dim, embed, opts...),
		Heads: heads, KVHeads: kvHeads, HeadDim: dim,
	}
}

// project returns the queries, keys and values of x, [B, S, embed], with
// the rotary embeddings of positions from offset.
func (a *SelfAttention) project(x *tensor.Tensor, offset int) (q, k, v *tensor.Tensor, err error) {
	if err := x.Err(); err != nil {
		return nil, nil, nil, err
	}
	if x.Rank() != 3 || a.Heads <= 0 || a.KVHeads <= 0 || a.HeadDim <= 0 || a.Heads%a.KVHeads != 0 {
		return nil, nil, nil, fmt.Errorf("nn: self-attention of %v in %d heads and %d key heads of %d features: %w",
			x.Shape(), a.Heads, a.KVHeads, a.HeadDim, tensor.ErrShape)
	}
	qkv := a.QKV.Forward(x)
	hq, hk := a.Heads*a.HeadDim, a.KVHeads*a.HeadDim
	parts := qkv.Split(-1, hq, hk, hk)
	q = splitHeads(parts[0], a.Heads, a.HeadDim)
	k = splitHeads(parts[1], a.KVHeads, a.HeadDim)
	v = splitHeads(parts[2], a.KVHeads, a.HeadDim)
	if a.Rotary != nil {
		q, k = a.Rotary.Apply(q, offset), a.Rotary.Apply(k, offset)
	}
	return q, k, v, nil
}

func (a *SelfAttention) mask(offset int) []tensor.AttentionOption {
	if a.NotCausal {
		return nil
	}
	return []tensor.AttentionOption{tensor.CausalOffset(offset)}
}

// Forward returns the self-attention of x, [B, S, embed], whose positions
// start at 0.
func (a *SelfAttention) Forward(x *tensor.Tensor) *tensor.Tensor {
	q, k, v, err := a.project(x, 0)
	if err != nil {
		return tensor.Failed(err)
	}
	return a.Out.Forward(mergeHeads(tensor.ScaledDotProductAttention(q, k, v, a.mask(0)...)))
}

// Decode returns the self-attention of x, [B, S, embed], the positions
// after those cache holds, to them and to themselves, and appends their
// keys and values to cache (see KVCache.Append).
func (a *SelfAttention) Decode(ctx context.Context, x *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	past := cache.Len()
	q, k, v, err := a.project(x, past)
	if err != nil {
		return nil, err
	}
	keys, values, err := cache.Append(ctx, k, v)
	if err != nil {
		return nil, err
	}
	out := tensor.ScaledDotProductAttention(q, keys.Cast(q.DType()), values.Cast(q.DType()), a.mask(past)...)
	return a.Out.Forward(mergeHeads(out)), nil
}

// MLP is the feed-forward layer of a transformer block: Down(act(Up(x))),
// with GPT-2's GELU by default.
type MLP struct {
	Up, Down   *Linear
	Activation Func
}

// NewMLP returns the feed-forward layer from embed features through
// hidden features and back, with the GELU activation.
func NewMLP(embed, hidden int, opts ...Option) *MLP {
	return &MLP{Up: NewLinear(embed, hidden, opts...), Down: NewLinear(hidden, embed, opts...), Activation: GELU()}
}

// Forward maps [..., embed] to [..., embed].
func (m *MLP) Forward(x *tensor.Tensor) *tensor.Tensor {
	act := m.Activation
	if act == nil {
		act = GELU()
	}
	return m.Down.Forward(act(m.Up.Forward(x)))
}

// GatedMLP is the SwiGLU feed-forward layer of Llama:
// Down(SiLU(Gate(x)) · Up(x)).
type GatedMLP struct {
	Gate, Up, Down *Linear
}

// NewGatedMLP returns the gated feed-forward layer from embed features
// through hidden features and back, without biases, as Llama's.
func NewGatedMLP(embed, hidden int, opts ...Option) *GatedMLP {
	opts = append(slices.Clone(opts), NoBias())
	return &GatedMLP{Gate: NewLinear(embed, hidden, opts...), Up: NewLinear(embed, hidden, opts...), Down: NewLinear(hidden, embed, opts...)}
}

// Forward maps [..., embed] to [..., embed].
func (m *GatedMLP) Forward(x *tensor.Tensor) *tensor.Tensor {
	return m.Down.Forward(m.Gate.Forward(x).SiLU().Mul(m.Up.Forward(x)))
}

// DecoderBlock is a pre-norm transformer decoder block:
// h = x + Attn(Norm1(x)), then h + MLP(Norm2(h)).
type DecoderBlock struct {
	Norm1   Module
	Attn    *SelfAttention
	Norm2   Module
	MLP     Module
	Dropout *Dropout // applied to each branch before its residual sum, or nil
}

// NewGPT2Block returns the block of GPT-2: LayerNorms, causal attention
// with biases in heads heads, and a GELU MLP four times as wide.
func NewGPT2Block(embed, heads int, opts ...Option) *DecoderBlock {
	return &DecoderBlock{
		Norm1: NewLayerNorm(embed, opts...),
		Attn:  NewSelfAttention(embed, heads, heads, opts...),
		Norm2: NewLayerNorm(embed, opts...),
		MLP:   NewMLP(embed, 4*embed, opts...),
	}
}

// NewLlamaBlock returns the block of Llama: RMSNorms (with Eps 1e-5
// unless the Eps option says otherwise), causal grouped-query attention
// with heads query and kvHeads key heads and the rotary embeddings rope,
// which the blocks of a model share, and a SwiGLU MLP of hidden features,
// all without biases.
func NewLlamaBlock(embed, heads, kvHeads, hidden int, rope *RotaryEmbedding, opts ...Option) *DecoderBlock {
	opts = append([]Option{Eps(1e-5)}, opts...)
	attn := NewSelfAttention(embed, heads, kvHeads, append(slices.Clone(opts), NoBias())...)
	attn.Rotary = rope
	return &DecoderBlock{
		Norm1: NewRMSNorm(embed, opts...),
		Attn:  attn,
		Norm2: NewRMSNorm(embed, opts...),
		MLP:   NewGatedMLP(embed, hidden, opts...),
	}
}

func (b *DecoderBlock) drop(x *tensor.Tensor) *tensor.Tensor { return dropout(b.Dropout, x) }

// dropout applies d to x, or passes x through if d is nil.
func dropout(d *Dropout, x *tensor.Tensor) *tensor.Tensor {
	if d == nil {
		return x
	}
	return d.Forward(x)
}

// Forward maps [B, S, embed] to [B, S, embed].
func (b *DecoderBlock) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := x.Add(b.drop(b.Attn.Forward(b.Norm1.Forward(x))))
	return h.Add(b.drop(b.MLP.Forward(b.Norm2.Forward(h))))
}

// Decode is Forward for the positions after those cache holds, whose keys
// and values it appends to cache (see SelfAttention.Decode).
func (b *DecoderBlock) Decode(ctx context.Context, x *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	a, err := b.Attn.Decode(ctx, b.Norm1.Forward(x), cache)
	if err != nil {
		return nil, err
	}
	h := x.Add(b.drop(a))
	return h.Add(b.drop(b.MLP.Forward(b.Norm2.Forward(h)))), nil
}

// TransformerEncoderLayer is self-attention and a feed-forward layer, each
// with a residual connection and a LayerNorm, like
// torch.nn.TransformerEncoderLayer with batch_first=True: post-norm by
// default, pre-norm with NormFirst.
type TransformerEncoderLayer struct {
	SelfAttn           *MultiheadAttention
	Linear1, Linear2   *Linear
	Norm1, Norm2       *LayerNorm
	Dropout            *Dropout // after the activation, or nil
	Dropout1, Dropout2 *Dropout // on the attention and feed-forward branches, or nil
	Activation         Func     // ReLU by default
	NormFirst          bool
}

// NewTransformerEncoderLayer returns an encoder layer of dModel features
// in heads heads with a feed-forward layer of hidden features, dropout
// 0.1 and the ReLU activation, PyTorch's defaults.
func NewTransformerEncoderLayer(dModel, heads, hidden int, opts ...Option) *TransformerEncoderLayer {
	return &TransformerEncoderLayer{
		SelfAttn: NewMultiheadAttention(dModel, heads, opts...),
		Linear1:  NewLinear(dModel, hidden, opts...),
		Linear2:  NewLinear(hidden, dModel, opts...),
		Norm1:    NewLayerNorm(dModel, opts...),
		Norm2:    NewLayerNorm(dModel, opts...),
		Dropout:  NewDropout(0.1, opts...), Dropout1: NewDropout(0.1, opts...), Dropout2: NewDropout(0.1, opts...),
		Activation: ReLU(),
	}
}

// Forward maps [B, S, dModel] to [B, S, dModel].
func (l *TransformerEncoderLayer) Forward(x *tensor.Tensor) *tensor.Tensor { return l.Encode(x) }

// Encode is Forward with the options of the self-attention, such as a
// mask (see MultiheadAttention.Attend).
func (l *TransformerEncoderLayer) Encode(x *tensor.Tensor, opts ...tensor.AttentionOption) *tensor.Tensor {
	act := l.Activation
	if act == nil {
		act = ReLU()
	}
	sa := func(x *tensor.Tensor) *tensor.Tensor { return dropout(l.Dropout1, l.SelfAttn.Attend(x, x, x, opts...)) }
	ff := func(x *tensor.Tensor) *tensor.Tensor {
		return dropout(l.Dropout2, l.Linear2.Forward(dropout(l.Dropout, act(l.Linear1.Forward(x)))))
	}
	if l.NormFirst {
		x = x.Add(sa(l.Norm1.Forward(x)))
		return x.Add(ff(l.Norm2.Forward(x)))
	}
	x = l.Norm1.Forward(x.Add(sa(x)))
	return l.Norm2.Forward(x.Add(ff(x)))
}
