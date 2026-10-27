package text

import (
	"fmt"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// BERTConfig sizes a BERT.
type BERTConfig struct {
	Dim      int // each token's vector
	Depth    int // encoder layers
	Heads    int // attention heads; Dim must be a multiple
	Hidden   int // the feed-forward width, usually 4·Dim
	MaxLen   int // the most tokens a text may have
	Types    int // segment types, 2 for sentence pairs
	Dropout  float64
	LayerEps float64
}

// BERTBase is Devlin et al.'s BERT-base: 768 numbers, 12 layers of 12
// heads, 512 tokens, 110 million parameters with its 30,522-token
// vocabulary.
var BERTBase = BERTConfig{Dim: 768, Depth: 12, Heads: 12, Hidden: 3072, MaxLen: 512, Types: 2, Dropout: 0.1, LayerEps: 1e-12}

// BERTMini is Turc et al.'s BERT-mini, 256 numbers in 4 layers of 4
// heads, and BERTTiny, 128 numbers in 2 layers of 2: small enough to teach
// from scratch on a few thousand texts.
var (
	BERTMini = BERTConfig{Dim: 256, Depth: 4, Heads: 4, Hidden: 1024, MaxLen: 512, Types: 2, Dropout: 0.1, LayerEps: 1e-12}
	BERTTiny = BERTConfig{Dim: 128, Depth: 2, Heads: 2, Hidden: 512, MaxLen: 512, Types: 2, Dropout: 0.1, LayerEps: 1e-12}
)

// BERT is a transformer encoder that reads token ids, as Devlin et al.'s
// BERT does: word, position and segment embeddings, post-norm encoder
// layers with GELU, and a pooler over the [CLS] token, here followed by a
// classifier. Padding, id PadID, is hidden from attention.
type BERT struct {
	Words, Positions, Types *nn.Embedding
	Norm                    *nn.LayerNorm
	Drop                    *nn.Dropout // or nil
	Layers                  []*nn.TransformerEncoderLayer
	Pooler                  *nn.Linear
	Head                    *nn.Linear
	// PadID is the id of [PAD]; 0 in BERT's vocabularies and BuildVocab's.
	PadID int32
	cfg   BERTConfig
}

// NewBERT returns a BERT of shape c over vocab tokens, with a head for
// classes classes.
func NewBERT(c BERTConfig, vocab, classes int) (*BERT, error) {
	if c.Dim < 1 || c.Heads < 1 || c.Dim%c.Heads != 0 || c.Depth < 1 || c.Hidden < 1 || c.MaxLen < 1 || vocab < 1 {
		return nil, fmt.Errorf("text: BERT %+v of %d tokens: %w", c, vocab, tensor.ErrShape)
	}
	eps := c.LayerEps
	if eps == 0 {
		eps = 1e-12
	}
	b := &BERT{
		Words:     nn.NewEmbedding(vocab, c.Dim),
		Positions: nn.NewEmbedding(c.MaxLen, c.Dim),
		Types:     nn.NewEmbedding(max(c.Types, 1), c.Dim),
		Norm:      nn.NewLayerNorm(c.Dim, nn.Eps(eps)),
		Pooler:    nn.NewLinear(c.Dim, c.Dim),
		Head:      nn.NewLinear(c.Dim, classes),
		cfg:       c,
	}
	if c.Dropout > 0 {
		b.Drop = nn.NewDropout(c.Dropout)
	}
	for range c.Depth {
		l := nn.NewTransformerEncoderLayer(c.Dim, c.Heads, c.Hidden, nn.Eps(eps))
		l.Activation = nn.GELU()
		if c.Dropout > 0 {
			l.Dropout.P, l.Dropout1.P, l.Dropout2.P = c.Dropout, c.Dropout, c.Dropout
		} else {
			l.Dropout, l.Dropout1, l.Dropout2 = nil, nil, nil
		}
		b.Layers = append(b.Layers, l)
	}
	// BERT draws its embeddings from N(0, 0.02²).
	small := tensor.Full(tensor.Float32, 0.02)
	for _, e := range []*nn.Embedding{b.Words, b.Positions, b.Types} {
		if err := e.Weight.Set(e.Weight.Value().Mul(small)); err != nil {
			return nil, err
		}
	}
	return b, nil
}

// Width returns Dim.
func (b *BERT) Width() int { return b.cfg.Dim }

// ids reads x, token ids of any number dtype shaped [B, S], as Int32.
func (b *BERT) ids(x *tensor.Tensor) (*tensor.Tensor, error) {
	if err := x.Err(); err != nil {
		return nil, err
	}
	if x.Rank() != 2 || x.Shape()[1] > b.cfg.MaxLen || x.Shape()[1] < 1 {
		return nil, fmt.Errorf("text: BERT of %v, want [batch, at most %d tokens]: %w", x.Shape(), b.cfg.MaxLen, tensor.ErrShape)
	}
	if x.DType() != tensor.Int32 {
		x = x.Cast(tensor.Int32)
	}
	return x, nil
}

// Encode maps token ids [B, S] to each token's vector, [B, S, Dim].
func (b *BERT) Encode(x *tensor.Tensor) *tensor.Tensor {
	ids, err := b.ids(x)
	if err != nil {
		return tensor.Failed(err)
	}
	n, s := ids.Shape()[0], ids.Shape()[1]
	pos := tensor.Arange[int32](0, int32(s), 1).Reshape(1, s).Expand(n, s)
	z := b.Words.Forward(ids).Add(b.Positions.Forward(pos)).Add(b.Types.Forward(tensor.Zeros(tensor.Int32, n, s)))
	z = b.Norm.Forward(z)
	if b.Drop != nil {
		z = b.Drop.Forward(z)
	}
	// Each query sees the keys that aren't padding: [B, 1, 1, S].
	keep := tensor.Full(tensor.Int32, 1).Sub(ids.Equal(tensor.Full(tensor.Int32, float64(b.PadID)))).Reshape(n, 1, 1, s)
	for _, l := range b.Layers {
		z = l.Encode(z, tensor.BoolMask(keep))
	}
	return z
}

// Features maps token ids [B, S] to the pooled [CLS] vector, [B, Dim].
func (b *BERT) Features(x *tensor.Tensor) *tensor.Tensor {
	z := b.Encode(x)
	if err := z.Err(); err != nil {
		return z
	}
	n, d := z.Shape()[0], z.Shape()[2]
	cls := z.Shrink([][2]int{{0, n}, {0, 1}, {0, d}}).Reshape(n, d)
	return b.Pooler.Forward(cls).Tanh()
}

// Forward maps token ids [B, S] to [B, classes] logits.
func (b *BERT) Forward(x *tensor.Tensor) *tensor.Tensor {
	f := b.Features(x)
	if b.Drop != nil {
		f = b.Drop.Forward(f)
	}
	return b.Head.Forward(f)
}

// SetHead starts a new classifier for classes classes, for transfer
// learning.
func (b *BERT) SetHead(classes int) { b.Head = nn.NewLinear(b.cfg.Dim, classes) }

// HeadParameters returns the classifier's weight and bias.
func (b *BERT) HeadParameters() []*tensor.Param { return []*tensor.Param{b.Head.Weight, b.Head.Bias} }
