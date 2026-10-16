package main

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Config is the shape of a GPT-2 model.
type Config struct {
	Vocab   int // tokens: 256 for the byte-level tokenizer
	Context int // positions the model attends over
	Embed   int // features of each position
	Heads   int // attention heads, which divide Embed
	Layers  int // decoder blocks
}

// sizes are the model sizes the -model flag names. "gpt2" is GPT-2 small
// with a byte-level vocabulary.
var sizes = map[string]Config{
	"tiny":  {Embed: 64, Heads: 4, Layers: 2},
	"small": {Embed: 128, Heads: 4, Layers: 4},
	"base":  {Embed: 256, Heads: 8, Layers: 6},
	"gpt2":  {Embed: 768, Heads: 12, Layers: 12},
}

func (c Config) validate() error {
	if c.Vocab <= 0 || c.Context <= 0 || c.Embed <= 0 || c.Heads <= 0 || c.Layers <= 0 || c.Embed%c.Heads != 0 {
		return fmt.Errorf("gpt2: model %+v: %w", c, tensor.ErrShape)
	}
	return nil
}

// GPT is GPT-2: token and position embeddings, pre-norm decoder blocks
// with causal attention and GELU MLPs, a final LayerNorm, and an output
// projection that shares the token embedding's weights.
type GPT struct {
	Config Config        `nn:"-"`
	Tok    *nn.Embedding // [Vocab, Embed], also the output projection
	Pos    *nn.Embedding // [Context, Embed]
	Blocks []*nn.DecoderBlock
	Norm   *nn.LayerNorm
}

// NewGPT returns a GPT-2 model of shape c initialized as GPT-2 is: the
// embeddings from N(0, 0.02²) and the rest as PyTorch initializes its
// layers, from seed.
func NewGPT(c Config, seed uint64) (*GPT, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	g := &GPT{
		Config: c,
		Tok:    &nn.Embedding{Weight: tensor.NewParam(tensor.Randn(seed, c.Vocab, c.Embed).Mul(tensor.Scalar[float32](0.02)))},
		Pos:    &nn.Embedding{Weight: tensor.NewParam(tensor.Randn(seed+1, c.Context, c.Embed).Mul(tensor.Scalar[float32](0.02)))},
		Norm:   nn.NewLayerNorm(c.Embed),
	}
	for i := range c.Layers {
		g.Blocks = append(g.Blocks, nn.NewGPT2Block(c.Embed, c.Heads, nn.Seed(seed+2+uint64(i))))
	}
	// Realize the scaled embeddings once, so a step reads their values
	// rather than recomputing them.
	if err := tensor.Realize(context.Background(), g.Tok.Weight.Value(), g.Pos.Weight.Value()); err != nil {
		return nil, err
	}
	return g, nil
}

// positions returns the Int32 positions [offset, offset+n) as [1, n] on
// the device of like.
func positions(offset, n int, like *tensor.Tensor) *tensor.Tensor {
	p := make([]int32, n)
	for i := range p {
		p[i] = int32(offset + i)
	}
	return tensor.FromSlice(p, 1, n).To(like.Device())
}

// embed returns the embeddings of the Int32 tokens idx, [B, S], at
// positions from offset.
func (g *GPT) embed(idx *tensor.Tensor, offset int) *tensor.Tensor {
	if err := idx.Err(); err != nil {
		return idx
	}
	s := idx.Shape()
	if len(s) != 2 || offset+s[1] > g.Config.Context {
		return tensor.Failed(fmt.Errorf("gpt2: tokens %v at position %d for a context of %d: %w", s, offset, g.Config.Context, tensor.ErrShape))
	}
	return g.Tok.Forward(idx).Add(g.Pos.Forward(positions(offset, s[1], idx)))
}

// head returns the logits of x, [..., Embed], over the vocabulary.
func (g *GPT) head(x *tensor.Tensor) *tensor.Tensor {
	return g.Norm.Forward(x).MatMulT(g.Tok.Weight.Value())
}

// Forward maps Int32 tokens, [B, S], to logits, [B, S, Vocab].
func (g *GPT) Forward(idx *tensor.Tensor) *tensor.Tensor {
	x := g.embed(idx, 0)
	for _, b := range g.Blocks {
		x = b.Forward(x)
	}
	return g.head(x)
}

// NewCaches returns a key and value cache for each block, for batch
// sequences of up to the model's context.
func (g *GPT) NewCaches(batch int) []*nn.KVCache {
	c := g.Config
	caches := make([]*nn.KVCache, c.Layers)
	for i := range caches {
		caches[i] = nn.NewKVCache(batch, c.Heads, c.Context, c.Embed/c.Heads, g.Tok.Weight.Value().DType())
	}
	return caches
}

// Decode runs the tokens idx, [B, S], which follow those the caches hold,
// appends their keys and values to the caches and returns the logits of
// the last of them, [B, Vocab].
func (g *GPT) Decode(ctx context.Context, idx *tensor.Tensor, caches []*nn.KVCache) (*tensor.Tensor, error) {
	if len(caches) != len(g.Blocks) {
		return nil, fmt.Errorf("gpt2: %d caches for %d blocks: %w", len(caches), len(g.Blocks), tensor.ErrShape)
	}
	x := g.embed(idx, caches[0].Len())
	for i, b := range g.Blocks {
		// A block's Decode computes the keys and values it caches alone.
		// Computing its input first keeps every later block from running
		// the products of the blocks before it once more.
		if err := x.Realize(ctx); err != nil {
			return nil, err
		}
		var err error
		if x, err = b.Decode(ctx, x, caches[i]); err != nil {
			return nil, err
		}
	}
	s := x.Shape()
	last := x.Shrink([][2]int{{0, s[0]}, {s[1] - 1, s[1]}, {0, s[2]}}).Reshape(s[0], s[2])
	return g.head(last), nil
}

// NumParams returns the number of parameters of g.
func (g *GPT) NumParams() int {
	n := 0
	for _, p := range nn.Parameters(g) {
		n += p.Value().Size()
	}
	return n
}
