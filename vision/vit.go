package vision

import (
	"fmt"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// ViT is Dosovitskiy et al.'s Vision Transformer, as torchvision's
// vit_b_16 builds it: the picture is cut into square patches, each
// becomes a vector, and transformer encoder layers read them with a
// class token whose final vector is the picture's.
type ViT struct {
	Patch  *nn.Conv2d    // [dim, C, p, p] with stride p
	Class  *tensor.Param // [1, 1, dim]
	Pos    *tensor.Param // [1, 1+patches, dim]
	Layers []*nn.TransformerEncoderLayer
	Norm   *nn.LayerNorm
	Head   *head
	size   int // the picture's side
	dim    int
	drop   float64
}

// ViTConfig sizes a ViT.
type ViTConfig struct {
	Size   int // the side of the square pictures it reads
	Patch  int // the side of each patch; Size must be a multiple
	Dim    int // each patch's vector
	Depth  int // encoder layers
	Heads  int // attention heads; Dim must be a multiple
	Hidden int // the feed-forward width, usually 4·Dim
}

// ViTB16 is ViT-B/16: 224-pixel pictures in 16-pixel patches, 768
// numbers, 12 layers of 12 heads, 86 million parameters.
var ViTB16 = ViTConfig{Size: 224, Patch: 16, Dim: 768, Depth: 12, Heads: 12, Hidden: 3072}

// ViTTiny is a ViT small enough to teach from scratch on a few thousand
// pictures: 64-pixel pictures in 8-pixel patches, 192 numbers, 6 layers
// of 3 heads.
var ViTTiny = ViTConfig{Size: 64, Patch: 8, Dim: 192, Depth: 6, Heads: 3, Hidden: 768}

// NewViT returns a ViT of shape v for classes classes. Channels and
// Dropout apply.
func NewViT(v ViTConfig, classes int, opts ...Option) (*ViT, error) {
	c := options(opts, config{})
	if v.Size < 1 || v.Patch < 1 || v.Size%v.Patch != 0 || v.Heads < 1 || v.Dim%v.Heads != 0 || v.Depth < 1 || v.Hidden < 1 {
		return nil, fmt.Errorf("vision: ViT %+v: %w", v, tensor.ErrShape)
	}
	n := (v.Size / v.Patch) * (v.Size / v.Patch)
	m := &ViT{
		Patch: nn.NewConv2d(c.in, v.Dim, v.Patch, nn.Stride(v.Patch)),
		Class: tensor.NewParam(tensor.Zeros(tensor.Float32, 1, 1, v.Dim)),
		// torchvision draws the positions from N(0, 0.02²).
		Pos:  tensor.NewParam(tensor.Randn(uint64(n), 1, 1+n, v.Dim).Mul(tensor.Full(tensor.Float32, 0.02))),
		Norm: nn.NewLayerNorm(v.Dim, nn.Eps(1e-6)),
		Head: newHead(v.Dim, classes, c.drop),
		size: v.Size, dim: v.Dim, drop: c.drop,
	}
	for range v.Depth {
		l := nn.NewTransformerEncoderLayer(v.Dim, v.Heads, v.Hidden, nn.Eps(1e-6))
		l.NormFirst, l.Activation = true, nn.GELU()
		l.Dropout, l.Dropout1, l.Dropout2 = nil, nil, nil
		m.Layers = append(m.Layers, l)
	}
	return m, nil
}

// Features maps [N, C, Size, Size] to the class token's [N, Dim].
func (m *ViT) Features(x *tensor.Tensor) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	if s := x.Shape(); len(s) != 4 || s[2] != m.size || s[3] != m.size {
		return tensor.Failed(fmt.Errorf("vision: ViT of %v, want [N, C, %d, %d]: %w", s, m.size, m.size, tensor.ErrShape))
	}
	n := x.Shape()[0]
	p := m.Patch.Forward(x).Flatten(2, -1).Transpose(1, 2) // [N, patches, Dim]
	cls := m.Class.Value().Expand(n, 1, m.dim)
	z := tensor.Cat(1, cls, p).Add(m.Pos.Value())
	for _, l := range m.Layers {
		z = l.Forward(z)
	}
	z = m.Norm.Forward(z)
	return z.Shrink([][2]int{{0, n}, {0, 1}, {0, m.dim}}).Reshape(n, m.dim)
}

// Forward maps [N, C, Size, Size] to [N, classes] logits.
func (m *ViT) Forward(x *tensor.Tensor) *tensor.Tensor { return m.Head.Forward(m.Features(x)) }

// Width returns Dim.
func (m *ViT) Width() int { return m.dim }

// SetHead starts a new head for classes classes.
func (m *ViT) SetHead(classes int) { m.Head = newHead(m.dim, classes, m.drop) }

// HeadParameters returns the head's weight and bias.
func (m *ViT) HeadParameters() []*tensor.Param { return m.Head.parameters() }
