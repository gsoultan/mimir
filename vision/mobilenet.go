package vision

import (
	"math"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// MobileNetV2 is Sandler et al.'s network for phones, as torchvision's
// mobilenet_v2 builds it: inverted residual blocks that widen, filter
// each channel on its own and narrow again. It has 3.5 million
// parameters at width 1, a seventh of ResNet-50's, for nearly its
// accuracy.
type MobileNetV2 struct {
	Body  *nn.Sequential
	Head  *head
	width int
	drop  float64
}

// inverted is MobileNetV2's block: a 1×1 convolution widening t-fold, a
// depthwise 3×3 and a 1×1 back down, with a shortcut when the shape
// stays.
type inverted struct {
	Expand  *convBN // nil when t is 1
	Depth   *convBN
	Project *convBN
	short   bool
}

func (b *inverted) Forward(x *tensor.Tensor) *tensor.Tensor {
	y := x
	if b.Expand != nil {
		y = b.Expand.Forward(y)
	}
	y = b.Project.Forward(b.Depth.Forward(y))
	if b.short {
		y = y.Add(x)
	}
	return y
}

// channels rounds c·w to a multiple of 8, as MobileNet does, keeping at
// least 90% of it.
func channels(c int, w float64) int {
	v := max(8, int(float64(c)*w+4)/8*8)
	if float64(v) < 0.9*float64(c)*w {
		v += 8
	}
	return v
}

// NewMobileNetV2 returns a MobileNetV2 for classes classes. Width scales
// its channels and Dropout, 0.2 by default, comes before the head.
func NewMobileNetV2(classes int, opts ...Option) *MobileNetV2 {
	c := options(opts, config{drop: 0.2})
	act := relu6()
	stride := 2
	if c.small {
		stride = 1
	}
	in := channels(32, c.width)
	m := &MobileNetV2{Body: nn.NewSequential(newConvBN(c.in, in, 3, stride, 1, act)), drop: c.drop}
	// t widens, c channels, n blocks, s the first one's stride.
	for _, row := range [][4]int{{1, 16, 1, 1}, {6, 24, 2, 2}, {6, 32, 3, 2}, {6, 64, 4, 2}, {6, 96, 3, 1}, {6, 160, 3, 2}, {6, 320, 1, 1}} {
		t, out, n, s := row[0], channels(row[1], c.width), row[2], row[3]
		if c.small && out == channels(24, c.width) {
			s = 1 // keep a small picture's size one stage longer
		}
		for j := range n {
			st := 1
			if j == 0 {
				st = s
			}
			hidden := in * t
			b := &inverted{
				Depth:   newConvBN(hidden, hidden, 3, st, hidden, act),
				Project: newConvBN(hidden, out, 1, 1, 1, nil),
				short:   st == 1 && in == out,
			}
			if t != 1 {
				b.Expand = newConvBN(in, hidden, 1, 1, 1, act)
			}
			m.Body.Layers = append(m.Body.Layers, b)
			in = out
		}
	}
	m.width = channels(1280, math.Max(1, c.width))
	m.Body.Layers = append(m.Body.Layers, newConvBN(in, m.width, 1, 1, 1, act))
	m.Head = newHead(m.width, classes, c.drop)
	return m
}

// Features maps [N, C, H, W] to [N, Width()].
func (m *MobileNetV2) Features(x *tensor.Tensor) *tensor.Tensor {
	return nn.GlobalAvgPool()(m.Body.Forward(x))
}

// Forward maps [N, C, H, W] to [N, classes] logits.
func (m *MobileNetV2) Forward(x *tensor.Tensor) *tensor.Tensor { return m.Head.Forward(m.Features(x)) }

// Width returns 1280, or more for a width above 1.
func (m *MobileNetV2) Width() int { return m.width }

// SetHead starts a new head for classes classes.
func (m *MobileNetV2) SetHead(classes int) { m.Head = newHead(m.width, classes, m.drop) }

// HeadParameters returns the head's weight and bias.
func (m *MobileNetV2) HeadParameters() []*tensor.Param { return m.Head.parameters() }
