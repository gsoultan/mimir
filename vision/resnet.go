package vision

import (
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// ResNet is a residual network, as He et al. and torchvision's resnet
// build it: a stem, four stages of residual blocks that halve the size
// and double the channels, a global average and a linear head.
type ResNet struct {
	Stem   *nn.Sequential
	Stages []*nn.Sequential
	Head   *head
	width  int
	drop   float64
}

// basicBlock is two 3×3 convolutions and a shortcut: ResNet-18 and 34.
type basicBlock struct {
	A, B *convBN
	Down *convBN // the shortcut when the shape changes, or nil
}

func (b *basicBlock) Forward(x *tensor.Tensor) *tensor.Tensor {
	short := x
	if b.Down != nil {
		short = b.Down.Forward(x)
	}
	return b.B.Forward(b.A.Forward(x)).Add(short).ReLU()
}

// bottleneck is a 1×1 convolution down to a quarter of the channels, a
// 3×3 and a 1×1 back up, and a shortcut: ResNet-50 and deeper.
type bottleneck struct {
	A, B, C *convBN
	Down    *convBN
}

func (b *bottleneck) Forward(x *tensor.Tensor) *tensor.Tensor {
	short := x
	if b.Down != nil {
		short = b.Down.Forward(x)
	}
	return b.C.Forward(b.B.Forward(b.A.Forward(x))).Add(short).ReLU()
}

// newResNet builds a ResNet with blocks[i] blocks in stage i, of
// bottlenecks when deep.
func newResNet(classes int, blocks []int, deep bool, opts []Option) *ResNet {
	c := options(opts, config{})
	relu := nn.ReLU()
	r := &ResNet{drop: c.drop}
	if c.small {
		r.Stem = nn.NewSequential(newConvBN(c.in, 64, 3, 1, 1, relu))
	} else {
		r.Stem = nn.NewSequential(newConvBN(c.in, 64, 7, 2, 1, relu), nn.MaxPool2d(3, nn.Stride(2), nn.Padding(1)))
	}
	in := 64
	for i, n := range blocks {
		mid := 64 << i
		out := mid
		if deep {
			out = mid * 4
		}
		stage := &nn.Sequential{}
		for j := range n {
			stride := 1
			if j == 0 && i > 0 {
				stride = 2
			}
			var down *convBN
			if stride != 1 || in != out {
				down = newConvBN(in, out, 1, stride, 1, nil)
			}
			if deep {
				stage.Layers = append(stage.Layers, &bottleneck{
					A: newConvBN(in, mid, 1, 1, 1, relu), B: newConvBN(mid, mid, 3, stride, 1, relu), C: newConvBN(mid, out, 1, 1, 1, nil), Down: down,
				})
			} else {
				stage.Layers = append(stage.Layers, &basicBlock{A: newConvBN(in, out, 3, stride, 1, relu), B: newConvBN(out, out, 3, 1, 1, nil), Down: down})
			}
			in = out
		}
		r.Stages = append(r.Stages, stage)
	}
	r.width = in
	r.Head = newHead(in, classes, c.drop)
	return r
}

// ResNet18 returns an 18-layer ResNet for classes classes: 11 million
// parameters.
func ResNet18(classes int, opts ...Option) *ResNet {
	return newResNet(classes, []int{2, 2, 2, 2}, false, opts)
}

// ResNet34 returns a 34-layer ResNet: 21 million parameters.
func ResNet34(classes int, opts ...Option) *ResNet {
	return newResNet(classes, []int{3, 4, 6, 3}, false, opts)
}

// ResNet50 returns a 50-layer ResNet of bottleneck blocks: 25 million
// parameters.
func ResNet50(classes int, opts ...Option) *ResNet {
	return newResNet(classes, []int{3, 4, 6, 3}, true, opts)
}

// Features maps [N, C, H, W] to [N, Width()].
func (r *ResNet) Features(x *tensor.Tensor) *tensor.Tensor {
	x = r.Stem.Forward(x)
	for _, s := range r.Stages {
		x = s.Forward(x)
	}
	return nn.GlobalAvgPool()(x)
}

// Forward maps [N, C, H, W] to [N, classes] logits.
func (r *ResNet) Forward(x *tensor.Tensor) *tensor.Tensor { return r.Head.Forward(r.Features(x)) }

// Width returns 512 for ResNet-18 and 34, and 2048 for ResNet-50.
func (r *ResNet) Width() int { return r.width }

// SetHead starts a new head for classes classes.
func (r *ResNet) SetHead(classes int) { r.Head = newHead(r.width, classes, r.drop) }

// HeadParameters returns the head's weight and bias.
func (r *ResNet) HeadParameters() []*tensor.Param { return r.Head.parameters() }
