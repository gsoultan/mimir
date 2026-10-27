// Package vision builds the networks that read pictures best, as
// torchvision's models do: ResNet, MobileNetV2 and the Vision
// Transformer, ViT, to classify them; a CenterNet Detector to find the
// things in them, with box IoU, NMS and mean average precision; and a
// U-Net to segment them, a class for every pixel.
//
//	m := vision.ResNet18(10)          // ten classes, 224×224 photos
//	logits := m.Forward(x)            // x is [N, 3, H, W]
//	small := vision.ResNet18(10, vision.Small()) // for 32×32 to 64×64 pictures
//
// Each is a Model: Features reads a batch of pictures into one vector
// each, and Head, a Linear layer, turns that into the logits. Transfer
// learning keeps Features and swaps the head: SetHead starts a new one
// for other classes, and Head's parameters alone are what an optimizer
// teaches to only fit the head (see Model.HeadParameters).
//
// The models start in training mode; nn.Eval switches their batch
// normalization and dropout to inference. Their layers are mimir's, so
// they run, train, save and load wherever a network does, and their
// parameters are named as mim.Save writes them.
package vision

import (
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Model is a picture network split into what it sees and what it says.
type Model interface {
	nn.Module
	// Features maps [N, C, H, W] pictures to [N, Width()] vectors.
	Features(x *tensor.Tensor) *tensor.Tensor
	// Width is how many numbers Features gives each picture.
	Width() int
	// SetHead replaces the last layer with a new one for classes
	// classes, for transfer learning.
	SetHead(classes int)
	// HeadParameters are the last layer's parameters, for an optimizer
	// that teaches only it.
	HeadParameters() []*tensor.Param
}

// Option configures a model.
type Option func(*config)

type config struct {
	in    int     // channels of each picture
	small bool    // for pictures of 32 to 64 pixels a side
	width float64 // the channels' multiplier
	drop  float64 // dropout before the head
}

// Channels sets the channels of each picture. The default is 3, for
// red, green and blue; 1 is for gray pictures.
func Channels(n int) Option { return func(c *config) { c.in = n } }

// Small fits the model to small pictures, 32 to 64 pixels a side, as
// for CIFAR: the first layer keeps the size, where it otherwise shrinks
// a 224-pixel photo fourfold.
func Small() Option { return func(c *config) { c.small = true } }

// Width multiplies a ResNet's or MobileNetV2's channels, such as 0.5 for
// a model a quarter the size. The default is 1.
func Width(w float64) Option { return func(c *config) { c.width = w } }

// Dropout drops this share of the features before the head while
// training. MobileNetV2's default is 0.2 and the others' 0.
func Dropout(p float64) Option { return func(c *config) { c.drop = p } }

func options(opts []Option, base config) config {
	base.in = 3
	if base.width == 0 {
		base.width = 1
	}
	for _, o := range opts {
		o(&base)
	}
	base.in = max(base.in, 1)
	return base
}

// head is a classifier: dropout, then a linear layer.
type head struct {
	Drop   *nn.Dropout // or nil
	Linear *nn.Linear
}

func newHead(width, classes int, drop float64) *head {
	h := &head{Linear: nn.NewLinear(width, classes)}
	if drop > 0 {
		h.Drop = nn.NewDropout(drop)
	}
	return h
}

func (h *head) Forward(x *tensor.Tensor) *tensor.Tensor {
	if h.Drop != nil {
		x = h.Drop.Forward(x)
	}
	return h.Linear.Forward(x)
}

func (h *head) parameters() []*tensor.Param {
	return []*tensor.Param{h.Linear.Weight, h.Linear.Bias}
}

// convBN is a convolution without bias, batch normalization and, when
// act is set, an activation, the unit every network here is built of.
type convBN struct {
	Conv *nn.Conv2d
	BN   *nn.BatchNorm
	act  nn.Func
}

func newConvBN(in, out, k, stride, groups int, act nn.Func) *convBN {
	return &convBN{
		Conv: nn.NewConv2d(in, out, k, nn.Stride(stride), nn.Padding((k-1)/2), nn.Groups(groups), nn.NoBias()),
		BN:   nn.NewBatchNorm2d(out),
		act:  act,
	}
}

func (c *convBN) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = c.BN.Forward(c.Conv.Forward(x))
	if c.act != nil {
		x = c.act(x)
	}
	return x
}

// relu6 is min(max(x, 0), 6), MobileNet's activation.
func relu6() nn.Func {
	return func(x *tensor.Tensor) *tensor.Tensor { return x.ReLU().Minimum(tensor.Full(tensor.Float32, 6)) }
}
