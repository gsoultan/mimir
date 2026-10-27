package audio

import (
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Classifier tells sounds apart from their log mel spectrogram, as
// PANNs' and keyword-spotting CNNs do: each clip's spectrogram is
// normalized, read like a gray picture by four blocks of two 3×3
// convolutions with batch normalization, each followed by pooling that
// halves both axes, averaged over time and frequency, and classified.
// Forward maps waveforms [N, samples] to [N, classes] logits; clips of
// one batch must have the same length (see Clip.Fit).
type Classifier struct {
	Mel    *MelSpectrogram
	Blocks []*block
	Drop   *nn.Dropout
	Head   *nn.Linear
	width  int
}

type block struct {
	A, B   *nn.Conv2d
	NA, NB *nn.BatchNorm
}

func (b *block) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = b.NA.Forward(b.A.Forward(x)).ReLU()
	x = b.NB.Forward(b.B.Forward(x)).ReLU()
	return x
}

// NewClassifier returns a Classifier of classes kinds of sound reading
// mel, of 64, 128, 256 and 512 channels: 4.7 million parameters. Width
// scales the channels, 0.25 for one of 0.3 million that learns from a few
// hundred clips.
func NewClassifier(classes int, mel *MelSpectrogram, width ...float64) *Classifier {
	w := 1.0
	if len(width) > 0 && width[0] > 0 {
		w = width[0]
	}
	c := &Classifier{Mel: mel, Drop: nn.NewDropout(0.2)}
	in := 1
	for i := range 4 {
		out := max(int(float64(int(64)<<i)*w), 4)
		c.Blocks = append(c.Blocks, &block{
			A: nn.NewConv2d(in, out, 3, nn.Padding(1), nn.NoBias()), NA: nn.NewBatchNorm2d(out),
			B: nn.NewConv2d(out, out, 3, nn.Padding(1), nn.NoBias()), NB: nn.NewBatchNorm2d(out),
		})
		in = out
	}
	c.width = in
	c.Head = nn.NewLinear(in, classes)
	return c
}

// Features maps waveforms [N, samples] to [N, Width()] vectors.
func (c *Classifier) Features(x *tensor.Tensor) *tensor.Tensor {
	m := c.Mel.Forward(x)
	if err := m.Err(); err != nil {
		return m
	}
	// Each clip's spectrogram to mean 0 and spread 1, so loudness and
	// microphone matter less.
	s := m.Shape()
	flat := m.Reshape(s[0], s[1]*s[2])
	mean := flat.Mean(1).Unsqueeze(1)
	d := flat.Sub(mean)
	std := d.Mul(d).Mean(1).Unsqueeze(1).Add(tensor.Scalar[float32](1e-5)).Sqrt()
	z := d.Div(std).Reshape(s[0], 1, s[1], s[2])
	for _, b := range c.Blocks {
		z = b.Forward(z)
		if sh := z.Shape(); sh[2] >= 2 && sh[3] >= 2 {
			z = z.AvgPool2d(2, tensor.ConvOptions{Stride: 2})
		}
	}
	return nn.GlobalAvgPool()(z)
}

// Forward maps waveforms [N, samples] to [N, classes] logits.
func (c *Classifier) Forward(x *tensor.Tensor) *tensor.Tensor {
	return c.Head.Forward(c.Drop.Forward(c.Features(x)))
}

// Width is how many numbers Features gives each clip.
func (c *Classifier) Width() int { return c.width }

// SetHead starts a new last layer for classes kinds of sound, for
// transfer learning.
func (c *Classifier) SetHead(classes int) { c.Head = nn.NewLinear(c.width, classes) }

// HeadParameters returns the last layer's weight and bias.
func (c *Classifier) HeadParameters() []*tensor.Param {
	return []*tensor.Param{c.Head.Weight, c.Head.Bias}
}
