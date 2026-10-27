package vision

import (
	"fmt"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// UNet is Ronneberger et al.'s U-Net for segmentation: an encoder that
// halves the picture and doubles the channels at each level, a decoder
// that grows it back with transposed convolutions, and skips that join
// each decoder level with the encoder level of its size, so the mask
// keeps the picture's fine edges. Forward maps [N, C, H, W] pictures,
// H and W multiples of 2^Depth, to [N, classes, H, W] logits, a class
// for every pixel; train it with loss.CrossEntropy, loss.Dice or both.
type UNet struct {
	Down []*doubleConv
	Mid  *doubleConv
	Up   []*nn.ConvTranspose2d
	Join []*doubleConv
	Out  *nn.Conv2d
}

// doubleConv is two 3×3 convolutions with batch normalization and ReLU.
type doubleConv struct{ A, B *convBN }

func newDoubleConv(in, out int) *doubleConv {
	return &doubleConv{A: newConvBN(in, out, 3, 1, 1, nn.ReLU()), B: newConvBN(out, out, 3, 1, 1, nn.ReLU())}
}

func (d *doubleConv) Forward(x *tensor.Tensor) *tensor.Tensor { return d.B.Forward(d.A.Forward(x)) }

// NewUNet returns a U-Net from in channels to classes classes, base
// channels at its first level and depth levels down: Ronneberger's is
// NewUNet(1, 2, 64, 4), 31 million parameters, and NewUNet(3, k, 16, 3)
// a small one for small pictures.
func NewUNet(in, classes, base, depth int) *UNet {
	u := &UNet{}
	c := in
	for i := range depth {
		u.Down = append(u.Down, newDoubleConv(c, base<<i))
		c = base << i
	}
	u.Mid = newDoubleConv(c, c*2)
	c *= 2
	for i := depth - 1; i >= 0; i-- {
		out := base << i
		u.Up = append(u.Up, nn.NewConvTranspose2d(c, out, 2, nn.Stride(2)))
		u.Join = append(u.Join, newDoubleConv(out*2, out))
		c = out
	}
	u.Out = nn.NewConv2d(c, classes, 1)
	return u
}

// Forward maps [N, C, H, W] to [N, classes, H, W] logits.
func (u *UNet) Forward(x *tensor.Tensor) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	if s := x.Shape(); len(s) != 4 || s[2]%(1<<len(u.Down)) != 0 || s[3]%(1<<len(u.Down)) != 0 {
		return tensor.Failed(fmt.Errorf("vision: U-Net of %d levels for pictures %v, want [N, C, H, W] with H and W multiples of %d: %w", len(u.Down), x.Shape(), 1<<len(u.Down), tensor.ErrShape))
	}
	var skips []*tensor.Tensor
	for _, d := range u.Down {
		x = d.Forward(x)
		skips = append(skips, x)
		x = x.MaxPool2d(2, tensor.ConvOptions{Stride: 2})
	}
	x = u.Mid.Forward(x)
	for i, up := range u.Up {
		x = up.Forward(x)
		x = u.Join[i].Forward(tensor.Cat(1, skips[len(skips)-1-i], x))
	}
	return u.Out.Forward(x)
}

// MeanIoU returns the intersection over union of each class's predicted
// and true pixels, averaged over the classes present in either: pred and
// truth hold one class per pixel. It is how segmentation is scored.
func MeanIoU(pred, truth []int32, classes int) float64 {
	inter, union := make([]float64, classes), make([]float64, classes)
	for i, p := range pred {
		t := truth[i]
		if p == t {
			inter[p]++
			union[p]++
		} else {
			if p >= 0 && int(p) < classes {
				union[p]++
			}
			if t >= 0 && int(t) < classes {
				union[t]++
			}
		}
	}
	var sum float64
	n := 0
	for c := range classes {
		if union[c] > 0 {
			sum += inter[c] / union[c]
			n++
		}
	}
	if n == 0 {
		return 1
	}
	return sum / float64(n)
}
