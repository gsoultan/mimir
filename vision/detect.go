package vision

import (
	"context"
	"fmt"
	"math"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Backbone reads pictures into feature maps for a Detector: [N, C, H, W]
// to [N, Channels(), H/Stride(), W/Stride()].
type Backbone interface {
	nn.Module
	Channels() int
	Stride() int
}

// resnetMaps is a ResNet's last feature map, before its average and head.
type resnetMaps struct {
	Net    *ResNet
	stride int
}

func (r *resnetMaps) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = r.Net.Stem.Forward(x)
	for _, s := range r.Net.Stages {
		x = s.Forward(x)
	}
	return x
}

func (r *resnetMaps) Channels() int { return r.Net.width }
func (r *resnetMaps) Stride() int   { return r.stride }

// Backbone returns r's feature maps, before the average and head, as a
// Detector's backbone: 32 times smaller than the picture, or 8 for a
// Small ResNet. Its weights are r's, so a ResNet taught to classify
// pictures starts a detector that already sees.
func (r *ResNet) Backbone() Backbone {
	stride := 32
	if len(r.Stem.Layers) == 1 {
		stride = 8
	}
	return &resnetMaps{Net: r, stride: stride}
}

// tiny is three convolutions down to a quarter of the size.
type tiny struct {
	Layers *nn.Sequential
	width  int
}

func (t *tiny) Forward(x *tensor.Tensor) *tensor.Tensor { return t.Layers.Forward(x) }
func (t *tiny) Channels() int                           { return t.width }
func (t *tiny) Stride() int                             { return 4 }

// TinyBackbone returns four convolutions from in channels to 2·width,
// four times smaller than the picture: a backbone that learns in seconds,
// for small pictures and few kinds of thing.
func TinyBackbone(in, width int) Backbone {
	relu := nn.ReLU()
	return &tiny{Layers: nn.NewSequential(
		newConvBN(in, width, 3, 2, 1, relu), newConvBN(width, width, 3, 1, 1, relu),
		newConvBN(width, 2*width, 3, 2, 1, relu), newConvBN(2*width, 2*width, 3, 1, 1, relu),
	), width: 2 * width}
}

// Detector finds things in pictures as Zhou et al.'s CenterNet does,
// "objects as points": on a grid four times smaller than the picture it
// scores, for each kind of thing, how likely each cell holds a thing's
// centre, and says how wide and tall that thing is and where in the cell
// its centre lies. The peaks of the scores are the things found, so it
// needs no anchors and no NMS.
//
// Forward maps [N, C, H, W] pictures, H and W multiples of the backbone's
// stride, to [N, classes+4, H/4, W/4]: the centre logits of each class,
// then the width, height and the x and y offset of the centre, in grid
// cells. Targets builds what a batch of true boxes should give, Loss
// scores Forward against it, and Detect reads the things found.
type Detector struct {
	Backbone Backbone
	Up       []*upBlock
	Heat     *detHead
	Size     *detHead
	Offset   *detHead
	classes  int
}

type upBlock struct {
	Conv *nn.ConvTranspose2d
	BN   *nn.BatchNorm
}

func (u *upBlock) Forward(x *tensor.Tensor) *tensor.Tensor {
	return u.BN.Forward(u.Conv.Forward(x)).ReLU()
}

type detHead struct{ A, B *nn.Conv2d }

func newDetHead(in, mid, out int) *detHead {
	return &detHead{A: nn.NewConv2d(in, mid, 3, nn.Padding(1)), B: nn.NewConv2d(mid, out, 1)}
}

func (h *detHead) Forward(x *tensor.Tensor) *tensor.Tensor { return h.B.Forward(h.A.Forward(x).ReLU()) }

// DetectorStride is how many times smaller than the picture a Detector's
// grid is.
const DetectorStride = 4

// NewDetector returns a Detector of classes kinds of thing on backbone,
// such as ResNet18(1).Backbone() or TinyBackbone(3, 16). A backbone
// coarser than the grid is grown to it by transposed convolutions, as
// CenterNet's ResNet does.
func NewDetector(classes int, backbone Backbone) (*Detector, error) {
	s := backbone.Stride()
	if classes < 1 || s < DetectorStride || s%DetectorStride != 0 || (s/DetectorStride)&(s/DetectorStride-1) != 0 {
		return nil, fmt.Errorf("vision: detector of %d classes on a backbone of stride %d, want a power of 2 at least 4: %w", classes, s, tensor.ErrShape)
	}
	d := &Detector{Backbone: backbone, classes: classes}
	c := backbone.Channels()
	widths := []int{256, 128, 64}
	ups := 0
	for f := s / DetectorStride; f > 1; f /= 2 {
		ups++
	}
	for i := range ups {
		// Each step up at most halves the channels, so a narrow backbone
		// keeps a narrow, cheap neck.
		out := min(widths[max(len(widths)-ups+i, 0)], max(c/2, 16))
		d.Up = append(d.Up, &upBlock{Conv: nn.NewConvTranspose2d(c, out, 4, nn.Stride(2), nn.Padding(1), nn.NoBias()), BN: nn.NewBatchNorm2d(out)})
		c = out
	}
	mid := min(64, max(c, 16))
	d.Heat, d.Size, d.Offset = newDetHead(c, mid, classes), newDetHead(c, mid, 2), newDetHead(c, mid, 2)
	// Start every centre score near 0.1, as CenterNet does, so the many
	// empty cells don't swamp the first steps.
	if err := d.Heat.B.Bias.Set(tensor.Full(tensor.Float32, -2.19).Expand(classes).Clone()); err != nil {
		return nil, err
	}
	return d, nil
}

// Classes returns the kinds of thing d finds.
func (d *Detector) Classes() int { return d.classes }

// Forward maps [N, C, H, W] to [N, classes+4, H/4, W/4].
func (d *Detector) Forward(x *tensor.Tensor) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	s := d.Backbone.Stride()
	if sh := x.Shape(); len(sh) != 4 || sh[2]%s != 0 || sh[3]%s != 0 {
		return tensor.Failed(fmt.Errorf("vision: detector of pictures %v, want [N, C, H, W] with H and W multiples of %d: %w", x.Shape(), s, tensor.ErrShape))
	}
	f := d.Backbone.Forward(x)
	for _, u := range d.Up {
		f = u.Forward(f)
	}
	return tensor.Cat(1, d.Heat.Forward(f), d.Size.Forward(f), d.Offset.Forward(f))
}

// DetectionTargets is what a Detector should give for a batch of
// pictures, on its grid.
type DetectionTargets struct {
	Heat   *tensor.Tensor // [N, classes, h, w], 1 at each centre, a Gaussian around it
	Size   *tensor.Tensor // [N, 2, h, w], width and height in cells at each centre
	Offset *tensor.Tensor // [N, 2, h, w], where in its cell each centre lies
	Mask   *tensor.Tensor // [N, 1, h, w], 1 at each centre
	Things int
}

// Targets returns the targets of pictures of height h and width w whose
// true boxes, in pixels, are truth[i] for picture i.
func (d *Detector) Targets(truth [][]Detection, h, w int) (DetectionTargets, error) {
	gh, gw := h/DetectorStride, w/DetectorStride
	n, k := len(truth), d.classes
	heat := make([]float32, n*k*gh*gw)
	size := make([]float32, n*2*gh*gw)
	off := make([]float32, n*2*gh*gw)
	mask := make([]float32, n*gh*gw)
	things := 0
	plane := gh * gw
	for i, boxes := range truth {
		for _, b := range boxes {
			if b.Class < 0 || b.Class >= k {
				return DetectionTargets{}, fmt.Errorf("vision: class %d of %d: %w", b.Class, k, tensor.ErrShape)
			}
			bw, bh := (b.X2-b.X1)/DetectorStride, (b.Y2-b.Y1)/DetectorStride
			if bw <= 0 || bh <= 0 {
				continue
			}
			cx, cy := (b.X1+b.X2)/2/DetectorStride, (b.Y1+b.Y2)/2/DetectorStride
			x, y := int(cx), int(cy)
			if x < 0 || y < 0 || x >= gw || y >= gh {
				continue
			}
			r := max(0, int(radius(bh, bw, 0.7)))
			sigma := float64(2*r+1) / 6
			hm := heat[(i*k+b.Class)*plane:]
			for yy := max(0, y-r); yy <= min(gh-1, y+r); yy++ {
				for xx := max(0, x-r); xx <= min(gw-1, x+r); xx++ {
					dx, dy := float64(xx-x), float64(yy-y)
					g := float32(math.Exp(-(dx*dx + dy*dy) / (2 * sigma * sigma)))
					if xx == x && yy == y {
						g = 1
					}
					hm[yy*gw+xx] = max(hm[yy*gw+xx], g)
				}
			}
			at := y*gw + x
			size[(i*2)*plane+at], size[(i*2+1)*plane+at] = float32(bw), float32(bh)
			off[(i*2)*plane+at], off[(i*2+1)*plane+at] = float32(cx-float64(x)), float32(cy-float64(y))
			if mask[i*plane+at] == 0 {
				things++
			}
			mask[i*plane+at] = 1
		}
	}
	return DetectionTargets{
		Heat:   tensor.FromSlice(heat, n, k, gh, gw),
		Size:   tensor.FromSlice(size, n, 2, gh, gw),
		Offset: tensor.FromSlice(off, n, 2, gh, gw),
		Mask:   tensor.FromSlice(mask, n, 1, gh, gw),
		Things: things,
	}, nil
}

// radius is CenterNet's Gaussian radius: how far a corner may move with
// the box still overlapping the true one by min.
func radius(h, w, min float64) float64 {
	b1, c1 := h+w, w*h*(1-min)/(1+min)
	r1 := (b1 + math.Sqrt(b1*b1-4*c1)) / 2
	b2, c2 := 2*(h+w), 4*(1-min)*w*h
	r2 := (b2 + math.Sqrt(b2*b2-16*c2)) / 2
	a3, b3, c3 := 4*min, -2*min*(h+w), (min-1)*w*h
	r3 := (b3 + math.Sqrt(b3*b3-4*a3*c3)) / 2
	return math.Min(r1, math.Min(r2, r3))
}

// part returns channels lo to hi of [N, C, h, w].
func part(x *tensor.Tensor, lo, hi int) *tensor.Tensor {
	s := x.Shape()
	return x.Shrink([][2]int{{0, s[0]}, {lo, hi}, {0, s[2]}, {0, s[3]}})
}

// Loss returns CenterNet's loss of out, from Forward, against t: the
// penalty-reduced focal loss of the centre scores, plus a tenth of the L1
// error of the sizes and the L1 error of the offsets at the centres.
func (d *Detector) Loss(out *tensor.Tensor, t DetectionTargets) *tensor.Tensor {
	if err := out.Err(); err != nil {
		return out
	}
	k := d.classes
	if s := out.Shape(); len(s) != 4 || s[1] != k+4 || t.Heat == nil || s[2] != t.Heat.Shape()[2] || s[3] != t.Heat.Shape()[3] || s[0] != t.Heat.Shape()[0] {
		return tensor.Failed(fmt.Errorf("vision: detector loss of %v for targets of %d classes: %w", out.Shape(), k, tensor.ErrShape))
	}
	one, zero := tensor.Scalar[float32](1), tensor.Scalar[float32](0)
	softplus := func(x *tensor.Tensor) *tensor.Tensor {
		return x.Maximum(zero).Add(one.Add(x.Abs().Neg().Exp()).Log())
	}
	logit := part(out, 0, k)
	p := logit.Sigmoid()
	logp, log1p := softplus(logit.Neg()).Neg(), softplus(logit).Neg()
	pos := t.Heat.Equal(one).Cast(tensor.Float32)
	neg := one.Sub(pos)
	far := one.Sub(t.Heat)
	far = far.Mul(far).Mul(far.Mul(far))
	q := one.Sub(p)
	focal := logp.Mul(q).Mul(q).Mul(pos).Sum().Add(log1p.Mul(p).Mul(p).Mul(far).Mul(neg).Sum())
	things := tensor.Scalar(float32(max(t.Things, 1)))
	heat := focal.Neg().Div(things)
	l1 := func(pred, want *tensor.Tensor) *tensor.Tensor {
		return pred.Sub(want).Abs().Mul(t.Mask).Sum().Div(things)
	}
	size := l1(part(out, k, k+2), t.Size)
	off := l1(part(out, k+2, k+4), t.Offset)
	return heat.Add(size.Mul(tensor.Scalar[float32](0.1))).Add(off)
}

// Detect returns the things found in each picture of out, from Forward:
// the peaks of the centre scores, those whose own 3×3 neighbourhood
// holds no higher score, up to top a picture scoring at least threshold,
// best first, with their boxes in pixels.
func (d *Detector) Detect(ctx context.Context, out *tensor.Tensor, threshold float64, top int) ([][]Detection, error) {
	if err := out.Err(); err != nil {
		return nil, err
	}
	s := out.Shape()
	k := d.classes
	if len(s) != 4 || s[1] != k+4 {
		return nil, fmt.Errorf("vision: detections of %v, want [N, %d, h, w]: %w", s, k+4, tensor.ErrShape)
	}
	n, gh, gw := s[0], s[2], s[3]
	heat := part(out, 0, k).Detach().Sigmoid()
	peak := heat.MaxPool2d(3, tensor.ConvOptions{Stride: 1, Padding: 1})
	kept := tensor.Where(heat.Equal(peak), heat, tensor.Zeros(tensor.Float32))
	top = min(top, k*gh*gw)
	scores, idx := kept.Reshape(n, k*gh*gw).TopK(top, 1, true)
	sv, err := scores.Data[float32](ctx)
	if err != nil {
		return nil, err
	}
	iv, err := idx.Cast(tensor.Int32).Data[int32](ctx)
	if err != nil {
		return nil, err
	}
	rest, err := part(out, k, k+4).Detach().Data[float32](ctx)
	if err != nil {
		return nil, err
	}
	plane := gh * gw
	res := make([][]Detection, n)
	for i := range n {
		for j := range top {
			score := float64(sv[i*top+j])
			if score < threshold {
				break
			}
			at := int(iv[i*top+j])
			c, cell := at/plane, at%plane
			y, x := cell/gw, cell%gw
			get := func(ch int) float64 { return float64(rest[(i*4+ch)*plane+cell]) }
			w, h := max(get(0), 0), max(get(1), 0)
			cx, cy := float64(x)+get(2), float64(y)+get(3)
			res[i] = append(res[i], Detection{
				Box:   Box{X1: (cx - w/2) * DetectorStride, Y1: (cy - h/2) * DetectorStride, X2: (cx + w/2) * DetectorStride, Y2: (cy + h/2) * DetectorStride},
				Class: c, Score: score,
			})
		}
	}
	return res, nil
}
