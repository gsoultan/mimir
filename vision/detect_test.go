package vision_test

import (
	"context"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/vision"
)

func TestBoxes(t *testing.T) {
	a := vision.Box{X1: 0, Y1: 0, X2: 10, Y2: 10}
	b := vision.Box{X1: 5, Y1: 0, X2: 15, Y2: 10}
	if got := vision.IoU(a, b); math.Abs(got-50.0/150) > 1e-9 {
		t.Errorf("IoU %v, want 1/3", got)
	}
	if got := vision.IoU(a, vision.Box{X1: 20, Y1: 20, X2: 30, Y2: 30}); got != 0 {
		t.Errorf("IoU of apart boxes %v", got)
	}
	dets := []vision.Detection{
		{Box: a, Class: 0, Score: 0.9},
		{Box: vision.Box{X1: 1, Y1: 1, X2: 10, Y2: 10}, Class: 0, Score: 0.8}, // overlaps a: dropped
		{Box: vision.Box{X1: 1, Y1: 1, X2: 10, Y2: 10}, Class: 1, Score: 0.7}, // another class: kept
		{Box: b, Class: 0, Score: 0.95},
	}
	kept := vision.NMS(dets, 0.5)
	if len(kept) != 3 || kept[0].Score != 0.95 || kept[1].Score != 0.9 || kept[2].Class != 1 {
		t.Errorf("NMS kept %+v", kept)
	}

	ctx := context.Background()
	ta := tensor.FromSlice([]float32{0, 0, 10, 10, 20, 20, 30, 30}, 2, 4)
	tb := tensor.FromSlice([]float32{5, 0, 15, 10}, 1, 4)
	iou, err := vision.BoxIoU(ta, tb).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(iou) != 2 || math.Abs(float64(iou[0])-1.0/3) > 1e-6 || iou[1] != 0 {
		t.Errorf("BoxIoU %v", iou)
	}
	// Same boxes: GIoU 1, loss 0. Apart: IoU 0 and the hull's empty share.
	same, _ := vision.GIoULoss(tb, tb).Item[float32](ctx)
	apart, _ := vision.GIoULoss(tensor.FromSlice([]float32{0, 0, 1, 1}, 1, 4), tensor.FromSlice([]float32{2, 0, 3, 1}, 1, 4)).Item[float32](ctx)
	if math.Abs(float64(same)) > 1e-6 || math.Abs(float64(apart)-(1+1.0/3)) > 1e-5 {
		t.Errorf("GIoU losses %v and %v", same, apart)
	}

	truth := [][]vision.Detection{{{Box: a, Class: 0}}, {{Box: b, Class: 1}}}
	if ap := vision.AveragePrecision(truth, truth, 2, 0.5); ap != 1 {
		t.Errorf("AP of the truth itself %v", ap)
	}
	half := [][]vision.Detection{{{Box: a, Class: 0, Score: 1}}, nil}
	if ap := vision.AveragePrecision(half, truth, 2, 0.5); ap != 0.5 {
		t.Errorf("AP finding one class of two %v", ap)
	}
}

// squares draws n pictures of size×size, each with one or two squares,
// red for class 0 and green for class 1, on gray noise.
func squares(n, size int, r *rand.Rand) (*tensor.Tensor, [][]vision.Detection, []int32) {
	px := make([]float32, n*3*size*size)
	mask := make([]int32, n*size*size)
	truth := make([][]vision.Detection, n)
	plane := size * size
	for i := range n {
		for k := range 3 * plane {
			px[i*3*plane+k] = float32(r.Float64() * 0.2)
		}
		for range 1 + r.IntN(2) {
			c := r.IntN(2)
			s := 6 + r.IntN(6)
			x, y := r.IntN(size-s), r.IntN(size-s)
			overlaps := false
			for _, d := range truth[i] {
				if vision.IoU(d.Box, vision.Box{X1: float64(x), Y1: float64(y), X2: float64(x + s), Y2: float64(y + s)}) > 0 {
					overlaps = true
				}
			}
			if overlaps {
				continue
			}
			for yy := y; yy < y+s; yy++ {
				for xx := x; xx < x+s; xx++ {
					px[(i*3+c)*plane+yy*size+xx] = 0.9
					mask[i*plane+yy*size+xx] = int32(c + 1)
				}
			}
			truth[i] = append(truth[i], vision.Detection{Box: vision.Box{X1: float64(x), Y1: float64(y), X2: float64(x + s), Y2: float64(y + s)}, Class: c})
		}
	}
	return tensor.FromSlice(px, n, 3, size, size), truth, mask
}

func TestDetectorLearns(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a detector")
	}
	ctx := context.Background()
	r := rand.New(rand.NewPCG(1, 2))
	d, err := vision.NewDetector(2, vision.TinyBackbone(3, 16))
	if err != nil {
		t.Fatal(err)
	}
	opt := optim.Adam(nn.Parameters(d), optim.LR(3e-3))
	const size = 32
	for step := range 300 {
		x, truth, _ := squares(16, size, r)
		tg, err := d.Targets(truth, size, size)
		if err != nil {
			t.Fatal(err)
		}
		l := d.Loss(d.Forward(x), tg)
		if err := l.Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(ctx); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
		if step == 299 {
			v, _ := l.Item[float32](ctx)
			t.Logf("loss after %d steps: %.3f", step+1, v)
		}
	}
	nn.Eval(d)
	x, truth, _ := squares(64, size, r)
	found, err := d.Detect(ctx, d.Forward(x), 0.3, 10)
	if err != nil {
		t.Fatal(err)
	}
	ap := vision.AveragePrecision(found, truth, 2, 0.5)
	t.Logf("mAP@0.5 %.3f", ap)
	if ap < 0.7 {
		t.Errorf("mAP@0.5 %.3f after training, want at least 0.7", ap)
	}
}

func TestDetectorShapes(t *testing.T) {
	ctx := context.Background()
	for name, tc := range map[string]struct {
		b    vision.Backbone
		size int
	}{
		"resnet":       {vision.ResNet18(1).Backbone(), 64},
		"small resnet": {vision.ResNet18(1, vision.Small()).Backbone(), 32},
		"tiny":         {vision.TinyBackbone(3, 8), 16},
	} {
		d, err := vision.NewDetector(3, tc.b)
		if err != nil {
			t.Fatal(err)
		}
		nn.Eval(d)
		out := d.Forward(tensor.Randn(1, 1, 3, tc.size, tc.size))
		if err := tensor.Realize(ctx, out); err != nil {
			t.Fatal(err)
		}
		want := []int{1, 7, tc.size / 4, tc.size / 4}
		if !slices.Equal(out.Shape(), want) {
			t.Errorf("%s: %v, want %v", name, out.Shape(), want)
		}
	}
	if _, err := vision.NewDetector(0, vision.TinyBackbone(3, 8)); err == nil {
		t.Error("a detector of no classes")
	}
}

func TestUNetLearns(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a U-Net")
	}
	ctx := context.Background()
	r := rand.New(rand.NewPCG(3, 4))
	u := vision.NewUNet(3, 3, 8, 2)
	opt := optim.Adam(nn.Parameters(u), optim.LR(1e-2))
	const size = 16
	for range 80 {
		x, _, mask := squares(8, size, r)
		y := tensor.FromSlice(mask, 8, size, size)
		out := u.Forward(x)
		l := loss.CrossEntropy(out, y).Add(loss.Dice(out, y))
		if err := l.Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(ctx); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
	}
	nn.Eval(u)
	x, _, mask := squares(16, size, r)
	pred, err := u.Forward(x).Argmax(1).Cast(tensor.Int32).Data[int32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	iou := vision.MeanIoU(pred, mask, 3)
	t.Logf("mean IoU %.3f", iou)
	if iou < 0.8 {
		t.Errorf("mean IoU %.3f after training, want at least 0.8", iou)
	}
	if _, err := u.Forward(tensor.Randn(1, 1, 3, 10, 10)).Data[float32](ctx); err == nil {
		t.Error("a U-Net of 2 levels read a 10-pixel picture")
	}
}
