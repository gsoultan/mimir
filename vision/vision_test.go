package vision_test

import (
	"context"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/vision"
)

func count(m nn.Module) int {
	n := 0
	for _, p := range nn.Parameters(m) {
		n += p.Value().Size()
	}
	return n
}

// The models have torchvision's parameter counts at 1000 classes.
func TestParameterCounts(t *testing.T) {
	for _, tc := range []struct {
		name string
		m    nn.Module
		want int
	}{
		{"resnet18", vision.ResNet18(1000), 11_689_512},
		{"resnet50", vision.ResNet50(1000), 25_557_032},
		{"mobilenet_v2", vision.NewMobileNetV2(1000), 3_504_872},
	} {
		if got := count(tc.m); got != tc.want {
			t.Errorf("%s: %d parameters, want %d", tc.name, got, tc.want)
		}
	}
}

func TestShapes(t *testing.T) {
	ctx := context.Background()
	vit, err := vision.NewViT(vision.ViTConfig{Size: 32, Patch: 8, Dim: 32, Depth: 2, Heads: 4, Hidden: 64}, 5)
	if err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct {
		m    vision.Model
		size int
	}{
		"resnet18":  {vision.ResNet18(5, vision.Small()), 32},
		"resnet50":  {vision.ResNet50(5), 64},
		"mobilenet": {vision.NewMobileNetV2(5, vision.Small(), vision.Width(0.5)), 32},
		"vit":       {vit, 32},
		"gray":      {vision.ResNet18(5, vision.Channels(1), vision.Small()), 16},
	} {
		in := 3
		if name == "gray" {
			in = 1
		}
		nn.Eval(tc.m)
		x := tensor.Randn(1, 2, in, tc.size, tc.size)
		out, err := tc.m.Forward(x).Data[float32](ctx)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(out) != 2*5 {
			t.Errorf("%s: %d logits, want 10", name, len(out))
		}
		if f := tc.m.Features(x).Shape(); f[0] != 2 || f[1] != tc.m.Width() {
			t.Errorf("%s: features %v, want [2 %d]", name, f, tc.m.Width())
		}
	}
	if _, err := vit.Forward(tensor.Randn(1, 1, 3, 16, 16)).Data[float32](ctx); err == nil {
		t.Error("a ViT reads a picture of the wrong size")
	}
}

// stripes returns n pictures of horizontal (class 0) or vertical (class
// 1) stripes, with noise.
func stripes(n, size int, seed uint64) (*tensor.Tensor, *tensor.Tensor) {
	x := make([]float32, n*3*size*size)
	y := make([]int32, n)
	noise, _ := tensor.Randn(seed, n*3*size*size).Data[float32](context.Background())
	for i := range n {
		y[i] = int32(i % 2)
		for c := range 3 {
			for r := range size {
				for k := range size {
					at := r
					if y[i] == 1 {
						at = k
					}
					v := float32(at % 2)
					idx := ((i*3+c)*size+r)*size + k
					x[idx] = v + 0.3*noise[idx]
				}
			}
		}
	}
	return tensor.FromSlice(x, n, 3, size, size), tensor.FromSlice(y, n)
}

// Each model learns to tell stripes apart, and a new head learns on
// features it keeps.
func TestTheyLearn(t *testing.T) {
	ctx := context.Background()
	vit, _ := vision.NewViT(vision.ViTConfig{Size: 16, Patch: 4, Dim: 32, Depth: 2, Heads: 2, Hidden: 64}, 2)
	for name, m := range map[string]vision.Model{
		"resnet18":  vision.ResNet18(2, vision.Small()),
		"mobilenet": vision.NewMobileNetV2(2, vision.Small(), vision.Width(0.35), vision.Dropout(0)),
		"vit":       vit,
	} {
		t.Run(name, func(t *testing.T) {
			x, y := stripes(16, 16, 7)
			opt := optim.AdamW(nn.Parameters(m), optim.LR(2e-3))
			step := func(o *optim.Optimizer) float32 {
				l := loss.CrossEntropy(m.Forward(x), y)
				if err := l.Backward(ctx); err != nil {
					t.Fatal(err)
				}
				if err := o.Step(ctx); err != nil {
					t.Fatal(err)
				}
				v, err := l.Data[float32](ctx)
				if err != nil {
					t.Fatal(err)
				}
				for _, p := range nn.Parameters(m) {
					p.ZeroGrad()
				}
				return v[0]
			}
			first := step(opt)
			var last float32
			for range 15 {
				last = step(opt)
			}
			if !(last < first*0.5) {
				t.Errorf("loss %v after 16 steps from %v", last, first)
			}

			// Transfer: a new head for three classes, taught alone.
			m.SetHead(3)
			before := snapshot(t, nn.Parameters(m)[0])
			head := optim.AdamW(m.HeadParameters(), optim.LR(1e-2))
			step(head)
			if after := snapshot(t, nn.Parameters(m)[0]); after != before {
				t.Error("teaching the head alone changed the first layer")
			}
			if out := m.Forward(x).Shape(); out[1] != 3 {
				t.Errorf("new head answers %v", out)
			}
		})
	}
}

func snapshot(t *testing.T, p *tensor.Param) float32 {
	t.Helper()
	v, err := p.Value().Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var s float32
	for _, x := range v {
		s += x
	}
	return s
}
