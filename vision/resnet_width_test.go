package vision_test

import (
	"context"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/vision"
)

func TestResNetWidth(t *testing.T) {
	r := vision.ResNet18(3, vision.Width(0.25))
	if r.Width() != 128 {
		t.Errorf("width %d, want 128", r.Width())
	}
	full, quarter := 0, 0
	for _, p := range nn.Parameters(vision.ResNet18(3)) {
		full += p.Value().Size()
	}
	for _, p := range nn.Parameters(r) {
		quarter += p.Value().Size()
	}
	if quarter*10 > full {
		t.Errorf("%d parameters at a quarter width, %d at full", quarter, full)
	}
	nn.Eval(r)
	out, err := r.Forward(tensor.Randn(1, 2, 3, 64, 64)).Data[float32](context.Background())
	if err != nil || len(out) != 6 {
		t.Errorf("%d outputs, %v", len(out), err)
	}
}
