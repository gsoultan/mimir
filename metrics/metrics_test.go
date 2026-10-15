package metrics_test

import (
	"errors"
	"testing"

	"github.com/gsoultan/mimir/metrics"
	"github.com/gsoultan/mimir/tensor"
)

func TestAccuracyAccumulatesOverBatches(t *testing.T) {
	acc := metrics.Accuracy()
	batches := []struct {
		pred   []float32
		target []int32
	}{
		{[]float32{0.9, 0.1, 0.2, 0.8}, []int32{0, 1}}, // both right
		{[]float32{0.3, 0.7, 0.6, 0.4}, []int32{0, 0}}, // one right
	}
	for _, b := range batches {
		if err := acc.Update(t.Context(), tensor.FromSlice(b.pred, 2, 2), tensor.FromSlice(b.target)); err != nil {
			t.Fatal(err)
		}
	}
	if got := acc.Value(); got != 0.75 || acc.Name() != "accuracy" {
		t.Errorf("%s = %v, want 0.75", acc.Name(), got)
	}
	err := acc.Update(t.Context(), tensor.Zeros(tensor.Float32, 2, 3), tensor.Zeros(tensor.Int32, 2, 1))
	if !errors.Is(err, tensor.ErrShape) {
		t.Errorf("[2, 1] targets: %v, want ErrShape", err)
	}
	if metrics.Accuracy().Value() != 0 {
		t.Error("empty accuracy is not 0")
	}
}
