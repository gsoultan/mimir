// Package metrics scores predictions over a dataset, a batch at a time.
package metrics

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Metric accumulates a score over batches.
type Metric interface {
	// Name identifies the metric, such as "accuracy".
	Name() string
	// Update adds a batch of predictions and targets.
	Update(ctx context.Context, pred, target *tensor.Tensor) error
	// Value returns the score over every batch so far.
	Value() float64
}

// Accuracy returns the fraction of examples whose highest-scoring class,
// along axis 1 of pred, equals the Int32 class in target.
func Accuracy() Metric { return &accuracy{} }

type accuracy struct {
	correct, total float64
}

func (a *accuracy) Name() string { return "accuracy" }

func (a *accuracy) Update(ctx context.Context, pred, target *tensor.Tensor) error {
	if err := pred.Err(); err != nil {
		return err
	}
	if err := target.Err(); err != nil {
		return err
	}
	if pred.Rank() < 2 || target.DType() != tensor.Int32 {
		return fmt.Errorf("metrics: accuracy of %s%v predictions for %s%v targets: %w", pred.DType(), pred.Shape(), target.DType(), target.Shape(), tensor.ErrShape)
	}
	guess := pred.Argmax(1)
	if !slices.Equal(guess.Shape(), target.Shape()) {
		return fmt.Errorf("metrics: accuracy of %v predictions for %v targets: %w", pred.Shape(), target.Shape(), tensor.ErrShape)
	}
	hits := guess.Equal(target).Cast(tensor.Float32).Sum()
	n, err := hits.Item[float32](ctx)
	if err != nil {
		return err
	}
	a.correct += float64(n)
	a.total += float64(target.Size())
	return nil
}

func (a *accuracy) Value() float64 {
	if a.total == 0 {
		return 0
	}
	return a.correct / a.total
}
