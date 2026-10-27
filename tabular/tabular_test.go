package tabular_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/linear"
	"github.com/gsoultan/mimir/metrics"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/tree"
)

func line(n int) tabular.Data {
	d := tabular.Data{Features: 2}
	for i := range n {
		x := float32(i%17) - 8
		d.X = append(d.X, x, float32(i%5))
		d.Y = append(d.Y, 2*x+1)
	}
	return d
}

func TestSplitKeepsEveryRowOnce(t *testing.T) {
	d := line(50)
	train, test := d.Split(0.2, 1)
	if train.Rows() != 40 || test.Rows() != 10 {
		t.Fatalf("%d and %d rows", train.Rows(), test.Rows())
	}
	got := append(slices.Clone(train.Y), test.Y...)
	want := slices.Clone(d.Y)
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Error("split rows differ from the table's")
	}
	if _, small := line(2).Split(0.01, 1); small.Rows() != 1 {
		t.Errorf("%d held-out rows of 2", small.Rows())
	}
}

func TestCrossValidateScoresEachFold(t *testing.T) {
	ctx := context.Background()
	fit := func(ctx context.Context, d tabular.Data) (tabular.Model, error) { return linear.Regression(ctx, d) }
	scores, err := tabular.CrossValidate(ctx, line(100), 5, 1, fit, metrics.R2)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 5 {
		t.Fatalf("%d scores", len(scores))
	}
	for _, s := range scores {
		if s < 0.999 {
			t.Errorf("fold R² %v of a straight line", s)
		}
	}
	boost := func(ctx context.Context, d tabular.Data) (tabular.Model, error) {
		return tree.GradientBoosting(ctx, d, tabular.Regression)
	}
	if _, err := tabular.CrossValidate(ctx, line(100), 1, 1, boost, metrics.R2); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("one fold: error %v", err)
	}
}
