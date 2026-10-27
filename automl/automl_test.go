package automl_test

import (
	"context"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/automl"
	"github.com/gsoultan/mimir/linear"
	"github.com/gsoultan/mimir/metrics"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tree"
)

// table returns rows of two features whose answer is a·b (regression),
// or whether a and b share a sign (classification), or a straight line
// of them (linear).
func table(n int, kind string, seed uint64) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 1))
	d := tabular.Data{Features: 3}
	for range n {
		a, b, noise := r.Float64()*2-1, r.Float64()*2-1, r.Float64()
		d.X = append(d.X, float32(a), float32(b), float32(noise))
		switch kind {
		case "product":
			d.Y = append(d.Y, float32(a*b*10))
		case "signs":
			d.Y = append(d.Y, float32(map[bool]int{false: 0, true: 1}[(a > 0) == (b > 0)]))
		case "line":
			d.Y = append(d.Y, float32(map[bool]int{false: 0, true: 1}[a+0.5*b+0.1*r.NormFloat64() > 0]))
		}
	}
	return d
}

func TestSearchPicksWhatFits(t *testing.T) {
	ctx := context.Background()
	// Trees for an interaction no line draws.
	r, err := automl.Search(ctx, table(600, "signs", 1), tabular.Classification, automl.Trials(4))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Trials) != 4 || r.Metric != "log loss" {
		t.Fatalf("%d trials of %s", len(r.Trials), r.Metric)
	}
	for i := 1; i < len(r.Trials); i++ {
		if r.Trials[i].Score < r.Trials[i-1].Score {
			t.Errorf("trials out of order: %v", r.Trials)
		}
	}
	if r.Winner.Family == "linear" {
		t.Errorf("a line won the signs table: %v", r.Trials)
	}
	if _, ok := r.Best.(*tree.Model); !ok {
		t.Errorf("best is %T", r.Best)
	}
	test := table(400, "signs", 2)
	if acc := metrics.ClassAccuracy(r.Best.Predict(test.X), test.Y); acc < 0.9 {
		t.Errorf("accuracy %v on new rows", acc)
	}

	// A line for a line.
	r, err = automl.Search(ctx, table(400, "line", 3), tabular.Classification, automl.Trials(3))
	if err != nil {
		t.Fatal(err)
	}
	if lin := r.Trials[len(r.Trials)-1]; lin.Family == "linear" && lin.Score > 1.5*r.Winner.Score {
		t.Errorf("logistic regression scored %v against %v on a straight line: %v", lin.Score, r.Winner.Score, r.Trials)
	}
}

func TestSearchRegressionAndLimits(t *testing.T) {
	ctx := context.Background()
	calls := 0
	r, err := automl.Search(ctx, table(500, "product", 4), tabular.Regression,
		automl.Families("linear", "forest"), automl.Trials(3), automl.Progress(func(int, int) { calls++ }))
	if err != nil {
		t.Fatal(err)
	}
	if r.Metric != "rmse" || calls != len(r.Trials) {
		t.Errorf("metric %s, %d progress calls for %d trials", r.Metric, calls, len(r.Trials))
	}
	for _, tr := range r.Trials {
		if tr.Family == "boost" {
			t.Errorf("boosting tried though left out: %v", tr)
		}
		if !strings.HasPrefix(tr.String(), tr.Family) {
			t.Errorf("trial named %q", tr.String())
		}
	}
	if r.Winner.Family != "forest" {
		t.Errorf("a line beat a forest on a product: %v", r.Trials)
	}
	// A budget of nothing still scores one candidate. The table is large
	// enough that one fit takes longer than a tick of the coarsest clock
	// (about 16 ms on Windows), so the budget is seen to be spent.
	r, err = automl.Search(ctx, table(50000, "line", 5), tabular.Classification, automl.Budget(1), automl.Families("linear"))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Trials) != 1 {
		t.Errorf("%d trials in no time", len(r.Trials))
	}
	if _, ok := r.Best.(*linear.Model); !ok || math.IsInf(r.Winner.Score, 0) {
		t.Errorf("best %T scoring %v", r.Best, r.Winner.Score)
	}
}
