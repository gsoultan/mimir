package tree

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// table returns n rows of f uniform features in [-1, 1) and the answer
// y(row) for each.
func table(n, f int, seed uint64, y func(row []float32) float32) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 1))
	d := tabular.Data{X: make([]float32, n*f), Features: f, Y: make([]float32, n)}
	for i := range d.X {
		d.X[i] = 2*r.Float32() - 1
	}
	for i := range n {
		d.Y[i] = y(d.X[i*f : (i+1)*f])
	}
	return d
}

// xor is a class no straight line separates: whether x0 and x1 share a sign.
func xor(row []float32) float32 {
	if (row[0] > 0) == (row[1] > 0) {
		return 1
	}
	return 0
}

// quadrant is one of three classes by the signs of x0 and x1.
func quadrant(row []float32) float32 {
	switch {
	case row[0] > 0 && row[1] > 0:
		return 2
	case row[0] > 0:
		return 1
	}
	return 0
}

func wave(row []float32) float32 {
	return float32(math.Sin(3*float64(row[0]))) + row[1]*row[1]
}

func accuracy(m *Model, d tabular.Data) float64 {
	c := m.Class(m.Predict(d.X))
	hits := 0
	for i, y := range d.Y {
		if c[i] == int(y) {
			hits++
		}
	}
	return float64(hits) / float64(len(d.Y))
}

// r2 returns the share of the variance of d.Y that m explains.
func r2(m *Model, d tabular.Data) float64 {
	p := m.Predict(d.X)
	var mean, ss, res float64
	for _, y := range d.Y {
		mean += float64(y)
	}
	mean /= float64(len(d.Y))
	for i, y := range d.Y {
		ss += (float64(y) - mean) * (float64(y) - mean)
		res += (float64(y) - float64(p[i])) * (float64(y) - float64(p[i]))
	}
	return 1 - res/ss
}

type fitter func(context.Context, tabular.Data, tabular.Task, ...Option) (*Model, error)

var fitters = map[string]fitter{
	"boosting": GradientBoosting,
	"forest":   RandomForest,
	"tree":     DecisionTree,
}

func TestModelsLearn(t *testing.T) {
	ctx := context.Background()
	for name, fit := range fitters {
		t.Run(name, func(t *testing.T) {
			for _, tc := range []struct {
				name string
				task tabular.Task
				y    func([]float32) float32
				min  float64
			}{
				{"two classes", tabular.Classification, xor, 0.93},
				{"three classes", tabular.Classification, quadrant, 0.93},
				{"regression", tabular.Regression, wave, 0.85},
			} {
				train, test := table(4000, 4, 1, tc.y), table(1000, 4, 2, tc.y)
				m, err := fit(ctx, train, tc.task)
				if err != nil {
					t.Fatal(err)
				}
				score := r2(m, test)
				if tc.task == tabular.Classification {
					score = accuracy(m, test)
				}
				if score < tc.min {
					t.Errorf("%s: test score %.3f, want at least %.2f", tc.name, score, tc.min)
				}
			}
		})
	}
}

func TestProbabilitiesSumToOne(t *testing.T) {
	ctx := context.Background()
	d := table(500, 3, 3, quadrant)
	for name, fit := range fitters {
		m, err := fit(ctx, d, tabular.Classification)
		if err != nil {
			t.Fatal(err)
		}
		p := m.Predict(d.X[:30])
		if len(p) != 10*m.Classes || m.Classes != 3 {
			t.Fatalf("%s: %d probabilities for 10 rows of %d classes", name, len(p), m.Classes)
		}
		for i := range 10 {
			var s float64
			for _, v := range p[i*3 : i*3+3] {
				if v < 0 || v > 1 {
					t.Fatalf("%s: probability %v", name, v)
				}
				s += float64(v)
			}
			if math.Abs(s-1) > 1e-5 {
				t.Errorf("%s: row %d probabilities sum to %v", name, i, s)
			}
		}
	}
}

func TestDecisionTreeFitsItsTrainingRows(t *testing.T) {
	d := table(300, 2, 4, xor)
	m, err := DecisionTree(context.Background(), d, tabular.Classification)
	if err != nil {
		t.Fatal(err)
	}
	if a := accuracy(m, d); a != 1 {
		t.Errorf("training accuracy %v, want 1", a)
	}
}

func TestStumpSplitsBetweenTheClasses(t *testing.T) {
	d := tabular.Data{X: []float32{1, 2, 3, 10, 11, 12}, Features: 1, Y: []float32{0, 0, 0, 1, 1, 1}}
	m, err := DecisionTree(context.Background(), d, tabular.Classification, MaxDepth(1))
	if err != nil {
		t.Fatal(err)
	}
	tr := m.Trees[0]
	if tr.Nodes() != 3 || tr.Feature[0] != 0 || tr.Threshold[0] != 6.5 {
		t.Fatalf("tree %+v, want one split at 6.5", tr)
	}
	if got := m.Predict([]float32{6, 7}); !reflect.DeepEqual(got, []float32{1, 0, 0, 1}) {
		t.Errorf("predictions %v", got)
	}
}

func TestMissingValuesGoWhereTheyFit(t *testing.T) {
	// The class is 1 exactly when x0 is missing.
	d := table(2000, 2, 5, func([]float32) float32 { return 0 })
	for i := range d.Y {
		if i%3 == 0 {
			d.X[i*2] = float32(math.NaN())
			d.Y[i] = 1
		}
	}
	for name, fit := range fitters {
		m, err := fit(context.Background(), d, tabular.Classification, PerNode(2))
		if err != nil {
			t.Fatal(err)
		}
		p := m.Predict([]float32{float32(math.NaN()), 0.3, 0.5, 0.3})
		if p[1] < 0.9 || p[3] > 0.1 {
			t.Errorf("%s: P(1) %v for a missing x0, %v for x0 = 0.5", name, p[1], p[3])
		}
	}
}

func TestImportanceFindsTheUsefulFeature(t *testing.T) {
	d := table(2000, 5, 6, func(row []float32) float32 { return 3 * row[2] })
	for name, fit := range fitters {
		m, err := fit(context.Background(), d, tabular.Regression)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for _, v := range m.Importance {
			sum += v
		}
		if m.Importance[2] < 0.9 || math.Abs(sum-1) > 1e-9 {
			t.Errorf("%s: importance %v", name, m.Importance)
		}
	}
}

func TestValidationStopsBoosting(t *testing.T) {
	ctx := context.Background()
	// Labels are noise, so the validation loss soon stops falling.
	r := rand.New(rand.NewPCG(7, 7))
	noise := func([]float32) float32 { return float32(r.IntN(2)) }
	train, val := table(1000, 3, 8, noise), table(500, 3, 9, noise)
	m, err := GradientBoosting(ctx, train, tabular.Classification, Trees(500), Validation(val, 5))
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Trees) >= 100 {
		t.Errorf("kept %d of 500 trees on noise", len(m.Trees))
	}
}

func TestBoostingOptionsLearn(t *testing.T) {
	ctx := context.Background()
	train, test := table(4000, 6, 10, xor), table(1000, 6, 11, xor)
	m, err := GradientBoosting(ctx, train, tabular.Classification, Subsample(0.7), ColSample(0.8),
		Bins(32), LearningRate(0.3), MaxDepth(4), Trees(150), L2(3), MinLeaf(5))
	if err != nil {
		t.Fatal(err)
	}
	if a := accuracy(m, test); a < 0.9 {
		t.Errorf("test accuracy %.3f", a)
	}
}

func TestSameSeedSameModel(t *testing.T) {
	ctx := context.Background()
	d := table(1000, 4, 12, quadrant)
	for name, fit := range fitters {
		a, err := fit(ctx, d, tabular.Classification, Seed(3), Subsample(0.8))
		if err != nil {
			t.Fatal(err)
		}
		b, err := fit(ctx, d, tabular.Classification, Seed(3), Subsample(0.8), Workers(1))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a.Predict(d.X), b.Predict(d.X)) {
			t.Errorf("%s: two fits with one seed differ", name)
		}
	}
}

func TestJSONRoundTrip(t *testing.T) {
	d := table(500, 3, 13, wave)
	m, err := GradientBoosting(context.Background(), d, tabular.Regression, Trees(20))
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var back Model
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m.Predict(d.X), back.Predict(d.X)) {
		t.Error("the decoded model answers differently")
	}
}

func TestBadInput(t *testing.T) {
	ctx := context.Background()
	good := table(10, 2, 14, xor)
	for _, tc := range []struct {
		name string
		d    tabular.Data
		task tabular.Task
		opts []Option
	}{
		{"no features", tabular.Data{X: good.X, Y: good.Y}, tabular.Classification, nil},
		{"ragged rows", tabular.Data{X: good.X[:19], Features: 2, Y: good.Y}, tabular.Classification, nil},
		{"answers", tabular.Data{X: good.X, Features: 2, Y: good.Y[:9]}, tabular.Classification, nil},
		{"fractional class", tabular.Data{X: good.X, Features: 2, Y: append([]float32{0.5}, good.Y[1:]...)}, tabular.Classification, nil},
		{"NaN answer", tabular.Data{X: good.X, Features: 2, Y: append([]float32{float32(math.NaN())}, good.Y[1:]...)}, tabular.Regression, nil},
		{"trees", good, tabular.Classification, []Option{Trees(0)}},
		{"bins", good, tabular.Classification, []Option{Bins(300)}},
		{"subsample", good, tabular.Classification, []Option{Subsample(0)}},
		{"validation width", good, tabular.Classification, []Option{Validation(table(5, 3, 1, xor), 3)}},
	} {
		for name, fit := range fitters {
			if _, err := fit(ctx, tc.d, tc.task, tc.opts...); !errors.Is(err, tensor.ErrShape) {
				t.Errorf("%s, %s: error %v, want ErrShape", tc.name, name, err)
			}
		}
	}
}

func TestCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	d := table(100, 2, 15, xor)
	for name, fit := range fitters {
		if _, err := fit(ctx, d, tabular.Classification); !errors.Is(err, context.Canceled) {
			t.Errorf("%s: error %v", name, err)
		}
	}
}

func BenchmarkGradientBoosting(b *testing.B) {
	d := table(100_000, 20, 16, wave)
	for b.Loop() {
		if _, err := GradientBoosting(context.Background(), d, tabular.Regression, Trees(50)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRandomForest(b *testing.B) {
	d := table(50_000, 20, 17, xor)
	for b.Loop() {
		if _, err := RandomForest(context.Background(), d, tabular.Classification, Trees(50)); err != nil {
			b.Fatal(err)
		}
	}
}
