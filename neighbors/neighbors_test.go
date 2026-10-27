package neighbors

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/metrics"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

func moons(n int, seed uint64) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 3))
	d := tabular.Data{Features: 2}
	for i := range n {
		t := r.Float64() * math.Pi
		a, b, y := math.Cos(t), math.Sin(t), float32(0)
		if i%2 == 1 {
			a, b, y = 1-math.Cos(t), 0.5-math.Sin(t), 1
		}
		d.X = append(d.X, float32(a+r.NormFloat64()*0.1), float32(b+r.NormFloat64()*0.1)*100)
		d.Y = append(d.Y, y)
	}
	return d
}

func TestClassifiesMoons(t *testing.T) {
	ctx := context.Background()
	train, test := moons(400, 1), moons(300, 2)
	for _, opts := range [][]Option{nil, {K(9), ByDistance()}, {Metric(Manhattan)}} {
		m, err := Fit(ctx, train, tabular.Classification, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if acc := metrics.ClassAccuracy(m.Predict(test.X), test.Y); acc < 0.95 {
			t.Errorf("accuracy %.3f with %d options", acc, len(opts))
		}
	}
	// In raw units the second feature, a hundred times larger, drowns the
	// first; standardized, both count.
	raw, err := Fit(ctx, train, tabular.Classification, Raw())
	if err != nil {
		t.Fatal(err)
	}
	if raw.Std[1] != 1 {
		t.Errorf("raw std %v", raw.Std)
	}
}

func TestRegressionAndNeighbors(t *testing.T) {
	d := tabular.Data{Features: 1, X: []float32{0, 1, 2, 3, 10}, Y: []float32{0, 10, 20, 30, 100}}
	m, err := Fit(context.Background(), d, tabular.Regression, K(2), Raw())
	if err != nil {
		t.Fatal(err)
	}
	if got := m.Predict([]float32{1.4}); math.Abs(float64(got[0])-15) > 1e-4 {
		t.Errorf("mean of 2 neighbours of 1.4 = %v, want 15", got)
	}
	ns := m.Neighbors([]float32{2.6})[0]
	if len(ns) != 2 || ns[0].Row != 3 || ns[1].Row != 2 {
		t.Errorf("neighbours of 2.6: %+v", ns)
	}
	w, _ := Fit(context.Background(), d, tabular.Regression, K(2), Raw(), ByDistance())
	// Weights 1/0.5 and 1/0.5 at 1.5: 15; at 1.25, 1/0.25 for 1 and 1/0.75 for 2.
	if got := w.Predict([]float32{1.25}); math.Abs(float64(got[0])-(10*4+20*4.0/3)/(4+4.0/3)) > 1e-3 {
		t.Errorf("distance-weighted answer %v", got)
	}
}

func TestModuleMatchesPredict(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		task tabular.Task
		opts []Option
	}{
		{tabular.Classification, nil},
		{tabular.Classification, []Option{K(7), ByDistance(), Metric(Cosine)}},
		{tabular.Classification, []Option{Metric(Manhattan)}},
		{tabular.Regression, []Option{ByDistance()}},
	} {
		d := moons(200, 5)
		if tc.task == tabular.Regression {
			for i := range d.Y {
				d.Y[i] = d.X[2*i] * 3
			}
		}
		m, err := Fit(ctx, d, tc.task, tc.opts...)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(m)
		var back Model
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatal(err)
		}
		q := moons(30, 6).X
		q[0] = float32(math.NaN())
		want := back.Predict(q)
		out := m.Module().Forward(tensor.FromSlice(q, 30, 2))
		if tc.task == tabular.Classification {
			out = out.Softmax(1)
		}
		got, err := out.Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		if NewModule(m.Shape()).Shape() != m.Shape() {
			t.Fatal("shapes differ")
		}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-3*max(1, math.Abs(float64(want[i]))) {
				t.Fatalf("%v: answer %d: module %v, Predict %v", tc.opts, i, got[i], want[i])
			}
		}
	}
}

func TestBadOptions(t *testing.T) {
	d := moons(10, 1)
	for _, o := range []Option{K(0), Metric("chebyshev"), Classes(1)} {
		if _, err := Fit(context.Background(), d, tabular.Classification, o); err == nil {
			t.Error("a bad option fit")
		}
	}
}
