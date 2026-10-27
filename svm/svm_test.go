package svm

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

// rings is two classes no line separates: inside a circle or outside it,
// with a noise feature in other units.
func rings(n int, seed uint64) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 1))
	d := tabular.Data{Features: 3}
	for range n {
		a, b := r.Float64()*4-2, r.Float64()*4-2
		y := float32(0)
		if a*a+b*b < 1.6 {
			y = 1
		}
		d.X = append(d.X, float32(a), float32(b), float32(r.Float64()*100))
		d.Y = append(d.Y, y)
	}
	return d
}

// blobs is k classes around their own centres.
func blobs(n, k int, seed uint64) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 2))
	d := tabular.Data{Features: 2}
	for i := range n {
		c := i % k
		ang := 2 * math.Pi * float64(c) / float64(k)
		d.X = append(d.X, float32(3*math.Cos(ang)+r.NormFloat64()*0.6), float32(3*math.Sin(ang)+r.NormFloat64()*0.6))
		d.Y = append(d.Y, float32(c))
	}
	return d
}

func accuracy(m tabular.Model, d tabular.Data) float64 {
	return metrics.ClassAccuracy(m.Predict(d.X), d.Y)
}

func TestKernelsSeparateRings(t *testing.T) {
	ctx := context.Background()
	train, test := rings(600, 1), rings(400, 2)
	for _, tc := range []struct {
		kernel string
		min    float64
	}{{RBF, 0.93}, {Polynomial, 0.9}, {Linear, 0.5}} {
		opts := []Option{Kernel(tc.kernel), C(10)}
		if tc.kernel == Polynomial {
			opts = append(opts, Degree(2), Coef0(1))
		}
		m, err := Fit(ctx, train, tabular.Classification, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if acc := accuracy(m, test); acc < tc.min {
			t.Errorf("%s: accuracy %.3f, want at least %v", tc.kernel, acc, tc.min)
		}
		if tc.kernel == RBF && m.SupportVectors() >= train.Rows()/2 {
			t.Errorf("%d support vectors of %d rows", m.SupportVectors(), train.Rows())
		}
	}
}

func TestLinearSeparatesBlobs(t *testing.T) {
	ctx := context.Background()
	for _, kernel := range []string{Linear, RBF} {
		m, err := Fit(ctx, blobs(600, 4, 3), tabular.Classification, Kernel(kernel))
		if err != nil {
			t.Fatal(err)
		}
		if acc := accuracy(m, blobs(400, 4, 4)); acc < 0.95 {
			t.Errorf("%s: accuracy %.3f on 4 blobs", kernel, acc)
		}
		p := m.Predict([]float32{3, 0})
		if len(p) != 4 || p[0] < 0.5 {
			t.Errorf("%s: probabilities %v at class 0's centre", kernel, p)
		}
		var sum float32
		for _, v := range p {
			sum += v
		}
		if math.Abs(float64(sum)-1) > 1e-4 {
			t.Errorf("%s: probabilities sum to %v", kernel, sum)
		}
	}
}

func TestRegression(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewPCG(5, 6))
	mk := func(n int) tabular.Data {
		d := tabular.Data{Features: 1}
		for range n {
			x := r.Float64()*6 - 3
			d.X = append(d.X, float32(x))
			d.Y = append(d.Y, float32(10*math.Sin(x)+50+r.NormFloat64()*0.3))
		}
		return d
	}
	train, test := mk(400), mk(200)
	m, err := Fit(ctx, train, tabular.Regression, C(100), Gamma(1))
	if err != nil {
		t.Fatal(err)
	}
	if rmse := metrics.RMSE(m.Predict(test.X), test.Y); rmse > 1 {
		t.Errorf("rbf rmse %.3f on 10·sin(x)+50", rmse)
	}
	lin, err := Fit(ctx, train, tabular.Regression, Kernel(Linear))
	if err != nil {
		t.Fatal(err)
	}
	// A line through a sine still beats the mean.
	if rmse := metrics.RMSE(lin.Predict(test.X), test.Y); rmse > 6.5 {
		t.Errorf("linear rmse %.3f", rmse)
	}
}

func TestModuleMatchesPredict(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name string
		d    tabular.Data
		task tabular.Task
		opts []Option
	}{
		{"rbf two", rings(300, 7), tabular.Classification, []Option{C(10)}},
		{"poly many", blobs(300, 3, 8), tabular.Classification, []Option{Kernel(Polynomial), Degree(3), Coef0(1)}},
		{"linear many", blobs(300, 3, 9), tabular.Classification, []Option{Kernel(Linear)}},
		{"sigmoid two", rings(300, 10), tabular.Classification, []Option{Kernel(Sigmoid), Gamma(0.1)}},
		{"rbf regression", rings(300, 11), tabular.Regression, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, err := Fit(ctx, tc.d, tc.task, tc.opts...)
			if err != nil {
				t.Fatal(err)
			}
			// Saved as JSON and read back, it answers alike.
			b, _ := json.Marshal(m)
			var back Model
			if err := json.Unmarshal(b, &back); err != nil {
				t.Fatal(err)
			}
			x := tc.d.X[:20*tc.d.Features]
			x[1] = float32(math.NaN())
			want := back.Predict(x)
			mod := m.Module()
			if mod.Shape() != NewModule(m.Shape()).Shape() {
				t.Fatal("shapes differ")
			}
			out := mod.Forward(tensor.FromSlice(x, 20, tc.d.Features))
			if tc.task == tabular.Classification {
				out = out.Softmax(1)
			}
			got, err := out.Data[float32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("%d answers, want %d", len(got), len(want))
			}
			for i := range got {
				if math.Abs(float64(got[i]-want[i])) > 1e-3*max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("answer %d: module %v, Predict %v", i, got[i], want[i])
				}
			}
		})
	}
}

func TestBadOptions(t *testing.T) {
	d := blobs(20, 2, 1)
	for _, o := range []Option{C(0), Kernel("cubic"), Gamma(-1), Classes(1)} {
		if _, err := Fit(context.Background(), d, tabular.Classification, o); err == nil {
			t.Error("a bad option fit")
		}
	}
}
