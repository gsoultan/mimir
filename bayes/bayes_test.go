package bayes

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

func gaussians(n int, seed uint64) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 4))
	d := tabular.Data{Features: 3}
	for i := range n {
		c := i % 3
		d.X = append(d.X, float32(float64(c)*2+r.NormFloat64()), float32(r.NormFloat64()*(1+float64(c))), float32(r.NormFloat64()*50))
		d.Y = append(d.Y, float32(c))
	}
	return d
}

// words is texts as counts of 6 words: class 0 uses the first three
// more, class 1 the last three.
func words(n int, seed uint64) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 5))
	d := tabular.Data{Features: 6}
	for i := range n {
		c := i % 2
		for j := range 6 {
			rate := 0.3
			if j/3 == c {
				rate = 2
			}
			k := 0
			for range 4 {
				if r.Float64() < rate/4 {
					k++
				}
			}
			d.X = append(d.X, float32(k))
		}
		d.Y = append(d.Y, float32(c))
	}
	return d
}

func TestFitAndClassify(t *testing.T) {
	ctx := context.Background()
	g, err := Gaussian(ctx, gaussians(600, 1))
	if err != nil {
		t.Fatal(err)
	}
	test := gaussians(300, 2)
	if acc := metrics.ClassAccuracy(g.Predict(test.X), test.Y); acc < 0.7 {
		t.Errorf("gaussian accuracy %.3f", acc)
	}
	for name, fit := range map[string]func(context.Context, tabular.Data, ...Option) (*Model, error){"multinomial": Multinomial, "bernoulli": Bernoulli} {
		m, err := fit(ctx, words(400, 3))
		if err != nil {
			t.Fatal(err)
		}
		test := words(300, 4)
		if acc := metrics.ClassAccuracy(m.Predict(test.X), test.Y); acc < 0.85 {
			t.Errorf("%s accuracy %.3f", name, acc)
		}
	}
}

func TestModuleMatchesPredict(t *testing.T) {
	ctx := context.Background()
	for name, d := range map[string]tabular.Data{"gaussian": gaussians(200, 8), "multinomial": words(200, 9), "bernoulli": words(200, 10)} {
		fit := map[string]func(context.Context, tabular.Data, ...Option) (*Model, error){"gaussian": Gaussian, "multinomial": Multinomial, "bernoulli": Bernoulli}[name]
		m, err := fit(ctx, d)
		if err != nil {
			t.Fatal(err)
		}
		bs, _ := json.Marshal(m)
		var back Model
		if err := json.Unmarshal(bs, &back); err != nil {
			t.Fatal(err)
		}
		x := d.X[:10*d.Features]
		x[2] = float32(math.NaN())
		want := back.Predict(x)
		got, err := m.Module().Forward(tensor.FromSlice(x, 10, d.Features)).Softmax(1).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-4 {
				t.Fatalf("%s: answer %d: module %v, Predict %v", name, i, got[i], want[i])
			}
		}
	}
}

func TestBadOptions(t *testing.T) {
	d := words(10, 1)
	if _, err := Multinomial(context.Background(), tabular.Data{Features: 1, X: []float32{-1, 2}, Y: []float32{0, 1}}); err == nil {
		t.Error("a negative count fit")
	}
	for _, o := range []Option{Alpha(-1), Classes(1), Priors(1)} {
		if _, err := Bernoulli(context.Background(), d, o); err == nil {
			t.Error("a bad option fit")
		}
	}
}
