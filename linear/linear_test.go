package linear

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

// table returns n rows of f normal features, feature j scaled by 10^j so
// the units differ, and the answer y(row) for each.
func table(n, f int, seed uint64, y func(r *rand.Rand, row []float32) float32) tabular.Data {
	r := rand.New(rand.NewPCG(seed, 1))
	d := tabular.Data{X: make([]float32, n*f), Features: f, Y: make([]float32, n)}
	for i := range d.X {
		d.X[i] = float32(r.NormFloat64() * math.Pow(10, float64(i%f)))
	}
	for i := range n {
		d.Y[i] = y(r, d.Row(i))
	}
	return d
}

func near(t *testing.T, name string, got float32, want, tol float64) {
	t.Helper()
	if math.Abs(float64(got)-want) > tol {
		t.Errorf("%s = %v, want %v ± %v", name, got, want, tol)
	}
}

func TestLeastSquaresRecoversTheLine(t *testing.T) {
	d := table(500, 3, 1, func(_ *rand.Rand, x []float32) float32 { return 2*x[0] - 0.3*x[1] + 5 })
	m, err := Regression(context.Background(), d, Tolerance(1e-9))
	if err != nil {
		t.Fatal(err)
	}
	near(t, "w0", m.Weight(0, 0), 2, 1e-3)
	near(t, "w1", m.Weight(0, 1), -0.3, 1e-4)
	near(t, "w2", m.Weight(0, 2), 0, 1e-5)
	near(t, "b", m.Bias[0], 5, 1e-3)
}

func TestRidgeMatchesItsClosedForm(t *testing.T) {
	d := table(400, 2, 2, func(r *rand.Rand, x []float32) float32 { return x[0] + 0.05*x[1] + float32(r.NormFloat64()) })
	const l2 = 0.5
	m, err := Regression(context.Background(), d, L2(l2), Tolerance(1e-12), MaxIter(100000))
	if err != nil {
		t.Fatal(err)
	}
	// Solve (ZᵀZ/n + l2 I) w = Zᵀ(y - ȳ)/n over the standardized Z.
	s := standardize(d)
	n := float64(s.n)
	var ybar float64
	for _, y := range d.Y {
		ybar += float64(y)
	}
	ybar /= n
	var a [2][2]float64
	var b [2]float64
	for i := range s.n {
		z := [2]float64{float64(s.cols[i]), float64(s.cols[s.n+i])}
		for p := range 2 {
			b[p] += z[p] * (float64(d.Y[i]) - ybar) / n
			for q := range 2 {
				a[p][q] += z[p] * z[q] / n
			}
		}
	}
	a[0][0] += l2
	a[1][1] += l2
	det := a[0][0]*a[1][1] - a[0][1]*a[1][0]
	w0 := (b[0]*a[1][1] - a[0][1]*b[1]) / det
	w1 := (a[0][0]*b[1] - a[1][0]*b[0]) / det
	near(t, "w0", m.Weight(0, 0), w0/s.std[0], 1e-4)
	near(t, "w1", m.Weight(0, 1), w1/s.std[1], 1e-5)
}

func TestLassoDropsUselessFeatures(t *testing.T) {
	d := table(1000, 5, 3, func(r *rand.Rand, x []float32) float32 { return 3*x[0] + float32(r.NormFloat64()) })
	m, err := Regression(context.Background(), d, L1(0.1))
	if err != nil {
		t.Fatal(err)
	}
	for j := 1; j < 5; j++ {
		if m.Weight(0, j) != 0 {
			t.Errorf("weight of useless feature %d is %v", j, m.Weight(0, j))
		}
	}
	if w := m.Weight(0, 0); w < 2.8 || w > 3 {
		t.Errorf("w0 = %v, want a little under 3", w)
	}
}

func TestLogisticRecoversTheLogOdds(t *testing.T) {
	d := table(20000, 2, 4, func(r *rand.Rand, x []float32) float32 {
		if r.Float64() < 1/(1+math.Exp(-(2*float64(x[0])-0.1*float64(x[1])+0.5))) {
			return 1
		}
		return 0
	})
	m, err := Logistic(context.Background(), d, L2(0))
	if err != nil {
		t.Fatal(err)
	}
	if m.Link != "sigmoid" || m.Classes != 2 {
		t.Fatalf("link %q for %d classes", m.Link, m.Classes)
	}
	near(t, "w0", m.Weight(0, 0), 2, 0.15)
	near(t, "w1", m.Weight(0, 1), -0.1, 0.01)
	near(t, "b", m.Bias[0], 0.5, 0.1)
	p := m.Predict([]float32{0, 5})
	near(t, "P(1)", p[1], 1/(1+math.Exp(0)), 0.06)
}

func TestLogisticSeparatesThreeClasses(t *testing.T) {
	// Three classes in a row along x0+x1/10 are not linearly separable
	// one against the rest, but softmax separates them.
	d := table(3000, 2, 5, func(_ *rand.Rand, x []float32) float32 {
		v := x[0] + x[1]/10
		switch {
		case v > 1:
			return 2
		case v < -1:
			return 0
		}
		return 1
	})
	m, err := Logistic(context.Background(), d, L2(1e-5))
	if err != nil {
		t.Fatal(err)
	}
	p := m.Predict(d.X)
	hits := 0
	for i, y := range d.Y {
		row := p[i*3 : i*3+3]
		best := 0
		for c := range row {
			if row[c] > row[best] {
				best = c
			}
		}
		if best == int(y) {
			hits++
		}
	}
	if a := float64(hits) / float64(len(d.Y)); a < 0.97 {
		t.Errorf("accuracy %.3f", a)
	}
}

func TestL1LogisticDropsUselessFeatures(t *testing.T) {
	d := table(3000, 4, 6, func(r *rand.Rand, x []float32) float32 {
		if x[0] > 0 {
			return 1
		}
		return 0
	})
	m, err := Logistic(context.Background(), d, L1(0.02))
	if err != nil {
		t.Fatal(err)
	}
	for j := 1; j < 4; j++ {
		if m.Weight(0, j) != 0 {
			t.Errorf("weight of useless feature %d is %v", j, m.Weight(0, j))
		}
	}
	if m.Weight(0, 0) <= 0 {
		t.Errorf("w0 = %v", m.Weight(0, 0))
	}
}

func TestMissingAndConstantFeatures(t *testing.T) {
	d := table(200, 3, 7, func(_ *rand.Rand, x []float32) float32 { return x[0] })
	for i := range d.Rows() {
		d.X[i*3+2] = 4 // constant
		if i%10 == 0 {
			d.X[i*3+1] = float32(math.NaN())
		}
	}
	m, err := Regression(context.Background(), d)
	if err != nil {
		t.Fatal(err)
	}
	if m.Weight(0, 2) != 0 {
		t.Errorf("constant feature's weight %v", m.Weight(0, 2))
	}
	p := m.Predict([]float32{1.5, float32(math.NaN()), 4})
	near(t, "prediction", p[0], 1.5, 1e-3)
}

func TestJSONRoundTrip(t *testing.T) {
	d := table(300, 3, 8, func(_ *rand.Rand, x []float32) float32 {
		return float32(int(math.Abs(float64(x[0]))*2) % 3)
	})
	m, err := Logistic(context.Background(), d)
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
	good := table(10, 2, 9, func(_ *rand.Rand, x []float32) float32 { return 1 })
	for _, tc := range []struct {
		name string
		d    tabular.Data
		opts []Option
	}{
		{"ragged", tabular.Data{X: good.X[:5], Features: 2, Y: good.Y}, nil},
		{"L1", good, []Option{L1(-1)}},
		{"L2", good, []Option{L2(math.NaN())}},
		{"iterations", good, []Option{MaxIter(0)}},
	} {
		if _, err := Regression(ctx, tc.d, tc.opts...); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("Regression, %s: error %v", tc.name, err)
		}
		if _, err := Logistic(ctx, tc.d, tc.opts...); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("Logistic, %s: error %v", tc.name, err)
		}
	}
	frac := tabular.Data{X: good.X, Features: 2, Y: append([]float32{0.5}, good.Y[1:]...)}
	if _, err := Logistic(ctx, frac); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("fractional class: error %v", err)
	}
}
