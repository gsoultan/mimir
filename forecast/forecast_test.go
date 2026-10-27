package forecast

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// weekly returns a series of n days with a weekly cycle and a trend-free
// level, scaled by size, with noise.
func weekly(n int, size float64, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 3))
	s := make([]float32, n)
	shape := []float64{1, 1.1, 1.2, 1.1, 1.4, 2, 1.8}
	for t := range s {
		s[t] = float32(size * (shape[t%7] + 0.05*r.NormFloat64()))
	}
	return s
}

func TestForecastLearnsTheWeek(t *testing.T) {
	var series [][]float32
	for i := range 20 {
		series = append(series, weekly(200, math.Pow(10, float64(i%4)), uint64(i)))
	}
	m, err := Fit(context.Background(), series, Season(7))
	if err != nil {
		t.Fatal(err)
	}
	if m.Error >= m.Naive {
		t.Errorf("one-step error %v is no better than the naive %v", m.Error, m.Naive)
	}
	// A new series of another size, and its next two weeks.
	h := weekly(203, 50, 99)
	got := m.Forecast(h[:189], 14)
	var mape float64
	for i, v := range got {
		mape += math.Abs(float64(v-h[189+i])) / float64(h[189+i])
	}
	if mape /= 14; mape > 0.08 {
		t.Errorf("two-week forecast %v is off by %.1f%% on average", got, 100*mape)
	}
	x, k := m.Rows(h[:189])
	if len(x) != m.Features() || math.Abs(float64(m.Trees.Predict(x)[0]*k-got[0])) > 1e-4 {
		t.Errorf("Rows gives a step answer other than the forecast's first")
	}
	b, _ := json.Marshal(m)
	var back Model
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if again := back.Forecast(h[:189], 14); again[13] != got[13] {
		t.Error("the decoded model forecasts differently")
	}
}

func TestForecastWithGaps(t *testing.T) {
	s := weekly(150, 10, 4)
	for i := 0; i < len(s); i += 9 {
		s[i] = float32(math.NaN())
	}
	m, err := Fit(context.Background(), [][]float32{s}, Season(7))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range m.Forecast(s, 7) {
		if v != v || v < 5 || v > 25 {
			t.Fatalf("forecast %v", m.Forecast(s, 7))
		}
	}
}

func TestForecastBadInput(t *testing.T) {
	for _, s := range [][][]float32{nil, {{1, 2}}} {
		if _, err := Fit(context.Background(), s); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%v: error %v", s, err)
		}
	}
	if _, err := Fit(context.Background(), [][]float32{{1, 2, 3, 4}}, Season(-1)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("negative season: error %v", err)
	}
}
