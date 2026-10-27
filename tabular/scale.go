package tabular

import "math"

// Scaler standardizes features: it subtracts each feature's mean and
// divides by its standard deviation, as scikit-learn's StandardScaler.
// A missing value, NaN, becomes 0, the mean, and so does every value of
// a constant feature, whose Std is 0.
type Scaler struct {
	Mean, Std []float32
}

// FitScaler returns the Scaler of x, rows of features numbers one after
// another, from the values that are not NaN.
func FitScaler(x []float32, features int) Scaler {
	s := Scaler{Mean: make([]float32, features), Std: make([]float32, features)}
	n := len(x) / max(features, 1)
	for j := range features {
		var sum, cnt float64
		for i := range n {
			if v := float64(x[i*features+j]); v == v {
				sum += v
				cnt++
			}
		}
		if cnt == 0 {
			continue
		}
		mean := sum / cnt
		var sq float64
		for i := range n {
			if v := float64(x[i*features+j]); v == v {
				sq += (v - mean) * (v - mean)
			}
		}
		s.Mean[j] = float32(mean)
		if std := math.Sqrt(sq / cnt); std > 1e-12*math.Max(1, math.Abs(mean)) {
			s.Std[j] = float32(std)
		}
	}
	return s
}

// Transform returns x standardized, in a new slice.
func (s Scaler) Transform(x []float32) []float32 {
	f := len(s.Mean)
	out := make([]float32, len(x))
	for i, v := range x {
		j := i % f
		if v == v && s.Std[j] != 0 {
			out[i] = (v - s.Mean[j]) / s.Std[j]
		}
	}
	return out
}
