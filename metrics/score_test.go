package metrics

import (
	"math"
	"reflect"
	"testing"
)

func approx(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-6 && !(math.IsNaN(got) && math.IsNaN(want)) {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

func TestClassScores(t *testing.T) {
	// Four rows of two classes: predicted 1, 0, 1, 1 for classes 1, 0, 0, 1.
	p := []float32{0.2, 0.8, 0.9, 0.1, 0.4, 0.6, 0.3, 0.7}
	y := []float32{1, 0, 0, 1}
	approx(t, "accuracy", ClassAccuracy(p, y), 0.75)
	if m := Confusion(p, y); !reflect.DeepEqual(m, [][]int{{1, 1}, {0, 2}}) {
		t.Errorf("confusion %v", m)
	}
	approx(t, "precision", Precision(p, y), 2.0/3)
	approx(t, "recall", Recall(p, y), 1)
	approx(t, "F1", F1(p, y), 0.8)
	approx(t, "log loss", LogLoss(p, y), -(math.Log(0.8)+math.Log(0.9)+math.Log(0.4)+math.Log(0.7))/4)
	// P(1) of the positives 0.8, 0.7 beat both negatives 0.1, 0.6.
	approx(t, "AUC", ROCAUC(p, y), 1)
}

func TestROCAUCTiesAndClasses(t *testing.T) {
	// sklearn.metrics.roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) = 0.75
	p := []float32{0.9, 0.1, 0.6, 0.4, 0.65, 0.35, 0.2, 0.8}
	approx(t, "AUC", ROCAUC(p, []float32{0, 0, 1, 1}), 0.75)
	// A tie between a positive and a negative counts half.
	approx(t, "tied AUC", ROCAUC([]float32{0.5, 0.5, 0.5, 0.5}, []float32{0, 1}), 0.5)
	approx(t, "one class", ROCAUC([]float32{0.5, 0.5}, []float32{1}), math.NaN())
	// Three perfectly ranked classes.
	p3 := []float32{0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8}
	y3 := []float32{0, 1, 2}
	approx(t, "three-class AUC", ROCAUC(p3, y3), 1)
	approx(t, "macro F1", F1(p3, y3), 1)
}

func TestRegressionScores(t *testing.T) {
	p := []float32{2.5, 0, 2, 8}
	y := []float32{3, -0.5, 2, 7}
	approx(t, "MAE", MAE(p, y), 0.5)
	approx(t, "RMSE", RMSE(p, y), math.Sqrt(0.375))
	// sklearn.metrics.r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) = 0.948...
	approx(t, "R2", R2(p, y), 1-1.5/29.1875)
	approx(t, "constant R2", R2([]float32{1, 2}, []float32{1, 1}), math.NaN())
}
