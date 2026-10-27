package metrics

import (
	"fmt"
	"math"
	"slices"
)

// The functions below score a whole set of answers at once, as
// scikit-learn's metrics do, from plain slices such as a tabular.Model
// returns: p holds a probability for each class of each row, class by
// class, or one value per row for regression, and y the true class index
// or value of each row. Each is a tabular.Score.

// classes returns the number of classes in p for len(y) rows.
func classes(p, y []float32) int {
	if len(y) == 0 || len(p)%len(y) != 0 {
		panic(fmt.Sprintf("metrics: %d scores for %d rows", len(p), len(y)))
	}
	return len(p) / len(y)
}

// predicted returns the most likely class of each row of p.
func predicted(p []float32, k int) []int {
	c := make([]int, len(p)/k)
	for i := range c {
		row := p[i*k : (i+1)*k]
		for j, v := range row {
			if v > row[c[i]] {
				c[i] = j
			}
		}
	}
	return c
}

// ClassAccuracy returns the fraction of rows whose most likely class in
// p is their class in y.
func ClassAccuracy(p, y []float32) float64 {
	c := predicted(p, classes(p, y))
	hits := 0
	for i, t := range y {
		if c[i] == int(t) {
			hits++
		}
	}
	return float64(hits) / float64(len(y))
}

// Confusion returns the confusion matrix: m[t][c] counts the rows of
// class t whose most likely class in p is c.
func Confusion(p, y []float32) [][]int {
	k := classes(p, y)
	m := make([][]int, k)
	for t := range m {
		m[t] = make([]int, k)
	}
	for i, c := range predicted(p, k) {
		m[int(y[i])][c]++
	}
	return m
}

// PerClass returns the precision, recall and F1 score of each class: of
// the rows predicted to be the class, the share that are; of the rows of
// the class, the share predicted so; and their harmonic mean. A class
// never predicted, or never present, scores 0.
func PerClass(p, y []float32) (precision, recall, f1 []float64) {
	m := Confusion(p, y)
	k := len(m)
	precision, recall, f1 = make([]float64, k), make([]float64, k), make([]float64, k)
	for c := range k {
		var predicted, actual int
		for t := range k {
			predicted += m[t][c]
			actual += m[c][t]
		}
		if predicted > 0 {
			precision[c] = float64(m[c][c]) / float64(predicted)
		}
		if actual > 0 {
			recall[c] = float64(m[c][c]) / float64(actual)
		}
		if s := precision[c] + recall[c]; s > 0 {
			f1[c] = 2 * precision[c] * recall[c] / s
		}
	}
	return precision, recall, f1
}

// average returns the mean of v, or v[1] for two classes: the score of
// the positive class, as scikit-learn's average="binary".
func average(v []float64) float64 {
	if len(v) == 2 {
		return v[1]
	}
	var s float64
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// Precision returns the precision of class 1 for two classes, and the
// mean over the classes for more.
func Precision(p, y []float32) float64 {
	v, _, _ := PerClass(p, y)
	return average(v)
}

// Recall returns the recall of class 1 for two classes, and the mean over
// the classes for more.
func Recall(p, y []float32) float64 {
	_, v, _ := PerClass(p, y)
	return average(v)
}

// F1 returns the F1 score of class 1 for two classes, and the mean over
// the classes for more.
func F1(p, y []float32) float64 {
	_, _, v := PerClass(p, y)
	return average(v)
}

// LogLoss returns the mean negative log probability p gives each row's
// class, with probabilities clipped to [1e-15, 1].
func LogLoss(p, y []float32) float64 {
	k := classes(p, y)
	var s float64
	for i, t := range y {
		s -= math.Log(math.Max(float64(p[i*k+int(t)]), 1e-15))
	}
	return s / float64(len(y))
}

// ROCAUC returns the area under the ROC curve: for two classes, the
// chance that a random row of class 1 has a higher probability of class
// 1 than a random row of class 0, ties counting half; for more, the mean
// of each class's area against the rest. A class with no rows, or every
// row, is left out; with none left it returns NaN.
func ROCAUC(p, y []float32) float64 {
	k := classes(p, y)
	auc := func(c int) float64 {
		idx := make([]int, len(y))
		for i := range idx {
			idx[i] = i
		}
		slices.SortFunc(idx, func(a, b int) int {
			pa, pb := p[a*k+c], p[b*k+c]
			switch {
			case pa < pb:
				return -1
			case pa > pb:
				return 1
			}
			return 0
		})
		// The Mann-Whitney U statistic from the positives' mean ranks,
		// ties sharing their ranks.
		var pos, rankSum float64
		for lo := 0; lo < len(idx); {
			hi := lo
			for hi < len(idx) && p[idx[hi]*k+c] == p[idx[lo]*k+c] {
				hi++
			}
			rank := float64(lo+hi+1) / 2
			for _, i := range idx[lo:hi] {
				if int(y[i]) == c {
					pos++
					rankSum += rank
				}
			}
			lo = hi
		}
		neg := float64(len(y)) - pos
		if pos == 0 || neg == 0 {
			return math.NaN()
		}
		return (rankSum - pos*(pos+1)/2) / (pos * neg)
	}
	if k == 2 {
		return auc(1)
	}
	var s float64
	var n int
	for c := range k {
		if a := auc(c); !math.IsNaN(a) {
			s += a
			n++
		}
	}
	if n == 0 {
		return math.NaN()
	}
	return s / float64(n)
}

// MAE returns the mean absolute difference between p and y.
func MAE(p, y []float32) float64 {
	classes(p, y)
	var s float64
	for i, t := range y {
		s += math.Abs(float64(p[i]) - float64(t))
	}
	return s / float64(len(y))
}

// RMSE returns the square root of the mean squared difference between p
// and y.
func RMSE(p, y []float32) float64 {
	classes(p, y)
	var s float64
	for i, t := range y {
		d := float64(p[i]) - float64(t)
		s += d * d
	}
	return math.Sqrt(s / float64(len(y)))
}

// R2 returns the coefficient of determination: 1 less the squared error
// of p over that of always answering y's mean. It is 1 for perfect
// answers, 0 for the mean, and negative for worse; NaN when every y is
// the same.
func R2(p, y []float32) float64 {
	classes(p, y)
	var mean float64
	for _, t := range y {
		mean += float64(t)
	}
	mean /= float64(len(y))
	var res, tot float64
	for i, t := range y {
		d, e := float64(p[i])-float64(t), float64(t)-mean
		res += d * d
		tot += e * e
	}
	if tot == 0 {
		return math.NaN()
	}
	return 1 - res/tot
}
