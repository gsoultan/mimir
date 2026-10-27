package loss

import (
	"context"
	"math"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

func value(t *testing.T, x *tensor.Tensor) float64 {
	t.Helper()
	v, err := x.Item[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return float64(v)
}

func near(t *testing.T, what string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-4*max(1, math.Abs(want)) {
		t.Errorf("%s = %v, want %v", what, got, want)
	}
}

func logSoftmax(row []float64) []float64 {
	m := math.Inf(-1)
	for _, v := range row {
		m = max(m, v)
	}
	var s float64
	for _, v := range row {
		s += math.Exp(v - m)
	}
	out := make([]float64, len(row))
	for i, v := range row {
		out[i] = v - m - math.Log(s)
	}
	return out
}

func TestClassLosses(t *testing.T) {
	logits := [][]float64{{2, 0.5, -1}, {0, 1, 3}}
	classes := []int32{0, 1}
	x := tensor.FromSlice([]float32{2, 0.5, -1, 0, 1, 3}, 2, 3)
	y := tensor.FromSlice(classes, 2)

	var ce, smooth, focal, hinge float64
	for i, row := range logits {
		lp := logSoftmax(row)
		c := classes[i]
		ce -= lp[c] / 2
		var mean float64
		for _, v := range lp {
			mean += v / 3
		}
		smooth += (-(1-0.1)*lp[c] - 0.1*mean) / 2
		p := math.Exp(lp[c])
		focal -= math.Pow(1-p, 2) * lp[c] / 2
		for j, v := range row {
			if j != int(c) {
				hinge += max(0, 1+v-row[c]) / 6
			}
		}
	}
	near(t, "cross entropy", value(t, CrossEntropy(x, y)), ce)
	near(t, "label smoothing", value(t, LabelSmoothing(0.1)(x, y)), smooth)
	near(t, "focal", value(t, Focal(2)(x, y)), focal)
	near(t, "focal 0", value(t, Focal(0)(x, y)), ce)
	near(t, "hinge", value(t, Hinge(x, y)), hinge)
}

func TestRegressionLosses(t *testing.T) {
	p := tensor.FromSlice([]float32{0, 2, -1, 0.5}, 4)
	y := tensor.FromSlice([]float32{0.2, 0, 1, 0.5}, 4)
	near(t, "smooth L1 beta 1", value(t, SmoothL1(1)(p, y)), value(t, Huber(p, y)))
	// |d| = 0.2, 2, 2, 0 with beta 0.5.
	near(t, "smooth L1 beta 0.5", value(t, SmoothL1(0.5)(p, y)), (0.04+1.75+1.75+0)/4)

	rate := tensor.FromSlice([]float32{0, 1}, 2)
	counts := tensor.FromSlice([]float32{2, 3}, 2)
	near(t, "poisson", value(t, PoissonNLL(rate, counts)), (1+(math.E-3))/2)

	mean := tensor.FromSlice([]float32{0, 1}, 2)
	variance := tensor.FromSlice([]float32{1, 4}, 2)
	near(t, "gaussian", value(t, GaussianNLL(mean, variance, counts)), (0.5*(0+4)+0.5*(math.Log(4)+1))/2)
}

func TestBinaryLosses(t *testing.T) {
	logits := []float64{1.5, -0.5, 0}
	target := []float64{1, 0, 1}
	x := tensor.FromSlice([]float32{1.5, -0.5, 0}, 3)
	y := tensor.FromSlice([]float32{1, 0, 1}, 3)
	var focal, bce float64
	for i, l := range logits {
		p := 1 / (1 + math.Exp(-l))
		ce := -(target[i]*math.Log(p) + (1-target[i])*math.Log(1-p))
		bce += ce / 3
		pt := p*target[i] + (1-p)*(1-target[i])
		a := 0.25*target[i] + 0.75*(1-target[i])
		focal += a * ce * math.Pow(1-pt, 2) / 3
	}
	near(t, "bce", value(t, BCE(x.Sigmoid(), y)), bce)
	near(t, "binary focal", value(t, BinaryFocal(2, 0.25)(x, y)), focal)
	near(t, "binary focal 0", value(t, BinaryFocal(0, -1)(x, y)), value(t, BCEWithLogits(x, y)))
}

func TestKLDiv(t *testing.T) {
	q := []float64{0.5, 0.5, 0.9, 0.1}
	p := []float64{0.25, 0.75, 1, 0}
	logq := tensor.FromSlice([]float32{float32(math.Log(q[0])), float32(math.Log(q[1])), float32(math.Log(q[2])), float32(math.Log(q[3]))}, 2, 2)
	target := tensor.FromSlice([]float32{0.25, 0.75, 1, 0}, 2, 2)
	var want float64
	for i := range p {
		if p[i] > 0 {
			want += p[i] * math.Log(p[i]/q[i]) / 2
		}
	}
	near(t, "kl", value(t, KLDiv(logq, target)), want)
}

func TestEmbeddingLosses(t *testing.T) {
	a := tensor.FromSlice([]float32{0, 0, 1, 0}, 2, 2)
	pos := tensor.FromSlice([]float32{3, 4, 1, 0}, 2, 2)
	neg := tensor.FromSlice([]float32{1, 0, 1, 2}, 2, 2)
	// Distances 5 and 1, then 0 and 2, with margin 1.
	near(t, "triplet", value(t, Triplet(1)(a, pos, neg)), (5.0+0)/2)

	u := tensor.FromSlice([]float32{1, 0, 1, 1}, 2, 2)
	v := tensor.FromSlice([]float32{0, 1, 1, 1}, 2, 2)
	lab := tensor.FromSlice([]float32{1, -1}, 2)
	near(t, "cosine embedding", value(t, CosineEmbedding(0)(u, v, lab)), (1+1)/2.0)

	s1 := tensor.FromSlice([]float32{1, 0}, 2)
	s2 := tensor.FromSlice([]float32{0, 0.5}, 2)
	ys := tensor.FromSlice([]float32{1, 1}, 2)
	near(t, "margin ranking", value(t, MarginRanking(1)(s1, s2, ys)), (0+1.5)/2)
}

func TestDice(t *testing.T) {
	// Logits so sure that the softmax is the one-hot of the classes.
	big := float32(30)
	x := tensor.FromSlice([]float32{big, 0, big, 0, 0, big, 0, big}, 1, 2, 4) // classes 0, 0, 1, 1... laid out [N, C, L]
	y := tensor.FromSlice([]int32{0, 1, 0, 1}, 1, 4)
	if got := value(t, Dice(x, y)); math.Abs(got) > 1e-3 {
		t.Errorf("dice of a perfect mask = %v", got)
	}
	wrong := tensor.FromSlice([]int32{1, 0, 1, 0}, 1, 4)
	if got := value(t, Dice(x, wrong)); got < 0.6 {
		t.Errorf("dice of the opposite mask = %v, want near 1", got)
	}
	m := tensor.FromSlice([]float32{big, -big, big, -big}, 4)
	if got := value(t, BinaryDice(m, tensor.FromSlice([]float32{1, 0, 1, 0}, 4))); math.Abs(got) > 1e-3 {
		t.Errorf("binary dice of a perfect mask = %v", got)
	}
}

func TestLossesLearn(t *testing.T) {
	// Each class loss has a gradient that points the right way.
	w := tensor.NewParam(tensor.Zeros(tensor.Float32, 2, 3))
	y := tensor.FromSlice([]int32{0, 2}, 2)
	for _, f := range []Func{LabelSmoothing(0.1), Focal(2), Hinge, Dice} {
		w.ZeroGrad()
		l := f(w.Value(), y)
		if err := l.Backward(context.Background()); err != nil {
			t.Fatal(err)
		}
		g, err := w.Grad().Data[float32](context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !(g[0] < 0 && g[5] < 0) {
			t.Errorf("gradient %v: want the true classes' logits to rise", g)
		}
	}
}
