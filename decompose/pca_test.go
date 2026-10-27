package decompose

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

func TestPCAFindsTheSharedDirection(t *testing.T) {
	// Features 0 and 1 move together; feature 2 is independent noise.
	r := rand.New(rand.NewPCG(1, 1))
	var x []float32
	for range 2000 {
		s := r.NormFloat64()
		x = append(x, float32(s+0.1*r.NormFloat64()), float32(1000*(s+0.1*r.NormFloat64())), float32(r.NormFloat64()))
	}
	p, err := PCA(context.Background(), x, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	d := p.Directions
	if math.Abs(float64(d[0])-math.Sqrt2/2) > 0.02 || math.Abs(float64(d[1])-math.Sqrt2/2) > 0.02 || math.Abs(float64(d[2])) > 0.1 {
		t.Errorf("first direction %v, want (0.71, 0.71, 0)", d[:3])
	}
	if math.Abs(float64(d[5])) < 0.99 {
		t.Errorf("second direction %v, want feature 2", d[3:])
	}
	if p.Explained[0] < 0.6 || p.Explained[0]+p.Explained[1] < 0.95 {
		t.Errorf("explained %v", p.Explained)
	}
	if got := p.Ranked(1)[0]; got != 2 {
		t.Errorf("feature %d weighs most in the second direction", got)
	}
	// Two directions keep almost everything: the round trip is close.
	back := p.Inverse(p.Transform(x[:30]))
	for i := range back {
		scale := []float64{1, 1000, 1}[i%3]
		if math.Abs(float64(back[i]-x[i]))/scale > 0.3 {
			t.Fatalf("value %d came back %v from %v", i, back[i], x[i])
		}
	}
}

func TestPCABadInput(t *testing.T) {
	for _, k := range []int{0, 3} {
		if _, err := PCA(context.Background(), []float32{1, 2, 3, 4}, 2, k); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("k %d: error %v", k, err)
		}
	}
}
