package recommend

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// tastes returns events of users in four groups, each of which likes its
// own quarter of the items, and one held-out event per user.
func tastes(users, items, per int, seed uint64) (train, held []Event) {
	r := rand.New(rand.NewPCG(seed, 4))
	for u := range users {
		g := u % 4
		liked := r.Perm(items / 4)[:per+1]
		for i, it := range liked {
			e := Event{User: u, Item: g*(items/4) + it, Value: 1}
			if i == 0 {
				held = append(held, e)
			} else {
				train = append(train, e)
			}
		}
	}
	return train, held
}

func TestImplicitBeatsPopularity(t *testing.T) {
	train, held := tastes(400, 200, 8, 1)
	m, err := Fit(context.Background(), 400, 200, train, Factors(8))
	if err != nil {
		t.Fatal(err)
	}
	hr := m.HitRate(train, held, 10)
	// Within a group every unseen item is alike, so the best is 10 of a
	// group's 42 unseen items, 0.24; any item at random hits 10 of 192.
	if hr < 0.2 {
		t.Errorf("hit rate at 10: %.3f", hr)
	}
	sim := m.Similar(3, 5)
	for _, it := range sim {
		if it/50 != 0 {
			t.Errorf("items similar to 3: %v, want items of its group", sim)
			break
		}
	}
	rec := m.Recommend(0, 5, []int{1, 2})
	if len(rec) != 5 || slices.Contains(rec, 1) {
		t.Errorf("recommended %v", rec)
	}
}

func TestRatingsPredictHeldOut(t *testing.T) {
	// A rating is 3 plus the user's taste times the item's, plus noise.
	r := rand.New(rand.NewPCG(2, 2))
	tu, ti := make([]float64, 200), make([]float64, 100)
	for i := range tu {
		tu[i] = r.NormFloat64()
	}
	for i := range ti {
		ti[i] = r.NormFloat64()
	}
	var train, held []Event
	for u := range 200 {
		for _, it := range r.Perm(100)[:30] {
			e := Event{u, it, float32(3 + tu[u]*ti[it] + 0.1*r.NormFloat64())}
			if r.IntN(10) == 0 {
				held = append(held, e)
			} else {
				train = append(train, e)
			}
		}
	}
	m, err := Fit(context.Background(), 200, 100, train, Ratings(), Factors(4), L2(0.05))
	if err != nil {
		t.Fatal(err)
	}
	var se, base float64
	for _, e := range held {
		d := float64(m.Score(e.User, e.Item) - e.Value)
		se += d * d
		b := float64(m.Mean - e.Value)
		base += b * b
	}
	rmse, mean := math.Sqrt(se/float64(len(held))), math.Sqrt(base/float64(len(held)))
	if rmse > 0.5*mean {
		t.Errorf("held-out RMSE %.3f against %.3f for the mean rating", rmse, mean)
	}
}

func TestModuleScoresAsScore(t *testing.T) {
	ctx := context.Background()
	train, _ := tastes(40, 20, 3, 3)
	m, err := Fit(ctx, 40, 20, train, Factors(4))
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Module().Forward(tensor.FromSlice([]float32{7, 31}, 2, 1)).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i, u := range []int{7, 31} {
		for it := range 20 {
			if math.Abs(float64(out[i*20+it]-m.Score(u, it))) > 1e-5 {
				t.Fatalf("module score of item %d for user %d is %v, want %v", it, u, out[i*20+it], m.Score(u, it))
			}
		}
	}
}

func TestBadEvents(t *testing.T) {
	ctx := context.Background()
	for _, es := range [][]Event{nil, {{User: 5, Item: 0, Value: 1}}, {{User: 0, Item: 0, Value: -1}}} {
		if _, err := Fit(ctx, 2, 2, es); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%v: error %v", es, err)
		}
	}
}
