package rules

import (
	"context"
	"errors"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// shop returns n baskets where bread and butter go together, jam follows
// both, and the rest is noise.
func shop(n int, seed uint64) [][]string {
	r := rand.New(rand.NewPCG(seed, 5))
	noise := []string{"milk", "eggs", "tea", "soap", "rice", "beans", "salt"}
	var baskets [][]string
	for range n {
		var b []string
		if r.Float64() < 0.4 {
			b = append(b, "bread", "butter")
			if r.Float64() < 0.8 {
				b = append(b, "jam")
			}
		} else if r.Float64() < 0.3 {
			b = append(b, "bread")
		}
		for _, item := range noise {
			if r.Float64() < 0.3 {
				b = append(b, item)
			}
		}
		if r.Float64() < 0.1 {
			b = append(b, "jam", "jam") // repeats count once
		}
		baskets = append(baskets, b)
	}
	return baskets
}

func find(rules []Rule, ifs []string, then string) *Rule {
	for i := range rules {
		if slices.Equal(rules[i].If, ifs) && rules[i].Then == then {
			return &rules[i]
		}
	}
	return nil
}

func TestMineFindsWhatGoesTogether(t *testing.T) {
	baskets := shop(2000, 1)
	rules, err := Mine(context.Background(), baskets, MinSupport(0.05), MinConfidence(0.5))
	if err != nil {
		t.Fatal(err)
	}
	r := find(rules, []string{"bread", "butter"}, "jam")
	if r == nil {
		t.Fatalf("no rule bread, butter -> jam in %v", rules)
	}
	if r.Confidence < 0.7 || r.Lift < 1.5 || r.Support < 0.25 {
		t.Errorf("bread, butter -> jam: %+v", *r)
	}
	if r := find(rules, []string{"butter"}, "bread"); r == nil || r.Confidence < 0.95 {
		t.Errorf("butter -> bread should be near certain: %v", r)
	}
	if r := find(rules, []string{"milk"}, "bread"); r != nil && (r.Lift < 0.85 || r.Lift > 1.15) {
		t.Errorf("milk goes with nothing, yet milk -> bread has lift %v", r.Lift)
	}
	for _, item := range append(slices.Clone(rules[0].If), rules[0].Then) {
		if !slices.Contains([]string{"bread", "butter", "jam"}, item) {
			t.Errorf("the rule of highest lift is %+v", rules[0])
		}
	}
	for i := 1; i < len(rules); i++ {
		if rules[i].Lift > rules[i-1].Lift {
			t.Errorf("rules are not by falling lift at %d", i)
		}
	}
	top, err := Mine(context.Background(), baskets, MinSupport(0.05), MinConfidence(0.5), Top(2))
	if err != nil || len(top) != 2 || !slices.Equal(top[0].If, rules[0].If) {
		t.Errorf("Top(2) gave %v, %v", top, err)
	}
}

func TestMineBadInput(t *testing.T) {
	for _, opts := range [][]Option{{MinSupport(0)}, {MinSupport(2)}, {MinConfidence(-1)}, {MaxItems(1)}} {
		if _, err := Mine(context.Background(), shop(10, 1), opts...); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%v: %v", opts, err)
		}
	}
	if _, err := Mine(context.Background(), nil); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("no baskets: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Mine(ctx, shop(10, 1)); !errors.Is(err, context.Canceled) {
		t.Errorf("canceled: %v", err)
	}
	if rules, err := Mine(context.Background(), [][]string{{"a"}, {"b"}}); err != nil || len(rules) != 0 {
		t.Errorf("nothing goes together: %v, %v", rules, err)
	}
}
