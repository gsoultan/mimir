package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

// blobs returns n rows around each center, spread by std, and each row's
// center. Feature 1 is in units 100 times larger than feature 0.
func blobs(n int, centers [][2]float64, std []float64, seed uint64) ([]float32, []int) {
	r := rand.New(rand.NewPCG(seed, 2))
	var x []float32
	var y []int
	for c, ctr := range centers {
		for range n {
			x = append(x, float32(ctr[0]+std[c]*r.NormFloat64()), float32(100*(ctr[1]+std[c]*r.NormFloat64())))
			y = append(y, c)
		}
	}
	return x, y
}

var three = [][2]float64{{0, 0}, {6, 0}, {0, 6}}

// agree returns the share of rows whose group matches their center, under
// the best matching of groups to centers.
func agree(got, want []int, k int) float64 {
	count := map[[2]int]int{}
	for i := range got {
		count[[2]int{got[i], want[i]}]++
	}
	hits := 0
	used := map[int]bool{}
	for c := range k {
		best, bg := -1, -1
		for g := range k {
			if !used[g] && count[[2]int{g, c}] > best {
				best, bg = count[[2]int{g, c}], g
			}
		}
		used[bg] = true
		hits += best
	}
	return float64(hits) / float64(len(got))
}

func TestKMeansFindsTheBlobs(t *testing.T) {
	x, y := blobs(300, three, []float64{1, 1, 1}, 1)
	m, err := KMeans(context.Background(), x, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if a := agree(m.Assign(x), y, 3); a < 0.98 {
		t.Errorf("agreement %.3f", a)
	}
	if s := m.Silhouette(x, 1); s < 0.6 {
		t.Errorf("silhouette %.3f", s)
	}
	one, _ := KMeans(context.Background(), x, 2, 3, Inits(1), Seed(9))
	if one.Inertia < m.Inertia-1e-6 {
		t.Errorf("4 starts kept inertia %v, worse than one start's %v", m.Inertia, one.Inertia)
	}
}

func TestMixtureFindsUnevenBlobs(t *testing.T) {
	x, y := blobs(400, [][2]float64{{0, 0}, {7, 7}}, []float64{0.5, 2}, 3)
	m, err := GaussianMixture(context.Background(), x, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if a := agree(m.Assign(x), y, 2); a < 0.98 {
		t.Errorf("agreement %.3f", a)
	}
	if math.Abs(float64(m.Weights[0])-0.5) > 0.05 {
		t.Errorf("weights %v", m.Weights)
	}
	p := m.Probabilities(x[:2])
	if math.Abs(float64(p[0]+p[1])-1) > 1e-5 {
		t.Errorf("probabilities %v", p)
	}
	ll := m.LogDensity([]float32{0, 0, 30, 3000})
	if ll[1] >= ll[0] {
		t.Errorf("an outlier's log density %v is not below a typical row's %v", ll[1], ll[0])
	}
}

func TestDBSCANFindsGroupsAndNoise(t *testing.T) {
	x, y := blobs(200, [][2]float64{{0, 0}, {10, 10}}, []float64{0.5, 0.5}, 4)
	x = append(x, 5, 500, -8, 900) // two strays
	d, err := DBSCAN(context.Background(), x, 2, 0.3)
	if err != nil {
		t.Fatal(err)
	}
	if d.Groups != 2 {
		t.Fatalf("%d groups", d.Groups)
	}
	n := len(y)
	if d.Labels[n] != Noise || d.Labels[n+1] != Noise {
		t.Errorf("strays labelled %v", d.Labels[n:])
	}
	if a := agree(d.Labels[:n], y, 2); a < 0.95 {
		t.Errorf("agreement %.3f", a)
	}
	if got := d.Assign([]float32{10, 1000, 50, 50}); got[0] != d.Labels[n-1] || got[1] != Noise {
		t.Errorf("assigned %v", got)
	}
}

func TestModuleAnswersAsAssign(t *testing.T) {
	ctx := context.Background()
	x, _ := blobs(100, three, []float64{1, 1, 1}, 5)
	x[4] = float32(math.NaN())
	km, err := KMeans(ctx, x, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	gm, err := GaussianMixture(ctx, x, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct {
		mod    *Module
		assign []int
		probs  []float32
	}{
		"kmeans":  {km.Module(), km.Assign(x), nil},
		"mixture": {gm.Module(), gm.Assign(x), gm.Probabilities(x)},
	} {
		var buf bytes.Buffer
		if err := mim.Save(ctx, &buf, tc.mod, nil); err != nil {
			t.Fatal(err)
		}
		f, err := mim.Parse(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		mod := NewModule(tc.mod.Shape())
		if err := f.Load(mod); err != nil {
			t.Fatal(err)
		}
		f.Close()
		logits := mod.Forward(tensor.FromSlice(x, len(x)/2, 2))
		got, err := logits.Argmax(1).Data[int32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i, a := range tc.assign {
			if int(got[i]) != a {
				t.Fatalf("%s: row %d in group %d, want %d", name, i, got[i], a)
			}
		}
		if tc.probs != nil {
			p, err := logits.Softmax(1).Data[float32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			for i := range p {
				if math.Abs(float64(p[i]-tc.probs[i])) > 1e-3 {
					t.Fatalf("%s: probability %d is %v, want %v", name, i, p[i], tc.probs[i])
				}
			}
		}
	}
}

func TestJSONRoundTrip(t *testing.T) {
	x, _ := blobs(50, three, []float64{1, 1, 1}, 6)
	m, err := KMeans(context.Background(), x, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(m)
	var back Means
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m.Assign(x), back.Assign(x)) {
		t.Error("the decoded grouping assigns differently")
	}
}

func TestBadInput(t *testing.T) {
	ctx := context.Background()
	x := []float32{1, 2, 3, 4}
	for _, err := range []error{
		func() error { _, err := KMeans(ctx, x, 3, 1); return err }(),
		func() error { _, err := KMeans(ctx, x, 2, 3); return err }(),
		func() error { _, err := KMeans(ctx, x, 2, 1, Inits(0)); return err }(),
		func() error { _, err := GaussianMixture(ctx, x, 2, 0); return err }(),
		func() error { _, err := DBSCAN(ctx, x, 2, 0); return err }(),
	} {
		if !errors.Is(err, tensor.ErrShape) {
			t.Errorf("error %v, want ErrShape", err)
		}
	}
}
