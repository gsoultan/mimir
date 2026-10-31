package cpu

import (
	"math/rand/v2"
	"testing"
)

// TestWinoWeightsPanelMatchesGo checks the vector weight transform of a
// panel of 8 output channels against the Go one, over windows read
// through strides.
func TestWinoWeightsPanelMatchesGo(t *testing.T) {
	if winoWeightsPanel == nil {
		t.Skip("no vector weight transform on this CPU")
	}
	rng := rand.New(rand.NewPCG(3, 4))
	const cin, mr = 5, 8
	// Windows of 8 output channels and cin input channels, transposed in
	// memory: element (kh, kw) of (o, i) at i·so' + o·si' ... with the
	// window's rows and columns swapped too.
	data := make([]float32, mr*cin*9+7)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	w := winoFilters{data: data, off: 7, so: 9, si: mr * 9, sh: 1, sw: 3}
	c := winoConv{g: 1, cin: cin, cout: mr}
	xs := cin*mr + winoSkew
	want := make([]float32, 16*xs)
	for o := range mr {
		winoWeightsGo(want, xs, c, mr, w, 0, o)
	}
	got := make([]float32, 16*xs)
	winoWeightsPanel(&got[0], xs, &data[w.off], w.so, w.si, w.sh, w.sw, cin)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d: got %v, want %v", i, got[i], want[i])
		}
	}
}
