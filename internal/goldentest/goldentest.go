// Package goldentest loads golden files written by tools/golden/gen.py and
// compares results against them.
package goldentest

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/io/safetensors"
)

// Case is one golden case: inputs in argument order and the expected output.
type Case struct {
	Inputs []safetensors.Tensor
	Want   safetensors.Tensor
}

// Dir returns the golden directory: $MIMIR_GOLDEN_DIR if set, so CI can test
// against freshly generated PyTorch goldens, else def.
func Dir(def string) string {
	if dir := os.Getenv("MIMIR_GOLDEN_DIR"); dir != "" {
		return dir
	}
	return def
}

// Load reads dir/name.safetensors and groups its tensors into cases. It also
// returns the file metadata, which names the oracle.
func Load(tb testing.TB, dir, name string) (map[string]Case, map[string]string) {
	tb.Helper()
	b, err := os.ReadFile(filepath.Join(dir, name+".safetensors"))
	if err != nil {
		tb.Fatal(err)
	}
	f, err := safetensors.Parse(b)
	if err != nil {
		tb.Fatal(err)
	}
	inputs := map[string]map[int]safetensors.Tensor{}
	cases := map[string]Case{}
	for _, full := range f.Names() {
		c, part, ok := strings.Cut(full, "/")
		if !ok {
			tb.Fatalf("golden tensor %q is not named <case>/<part>", full)
		}
		t, _ := f.Tensor(full)
		if part == "out" {
			cs := cases[c]
			cs.Want = t
			cases[c] = cs
			continue
		}
		i, err := strconv.Atoi(strings.TrimPrefix(part, "in"))
		if err != nil || !strings.HasPrefix(part, "in") {
			tb.Fatalf("golden tensor %q is not named <case>/in<N> or <case>/out", full)
		}
		if inputs[c] == nil {
			inputs[c] = map[int]safetensors.Tensor{}
		}
		inputs[c][i] = t
	}
	for c, cs := range cases {
		for i := range len(inputs[c]) {
			t, ok := inputs[c][i]
			if !ok {
				tb.Fatalf("golden case %q is missing in%d", c, i)
			}
			cs.Inputs = append(cs.Inputs, t)
		}
		cases[c] = cs
	}
	for c := range inputs {
		if _, ok := cases[c]; !ok {
			tb.Fatalf("golden case %q has no out", c)
		}
	}
	return cases, f.Metadata
}

// Tol bounds the allowed error: |got - want| <= Atol + Rtol*|want|.
type Tol struct {
	Atol, Rtol float64
}

// Compare fails tb unless got, of shape gotShape, matches want within tol.
// NaN must match NaN and infinities must match exactly. Int32 results must
// match exactly whatever tol says.
func Compare[T float32 | int32](tb testing.TB, want safetensors.Tensor, got []T, gotShape []int, tol Tol) {
	tb.Helper()
	if !slices.Equal(want.Shape, gotShape) {
		tb.Fatalf("shape %v, want %v", gotShape, want.Shape)
	}
	var w []float64
	switch any(got).(type) {
	case []float32:
		v, err := want.Float32s()
		if err != nil {
			tb.Fatal(err)
		}
		w = widen(v)
	default:
		v, err := want.Int32s()
		if err != nil {
			tb.Fatal(err)
		}
		w = widen(v)
		tol = Tol{}
	}
	g := widen(got)
	var bad []string
	worst := 0.0
	for i := range w {
		if within(g[i], w[i], tol) {
			continue
		}
		worst = max(worst, math.Abs(g[i]-w[i]))
		if len(bad) < 5 {
			bad = append(bad, fmt.Sprintf("[%d] got %v, want %v", i, g[i], w[i]))
		}
	}
	if bad != nil {
		tb.Errorf("mismatch (atol %g, rtol %g; worst abs error %g):\n  %s", tol.Atol, tol.Rtol, worst, strings.Join(bad, "\n  "))
	}
}

func within(got, want float64, tol Tol) bool {
	switch {
	case math.IsNaN(want) || math.IsNaN(got):
		return math.IsNaN(want) && math.IsNaN(got)
	case math.IsInf(want, 0) || math.IsInf(got, 0):
		return got == want
	}
	return math.Abs(got-want) <= tol.Atol+tol.Rtol*math.Abs(want)
}

func widen[T float32 | int32](v []T) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}
